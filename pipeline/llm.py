"""统一的 LLM 调用层。

复用自 `LangGraph_Agent/src/llm.py` 的 `call_llm`，但有四处刻意的改动：

1. **模型 / 地址从环境变量读**，不写死在 config.py（旧项目就是这么踩的：
   代码里写 `deepseek-v4-flash`、README 里写 `deepseek-chat`，两边不一致）
2. **内置重试**（旧项目没有，网络抖一下直接抛异常，批处理跑到一半全废）
3. **支持 JSON mode**（`response_format={"type":"json_object"}`），
   配合 `Structured_Output_Tool` 的解析层，比"提示词要求 JSON"稳得多
4. **同步实现**：W2 是批处理脚本，串行跑几百条，异步只会增加复杂度

## 配置（项目根目录 `.env`）

    DEEPSEEK_API_KEY=sk-...
    DEEPSEEK_BASE_URL=https://api.deepseek.com   # 可选
    DEEPSEEK_MODEL=deepseek-chat                 # 可选
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import TypeVar

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel

# 项目根目录的 .env（pipeline/ 的上一级）
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
TIMEOUT = float(os.getenv("LLM_TIMEOUT_SECONDS", "90"))
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))


class LLMError(RuntimeError):
    """LLM 调用失败。单独一个异常类型，方便上层统计「哪几条没抽出来」。"""


# ============================================================
# 用量记账
# ============================================================
# 每次成功的调用都往这里追加一条。脚本是单线程批处理，不需要锁；
# 如果以后要并发，换成 contextvar 或传一个 Stats 对象进来。
# **只记 token，不换算成钱**——各家定价会变，写死的汇率迟早是错的。

USAGE: list[dict] = []


def usage_summary() -> dict:
    """汇总本次进程内的 LLM 用量。"""
    prompt = sum(u.get("prompt_tokens", 0) for u in USAGE)
    completion = sum(u.get("completion_tokens", 0) for u in USAGE)
    cached = sum(u.get("prompt_cache_hit_tokens", 0) for u in USAGE)
    return {
        "calls": len(USAGE),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cached_tokens": cached,
    }


def call_llm(
    messages: list[dict],
    *,
    temperature: float = 0.0,
    model: str | None = None,
    json_mode: bool = False,
) -> tuple[str, dict]:
    """调用一次 chat completion，返回 `(文本, usage)`。

    - `json_mode=True` 时带 `response_format`，让模型尽量吐合法 JSON
    - 429 / 5xx / 网络错误重试（2、4、8 秒）；其他 4xx **立刻失败**——
      那说明请求本身写错了，重试一百次也一样
    """
    if not API_KEY:
        raise LLMError("没有找到 DEEPSEEK_API_KEY（检查项目根目录的 .env）")

    payload: dict = {
        "model": model or MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    last_err = "未知错误"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = httpx.post(
                f"{BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                usage = data.get("usage", {})
                USAGE.append(usage)
                return data["choices"][0]["message"]["content"], usage

            if resp.status_code not in (429, 500, 502, 503, 504):
                raise LLMError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            last_err = f"HTTP {resp.status_code}"

        except httpx.HTTPError as exc:
            last_err = f"{type(exc).__name__}: {exc}"

        if attempt < MAX_RETRIES:
            time.sleep(2 ** attempt)

    raise LLMError(f"重试 {MAX_RETRIES} 次仍失败：{last_err}")


# ============================================================
# 结构化输出：让模型吐 JSON 并校验成 Pydantic 模型
# ============================================================
# 三层管线（提取 → 解析 → 校验）来自 `Structured_Output_Tool`，
# 「校验失败把畸形输出回灌模型再试一次」的循环来自 `Multi_Agent/supervisor`。

T = TypeVar("T", bound=BaseModel)


def _extract_json(text: str) -> str:
    """从模型输出里抠出 JSON 对象（可能被 markdown 代码块包着）。"""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("输出里没有 JSON 对象")
    return text[start : end + 1]


def _loads_fixed(text: str) -> dict:
    """解析 JSON；失败时修一次尾逗号（模型最常见的格式毛病）。"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", text))


def call_structured(
    system: str,
    user: str,
    model_class: type[T],
    *,
    max_attempts: int = 2,
    temperature: float = 0.0,
) -> tuple[T, dict]:
    """调一次 LLM，把输出校验成 `model_class`，返回 `(对象, usage)`。

    校验失败时把畸形输出**原样回灌**，让它自己改——比重问一遍省 token。
    两次都失败就抛 `LLMError`，由调用方决定是跳过这一条还是整批中止。
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    last_error: Exception | None = None
    for _ in range(max_attempts):
        raw, usage = call_llm(messages, temperature=temperature, json_mode=True)
        try:
            return model_class.model_validate(_loads_fixed(_extract_json(raw))), usage
        except Exception as exc:                       # noqa: BLE001
            last_error = exc
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"上面的输出不合法：{exc}。请只输出一个合法的 JSON 对象。",
            })

    raise LLMError(f"结构化输出失败：{last_error}")
