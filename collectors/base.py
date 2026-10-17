"""采集器基类：把「怎么礼貌地拿数据」和「怎么解析某一家」分开。

## 为什么要这个基类

限频、退避、超时、原始响应归档——这些是**每个采集器都必须遵守**的东西。
如果让每个子类各自实现，早晚有一家会漏掉一条。
所以规矩收拢在这里强制执行，子类只负责两件事：
**怎么发请求、怎么把响应解析成 RawPosting。**

## 合规约定（README 里有完整版）

- 只打公司招聘官网**自身页面在调用**的公开接口
- **严格串行**，两次请求之间至少 `min_interval` 秒（默认 3 秒），绝不并发
- 响应**全量归档**到 `data/raw/`：一是 schema 变了能重放，二是避免重复拉
- 只取公开岗位信息，**不碰任何候选人个人信息**
- 不在仓库里转发原始 dump；下游报告保留原始 URL 做归属

## 关于 User-Agent（一句必须说清的话）

字节的 WAF 实测**只认 macOS 浏览器 UA**：诚实 UA / curl / python-requests /
Windows Chrome UA 一律 405（见 README 的实测表）。

所以在这里，「拿到数据」和「诚实声明自己是谁」这两件事**无法同时做到**。
我们的选择是：用浏览器 UA 换取访问，同时用**更严格的自我约束**补偿——
严格串行、每次 ≥3 秒、全量归档只拉增量、只取公开岗位信息。
这是一个有意识的取舍，不是随手抄的 UA 字符串。
"""
from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import requests

logger = logging.getLogger(__name__)

# 实测唯一能过字节 WAF 的 UA 形态。见模块文档里那段说明。
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


class CollectorError(RuntimeError):
    """采集失败。单独一个异常类型，便于上层区分「这家挂了」和「我们写错了」。"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class RawPosting:
    """采集器吐出的**原始**岗位。

    字段刻意贴各家的原始命名——归一化是 pipeline 的事，不是采集器的。
    采集器只保证「拿到了什么、从哪拿的、什么时候拿的」，
    这样 schema 变了只需要重放归档，不用重新爬。
    """
    source: str            # 适配器名，如 "feishu"
    company: str           # 公司标识，如 "bytedance"
    source_id: str         # 该源内唯一 ID
    url: str               # 可点开的原始页面
    raw: dict[str, Any] = field(default_factory=dict)   # 原始 JSON，归档用
    fetched_at: str = field(default_factory=now_iso)


class BaseCollector(ABC):
    """所有采集器的父类：限频、退避、超时、归档都在这里，子类不用再操心。"""

    source: str = ""

    def __init__(
        self,
        *,
        min_interval: float = 3.0,
        timeout: float = 30.0,
        max_retries: int = 3,
        archive_dir: Path | str | None = None,
        user_agent: str = BROWSER_UA,
    ) -> None:
        self.min_interval = min_interval
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent
        self.archive_dir = Path(archive_dir) if archive_dir else None
        self._last_request_at = 0.0
        self._session = requests.Session()
        # 统计，供 run 汇总用
        self.stats = {"requests": 0, "retries": 0}

    # ---------- 子类要实现的两个方法 ----------

    @abstractmethod
    def fetch(self, keyword: str, limit: int = 50) -> Iterator[RawPosting]:
        """按关键词拉岗位，逐条 yield。"""
        raise NotImplementedError

    # ---------- 给子类用的请求工具 ----------

    def default_headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": self.user_agent,
        }

    def _throttle(self) -> None:
        """串行限频：离上次请求不够 min_interval 就睡够再走。

        这是本采集器最重要的自我约束——宁可慢，不可把别人的站点压着打。
        """
        elapsed = time.monotonic() - self._last_request_at
        wait = self.min_interval - elapsed
        if wait > 0:
            time.sleep(wait)

    def post_json(self, url: str, payload: dict, headers: dict | None = None) -> dict:
        """带限频 + 指数退避的 POST，返回解析后的 JSON。

        重试策略：429 / 5xx 值得重试（服务端忙）；其他 4xx **立刻失败不重试**——
        那说明我们的请求写错了，重试一百次也一样，只会白等。
        """
        merged = {**self.default_headers(), **(headers or {})}
        last_err = "未知错误"

        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                resp = self._session.post(
                    url, json=payload, headers=merged, timeout=self.timeout
                )
                self.stats["requests"] += 1
                self._last_request_at = time.monotonic()

                if resp.status_code == 200:
                    return resp.json()

                if resp.status_code not in (429, 500, 502, 503, 504):
                    raise CollectorError(
                        f"{url} 返回 HTTP {resp.status_code}（客户端错误，不重试）"
                    )
                last_err = f"HTTP {resp.status_code}"

            except requests.RequestException as exc:
                self._last_request_at = time.monotonic()
                last_err = f"{type(exc).__name__}: {exc}"

            if attempt < self.max_retries:
                backoff = 2 ** attempt          # 2、4、8 秒
                self.stats["retries"] += 1
                logger.warning(
                    "第 %d 次失败（%s），%.0f 秒后重试", attempt, last_err, backoff
                )
                time.sleep(backoff)

        raise CollectorError(f"{url} 重试 {self.max_retries} 次仍失败：{last_err}")

    # ---------- 原始响应归档 ----------

    def archive(self, name: str, payload: dict) -> None:
        """把原始响应落盘，**按采集日期分目录**：
        `<archive_dir>/<YYYY-MM-DD>/<name>.json`。

        两个用途：
          1. schema 变了（比如飞书改了字段名）→ 能重放，不用重新爬
          2. 出问题时，原始数据是唯一的事实来源

        为什么不直接写 `<archive_dir>/<name>.json`：
        `replace()` 是原子替换，同名会被**静默覆盖**——同一个查询第二天再跑，
        「昨天的事实来源」就没了。后果是三重的：历史批次不能回放、
        读归档做的验收隔天不可复现、也没法用归档验证跨批次去重。
        日更场景下每天一份，互不干扰。
        """
        if not self.archive_dir:
            return
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = self.archive_dir / day / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(path)   # 原子替换：避免读到写了一半的文件
