"""LLM 层的纯函数与重试逻辑单测。**不联网**：`call_llm` 被 monkeypatch 掉。

这一层最该锁住的是「模型输出不合法时，流程不能整批崩」——
抽取几百条 JD，只要有 1 条判死就浪费一次配额，有 10 条就让整批重跑。
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from pipeline.llm import LLMError, _extract_json, _loads_fixed, call_structured


class Demo(BaseModel):
    name: str
    score: int


# ---------- JSON 提取 ----------

def test_extract_json_plain():
    assert _extract_json('{"a": 1}') == '{"a": 1}'


def test_extract_json_inside_markdown_fence():
    text = '```json\n{"a": 1}\n```'
    assert _extract_json(text) == '{"a": 1}'


def test_extract_json_ignores_surrounding_prose():
    text = '好的，结果如下：{"a": 1} 希望有帮助。'
    assert _extract_json(text) == '{"a": 1}'


def test_extract_json_missing_raises():
    with pytest.raises(ValueError):
        _extract_json("里面没有 JSON")


# ---------- 尾逗号修复 ----------

def test_loads_fixed_trailing_comma_object():
    assert _loads_fixed('{"a": 1,}') == {"a": 1}


def test_loads_fixed_trailing_comma_nested():
    assert _loads_fixed('{"a": [1, 2,],}') == {"a": [1, 2]}


# ---------- 结构化输出重试 ----------

def test_call_structured_retries_after_validation_error(monkeypatch):
    """第一次缺字段 → 校验失败 → 把错误回灌 → 第二次成功。"""
    outputs = iter(['{"name": "x"}', '{"name": "x", "score": 9}'])
    monkeypatch.setattr("pipeline.llm.call_llm", lambda *a, **k: (next(outputs), {}))

    obj, _usage = call_structured("sys", "user", Demo)
    assert obj.score == 9


def test_call_structured_raises_after_exhausting_attempts(monkeypatch):
    """两次都不合法 → 抛 LLMError，让上层跳过这一条。"""
    monkeypatch.setattr("pipeline.llm.call_llm", lambda *a, **k: ('{"name": "x"}', {}))

    with pytest.raises(LLMError):
        call_structured("sys", "user", Demo, max_attempts=2)
