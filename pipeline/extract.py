"""JD 结构化抽取：把一段岗位正文变成 `JobRequirement`。

## 为什么这里这么短

「调模型 → 抠 JSON → 修格式 → 校验 → 失败回灌重试」这套脏活全在
`pipeline/llm.py::call_structured()` 里，本模块只负责两件事：
**写好 prompt**、**管好缓存**。

## 为什么抽取结果要缓存

同一条 JD 在调 prompt、重跑流程时会被抽很多次。
按 `content_hash` 落盘到 `data/extracted/`，重跑直接读缓存，不重复花钱。
缓存 key 是内容指纹，**JD 内容变了缓存自动失效**。
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from .llm import call_structured
from .normalize import Posting

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "extracted"

# JD 截断长度：6000 字符 ≈ 3000 token，正文再长也够用
MAX_CHARS = 6000


class JobRequirement(BaseModel):
    """从一段 JD 里抽出来的结构化要求。"""
    skills: list[str] = Field(default_factory=list)
    min_years: float | None = None            # 最低年限；None = JD 没写
    education: str | None = None
    responsibilities: list[str] = Field(default_factory=list)
    bonus_points: list[str] = Field(default_factory=list)
    accepts_fresh_grad: bool | None = None    # 明确接受应届？None = 没提
    ai_assist_banned: bool = False            # 明确禁止 AI 辅助


SYSTEM_PROMPT = """你是招聘 JD 解析器。从给定的岗位描述里抽取结构化信息。

**只输出一个 JSON 对象**，不要解释，不要 markdown 代码块。字段如下：

{
  "skills": ["硬技能，如 Python / PyTorch / RAG / Kubernetes"],
  "min_years": 3 或 null,
  "education": "本科" 或 null,
  "responsibilities": ["岗位职责要点，每条不超过 20 字"],
  "bonus_points": ["加分项 / 优先条件"],
  "accepts_fresh_grad": true / false / null,
  "ai_assist_banned": true / false
}

规则：
- `skills` 只抽**技术能力**。公司名、团队名、产品名、业务名词（如「抖音」「火山引擎」）不算技能。
- `min_years`：文中说「3 年以上」就填 3；没写年限填 null，**不要猜**。
- `education`：文中说「本科及以上」填「本科」；没说填 null。
- `accepts_fresh_grad`：**明确**写「应届生」「欢迎优秀应届生」才填 true；
  明确要求工作经验就填 false；没提填 null。
- `ai_assist_banned`：明确禁止使用 AI 编程助手 / ChatGPT / Copilot 才填 true；没提填 false。
"""


def extract_requirement(posting: Posting, *, use_cache: bool = True) -> JobRequirement:
    """一条岗位 → `JobRequirement`。带磁盘缓存。

    失败时抛 `LLMError`，由调用方统计「哪几条没抽出来」，不要中断整批。
    """
    cache = CACHE_DIR / f"{posting.content_hash}.json"
    if use_cache and cache.exists():
        return JobRequirement.model_validate_json(cache.read_text(encoding="utf-8"))

    doc = (
        f"标题：{posting.title}\n\n"
        f"岗位描述：\n{posting.description}\n\n"
        f"任职要求：\n{posting.requirement}"
    )[:MAX_CHARS]

    obj, _usage = call_structured(SYSTEM_PROMPT, doc, JobRequirement)

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(obj.model_dump_json(indent=2), encoding="utf-8")
    return obj
