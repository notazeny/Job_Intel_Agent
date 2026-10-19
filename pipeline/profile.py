"""简历画像：把 `data/resume.md` 变成 `CandidateProfile`。

## 为什么单独抽一层

匹配时需要的不是简历原文，而是**结构化的候选人特征**：
会什么技能、项目做了什么、几年经验、想要什么。

把它和「匹配打分」分开，好处是：
- 画像只生成一次，缓存到 `data/profile.json`，不用每条 JD 都重算
- 换匹配 prompt 时不用重新解析简历
- 简历更新后（`resume.md` 变新）缓存自动失效

## 一句关键约束

`sazeny` 是 **2026-06 毕业的应届生，0 年全职经验，但有多个完整项目**。
画像和匹配都必须能区分「全职年限」和「项目深度」——
否则所有 JD 都会因为「要求 3 年经验」被判死，而那恰好是这个项目要解决的问题。
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from .llm import call_structured

ROOT = Path(__file__).resolve().parent.parent
RESUME_PATH = ROOT / "data" / "resume.md"
PROFILE_PATH = ROOT / "data" / "profile.json"


class CandidateProfile(BaseModel):
    """候选人画像。字段刻意保持少而硬——下游要拿它和 JD 逐项比。"""
    skills: list[str] = Field(default_factory=list)
    project_highlights: list[str] = Field(default_factory=list)
    fulltime_years: float = 0.0          # 全职工作经验（实习不算）
    education: str = ""
    graduation_year: int | None = None
    target_cities: list[str] = Field(default_factory=list)
    target_roles: list[str] = Field(default_factory=list)
    expected_salary: str = ""
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)


SYSTEM_PROMPT = """你是简历解析器。从给定的简历文字里抽取候选人画像。

**只输出一个 JSON 对象**，不要解释，不要 markdown 代码块。字段如下：

{
  "skills": ["候选人掌握的硬技能，只写技术能力，不要写公司/项目名"],
  "project_highlights": ["每个项目一句话：做了什么 + 用了什么技术，不超过 40 字"],
  "fulltime_years": 0,
  "education": "本科" 或 "硕士" 或 "大专",
  "graduation_year": 2026,
  "target_cities": ["深圳"],
  "target_roles": ["AI Agent 开发", "AI 应用开发"],
  "expected_salary": "5-10K",
  "strengths": ["相对其他应届生的强项，3-5 条，必须能从简历里找到依据"],
  "weaknesses": ["从简历本身能看出的短板，3-5 条，比如无实习、项目时间短、非名校"]
}

规则：
- `fulltime_years` 只算**正式工作**。实习、课程项目、个人项目都不算。
- `strengths` / `weaknesses` 必须**基于简历内容**，不要编造，也不要写空话（如「学习能力强」）。
- `weaknesses` 要具体，例如「无实习经历」「求职方向与学历背景不完全匹配」。
"""


def _cache_is_fresh() -> bool:
    """profile.json 存在，且比 resume.md 新。"""
    return (
        PROFILE_PATH.exists()
        and RESUME_PATH.exists()
        and PROFILE_PATH.stat().st_mtime >= RESUME_PATH.stat().st_mtime
    )


def build_profile(*, use_cache: bool = True) -> CandidateProfile:
    """读 `data/resume.md` → `CandidateProfile`，结果缓存到 `data/profile.json`。"""
    if use_cache and _cache_is_fresh():
        return CandidateProfile.model_validate_json(
            PROFILE_PATH.read_text(encoding="utf-8")
        )

    if not RESUME_PATH.exists():
        raise FileNotFoundError(f"找不到简历文字版：{RESUME_PATH}")

    resume_text = RESUME_PATH.read_text(encoding="utf-8")
    obj, _usage = call_structured(SYSTEM_PROMPT, resume_text, CandidateProfile)

    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(obj.model_dump_json(indent=2), encoding="utf-8")
    return obj
