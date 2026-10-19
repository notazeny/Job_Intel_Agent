"""岗位匹配：候选人画像 + JD 要求 → `MatchResult`。

## 设计上最关键的一条

**经验年限不是硬门槛。**

`sazeny` 是应届生，但「0 年全职经验」和「什么都不会」是两回事。
如果因为 JD 写了「3 年以上」就直接判死，这个项目就退化成「筛出应届岗」，
而市场上一大半有价值的岗位会被扔掉。

所以：
- `min_years` **不参与过滤，也不扣分**
- 经验差距只写进 `seniority_note` 一句话
- `skill_coverage` 只看**技能是否具备**，和年限无关

## 分数是分开的

- `skill_coverage`：技能覆盖度（报告正文展示）
- `sort_score`：仅用于排序（报告不单独展示）
- `fit_label`：给人看的结论

刻意**不合并成一个综合分**——「缺 3 个技能」和「差 2 年经验」是性质完全不同的两件事。
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from .extract import JobRequirement
from .llm import call_structured
from .profile import CandidateProfile


class MatchResult(BaseModel):
    skill_coverage: int = Field(ge=0, le=100)
    gaps: list[str] = Field(default_factory=list)
    seniority_note: str = ""
    fit_label: Literal["优先投", "可以投", "用来量差距"]
    sort_score: int = Field(ge=0, le=100)
    reason: str = ""


SYSTEM_PROMPT = """你是求职匹配分析器。下面是候选人画像和一个岗位的结构化要求。

**只输出一个 JSON 对象**，不要解释，不要 markdown 代码块。字段如下：

{
  "skill_coverage": 0-100 的整数,
  "gaps": ["候选人确实缺少的技能，具体到能去学的程度"],
  "seniority_note": "一句话说明经验方面的差距",
  "fit_label": "优先投" 或 "可以投" 或 "用来量差距",
  "sort_score": 0-100 的整数,
  "reason": "一句话说清为什么"
}

判断规则：
1. **经验不是硬门槛**。不要因为岗位要求 3 年经验就判「不够格」——
   候选人 0 年全职经验，但有多个完整项目。经验差距只写进 `seniority_note`，
   **不要**因此压低 `skill_coverage`。
2. `skill_coverage` 只看技能：岗位要求的技能里，候选人具备多少（按重要性加权）。
3. `gaps` 只写候选人**确实没有**的技能。他已经会的不要写进缺口。
   gaps 要具体，比如「PyTorch 分布式训练」而不是「深度学习」。
4. `fit_label` 三档：
   - 「优先投」：技能覆盖 ≥70，且项目方向高度相关
   - 「可以投」：技能有缺口，但项目经历能撑住
   - 「用来量差距」：缺口大或方向不匹配，但 gaps 清单对补技能有价值
5. `sort_score` 综合技能覆盖、项目相关性和岗位方向，用于排序。
"""


def match(
    job: JobRequirement,
    profile: CandidateProfile,
    *,
    job_title: str = "",
) -> MatchResult:
    """单条岗位的匹配结果。"""
    payload = {
        "job_title": job_title,
        "job_requirement": job.model_dump(),
        "candidate": profile.model_dump(),
    }
    user = json.dumps(payload, ensure_ascii=False, indent=2)

    obj, _usage = call_structured(SYSTEM_PROMPT, user, MatchResult)
    return obj
