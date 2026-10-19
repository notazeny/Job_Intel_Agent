"""主流程：硬筛 → LLM 抽取 → LLM 匹配 → Markdown 报告。

用法：
    python scripts/match_jobs.py                 # 跑完整候选池
    python scripts/match_jobs.py -n 10           # 只跑前 10 条（试水，省 token）
    python scripts/match_jobs.py --cities 深圳 广州

## 流程里哪一步花 token

- 抽取：每条 JD 一次（**有磁盘缓存**，重跑不花钱）
- 匹配：每条 JD 一次（结果也缓存到 `data/matched/`）

所以第一次跑完整池需要几分钟，之后重跑几乎瞬时。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.extract import JobRequirement, extract_requirement  # noqa: E402
from pipeline.llm import LLMError, usage_summary  # noqa: E402
from pipeline.match import MatchResult, match  # noqa: E402
from pipeline.normalize import Posting  # noqa: E402
from pipeline.profile import CandidateProfile, build_profile  # noqa: E402
from storage.jsonl_store import JsonlStore  # noqa: E402

# 视为「技术岗」的类别。产品/设计/运营/UI 等不在内。
TECH_CATEGORIES = {
    "后端", "研发", "算法", "基础架构", "前端", "客户端",
    "机器学习", "数据", "测试", "全栈", "计算机视觉",
    "安全", "多媒体", "高级技术职位", "IT支持",
}

DEFAULT_CITIES = ["深圳", "广州", "佛山"]

CACHE_DIR = ROOT / "data" / "matched"


def candidate_pool(rows: list[Posting], cities: set[str]) -> list[Posting]:
    """硬筛：目标城市 ∩ 技术岗。**不按经验、不按时间过滤**（见 match.py 的说明）。"""
    return [p for p in rows if cities & set(p.cities) and p.category in TECH_CATEGORIES]


def match_with_cache(
    posting: Posting,
    req: JobRequirement,
    profile: CandidateProfile,
    *,
    use_cache: bool = True,
) -> MatchResult:
    cache = CACHE_DIR / f"{posting.content_hash}.json"
    if use_cache and cache.exists():
        return MatchResult.model_validate_json(cache.read_text(encoding="utf-8"))

    result = match(req, profile, job_title=posting.title)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result


def write_report(
    results: list[tuple[Posting, JobRequirement, MatchResult]],
    skipped: list[tuple[Posting, str]],
    profile: CandidateProfile,
    out_path: Path,
    cities: list[str],
) -> None:
    buckets: dict[str, list] = {"优先投": [], "可以投": [], "用来量差距": []}
    for item in results:
        buckets[item[2].fit_label].append(item)

    lines: list[str] = [
        "# 岗位匹配报告",
        "",
        f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}　"
        f"目标城市：{'、'.join(cities)}",
        "> 候选池：目标城市 × 技术岗。**经验年限不作为门槛**，只作为参考说明。",
        "",
        "## 概览",
        "",
        f"- 匹配成功：**{len(results)}** 条",
        f"- 优先投：**{len(buckets['优先投'])}** 条",
        f"- 可以投：**{len(buckets['可以投'])}** 条",
        f"- 用来量差距：**{len(buckets['用来量差距'])}** 条",
        f"- 排除/失败：{len(skipped)} 条",
        "",
    ]

    # LLM 用量（只统计本次进程实际发生的调用；缓存命中不产生调用）
    u = usage_summary()
    lines += [
        f"- LLM 调用：**{u['calls']}** 次　"
        f"输入 {u['prompt_tokens']:,} token（缓存命中 {u['cached_tokens']:,}）　"
        f"输出 {u['completion_tokens']:,} token",
        "",
    ]

    # ---- 高频缺口：跨所有岗位统计，直接回答「最该补什么」----
    gap_counter: Counter[str] = Counter()
    for _, _, m in results:
        gap_counter.update(m.gaps)
    if gap_counter:
        lines += [
            "## 高频缺口（跨所有岗位）",
            "",
            "> 缺口是 LLM 逐条写的，措辞不完全统一（「Go」和「Go 语言开发」会分开计），",
            "> 看数量级即可。",
            "",
        ]
        for gap, n in gap_counter.most_common(20):
            lines.append(f"- **{n} 次** — {gap}")
        lines.append("")

    for label in ("优先投", "可以投", "用来量差距"):
        items = sorted(buckets[label], key=lambda x: x[2].sort_score, reverse=True)
        if not items:
            continue
        lines += [f"## {label}（{len(items)} 条）", ""]
        for i, (p, req, m) in enumerate(items, 1):
            lines += [
                f"### {i}. {p.title}",
                "",
                f"- **公司**：{p.company}　**城市**：{' / '.join(p.cities)}　"
                f"**类别**：{p.category}",
                f"- **技能覆盖**：{m.skill_coverage}%　**排序分**：{m.sort_score}",
                f"- **经验说明**：{m.seniority_note or '—'}",
                f"- **理由**：{m.reason}",
            ]
            if m.gaps:
                lines.append("- **缺口**：")
                lines += [f"  - {g}" for g in m.gaps]
            if req.min_years is not None:
                lines.append(f"- **JD 要求年限**：{req.min_years} 年（仅参考）")
            lines.append(f"- **链接**：{p.url}")
            lines.append("")

    if skipped:
        lines += ["## 排除 / 失败", ""]
        for p, why in skipped[:20]:
            lines.append(f"- {p.title}（{why}）")
        lines.append("")

    lines += [
        "## 说明",
        "",
        "- `技能覆盖` 只看技能是否具备，和经验年限无关。",
        "- `缺口` 是可以直接拿去补的具体技能，不是泛泛的「需要提升」。",
        "- 本报告由 LLM 生成，**结论需要人工复核**再决定投不投。",
        "",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="岗位匹配：硬筛 → 抽取 → 匹配 → 报告")
    ap.add_argument("-n", "--limit", type=int, default=0, help="只跑前 N 条（0 = 全部）")
    ap.add_argument("--cities", nargs="+", default=DEFAULT_CITIES)
    ap.add_argument("--out", default="reports/w2_match.md")
    ap.add_argument("--no-cache", action="store_true", help="忽略抽取/匹配缓存")
    args = ap.parse_args()

    profile = build_profile()
    rows = JsonlStore(ROOT / "data" / "postings.jsonl").load_all()
    pool = candidate_pool(rows, set(args.cities))
    if args.limit:
        pool = pool[: args.limit]

    print(f"候选池 {len(pool)} 条（城市：{'、'.join(args.cities)}）")
    print(f"候选人：{profile.education} {profile.graduation_year} 届，"
          f"全职 {profile.fulltime_years} 年，期望 {'、'.join(profile.target_cities)}\n")

    results: list[tuple[Posting, JobRequirement, MatchResult]] = []
    skipped: list[tuple[Posting, str]] = []

    for i, p in enumerate(pool, 1):
        print(f"  [{i}/{len(pool)}] {p.title[:38]}", end=" ... ", flush=True)
        try:
            req = extract_requirement(p, use_cache=not args.no_cache)
            if req.ai_assist_banned:
                skipped.append((p, "JD 明确禁止 AI 辅助"))
                print("排除（禁止 AI 辅助）")
                continue
            m = match_with_cache(p, req, profile, use_cache=not args.no_cache)
            results.append((p, req, m))
            print(f"{m.fit_label}（覆盖 {m.skill_coverage}%）")
        except LLMError as exc:
            skipped.append((p, str(exc)[:60]))
            print(f"失败：{exc}")

    results.sort(key=lambda x: x[2].sort_score, reverse=True)

    out_path = ROOT / args.out
    write_report(results, skipped, profile, out_path, args.cities)

    u = usage_summary()
    print(f"\n完成：{len(results)} 条匹配，{len(skipped)} 条排除/失败")
    print(f"LLM ：{u['calls']} 次调用，输入 {u['prompt_tokens']:,} token"
          f"（缓存命中 {u['cached_tokens']:,}），输出 {u['completion_tokens']:,} token")
    print(f"报告：{out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
