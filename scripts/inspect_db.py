"""看一眼库里最近的数据。

这是验证「真的采到了东西」最快的方式——**不用连数据库、不用起服务**，
直接把人能读的字段打出来，并且给出原始 URL（能点开才算真数据）。

用法：
    python scripts/inspect_db.py            # 最近 10 条
    python scripts/inspect_db.py -n 5       # 最近 5 条
    python scripts/inspect_db.py --stats    # 只出统计
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.freshness import MAX_AGE_DAYS, is_fresh  # noqa: E402
from storage.jsonl_store import JsonlStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="查看采集到的岗位")
    ap.add_argument("-n", "--limit", type=int, default=10)
    ap.add_argument("--stats", action="store_true", help="只出统计，不列岗位")
    args = ap.parse_args()

    store = JsonlStore(ROOT / "data" / "postings.jsonl")
    postings = store.load_all()

    if not postings:
        print("库里还没有数据。先跑：python -m collectors.run --max-pages 1")
        return 1

    print(f"总计 {len(postings)} 条\n")
    print(f"  公司  : {dict(Counter(p.company for p in postings).most_common())}")
    print(f"  城市  : {dict(Counter(p.city for p in postings).most_common(8))}")
    print(f"  类型  : {dict(Counter(p.recruit_type for p in postings).most_common())}")
    print(f"  大类  : {dict(Counter(p.category for p in postings).most_common(6))}")
    published = [p.published_at[:10] for p in postings if p.published_at]
    if published:
        print(f"  发布日: {min(published)} ~ {max(published)}")

    # 时间衰减：库里全留，但推荐时只放行 MAX_AGE_DAYS 以内的
    fresh_n = sum(1 for p in postings if is_fresh(p.published_at))
    print(
        f"  时效  : {MAX_AGE_DAYS:.0f} 天内 {fresh_n} 条 / "
        f"更老 {len(postings) - fresh_n} 条（推荐时过滤更老的）"
    )

    if args.stats:
        return 0

    print(f"\n最近 {args.limit} 条：\n")
    for p in postings[-args.limit:]:
        print(f"  [{p.company}] {p.title}")
        print(f"      {p.city} · {p.category} · {p.recruit_type} · 发布 {p.published_at[:10]}")
        print(f"      描述 {len(p.description)} 字 / 要求 {len(p.requirement)} 字")
        print(f"      {p.url}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
