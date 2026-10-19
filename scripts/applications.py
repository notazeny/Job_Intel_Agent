"""投递记录 CLI —— 这个项目的「真实闭环」。

用法：
    # 投了一家之后记一笔
    python scripts/applications.py add --company bytedance \
        --title "AI Agent开发工程师" --url "https://..." --channel 官网

    # 看全部记录
    python scripts/applications.py list

    # 看回复率 / 面试率
    python scripts/applications.py stats

    # 收到回复后改状态（序号从 list 里看）
    python scripts/applications.py update 3 --status 已回复
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.applications import (  # noqa: E402
    STATUSES,
    Application,
    add,
    load_all,
    rewrite,
    stats,
)


def cmd_add(args: argparse.Namespace) -> int:
    add(Application(
        company=args.company,
        title=args.title,
        url=args.url,
        status=args.status,
        channel=args.channel,
        note=args.note,
    ))
    print(f"已记录：{args.company} · {args.title}（{args.status}）")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    apps = load_all()
    if not apps:
        print("还没有投递记录。投了一家后跑：")
        print("  python scripts/applications.py add --company ... --title ... --url ...")
        return 1

    print(f"{'#':>3}  {'日期':<11} {'状态':<7} {'公司':<12} {'岗位'}")
    for i, a in enumerate(apps, 1):
        print(f"{i:>3}  {a.applied_at:<11} {a.status:<7} {a.company:<12} {a.title[:40]}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    apps = load_all()
    s = stats(apps)
    if not s["total"]:
        print("还没有投递记录。")
        return 1

    print(f"总投递    : {s['total']} 家")
    print(f"有回音    : {s['responded']} 家（回复率 {s['response_rate']:.1%}）")
    print(f"拿到面试  : {s['interviewed']} 家（面试率 {s['interview_rate']:.1%}）")
    print()
    print("说明：分母是全部已投，不是「已回复的」——")
    print("      石沉大海的也得算进去，否则数字就没意义了。")
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    apps = load_all()
    if not 1 <= args.index <= len(apps):
        print(f"序号超出范围（当前共 {len(apps)} 条）")
        return 1
    old = apps[args.index - 1]
    apps[args.index - 1] = Application(
        company=old.company,
        title=old.title,
        url=old.url,
        applied_at=old.applied_at,
        status=args.status,
        channel=old.channel,
        note=args.note or old.note,
    )
    rewrite(apps)
    print(f"第 {args.index} 条状态：{old.status} → {args.status}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="投递记录")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="新增一条投递记录")
    p_add.add_argument("--company", required=True)
    p_add.add_argument("--title", required=True)
    p_add.add_argument("--url", required=True)
    p_add.add_argument("--status", default="已投", choices=STATUSES)
    p_add.add_argument("--channel", default="官网")
    p_add.add_argument("--note", default="")
    p_add.set_defaults(func=cmd_add)

    p_list = sub.add_parser("list", help="列出全部记录")
    p_list.set_defaults(func=cmd_list)

    p_stats = sub.add_parser("stats", help="回复率 / 面试率")
    p_stats.set_defaults(func=cmd_stats)

    p_upd = sub.add_parser("update", help="改某一条的状态")
    p_upd.add_argument("index", type=int, help="list 里的序号，从 1 开始")
    p_upd.add_argument("--status", required=True, choices=STATUSES)
    p_upd.add_argument("--note", default="")
    p_upd.set_defaults(func=cmd_update)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
