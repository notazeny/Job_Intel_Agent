"""生成候选人画像：`data/resume.md` → `data/profile.json`。

用法：
    python scripts/build_profile.py            # 有缓存就复用
    python scripts/build_profile.py --refresh  # 强制重新解析简历
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.profile import PROFILE_PATH, RESUME_PATH, build_profile  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="从简历文字版生成候选人画像")
    ap.add_argument("--refresh", action="store_true", help="忽略缓存，重新生成")
    args = ap.parse_args()

    if not RESUME_PATH.exists():
        print(f"找不到简历文字版：{RESUME_PATH}")
        print("先把简历图片转写成 markdown 放到这个路径。")
        return 1

    profile = build_profile(use_cache=not args.refresh)

    print(f"画像已写入 {PROFILE_PATH}")
    print(f"  技能      : {len(profile.skills)} 项")
    print(f"  项目      : {len(profile.project_highlights)} 条")
    print(f"  全职年限  : {profile.fulltime_years} 年")
    print(f"  学历      : {profile.education}（{profile.graduation_year} 届）")
    print(f"  期望城市  : {'、'.join(profile.target_cities)}")
    print(f"  期望薪资  : {profile.expected_salary}")
    print(f"  强项      : {'；'.join(profile.strengths[:3])}")
    print(f"  短板      : {'；'.join(profile.weaknesses[:3])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
