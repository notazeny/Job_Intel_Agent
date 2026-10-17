"""把「被判为近似重复」的岗位对摊开，供人工判断是真重复还是误杀。

## 为什么需要这个脚本

去重靠 SimHash 判「文本像不像」，但**「像」不等于「是同一个岗位」**。

现在比对**只在同一个桶内**进行（同公司 + 同可选城市集合 + 同标题），
跨城市的误杀已经不会再出现。所以剩下的近重复只有两种可能：

  · 同一个岗位被 HR 改了几个字重发  → **应该杀**
  · 两个不同岗位撞脸（比如字节所有 JD 套同一份模板）→ **误杀**

这个脚本就是把判断所需的材料摆到你眼前，让你用眼睛分辨这两者。

这个脚本**不改任何判重逻辑**，只负责把判断所需的材料摆到人眼前：
每一对的两边，各自的标题、城市、发布日、正文开头。

## 用法

    python scripts/inspect_near_dups.py            # 列出全部近似重复对
    python scripts/inspect_near_dups.py -n 5       # 只看前 5 对
    python scripts/inspect_near_dups.py -w 200     # 正文开头显示 200 字

## 数据从哪来（离线，不用联网）

采集时归档的原始响应：
    data/raw/bytedance/bytedance_Agent_0.json     (50 条)
    data/raw/bytedance/bytedance_Agent_50.json    (50 条)

文件结构：{"keyword": str, "offset": int, "body": {...}}
岗位列表在 body["data"]["job_post_list"]，每一项就是采集时的原始 dict。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collectors.base import RawPosting              # noqa: E402
from pipeline.dedup import dedup                    # noqa: E402
from pipeline.normalize import Posting, normalize   # noqa: E402


def load_archived_postings(raw_dir: Path) -> list[Posting]:
    """读 data/raw/ 下的归档，洗成 Posting 列表。**不去重**。"""
    # 找文件
    files = raw_dir.rglob("*.json")
    postings = []

    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        # 取岗位信息
        posts = d["body"]["data"]["job_post_list"]
        for p in posts:
            jd = RawPosting(
                source = "feishu",
                company = "bytedance",
                source_id = p["id"],
                url = f"https://jobs.bytedance.com/experienced/position/{p['id']}/detail",
                raw = p,
            )
            postings.append(normalize(jd)) 
    return postings


def print_pair(index: int, dropped: Posting, keeper: Posting, distance: int, width: int) -> None:
    """打出一对，供人工判断「机器判的重复到底对不对」。

    城市显示的是 `p.cities`（可选办公城市集合），**不是** `p.city`——
    `city` 是派生的单值，同一个岗位两次采集可能拿到不同的主城市；
    显示它反而会让你误以为「北京 vs 上海凭什么是重复」。
    真正决定分组的依据是 `cities`，所以要看到的是它。
    """
    def block(label: str, p: Posting) -> None:
        body = " ".join(p.description.split())[:width]
        cities = " / ".join(p.cities) if p.cities else "(无城市信息)"
        print(f"  {label}  {p.title}")
        print(f"        {cities} · {p.published_at[:10]} · 描述 {len(p.description)} 字")
        print(f"        {body}…")

    print(f"\n───── 第 {index} 对   距离 {distance} ─────")
    block("保留", keeper)
    block("被丢", dropped)


def main() -> int:
    ap = argparse.ArgumentParser(description="摊开近似重复对，供人工判断")
    ap.add_argument("-n", "--limit", type=int, default=0, help="只看前 N 对；0 = 全部")
    ap.add_argument("-w", "--width", type=int, default=120, help="正文开头显示多少字")
    ap.add_argument("--raw-dir", default="data/raw", help="归档目录（相对项目根）")
    args = ap.parse_args()

    postings = load_archived_postings(ROOT / args.raw_dir)
    print(f"读入 {len(postings)} 条原始岗位")

    result = dedup(postings)
    print(f"去重后唯一 {len(result.unique)} 条，"
          f"精确重复 {len(result.exact_duplicates)}，"
          f"近似重复 {len(result.near_duplicates)} 对")

    for i, r in enumerate(result.near_duplicates, 1):
        if args.limit and i > args.limit:
            break
        print_pair(i,r[0],r[1],r[2],args.width)

    if result.near_duplicates:
        print("请人工确认：这些是不是同一个岗位的重复发布？")
        print("  是   → 去重判对了")
        print("  不是 → 同一个桶里出现了两个不同岗位，说明 JD 模板太同质，该调 hamming_threshold")
    else:
        print("桶内没有近似重复（同公司 + 同城市集合 + 同标题的岗位，正文都不像）。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
