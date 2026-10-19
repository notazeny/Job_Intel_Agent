"""采集入口：走一遍「采集 → 归一化 → 去重 → 落盘」，打印本次增量报告。

用法：
    python -m collectors.run                                  # 按 config+companies 全量跑
    python -m collectors.run --company bytedance              # 只跑一家
    python -m collectors.run --keyword Agent --max-pages 2    # 调试：只拉 2 页

⚠️ 第一次跑**务必加 `--max-pages`**。字节单关键词 count=2111，按 50 一页
   就是 43 次请求；限频 3 秒一条，全量要两分多钟。先拉两页看看数据长什么样，
   确认没问题再放开。
"""
from __future__ import annotations

import argparse
import logging
import sys
from itertools import islice
from pathlib import Path

import yaml

from pipeline.dedup import dedup
from pipeline.normalize import Posting, normalize
from storage.jsonl_store import JsonlStore

from .base import CollectorError
from .feishu import FeishuCollector

ROOT = Path(__file__).resolve().parent.parent

# 适配器注册表：加一家新后端只要往这里加一行
ADAPTERS = {
    "feishu": FeishuCollector,
}


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"配置文件不存在：{path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def build_collector(spec: dict, collect_cfg: dict, archive_dir: Path):
    adapter = spec.get("adapter")
    cls = ADAPTERS.get(adapter)
    if cls is None:
        raise CollectorError(
            f"未知适配器 {adapter!r}（公司 {spec.get('name')}）。"
            f"可选：{list(ADAPTERS)}"
        )

    common = {
        "min_interval": collect_cfg.get("min_interval_seconds", 3.0),
        "timeout": collect_cfg.get("timeout_seconds", 30),
        "max_retries": collect_cfg.get("max_retries", 3),
        "archive_dir": archive_dir,
    }
    if adapter == "feishu":
        return cls(
            host=spec["host"],
            company=spec["name"],
            portal_type=spec.get("portal_type", 6),
            job_path=spec.get("job_path"),
            **common,
        )
    return cls(company=spec["name"], **common)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="求职情报 Agent · 采集")
    ap.add_argument("--company", help="只跑 companies.yaml 里的某一家（按 name）")
    ap.add_argument("--keyword", help="只跑一个关键词（默认用 config.yaml 里的全部）")
    ap.add_argument(
        "--max-pages", type=int, default=0,
        help="每(公司×关键词)最多拉几页；0 = 不限。第一次跑请设成 1-2",
    )
    ap.add_argument("--dry-run", action="store_true", help="只拉不写盘")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)

    cfg = load_yaml(ROOT / "config.yaml")
    collect_cfg = cfg.get("collect", {})
    paths = cfg.get("paths", {})
    page_size = int(collect_cfg.get("page_size", 50))

    companies = load_yaml(ROOT / "companies.yaml").get("companies", [])
    if args.company:
        companies = [c for c in companies if c.get("name") == args.company]
        if not companies:
            raise SystemExit(f"companies.yaml 里没有 name={args.company!r}")

    keywords = [args.keyword] if args.keyword else collect_cfg.get("keywords", [])
    if not companies or not keywords:
        raise SystemExit("没有可跑的公司或关键词，检查 companies.yaml / config.yaml")

    store = JsonlStore(ROOT / paths.get("postings_file", "data/postings.jsonl"))
    archive_root = ROOT / paths.get("raw_archive_dir", "data/raw")

    before_total = len(store)

    print(f"\n目标：{len(companies)} 家公司 × {len(keywords)} 个关键词")
    print(f"限频：每请求 ≥{collect_cfg.get('min_interval_seconds', 3.0)} 秒（串行）")
    print(f"已有记录：{before_total} 条\n")

    fetched: list[Posting] = []
    errors: list[str] = []
    request_count = 0

    for spec in companies:
        label = spec.get("label", spec["name"])
        try:
            collector = build_collector(spec, collect_cfg, archive_root / spec["name"])
        except Exception as exc:                      # noqa: BLE001
            errors.append(f"{label}: 适配器构造失败：{exc}")
            continue

        for kw in keywords:
            print(f"  → 拉取 {label} × 「{kw}」 …")
            try:
                stream = collector.fetch(kw, limit=page_size)
                if args.max_pages:
                    stream = islice(stream, args.max_pages * page_size)
                for raw in stream:
                    fetched.append(normalize(raw))
            except CollectorError as exc:
                errors.append(f"{label} × {kw}: {exc}")
                print(f"     ✗ 失败：{exc}")
            except Exception as exc:                  # noqa: BLE001
                errors.append(f"{label} × {kw}: {type(exc).__name__}: {exc}")
                print(f"     ✗ 异常：{type(exc).__name__}: {exc}")
            finally:
                request_count += collector.stats["requests"]

    # ---------- 批内去重 ----------
    # ⚠️ 这里**不再按 source_id 过滤**。实测字节会给同一个岗位（正文一字不差）
    # 发不同的 id —— `source_id` 标识的是「一次发布」，不是「一个岗位」。
    # 跨批次是不是同一个岗位，交给下面的 upsert 按业务主键（公司+城市集合+标题）判。
    intra = dedup(fetched)

    print("\n" + "=" * 62)
    print(f"本次请求 {request_count} 次；抓到 {len(fetched)} 条")
    print(
        f"  批内去重  -{len(fetched) - len(intra.unique):>4}  → {len(intra.unique)}"
        f"（精确重复 {len(intra.exact_duplicates)}，近似重复 {len(intra.near_duplicates)}）"
    )

    if intra.near_duplicates:
        print("\n  近似重复样例（人工确认一下是不是真重复）：")
        for dropped, keeper, dist in intra.near_duplicates[:3]:
            print(f"    距离{dist}  「{dropped.title[:28]}」 ≈ 已收「{keeper.title[:28]}」")

    if errors:
        print(f"\n  ⚠️ {len(errors)} 处失败：")
        for e in errors:
            print(f"    · {e}")

    if args.dry_run:
        print("\n  (--dry-run，未写盘)")
        return 0

    added, updated, total = store.upsert(intra.unique)
    print(f"\n  ✅ 新增 {added} 条，更新 {updated} 条，库内共 {total} 条")
    print(f"已写入 → {store.path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
