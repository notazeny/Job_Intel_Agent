"""落盘存储：JSONL。

## 为什么先用 JSONL 而不是直接上 MySQL

这一版的目标是**尽快验证「能不能拿到真实数据」**——
先让数据落地、能看、能数，再谈数据库。JSONL 有个额外好处：
**人能直接 `head -1` 看一条**，调试时比连数据库快得多。

存储层做了接口隔离（`append` / `load_all` / `known_ids`），
后面换 MySQL 时只改这个文件，pipeline 和 collectors 不用动。

## 读回来为什么要多绕一层 `_posting_from_dict`

不能直接写 `Posting(**json.loads(line))`，因为盘上的数据有两种会咬人的情形：

1. **schema 会变。** 早期落盘的行没有后来新增的字段（比如 `cities` /
   `is_overseas`），直接 `Posting(**d)` 会抛 TypeError——而 `load_all()`
   的 `except` 又会把它**静默跳过**：整库读出来是空的，还不报错。最难查的一种错。
2. **JSON 没有元组。** `cities` 是 `tuple`，写出去变成 `["深圳","广州"]`，
   读回来是 `list`——而 `list` **不能当 dict 的 key**
   （`pipeline/dedup.py` 里就拿它当分组 key，见那里 `seen_hash_votes`）。

所以还原这一步是**有意为之**，不是多余的封装。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from pipeline.normalize import Posting, content_fingerprint


def _posting_from_dict(raw: dict) -> Posting:
    """盘上的 dict → `Posting`，**顺便做向后兼容**（原因见模块头）。

    被 `load_all()` 逐行调用。
    """
    d = dict(raw)

    # ① 老行缺字段 → 补默认值，别让它 TypeError 后被静默吞掉
    d.setdefault("cities", ())
    d.setdefault("is_overseas", False)

    # ② JSON 把 tuple 还原成了 list → 转回来，否则不能当 dict key
    if isinstance(d.get("cities"), list):
        d["cities"] = tuple(d["cities"])

    # ③ 指纹**重算**，不信盘上那份。
    #    content_hash 是 (公司, 标题, 城市集合, 正文) 的纯函数，存盘属于冗余；
    #    而它算过一次就变过（加 cities 那次），旧行里是旧算法的值——
    #    跨批次去重拿它对不上新算的，会把老岗位当新增再写一遍。
    #    与其做数据迁移，不如读的时候重算，算法改动自动跟上。
    d["content_hash"] = content_fingerprint(
        d.get("company", ""),
        d.get("title", ""),
        d.get("cities", ()),
        d.get("description", ""),
        d.get("requirement", ""),
    )

    return Posting(**d)


class JsonlStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    # ---------- 读 ----------

    def load_all(self) -> list[Posting]:
        """读全部。数据量到十万级再换数据库，现在读全量最省事。"""
        if not self.path.exists():
            return []
        out: list[Posting] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(_posting_from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError):
                    # 单行坏了不该让整个文件读不出来——记下来继续
                    continue
        return out

    def known_source_ids(self) -> set[tuple[str, str, str]]:
        """已见过的 (source, company, source_id)。

        用它做**跨批次去重**：昨天拉过的岗位，今天不该再算「新增」。
        """
        return {(p.source, p.company, p.source_id) for p in self.load_all()}

    def known_content_hashes(self) -> set[str]:
        """已见过的内容指纹。`collectors/run.py` 的第 ③ 级去重就是查它。"""
        return {p.content_hash for p in self.load_all()}

    # ---------- 写 ----------

    def upsert(self, postings: Iterable[Posting]) -> tuple[int, int, int]:
        """按**业务主键** upsert：同主键的用新的覆盖旧的。

        返回 `(真正新增, 内容有更新, 合并后总数)`。

        ## 为什么需要它

        实测字节会给同一个岗位（正文一字不差）发**不同的 `source_id`**，
        所以「按 source_id 去重」在跨批次场景基本失效。真正稳定的标识是
        `Posting.business_key()`（公司 + 可选城市集合 + 标题）。

        同一个主键的新数据有两种可能：
          · 正文没变   → 纯重复，不计数
          · 正文变了   → **岗位信息更新了**，用新的覆盖旧的（这就是 upsert）

        ## 为什么是重写整个文件

        JSONL 只能追加，改一行就得整文件重写。现在几千条，O(n) 完全无所谓；
        以后上 MySQL 就是一句 `INSERT … ON DUPLICATE KEY UPDATE`。
        """
        incoming = list(postings)
        if not incoming:
            return 0, 0, len(self.load_all())

        # 先按主键收敛成一张表：后来的自然覆盖先来的
        existing = {p.business_key(): p for p in self.load_all()}

        added = updated = 0
        for p in incoming:
            key = p.business_key()
            old = existing.get(key)
            if old is None:
                added += 1
            elif old.content_hash != p.content_hash:
                # 主键相同但内容变了 = 岗位更新，计一次 update
                updated += 1
            existing[key] = p

        if added or updated:
            self._rewrite(list(existing.values()))
        return added, updated, len(existing)

    def _rewrite(self, postings: list[Posting]) -> None:
        """整文件重写（原子替换：先写 .tmp 再 rename，中途崩了不会毁库）。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            "".join(
                json.dumps(p.to_dict(), ensure_ascii=False) + "\n" for p in postings
            ),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def append(self, postings: Iterable[Posting]) -> int:
        """追加写入。返回写入条数。

        用追加而不是整体重写：日更场景下每天只多几百条，
        重写整个文件是 O(n) 的浪费，而且中途崩了会毁掉全部历史。
        """
        postings = list(postings)
        if not postings:
            return 0

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            for p in postings:
                f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
        return len(postings)

    def __iter__(self) -> Iterator[Posting]:
        return iter(self.load_all())

    def __len__(self) -> int:
        return len(self.load_all())
