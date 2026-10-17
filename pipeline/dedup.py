"""去重：精确 + 近似，两级。

**全是规则，不用 LLM。** 让 LLM 去判断「这两条是不是同一个岗位」是纯浪费钱，
而且会引入不确定性——去重结果每次跑都不一样，那还怎么测？


## 数据在这条链上的形状变化

    list[Posting]                        ← 来自 pipeline/normalize.py
      ↓  dedup(postings)
           一级：精确 —— 内容指纹撞了就丢（O(1) 查表）
           二级：近似 —— 按 (公司, 城市集合, 标题) 开桶，**桶内**比正文的 SimHash
      ↓  DedupResult(unique / exact_duplicates / near_duplicates)
      ↓  storage/jsonl_store.py 落盘；scripts/inspect_near_dups.py 人工查验

本文件里的函数：

    dedup()        ← 主流程，两级去重（唯一有状态的地方）
      ├ simhash64()  ← 一段文本 → 64 位指纹（近似比较用的「钥匙」）
      └ hamming()    ← 两个指纹差几位（越小越像）

## 为什么需要两级

- **精确去重**（md5 内容指纹）：抓「同一条岗位被原样重发」——
  HR 复制粘贴直接再发一遍，内容**逐字相同**。最快也最准。
- **近似去重**（SimHash + 汉明距离）：抓「改了一点点的重复」——
  标题加了「（急招）」「-2026」、正文里换了几个词。
  这类重复 md5 抓不到，但它们占着一个候选位，会污染排序和评测。

## 近似去重为什么必须「先分组、再比正文」

SimHash 比的是「两段文本像不像」。**如果不分组**，字节的 JD 全都套同一份模板，
正文骨架一模一样 → 不同产品线（ArkClaw vs 火山引擎）的岗位会被判成重复，
实测 100 条里这样误杀 **13 对**。

所以先用**主键 `(公司, 城市集合, 标题)` 开桶**，只在桶内比正文：

    同一个桶（同公司 + 同可选城市 + 同标题）→ 才可能是同一岗位的重复发布
    不同桶                                  → 一定是不同岗位，根本不用比

⚠️ **桶的 key 只用来分组，绝不能喂进 SimHash。** 这里踩过坑：
曾经写成 `simhash64(f"{公司}{城市}{标题}")`，看着像「把城市带上了」，
实际上**正文被挤没了**——同桶任意两条的文本一字不差、距离恒为 0，
近似去重直接退化成「按主键再精确去重一次」，正文完全不参与。见 `dedup()` 里二级那段。

## 为什么用字符 n-gram 而不是分词

中文分词要引入 jieba，而 SimHash 只需要「把文本变成一堆带权 token」。
字符 3-gram 对中文的近重复检测效果足够好，而且**零依赖**——
少一个依赖就少一处版本坑（这个坑在 CI 里吃过一次了）。
"""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .normalize import Posting

# 经验值：64 位 SimHash 上汉明距离 ≤3 基本可以认定是近重复。
# 调大会误杀（把不同岗位判成重复），调小会漏杀。
# 注意：这个阈值只在**同一个桶内**生效，跨桶根本不比，所以不用为了防误杀调太小。
DEFAULT_HAMMING_THRESHOLD = 3


def simhash64(text: str, ngram: int = 3) -> int:
    """字符 n-gram + 频率加权的 SimHash，返回 64 位整数。

    原理：把每个 token 哈希成一个 64 位数，按位投票——
    该位是 1 就 +权重，是 0 就 -权重。最后看每位的正负决定结果。
    相近的文本，投票结果也相近。

    入参是**一段文本**（`dedup()` 里传的是 JD 正文），出参是 64 位整数指纹——
    「像不像」交给 `hamming()` 去量。文本短于 ngram 时返回 0（没法比）。
    """
    text = (text or "").strip()
    if len(text) < ngram:
        return 0

    grams = Counter(text[i : i + ngram] for i in range(len(text) - ngram + 1))
    if not grams:
        return 0

    votes = [0] * 64
    for gram, weight in grams.items():
        h = int.from_bytes(hashlib.md5(gram.encode("utf-8")).digest()[:8], "big")
        for bit in range(64):
            votes[bit] += weight if (h >> bit) & 1 else -weight

    out = 0
    for bit in range(64):
        if votes[bit] > 0:
            out |= 1 << bit
    return out


def hamming(a: int, b: int) -> int:
    """两个 SimHash 的汉明距离 = 有多少位不同。越小越像。

    `a ^ b` 把不同的位变 1，再数 1 的个数。被 `dedup()` 调用。
    """
    return bin(a ^ b).count("1")


@dataclass
class DedupResult:
    """去重结果。刻意把「被丢掉的是什么、因为什么丢的」留下来——
    只看一个「去重率 Y%」的数字，是没法判断去重做得对不对的。

    三个列表的**单位不一样**，用的时候别搞混：

    - `unique`           每条是一个岗位
    - `exact_duplicates` 每条是一个岗位（被丢的那个）
    - `near_duplicates`  每条是**一对** `(被丢的, 撞上的那个, 汉明距离)`
    """
    unique: list[Posting]
    exact_duplicates: list[Posting]
    near_duplicates: list[tuple[Posting, Posting, int]]   # (被丢的, 撞上的那个, 汉明距离)

    @property
    def total(self) -> int:
        """去重前的总条数。

        `near_duplicates` 存的是「对」，但每对里**恰好只有 1 条**是被丢的
        （`dedup()` 里一个岗位最多进一对），所以对数 == 被丢的条数。
        """
        return len(self.unique) + len(self.exact_duplicates) + len(self.near_duplicates)

    @property
    def dedup_rate(self) -> float:
        return 0.0 if self.total == 0 else (self.total - len(self.unique)) / self.total

    def summary(self) -> str:
        return (
            f"共 {self.total} 条 → 唯一 {len(self.unique)} 条"
            f"（精确重复 {len(self.exact_duplicates)}，"
            f"近似重复 {len(self.near_duplicates)}），"
            f"去重率 {self.dedup_rate:.1%}"
        )


# 分组 key 的类型别名：公司 + 可选城市集合 + 标题 = 岗位主键
# 单独起个名字，是为了让下面两张账本的类型标注能看懂
GroupKey = tuple[str, tuple[str, ...], str]


def dedup(
    postings: Iterable[Posting],
    *,
    hamming_threshold: int = DEFAULT_HAMMING_THRESHOLD,
) -> DedupResult:
    """两级去重。**顺序有讲究**：先精确（O(1) 查表），再近似（要跟同桶的比）。

    内部就两张账本，别被名字绕晕：

        seen_hash        : 内容指纹 → 已收下的岗位                （一级用）
        seen_hash_votes  : 分组 key → 该桶的 [(正文指纹, 岗位), …] （二级用）

    分组 key = `(company, cities, title)`，也就是岗位主键。

    ⚠️ `seen_hash_votes` 必须是 **dict**（一个桶一个列表），**不能是单个 list**——
    否则不同分组的指纹混进同一个池子，等于没分组，跨产品线的误杀立刻回来。

    入参：归一化后的 `Posting` 列表（来自 pipeline/normalize.py）
    出参：`DedupResult`
    """
    unique: list[Posting] = []
    exact: list[Posting] = []
    near: list[tuple[Posting, Posting, int]] = []

    # 一级账本：内容指纹 → 已收岗位
    seen_hash: dict[str, Posting] = {}
    # 二级账本：分组 key → 该桶已收岗位的 (SimHash, 岗位)
    # 「桶」这个东西就是这一层 dict 存在的全部理由，别简化成 list
    seen_hash_votes: dict[GroupKey, list[tuple[int, Posting]]] = {}

    for p in postings:
        # ---- 一级：精确 ----
        # 指纹在 normalize 里就算好了（公司+标题+城市集合+正文 的 md5）。
        # 撞上说明是同一岗位的逐字重复，直接丢。
        if p.content_hash in seen_hash:
            exact.append(p)
            continue

        # ---- 二级：近似 ----
        # 只拿**正文**算指纹：公司/城市/标题已经在分组 key 里了，
        # 再喂进 SimHash 是重复劳动，而且会把正文挤掉（见模块头那个坑）。
        votes = simhash64(f"{p.description}\n{p.requirement}")

        # 开桶：同 key 的共用一张比较清单
        # key 用 `p.business_key()`，不用 `(company, cities, title)` 手拼——
        # 主键的定义只应该有一处（在 Posting 上），否则两边早晚会漂移。
        bucket = seen_hash_votes.setdefault(p.business_key(), [])

        matched = False
        if votes:
            # 只跟**同一个桶**里的比，跨桶的一定是不同岗位
            for other_votes, other in bucket:
                dist = hamming(votes, other_votes)
                if dist <= hamming_threshold:
                    near.append((p, other, dist))
                    matched = True
                    break

        if matched:
            # 判为近似重复，丢；它不进桶，也不进 unique
            continue

        # 没撞上（或正文太短、votes 为 0 没法比）→ 收下当桶里的新比较基准
        bucket.append((votes, p))
        seen_hash[p.content_hash] = p
        unique.append(p)

    return DedupResult(unique=unique, exact_duplicates=exact, near_duplicates=near)
