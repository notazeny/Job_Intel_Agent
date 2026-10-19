"""时间衰减：把「这个岗位有多新」变成一个 0~1 的分数。

## 为什么需要它

实测 866 条真实 JD 的年龄分布（2026-10-17）：

    0-30 天    2.8%      31-90 天   10.5%
    91-180 天 28.1%     181-365 天 24.0%
    1-2 年    19.4%      >2 年      15.2%

    P50 = 227 天，最老 8.2 年。**一半岗位挂了 7 个多月。**

不做衰减，推荐列表会被 2020 年的岗位占满——而求职者真正要的是
「现在正在招人、HR 正在看简历」的机会。

## 公式：指数衰减（半衰期）

    freshness = 0.5 ** (age_days / HALF_LIFE_DAYS)

半衰期 90 天（拍板值）：

    30 天 → 0.79     90 天 → 0.50     180 天 → 0.25
    365 天 → 0.06    730 天 → 0.004

**为什么用半衰期而不是线性**：招聘的时效性是「每过一段时间价值减半」，
不是「过了 N 天突然一文不值」。而且半衰期直接回答「多久之后价值减半」，
参数可解释、可调，也方便在面试里讲清楚为什么选 90 天。

## 两条使用约定

1. **不存进 `Posting`。** freshness 是 `published_at` 和「现在」的纯函数，
   **每天都会变**。存库第二天就过期了。所以每次用时现算。
2. **库里全留，查询时过滤。** 老岗位仍然是真实在招的岗位，只是不值得推荐；
   删了就再也不知道「这家公司把这个岗挂了多久」——那本身就是有价值的信息。

## 和 `is_overseas` 的对称性

两者都是「打标记保留 → 查询/推荐时再切」，不是采集时删数据。
过滤属于**查询层**的事，不属于**存储层**。
"""
from __future__ import annotations

from datetime import datetime, timezone

# 拍板值：半衰期 90 天
HALF_LIFE_DAYS = 90.0

# 进「LLM 抽取 + 推荐」的时间窗口：1 个月（拍板值，2026-10-17）
#
# 理由：AI 岗位的技术要求按月变化，太老的 JD 写的技术栈可能已经过时。
# 注意这个阈值**只决定「拿多少条去抽取和匹配」，不决定「删多少条」**——
# 库里仍然全留（见模块头第 2 条约定）。
#
# 实测（866 条，2026-10-17）：近 1 个月 24 条 / 近 3 个月 115 条 / 近 1 年 566 条。
# 24 条偏薄，前提是先把字节拉全（当前只拉了 500/2111）。
MAX_AGE_DAYS = 30.0

# 没有发布时间时用的中性分：不奖励、不惩罚
NEUTRAL = 0.5


def parse_published(ts: str) -> datetime | None:
    """把 ISO 8601 字符串解析成带时区的 datetime；解析不了返回 None。"""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return None
    # `ms_to_iso()` 产出的是带 +00:00 的，但外部数据不一定，统一补上 UTC
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def age_days(ts: str, *, now: datetime | None = None) -> float | None:
    """岗位发布至今多少天。解析不了返回 None；未来时间按 0 天算。"""
    dt = parse_published(ts)
    if dt is None:
        return None
    now = now or datetime.now(timezone.utc)
    return max((now - dt).total_seconds() / 86400.0, 0.0)


def freshness(
    ts: str,
    *,
    now: datetime | None = None,
    half_life_days: float = HALF_LIFE_DAYS,
) -> float:
    """0~1 的新鲜度。没有发布时间时返回中性值 `NEUTRAL`。

    被 W2 的匹配打分调用（乘在总分上），也可以单独用来排序。
    """
    age = age_days(ts, now=now)
    if age is None:
        return NEUTRAL
    return 0.5 ** (age / half_life_days)


def is_fresh(
    ts: str,
    *,
    now: datetime | None = None,
    max_age_days: float = MAX_AGE_DAYS,
) -> bool:
    """是否还在「值得推荐」的年龄范围内。

    没有发布时间的**放行**——宁可在排序上靠后，也不要因为缺字段就把真岗位筛掉。
    """
    age = age_days(ts, now=now)
    return True if age is None else age <= max_age_days
