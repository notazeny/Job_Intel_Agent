"""归一化：把各家形状不同的原始岗位，洗成同一个 `Posting`。

**这一步全是规则，不用 LLM。** 理由：去 HTML、字段改名、时间戳换算、
城市/类型映射——都是确定性的纯函数，可单测、零 token、跑得飞快。
把 LLM 花在这种地方是纯浪费。

采集器尽量保留原始字段（schema 变了能重放），归一化的脏活都在这里干。


## 数据在这条链上长什么样（从归档文件一路到下游）

    data/raw/<公司>/<公司>_<关键词>_<offset>.json   ← 采集时原样归档的响应
      │   形如 {"keyword": …, "offset": …, "body": {…}}
      │   岗位数组在 body["data"]["job_post_list"]
      ↓  RawPosting(source=…, company=…, raw=p)      ← 采集器只搬运，字段一个不改
      ↓  normalize(raw)                              ← 只做「按 source 选适配器」
      ↓  _NORMALIZERS[raw.source](raw)               ← 目前表里只有 "feishu"
      ↓  _from_feishu(raw)                           ← 真正干活：取字段、清洗、算派生
      ↓  Posting                                     ← 下游唯一认的形状
      ↓  dedup(postings)                             ← 见 pipeline/dedup.py

本文件的调用是**一层包一层**，照着上面那个顺序读就不会绕：

    normalize()              只查表分派，不含任何解析逻辑
      └ _from_feishu()       飞书字段 → Posting 字段的翻译表
          ├ strip_html()           去 <br>/<p>、解实体、收空白
          ├ ms_to_iso()            毫秒时间戳 → ISO 字符串
          ├ _extract_cities()      可选办公城市 → 排序去重的元组（**去重主键的一部分**）
          ├ _is_overseas()         海外岗位打标（W2 硬筛时切）
          └ content_fingerprint()  内容指纹（精确去重的钥匙）
"""
from __future__ import annotations

import hashlib
import html
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from collectors.base import RawPosting

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t　]+")
_BLANK_RE = re.compile(r"\n{3,}")

# 飞书 address_list[].country.name 里，国内岗位一律是这个值（海外是「美国」等）
_MAINLAND = "中国大陆"


@dataclass
class Posting:
    """归一化后的岗位。全项目下游统一用这个。

    字段分三类，别混：

    - **身份**（决定「是不是同一个岗位」）：`company` + `cities` + `title`
      去重、比对都认这三个，它们必须**稳定可比**。
    - **内容**（喂给检索和 LLM 的）：`description` / `requirement`
    - **属性**（筛选、展示、排序用）：`city` / `category` / `recruit_type` /
      `published_at` / `is_overseas` —— 这些可以随接口变，**不该拿去做去重**。

    最容易被搞混的一对是 `city` 和 `cities`，见各自的注释。
    """
    source: str
    company: str
    source_id: str
    url: str
    title: str
    description: str          # 岗位描述正文（已去 HTML）
    requirement: str          # 任职要求正文（已去 HTML）
    city: str                 # **展示用**单值城市，来自 city_info（派生字段，会变）
    cities: tuple[str, ...]   # **身份用**可选办公城市集合，来自 city_list（已排序去重）
    category: str             # 岗位大类，如「后端」；W2 硬筛的过滤器，不进主键
    recruit_type: str         # 招聘类型，如「正式」「实习」
    published_at: str         # ISO 8601
    content_hash: str         # 内容指纹，用于精确去重
    fetched_at: str
    is_overseas: bool         # 岗位是否全在海外；W2 硬筛时切掉

    def to_dict(self) -> dict:
        return asdict(self)

    def business_key(self) -> tuple[str, tuple[str, ...], str]:
        """岗位的**业务主键** = 公司 + 可选城市集合 + 标题。

        ## 为什么不用 `source_id`

        实测（2026-10-17，字节 500 条）：同一个岗位、正文**一字不差**，
        会拿到**不同的 source_id**——

            新 id=7548358333829499144   旧 id=7549106477568575752

        所以 `source_id` 只能标识「一次发布」，不能标识「一个岗位」。
        拿它做跨批次去重，漏杀是必然的。

        ## 为什么是这三个字段

        - 公司不同     → 一定是不同岗位
        - 可选城市不同 → 求职者面对的是两个不同的选择
        - 标题不同     → 承载团队/产品线，是求职者真正在选的东西

        `category` 不进主键（只有 10 个取值，太粗），它是 W2 的过滤器。
        在 409 条真实数据上验证过：这个三元组冲突 = 0。
        """
        return (self.company, self.cities, self.title)


def strip_html(text: str | None) -> str:
    """去标签 + 解实体 + 收拾空白。

    岗位正文偶尔带 <br> / <p>，直接喂给 LLM 会污染输入；
    而多余的连续空格和空行会让后面做 SimHash 时产生噪音。
    """
    if not text:
        return ""
    text = _TAG_RE.sub("\n", text)
    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANK_RE.sub("\n\n", text).strip()


def ms_to_iso(ms: int | float | None) -> str:
    """飞书的 publish_time 是**毫秒**时间戳——直接当秒用会得到 1970 年。"""
    if not ms:
        return ""
    try:
        return datetime.fromtimestamp(float(ms) / 1000, tz=timezone.utc).isoformat(
            timespec="seconds"
        )
    except (ValueError, OSError, OverflowError):
        return ""


def _extract_cities(p: dict) -> tuple[str, ...]:
    """从飞书的 `city_list` 里抽出「可选办公城市」集合。

    **这个元组是岗位身份的一部分**（去重主键 = 公司 + 本字段 + 标题），
    所以有两条硬要求：

    1. **只取 name，丢掉整个 dict。** 原始结构是
       `[{"name": "深圳", "code": "CT_…", …}, …]`，比对时只有名字有意义。
    2. **必须排序去重。** 飞书返回的顺序**不保证稳定**——今天给你
       `['深圳','广州']`，明天可能是 `['广州','深圳']`。不排序的话，
       同一个岗位会被判成两条，去重直接漏杀。

    空名字一律丢掉（有些岗位会给个空 name 占位）。

    被 `_from_feishu()` 调用，结果同时进 `Posting.cities` 和 `content_fingerprint()`。
    """
    raw = p.get("city_list") or []
    names = {(c.get("name") or "").strip() for c in raw}
    names.discard("")
    return tuple(sorted(names))


def _is_overseas(p: dict) -> bool:
    """岗位是不是**全部**在海外（打标记保留，W2 硬筛时才切）。

    依据是 `job_post_info.address_list[].country.name`，飞书给的是中文国名
    （国内岗位一律 `"中国大陆"`，海外是 `"美国"` 等）。

    注意判据是 **「所有地址都不在中国大陆」**，不是「第一个地址不在」——
    一个岗位可能同时挂北京和圣何塞，那它国内是去得了的，**不算海外**。

    没有地址信息时保守返回 False（宁可多留一条，也别误删）——
    address_list 的填充率实测 100%，这条路正常走不到。

    被 `_from_feishu()` 调用。
    """
    addrs = (p.get("job_post_info") or {}).get("address_list") or []
    countries = {((a.get("country") or {}).get("name") or "").strip() for a in addrs}
    countries.discard("")
    if not countries:
        return False
    return _MAINLAND not in countries


def content_fingerprint(
    company: str,
    title: str,
    cities: tuple[str, ...],
    description: str,
    requirement: str,
) -> str:
    """内容指纹 —— **精确去重**的钥匙。

    为什么用「公司 + 标题 + 城市集合 + 正文」而不是 `source_id`：
    同一条岗位在不同源、不同批次会有不同的 id，但内容是一样的；
    反过来，这四个全同的两条，几乎不可能是两个不同岗位。

    `cities` 必须算进来：同一个标题挂在北京和深圳，是**两个岗位**，不算重复。

    算之前会先用 `_normalize_for_hash()` 抹掉排版差异（空格、换行、大小写），
    否则同一段 JD 因为多一个空格就被当成两条不同岗位——去重就白做了。

    被 `_from_feishu()` 调用，结果进 `Posting.content_hash`。
    """
    joined = "\n".join(
        _normalize_for_hash(x) for x in (company, title, cities, description, requirement)
    )
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def _normalize_for_hash(text: str | tuple[str, ...] | list[str]) -> str:
    """算指纹前先把「排版差异」抹掉。

    入参可能是字符串，也可能是 `_extract_cities()` 给的元组
    （`cities` 字段就是元组），所以这里先把它拍平成字符串再处理——
    否则元组没有 `.replace()`，会直接崩。
    """
    if isinstance(text, (tuple, list)):
        text = "".join(text)
    return _WS_RE.sub("", (text or "").replace("\n", "")).lower()


# ============================================================
# 各源的适配函数：RawPosting -> Posting
# ============================================================

def normalize(raw: RawPosting) -> Posting:
    """按 `raw.source` 查表分派到对应的适配函数。

    本函数**不含任何解析逻辑**，只负责「选对翻译官」：
    接一家新公司时，在这里的 `_NORMALIZERS` 里挂一个新的 `_from_xxx` 就行，
    上面那些公共工具（strip_html / ms_to_iso / …）直接复用。

    入参：`RawPosting`（采集器搬来的原始响应 + 几个定位字段）
    出参：`Posting`（下游唯一认的形状）
    """
    fn = _NORMALIZERS.get(raw.source)
    if fn is None:
        raise ValueError(f"没有为 source={raw.source!r} 写归一化函数")
    return fn(raw)


def _from_feishu(raw: RawPosting) -> Posting:
    """飞书招聘的字段 → `Posting` 字段。**本文件唯一真正干活的地方。**

    被 `normalize()` 通过 `_NORMALIZERS` 查到并调用。
    入参 `raw.raw` 是飞书接口返回的单条岗位 dict（结构见模块头的数据流图），
    出参是下游唯一认的 `Posting`。

    飞书给的城市信息有**两个字段、两种语义**，这是最容易搞混的地方，一次说清：

        city_info : 单个 dict，是**派生**出来的「主城市」
        city_list : 列表，才是岗位的**真实属性** —— 「可选办公城市」

    例：一个岗位 `city_info = 深圳`，`city_list = ['深圳','广州','上海']`，
    意思是「这个岗位在深/广/上都能入职」，**不是**「三个城市各招一个」。

    → 所以 `city`（单值）读 `city_info`，**只用于展示和 W2 硬筛**；
      `cities`（集合）读 `city_list`，**进去重主键**。两者各管各的，别互相顶替。
    """
    p = raw.raw
    title = (p.get("title") or "").strip()
    description = strip_html(p.get("description"))
    requirement = strip_html(p.get("requirement"))

    # ---- 城市：单值和集合分开算（语义不同，见 docstring）----
    # cities 是主键的一部分，**必须在函数开头无条件算出来**，
    # 不能塞进下面的兜底分支里——绝大多数岗位都有 city_info，走不到那个分支。
    cities = _extract_cities(p)

    city_info = p.get("city_info") or {}
    city = (city_info.get("name") or "").strip()
    if not city:
        # city_info 偶尔缺失，退回集合里的第一个，纯粹为了展示不空着
        city = cities[0] if cities else ""

    return Posting(
        source=raw.source,
        company=raw.company,
        source_id=raw.source_id,
        url=raw.url,
        title=title,
        description=description,
        requirement=requirement,
        city=city,
        cities=cities,
        category=((p.get("job_category") or {}).get("name") or "").strip(),
        recruit_type=((p.get("recruit_type") or {}).get("name") or "").strip(),
        published_at=ms_to_iso(p.get("publish_time")),
        content_hash=content_fingerprint(raw.company, title, cities, description, requirement),
        fetched_at=raw.fetched_at,
        is_overseas=_is_overseas(p),
    )


_NORMALIZERS = {
    "feishu": _from_feishu,
}
