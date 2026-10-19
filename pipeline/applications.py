"""投递追踪：把「投了哪些、结果如何」记下来。

## 为什么它是整个项目的闭环

采集、抽取、匹配只能证明**系统认为该投**。
真正能证明系统有用的是：**按它的推荐投出去，回复率是多少**。

这也是原计划里唯一「不可能自嗨」的指标——
题目自己出、答案自己标、标准自己定，这些都是可以伪造的；
但「投了 20 家、回了 3 家」造不了假。

## 为什么单独一个 JSONL

投递记录的量级是几十到几百条，不需要数据库。
放 `data/applications.jsonl`，一天追加一行，git 不跟踪（`data/` 已忽略）。

## 状态机

    ｜已投 → 已回复 → 面试中 → 已offer
    ｜  └→ 被拒
    ｜  └→ 无回复（超过 2 周没消息）

`已回复` 和 `面试中` 都算「有回音」；`已offer` 和 `面试中` 算「过了初筛」。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORE_PATH = ROOT / "data" / "applications.jsonl"

STATUSES = ("已投", "已回复", "面试中", "已offer", "被拒", "无回复")

# 「有回音」的定义：不是石沉大海
RESPONDED = {"已回复", "面试中", "已offer", "被拒"}
# 「过了初筛」的定义：拿到面试或 offer
INTERVIEWED = {"面试中", "已offer"}


@dataclass
class Application:
    company: str
    title: str
    url: str
    applied_at: str = field(default_factory=lambda: date.today().isoformat())
    status: str = "已投"
    channel: str = "官网"          # 官网 / Boss / 内推 / 校招
    note: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status 必须是 {STATUSES} 之一，收到 {self.status!r}")


def load_all() -> list[Application]:
    """读全部投递记录。坏行跳过，不让一条脏数据毁掉整个看板。"""
    if not STORE_PATH.exists():
        return []
    out: list[Application] = []
    for line in STORE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(Application(**json.loads(line)))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return out


def add(app: Application) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STORE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(app), ensure_ascii=False) + "\n")


def rewrite(apps: list[Application]) -> None:
    """整文件重写（原子替换），`update` 用。"""
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STORE_PATH.with_suffix(".tmp")
    tmp.write_text(
        "".join(json.dumps(asdict(a), ensure_ascii=False) + "\n" for a in apps),
        encoding="utf-8",
    )
    tmp.replace(STORE_PATH)


def stats(apps: list[Application]) -> dict:
    """回复率 / 面试率。**分母是全部已投**，不是「已回复的」。"""
    total = len(apps)
    responded = sum(1 for a in apps if a.status in RESPONDED)
    interviewed = sum(1 for a in apps if a.status in INTERVIEWED)
    return {
        "total": total,
        "responded": responded,
        "interviewed": interviewed,
        "response_rate": (responded / total) if total else 0.0,
        "interview_rate": (interviewed / total) if total else 0.0,
    }
