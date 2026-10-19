"""飞书招聘（ATSX）通用适配器 —— 本项目的主源。

## 为什么一个适配器能接一串公司

字节跳动、MiniMax、月之暗面等公司的招聘站，用的是**同一套飞书招聘后端**：
请求路径、请求体、响应结构完全一致，**只有 host 不同**。
所以这里写一份，配 N 家。

端点：`POST https://{host}/api/v1/search/job/posts`

## 响应结构（实测，2026-10-15）

```
{"code": 0, "message": "ok", "data": {"count": 2111, "job_post_list": [...]}}
```

单个 job_post 里我们用的字段：

| 字段 | 含义 | 备注 |
|---|---|---|
| `id` | 岗位 ID | 雪花 ID，字符串化后当 source_id |
| `code` | 岗位编号 | 如 A08060A |
| `title` | 岗位名 | |
| `description` | **岗位描述正文** | 完整，可直接喂 LLM |
| `requirement` | **任职要求正文** | 完整 |
| `publish_time` | 发布时间 | **毫秒**时间戳，注意换算 |
| `city_info.name` | 城市 | |
| `job_category.name` | 岗位大类 | 如「后端」 |
| `recruit_type.name` | 招聘类型 | 如「正式」「实习」 |
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .base import BaseCollector, CollectorError, RawPosting, now_iso


class FeishuCollector(BaseCollector):
    source = "feishu"

    def __init__(
        self,
        host: str,
        company: str,
        *,
        portal_type: int = 6,
        job_path: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.host = host
        self.company = company
        # portal_type 实测 6 = 社招。不同租户可能不同，放进 companies.yaml 可配。
        self.portal_type = portal_type
        # 详情页路径。**每家都不一样**（2026-10-19 逐个实测过）：
        #   字节     /experienced/position/{id}/detail
        #   MiniMax  /index/position/{id}/detail
        #   百川     /baichuanzhaopin/position/{id}/detail
        #   零一万物 /index/position/{id}/detail
        #   面壁     /career/position/{id}/detail
        #   智谱     /referral/position/{id}/detail
        #   商汤     /exp/position/{id}/detail
        # 默认值只是字节的兜底，正式接入必须写进 companies.yaml，
        # 否则报告里的链接点开是 404。
        self.job_path = job_path or "/experienced/position/{id}/detail"

    def job_url(self, source_id: str) -> str:
        """岗位详情页地址。路径由 `job_path` 模板决定（见 `__init__` 的实测清单）。"""
        return f"https://{self.host}{self.job_path.format(id=source_id)}"

    def fetch(self, keyword: str, limit: int = 50) -> Iterator[RawPosting]:
        """按关键词翻页拉取。

        分页靠 offset 递增；`count` 是总数，用它判断什么时候停——
        不要靠「这一页不满 50 条」来判断，因为总数恰好是 50 的倍数时会多拉一次。
        """
        url = f"https://{self.host}/api/v1/search/job/posts"
        offset = 0
        total: int | None = None

        while True:
            payload = {
                "keyword": keyword,
                "limit": limit,
                "offset": offset,
                "job_category_id_list": [],
                "location_code_list": [],
                "subject_id_list": [],
                "recruitment_id_list": [],
                "portal_type": self.portal_type,
            }
            body = self.post_json(url, payload)

            if body.get("code") != 0:
                raise CollectorError(
                    f"{self.company} 接口返回 code={body.get('code')} "
                    f"message={body.get('message')}"
                )

            data = body.get("data") or {}
            if total is None:
                total = int(data.get("count") or 0)
                if total == 0:
                    logger_no_result(self.company, keyword)
                    return

            posts = data.get("job_post_list") or []
            if not posts:
                return

            # 原始响应归档：schema 变了能重放，不用重新爬
            self.archive(
                f"{self.company}_{keyword}_{offset}",
                {"keyword": keyword, "offset": offset, "body": body},
            )

            for post in posts:
                yield self._to_raw(post)

            offset += len(posts)
            if offset >= total:
                return

    def _to_raw(self, post: dict) -> RawPosting:
        source_id = str(post.get("id") or post.get("code") or "")
        return RawPosting(
            source=self.source,
            company=self.company,
            source_id=source_id,
            url=self.job_url(source_id),
            raw=post,
            fetched_at=now_iso(),
        )


def logger_no_result(company: str, keyword: str) -> None:
    import logging
    logging.getLogger(__name__).info("%s 上用关键词 %r 没搜到岗位", company, keyword)
