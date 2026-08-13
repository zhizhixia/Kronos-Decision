"""公告/新闻事件风险展示：只作事件风险证据，不直接决定动作。"""
from __future__ import annotations

from typing import Callable


class EventRiskProvider:
    """注入式事件源；未配置时返回不可用，任何情况下不改写量化动作。"""

    def __init__(self, fetch_events: Callable | None = None) -> None:
        self.fetch_events = fetch_events

    def events(self, stock_code: str, since: str | None = None, limit: int = 20, as_of: str | None = None) -> dict:
        """返回事件展示；历史报告只保留不晚于截止日的可标日期事件。"""
        try:
            raw = self.fetch_events(stock_code, since) if self.fetch_events else None
        except Exception:
            return {"available": False, "reason": "EVENT_SOURCE_UNAVAILABLE"}
        if raw is None:
            return {"available": False, "reason": "EVENT_SOURCE_NOT_CONFIGURED"}
        events = [dict(event) for event in raw]
        if as_of is not None:
            cutoff = str(as_of)[:10]
            events = [event for event in events if (event_date := str(event.get("date") or event.get("published_at") or "")[:10]) and event_date <= cutoff]
        events = events[:limit]
        return {"available": True, "events": events, "count": len(events), "disclaimer": "公告与新闻仅作为事件风险证据展示，不直接决定增持、减持或回避动作。"}
