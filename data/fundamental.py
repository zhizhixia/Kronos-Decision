"""免费基本面证据适配器：展示信息，当前数据不得用于历史日期。"""
from __future__ import annotations

from datetime import datetime
from typing import Callable


class FundamentalAdapter:
    """注入式基本面快照；真实数据源未接入或缺少点时数据时返回不可用。"""

    def __init__(self, fetch_snapshot: Callable | None = None, latest_complete_session: Callable | None = None) -> None:
        self.fetch_snapshot = fetch_snapshot
        self.latest_complete_session = latest_complete_session or (lambda: datetime.now().date())

    def snapshot(self, stock_code: str, as_of=None) -> dict:
        """返回基本面展示信息；as_of 为历史日期时明确标记非点时可用。"""
        try:
            raw = self.fetch_snapshot(stock_code) if self.fetch_snapshot else None
        except Exception:
            return {"available": False, "reason": "FUNDAMENTAL_SOURCE_UNAVAILABLE"}
        if raw is None:
            return {"available": False, "reason": "FUNDAMENTAL_SOURCE_NOT_CONFIGURED"}
        items = dict(raw.get("items", {}))
        data_date = raw.get("as_of")
        if as_of is not None and data_date is not None and as_of < data_date:
            items = {key: None for key in items}  # 历史锚点无点时基本面，不展示数值
            return {"available": True, "point_in_time_available": False, "as_of": str(data_date), "items": items, "disclaimer": "历史日期无点时基本面数据，数值已隐藏；本信息不进入量化门禁。"}
        return {"available": True, "point_in_time_available": True, "as_of": str(data_date) if data_date else None, "items": items, "disclaimer": "基本面证据仅作展示，不直接决定量化动作。"}
