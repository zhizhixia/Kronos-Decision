"""基本面与事件风险展示适配器测试。"""
from __future__ import annotations

from data.events import EventRiskProvider
from data.fundamental import FundamentalAdapter


def test_fundamental_unavailable_without_source() -> None:
    result = FundamentalAdapter().snapshot("600519")
    assert result["available"] is False
    assert result["reason"] == "FUNDAMENTAL_SOURCE_NOT_CONFIGURED"


def test_fundamental_hides_values_for_historical_as_of() -> None:
    adapter = FundamentalAdapter(fetch_snapshot=lambda code: {"as_of": "2026-08-10", "items": {"pe": 25.0, "roe": 0.3}})
    result = adapter.snapshot("600519", as_of="2024-01-01")
    assert result["point_in_time_available"] is False
    assert result["items"]["pe"] is None
    assert "不进入量化门禁" in result["disclaimer"]


def test_fundamental_returns_items_for_current_date() -> None:
    adapter = FundamentalAdapter(fetch_snapshot=lambda code: {"as_of": "2026-08-10", "items": {"pe": 25.0}})
    result = adapter.snapshot("600519")
    assert result["available"] is True
    assert result["items"]["pe"] == 25.0


def test_event_risk_never_changes_action() -> None:
    provider = EventRiskProvider(fetch_events=lambda code, since: [{"type": "公告", "date": "2026-08-09", "title": "业绩预告"}])
    result = provider.events("600519")
    assert result["available"] is True
    assert result["count"] == 1
    assert "不直接决定" in result["disclaimer"]
    assert EventRiskProvider().events("600519")["reason"] == "EVENT_SOURCE_NOT_CONFIGURED"


def test_historical_events_hide_future_and_undated_records() -> None:
    provider = EventRiskProvider(fetch_events=lambda code, since: [{"date": "2024-01-02", "title": "保留"}, {"date": "2024-01-04", "title": "隐藏"}, {"title": "无日期"}])

    result = provider.events("600519", as_of="2024-01-03")

    assert [item["title"] for item in result["events"]] == ["保留"]
