"""完整日线截止日与盘中排除规则测试。"""
from __future__ import annotations

import pandas as pd

from data.fetcher import DataFetcher


class FixedCalendar:
    """固定完整交易日，避免测试依赖当前时间与节假日。"""

    version = "fixed-test-v1"

    def __init__(self, complete: str) -> None:
        self.complete = pd.Timestamp(complete)

    def latest_complete_session(self):
        return self.complete, ()


def _bars(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(dates), "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1000.0, "amount": 10000.0})


def _fetcher(complete: str) -> DataFetcher:
    fetcher = DataFetcher()
    fetcher._calendar = FixedCalendar(complete)
    return fetcher


def test_missing_latest_complete_session_is_stale_and_hard_error() -> None:
    bundle = _fetcher("2024-01-03")._build_bundle(_bars(["2024-01-02"]), "fixture", False, ("fixture",), None)
    assert bundle.as_of == pd.Timestamp("2024-01-02")
    assert bundle.is_stale is True
    assert "ERROR_INCOMPLETE_LATEST_SESSION" in bundle.quality_flags
    assert bundle.has_hard_quality_error is True


def test_source_can_use_today_only_when_the_complete_bar_is_present() -> None:
    bundle = _fetcher("2024-01-03")._build_bundle(_bars(["2024-01-02", "2024-01-03"]), "fixture", False, ("fixture",), None)
    assert bundle.as_of == pd.Timestamp("2024-01-03")
    assert bundle.is_stale is False
    assert "ERROR_INCOMPLETE_LATEST_SESSION" not in bundle.quality_flags


def test_requested_historical_as_of_excludes_later_bar_without_staleness() -> None:
    bundle = _fetcher("2024-01-03")._build_bundle(_bars(["2024-01-02", "2024-01-03"]), "fixture", False, ("fixture",), pd.Timestamp("2024-01-02"))
    assert bundle.as_of == pd.Timestamp("2024-01-02")
    assert bundle.bars["date"].tolist() == [pd.Timestamp("2024-01-02")]
    assert bundle.is_stale is False


def test_explicit_as_of_cannot_keep_bar_after_latest_complete_session() -> None:
    bundle = _fetcher("2024-01-02")._build_bundle(_bars(["2024-01-02", "2024-01-03"]), "fixture", False, ("fixture",), pd.Timestamp("2024-01-03"))

    assert bundle.bars["date"].tolist() == [pd.Timestamp("2024-01-02")]
    assert bundle.as_of == pd.Timestamp("2024-01-02")
    assert bundle.is_stale is False
