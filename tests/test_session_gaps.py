"""日线缺失交易日与零成交状态的失败闭合测试。"""
from __future__ import annotations

import pandas as pd

from data.fetcher import DataFetcher


class SessionCalendar:
    """以固定交易日历消除测试对当前日期和网络的依赖。"""

    version = "session-gap-test-v1"

    def __init__(self, sessions: pd.DatetimeIndex) -> None:
        self.sessions = sessions

    def latest_complete_session(self):
        return self.sessions[-1], ()

    def sessions_between(self, start: pd.Timestamp, end: pd.Timestamp):
        selected = self.sessions[(self.sessions >= start) & (self.sessions <= end)]
        return selected, ()


def _bars() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=60)
    return pd.DataFrame({"date": dates, "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1000.0, "amount": 10000.0})


def _fetcher(tmp_path) -> tuple[DataFetcher, pd.DataFrame]:
    bars = _bars()
    fetcher = DataFetcher()
    fetcher._cache_dir = tmp_path
    fetcher._retry_max = 1
    fetcher._calendar = SessionCalendar(pd.DatetimeIndex(bars["date"]))
    return fetcher, bars


def test_missing_expected_session_is_a_hard_data_gap(tmp_path) -> None:
    fetcher, bars = _fetcher(tmp_path)

    flags = fetcher._validate(bars.drop(index=20).reset_index(drop=True), "600519")
    bundle = fetcher._build_bundle(bars.drop(index=20).reset_index(drop=True), "fixture", False, ("fixture",), None, flags)

    assert "ERROR_UNRESOLVED_SESSION_GAP" in flags
    assert bundle.has_hard_quality_error is True


def test_zero_volume_is_not_mislabeled_as_a_missing_session(tmp_path) -> None:
    fetcher, bars = _fetcher(tmp_path)
    bars.loc[20, "volume"] = 0.0

    flags = fetcher._validate(bars, "600519")

    assert "POSSIBLE_SUSPENSION_ZERO_VOLUME" in flags
    assert "ERROR_UNRESOLVED_SESSION_GAP" not in flags


def test_backup_source_can_recover_a_primary_session_gap(tmp_path, monkeypatch) -> None:
    fetcher, bars = _fetcher(tmp_path)
    incomplete = bars.drop(index=20).reset_index(drop=True)
    monkeypatch.setattr(fetcher, "_fetch_akshare", lambda code: incomplete)
    monkeypatch.setattr(fetcher, "_fetch_baostock", lambda code: bars)

    bundle = fetcher.fetch_daily_bundle("600519")

    assert bundle.source == "baostock"
    assert bundle.fallback_chain == ("akshare", "baostock")
    assert "ERROR_UNRESOLVED_SESSION_GAP" not in bundle.quality_flags


def test_historical_validation_ignores_future_gap_and_price_jump(tmp_path) -> None:
    fetcher, bars = _fetcher(tmp_path)
    dates = pd.bdate_range("2024-01-02", periods=100)
    bars = pd.DataFrame({"date": dates, "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1000.0, "amount": 10000.0})
    fetcher._calendar = SessionCalendar(pd.DatetimeIndex(dates))
    future_problem = bars.drop(index=80).reset_index(drop=True)
    future_problem.loc[70, "close"] = 20.0

    flags = fetcher._validate(future_problem, "600519", dates[60])

    assert "ERROR_UNRESOLVED_SESSION_GAP" not in flags
    assert "ABNORMAL_PRICE_JUMP" not in flags
