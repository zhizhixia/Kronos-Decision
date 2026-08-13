"""CSV 缓存的来源审计与质量失败闭环测试。"""
from __future__ import annotations

import pandas as pd

from data.fetcher import DataFetcher


class FixedCalendar:
    version = "fixed-test-v1"

    def __init__(self, complete: pd.Timestamp) -> None:
        self.complete = complete

    def latest_complete_session(self):
        return self.complete, ()


def _bars() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=60)
    return pd.DataFrame({"date": dates, "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1000.0, "amount": 10000.0})


def _fetcher(tmp_path) -> DataFetcher:
    fetcher = DataFetcher()
    fetcher._cache_dir = tmp_path
    fetcher._retry_max = 1
    fetcher._calendar = FixedCalendar(_bars()["date"].max())
    return fetcher


def test_fresh_cache_preserves_original_fallback_provenance(tmp_path) -> None:
    fetcher = _fetcher(tmp_path)
    bundle = fetcher._build_bundle(_bars(), "akshare", False, ("akshare",), None)
    fetcher._save_cache("600519", bundle.bars, bundle)

    cached = fetcher.fetch_daily_bundle("600519")

    assert cached.source == "csv_cache"
    assert cached.fallback_chain == ("akshare", "csv_cache")
    assert cached.content_hash == bundle.content_hash


def test_invalid_cache_is_not_used_when_a_source_can_recover(tmp_path, monkeypatch) -> None:
    fetcher = _fetcher(tmp_path)
    invalid = _bars()
    invalid.loc[0, "high"] = 1.0
    invalid.to_csv(fetcher._cache_path("600519"), index=False)
    monkeypatch.setattr(fetcher, "_fetch_akshare", lambda code: _bars())
    monkeypatch.setattr(fetcher, "_fetch_baostock", lambda code: (_ for _ in ()).throw(AssertionError("不应调用备源")))

    bundle = fetcher.fetch_daily_bundle("600519")

    assert bundle.source == "akshare"
    assert bundle.fallback_chain == ("akshare",)


def test_unreadable_cache_is_not_used_when_a_source_can_recover(tmp_path, monkeypatch) -> None:
    """中断留下的半个 CSV 不能阻断主数据源回退。"""
    fetcher = _fetcher(tmp_path)
    _bars().to_csv(fetcher._cache_path("600519"), index=False)
    monkeypatch.setattr("data.fetcher.pd.read_csv", lambda *args, **kwargs: (_ for _ in ()).throw(pd.errors.ParserError("截断 CSV")))
    monkeypatch.setattr(fetcher, "_fetch_akshare", lambda code: _bars())
    monkeypatch.setattr(fetcher, "_fetch_baostock", lambda code: (_ for _ in ()).throw(AssertionError("不应调用备源")))

    bundle = fetcher.fetch_daily_bundle("600519")

    assert bundle.source == "akshare"
