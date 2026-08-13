"""Tushare Pro 可选适配器的离线合同测试。"""
from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from data.fetcher import DataFetcher
from decision.errors import DataSourceError


def _bars() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.bdate_range("2024-01-02", periods=60), "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1000.0, "amount": 10000.0})


def test_tushare_is_not_in_default_free_source_chain() -> None:
    fetcher = DataFetcher()
    assert [source for source, _ in fetcher._source_candidates()] == ["akshare", "baostock"]


def test_tushare_joins_chain_only_with_configuration_and_token(monkeypatch) -> None:
    fetcher = DataFetcher()
    fetcher._optional_sources = ("tushare",)
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    assert [source for source, _ in fetcher._source_candidates()] == ["akshare", "baostock"]
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")
    assert [source for source, _ in fetcher._source_candidates()] == ["akshare", "baostock", "tushare"]


def test_tushare_normalizes_pro_bar_and_uses_environment_token(monkeypatch) -> None:
    fetcher = DataFetcher()
    api_instance = object()
    tokens: list[str] = []
    calls: dict[str, str] = {}
    raw = pd.DataFrame({"ts_code": ["600519.SH", "600519.SH"], "trade_date": ["20240103", "20240102"], "open": [10.0, 9.0], "high": [11.0, 10.0], "low": [9.0, 8.0], "close": [10.0, 9.0], "vol": [100.0, 200.0], "amount": [1.5, 2.0]})

    def pro_bar(**kwargs):
        calls.update(kwargs)
        return raw

    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")
    monkeypatch.setitem(sys.modules, "tushare", types.SimpleNamespace(pro_api=lambda token: tokens.append(token) or api_instance, pro_bar=pro_bar))
    bars = fetcher._fetch_tushare("600519")

    assert tokens == ["test-token"]
    assert calls["ts_code"] == "600519.SH"
    assert calls["pro_api"] is api_instance
    assert calls["adj"] == "qfq"
    assert bars["date"].is_monotonic_increasing
    assert set(bars["volume"]) == {10000.0, 20000.0}
    assert set(bars["amount"]) == {1500.0, 2000.0}


def test_tushare_requires_environment_token(monkeypatch) -> None:
    fetcher = DataFetcher()
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    with pytest.raises(DataSourceError, match="TUSHARE_TOKEN"):
        fetcher._fetch_tushare("600519")


def test_tushare_sdk_error_does_not_echo_sensitive_context(monkeypatch) -> None:
    fetcher = DataFetcher()
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")
    module = types.SimpleNamespace(pro_api=lambda token: (_ for _ in ()).throw(RuntimeError("secret-context")))
    monkeypatch.setitem(sys.modules, "tushare", module)
    with pytest.raises(DataSourceError) as caught:
        fetcher._fetch_tushare("600519")
    assert "secret-context" not in str(caught.value)


def test_opted_in_tushare_is_used_after_free_sources(tmp_path, monkeypatch) -> None:
    fetcher = DataFetcher()
    fetcher._cache_dir = tmp_path
    fetcher._retry_max = 1
    fetcher._optional_sources = ("tushare",)
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")
    monkeypatch.setattr(fetcher, "_fetch_akshare", lambda code: (_ for _ in ()).throw(DataSourceError("down")))
    monkeypatch.setattr(fetcher, "_fetch_baostock", lambda code: (_ for _ in ()).throw(DataSourceError("down")))
    monkeypatch.setattr(fetcher, "_fetch_tushare", lambda code: _bars())

    bundle = fetcher.fetch_daily_bundle("600519", as_of=pd.Timestamp("2024-03-31"))

    assert bundle.source == "tushare"
    assert bundle.fallback_chain == ("akshare", "baostock", "tushare")
