"""数据源超时降级测试（不依赖网络）。"""
from __future__ import annotations

import time

import pandas as pd

from data.contracts import MarketDataBundle
from data.fetcher import DataFetcher


def test_source_timeout_falls_through_to_next_source(tmp_path, monkeypatch) -> None:
    fetcher = DataFetcher()
    fetcher._cache_dir = tmp_path
    fetcher._source_timeout_seconds = 0.2
    fetcher._retry_max = 1

    def slow_source(code):
        time.sleep(5)
        raise AssertionError("不应到达")

    bars = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=60), "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1000.0, "amount": 10000.0})
    bundle = MarketDataBundle(bars, "fixture", pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-01"), "qfq", "v1", None, False, (), ("fixture",), "hash")

    monkeypatch.setattr(fetcher, "_fetch_akshare", slow_source)
    monkeypatch.setattr(fetcher, "_fetch_baostock", lambda code: bars)
    monkeypatch.setattr(fetcher, "_build_bundle", lambda bars_, source, stale, chain, as_of, flags=(): bundle)
    started = time.time()
    result = fetcher.fetch_daily_bundle("600519")
    assert result is bundle
    assert time.time() - started < 3


def test_pool_source_timeout_falls_back_to_cache(tmp_path, monkeypatch) -> None:
    """股票池 AkShare 拉取挂起时按 60 秒超时降级到过期缓存，而不是无限等待。"""
    import data.pool as pool_module

    monkeypatch.setattr(pool_module, "_CACHE_PATH", tmp_path / "pool_hs300.csv")
    monkeypatch.setattr(pool_module, "_is_fresh", lambda: False)
    monkeypatch.setattr(pool_module, "_SOURCE_TIMEOUT_SECONDS", 0.3)

    def hanging_fetch():
        time.sleep(10)
        return {}

    monkeypatch.setattr(pool_module, "_fetch_from_akshare", hanging_fetch)
    pool_module._save_cache({"600519": "贵州茅台"})
    started = time.time()
    result = pool_module.get_hs300_pool()
    assert result == {"600519": "贵州茅台"}
    assert time.time() - started < 5


def test_call_with_timeout_budget_and_error_propagation() -> None:
    """公共超时工具：超时抛 TimeoutError、成功返回值、异常原样传播。"""
    import pytest

    from data.timeout import call_with_timeout

    assert call_with_timeout(lambda: 42, 1, name="fast") == 42
    with pytest.raises(ValueError, match="boom"):
        call_with_timeout(lambda: (_ for _ in ()).throw(ValueError("boom")), 1, name="err")
    started = time.time()
    with pytest.raises(TimeoutError):
        call_with_timeout(lambda: time.sleep(10) or 1, 0.3, name="slow")
    assert time.time() - started < 2
