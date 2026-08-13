"""市场状态与行业相对强弱测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.market_context import MarketContextProvider, market_state, sector_relative_strength


def _index() -> pd.Series:
    index = pd.bdate_range("2023-01-01", periods=300)
    return pd.Series(100 * (1 + np.linspace(0, 0.3, len(index))), index=index)


def test_market_state_detects_trend_and_volatility() -> None:
    state = market_state(_index())
    assert state["status"] == "ok"
    assert state["trend"] == "above_ma"
    assert state["annualized_volatility"] > 0


def test_sector_relative_strength_ranks_sectors() -> None:
    index = _index()
    strong = 100 * (1 + np.linspace(0, 0.6, len(index)))
    weak = 100 * (1 + np.linspace(0, 0.1, len(index)))
    result = sector_relative_strength({"strong": pd.Series(strong, index=index.index), "weak": pd.Series(weak, index=index.index)}, index)
    assert result["status"] == "ok"
    assert list(result["sectors"]) == ["strong", "weak"]
    assert result["sectors"]["strong"] > result["sectors"]["weak"]


def test_provider_returns_unavailable_without_data() -> None:
    provider = MarketContextProvider(fetch_index_bars=lambda: None)
    context = provider.context()
    assert context["available"] is False


def test_provider_returns_display_only_context() -> None:
    index_bars = pd.DataFrame({"date": _index().index, "close": _index().values})
    sector_bars = pd.DataFrame({"date": _index().index, "sector": "strong", "close": 100 * (1 + np.linspace(0, 0.6, len(_index())))})
    provider = MarketContextProvider(fetch_index_bars=lambda: index_bars, fetch_sector_bars=lambda: sector_bars)
    context = provider.context()
    assert context["available"] is True
    assert "disclaimer" in context
    assert context["market_state"]["trend"] in ("above_ma", "below_ma")


def test_engine_market_context_respects_total_budget(monkeypatch) -> None:
    """展示信息总预算超时即返回 unavailable，不阻塞 v2 报告。"""
    import time

    from decision.engine import DecisionEngine

    engine = DecisionEngine()

    def slow_fetch(code: str):
        time.sleep(3)
        return None

    monkeypatch.setattr(engine._fetcher, "fetch_daily_bundle", slow_fetch)
    started = time.time()
    payload = engine._market_context_payload(timeout=0.5)
    elapsed = time.time() - started
    assert elapsed < 2.0
    assert payload["available"] is False
    assert payload["reason"] == "MARKET_CONTEXT_TIMEOUT_OR_UNAVAILABLE"
