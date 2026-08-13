"""市场状态与行业相对强弱（仅展示信息，不进量化门禁）。"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def market_state(index_close: pd.Series, lookback: int = 252) -> dict:
    """用宽基指数收盘序列计算趋势与波动状态。"""
    series = index_close.dropna().astype(float)
    if len(series) < 60:
        return {"status": "insufficient_data", "trend": None, "volatility_state": None}
    window = min(lookback, len(series))
    current = float(series.iloc[-1])
    ma = float(series.iloc[-window:].mean())
    daily = series.pct_change(fill_method=None).dropna().iloc[-min(20, len(series) - 1):]
    annualized_vol = float(daily.std(ddof=1) * np.sqrt(252)) if len(daily) >= 2 else 0.0
    return {
        "status": "ok",
        "trend": "above_ma" if current >= ma else "below_ma",
        "ma_level": ma,
        "current_level": current,
        "annualized_volatility": round(annualized_vol, 4),
        "volatility_state": "high" if annualized_vol > 0.30 else ("low" if annualized_vol < 0.15 else "medium"),
    }


def sector_relative_strength(prices_by_sector: dict[str, pd.Series], benchmark: pd.Series, lookback: int = 60) -> dict:
    """每行业相对基准的 lookback 日收益差，按强弱排序。"""
    bench = benchmark.dropna().astype(float)
    if bench.empty:
        return {"status": "insufficient_data", "sectors": {}}
    bench_return = float(bench.iloc[-1] / bench.iloc[max(0, len(bench) - 1 - lookback)] - 1)
    strengths: dict[str, float] = {}
    for sector, series in prices_by_sector.items():
        clean = series.dropna().astype(float)
        if len(clean) < lookback + 1:
            continue
        sector_return = float(clean.iloc[-1] / clean.iloc[-1 - lookback] - 1)
        strengths[sector] = round(sector_return - bench_return, 4)
    ranked = dict(sorted(strengths.items(), key=lambda item: item[1], reverse=True))
    return {"status": "ok" if ranked else "insufficient_data", "benchmark_lookback_return": round(bench_return, 4), "sectors": ranked}


class MarketContextProvider:
    """注入式市场背景信息：失败或缺失时返回 unavailable，绝不参与量化决策。"""

    def __init__(self, fetch_index_bars: Callable | None = None, fetch_sector_bars: Callable | None = None) -> None:
        self.fetch_index_bars = fetch_index_bars
        self.fetch_sector_bars = fetch_sector_bars

    def context(self) -> dict:
        try:
            index_bars = self.fetch_index_bars() if self.fetch_index_bars else None
            sector_bars = self.fetch_sector_bars() if self.fetch_sector_bars else None
        except Exception:
            return {"available": False, "reason": "MARKET_DATA_UNAVAILABLE"}
        if index_bars is None or index_bars.empty:
            return {"available": False, "reason": "INDEX_DATA_UNAVAILABLE"}
        index_close = index_bars.set_index("date")["close"].astype(float)
        state = market_state(index_close)
        sectors = {}
        if sector_bars is not None and not sector_bars.empty and len(index_close) > 60:
            grouped = {name: group.set_index("date")["close"].astype(float) for name, group in sector_bars.groupby("sector")}
            sectors = sector_relative_strength(grouped, index_close)
        return {"available": True, "market_state": state, "sector_strength": sectors, "disclaimer": "市场状态与行业强弱仅作展示信息，不进入量化证据门禁。"}
