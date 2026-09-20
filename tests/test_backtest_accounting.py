"""独立给定预期的回测账务场景。"""
from __future__ import annotations

import pandas as pd
import pytest

from research.backtest import BacktestEngine, ExecutionRuleError, ExecutionRules


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "code": ["000001"] * 3,
            "open": [10.0, 12.0, 8.0],
            "close": [10.0, 12.0, 8.0],
            "volume": [1000, 1000, 1000],
            "execution_volume": [1000, 1000, 1000],
            "can_trade": [True] * 3,
            "suspended": [False] * 3,
            "limit_up": [False] * 3,
            "limit_down": [False] * 3,
        }
    )


def test_insufficient_cash_is_not_converted_to_negative_cash() -> None:
    signals = pd.DataFrame(
        {"signal_date": [pd.Timestamp("2024-01-02")], "code": ["000001"], "action": ["BUY"], "target_shares": [1000]}
    )
    result = BacktestEngine(ExecutionRules(slippage_bps=0)).run(_bars(), signals, 100.0)
    assert result.fills == ()
    assert result.failures[0].reason == "INSUFFICIENT_CASH"
    assert result.final_cash == 100.0
    assert result.final_positions == {}


def test_missing_next_bar_is_a_recorded_failure() -> None:
    signals = pd.DataFrame(
        {"signal_date": [pd.Timestamp("2024-01-04")], "code": ["000001"], "action": ["BUY"], "target_shares": [100]}
    )
    result = BacktestEngine(ExecutionRules(slippage_bps=0)).run(_bars(), signals, 2000.0)
    assert result.failures[0].reason == "NO_NEXT_BAR"
    assert result.final_cash == pytest.approx(2000.0)


def test_equity_marks_unfilled_position_to_close() -> None:
    signals = pd.DataFrame(
        {"signal_date": [pd.Timestamp("2024-01-02")], "code": ["000001"], "action": ["BUY"], "target_shares": [100]}
    )
    result = BacktestEngine(ExecutionRules(slippage_bps=0)).run(_bars(), signals, 2000.0)
    last = result.equity_curve.iloc[-1]
    assert last["cash"] == pytest.approx(799.64)
    assert last["market_value"] == pytest.approx(800.0)
    assert last["equity"] == pytest.approx(1599.64)


def test_missing_bar_for_held_position_fails_closed() -> None:
    """持仓证券缺少某日行情时不能把市值静默记为零。"""
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03", "2024-01-04"]
            ),
            "code": ["000001", "000001", "000002", "000002", "000002"],
            "open": [10.0, 10.0, 20.0, 20.0, 20.0],
            "close": [10.0, 10.0, 20.0, 20.0, 20.0],
            "volume": [1000] * 5,
            "execution_volume": [1000] * 5,
            "can_trade": [True] * 5,
            "suspended": [False] * 5,
            "limit_up": [False] * 5,
            "limit_down": [False] * 5,
        }
    )
    signals = pd.DataFrame(
        {
            "signal_date": [pd.Timestamp("2024-01-02")],
            "code": ["000001"],
            "action": ["BUY"],
            "target_shares": [100],
        }
    )

    with pytest.raises(ExecutionRuleError, match="缺少行情"):
        BacktestEngine(ExecutionRules(slippage_bps=0)).run(bars, signals, 2000.0)
