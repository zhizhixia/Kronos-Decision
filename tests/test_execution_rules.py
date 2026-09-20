"""执行规则和独立账务期望。"""
from __future__ import annotations

import pandas as pd
import pytest

from research.backtest import BacktestEngine, ExecutionRuleError, ExecutionRules


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
            "code": ["600519"] * 4,
            "open": [10.0, 10.0, 11.0, 12.0],
            "close": [10.0, 10.5, 11.5, 12.0],
            "volume": [1000, 1000, 1000, 1000],
            "execution_volume": [1000, 1000, 1000, 1000],
            "can_trade": [True] * 4,
            "suspended": [False] * 4,
            "limit_up": [False] * 4,
            "limit_down": [False] * 4,
        }
    )


def test_signal_executes_next_open_and_t_plus_one_is_enforced() -> None:
    engine = BacktestEngine(
        ExecutionRules(commission_rate=0.001, stamp_duty_rate=0.001, slippage_bps=0)
    )
    signals = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "code": ["600519"] * 2,
            "action": ["BUY", "SELL"],
            "target_shares": [100, 0],
        }
    )

    result = engine.run(_bars(), signals, 2000.0)

    assert [fill.execution_date for fill in result.fills] == [pd.Timestamp("2024-01-03")]
    assert result.failures[0].reason == "T_PLUS_ONE_OR_EMPTY"
    # 独立账务：买入 100 股，10 元成交，买入佣金 1 元；当日不能卖出。
    assert result.final_cash == pytest.approx(2000.0 - 1000.0 - 1.0)
    assert result.final_positions == {"600519": 100}


def test_partial_volume_and_limit_are_recorded_as_failures_not_fabricated() -> None:
    bars = _bars().copy()
    bars.loc[1, "execution_volume"] = 150
    bars.loc[2, "limit_up"] = True
    signals = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "code": ["600519", "600519"],
            "action": ["BUY", "BUY"],
            "target_shares": [200, 300],
        }
    )

    result = BacktestEngine(ExecutionRules(slippage_bps=0)).run(bars, signals, 10000.0)

    assert result.fills[0].status == "PARTIAL"
    assert result.fills[0].requested_shares == 200
    assert result.fills[0].filled_shares == 100
    assert result.failures[0].reason == "LIMIT_UP"


def test_missing_volume_and_unsupported_corporate_action_fail_closed() -> None:
    with pytest.raises(ExecutionRuleError, match="成交量"):
        BacktestEngine().run(_bars().drop(columns=["execution_volume"]), pd.DataFrame(), 1000.0)

    bars = _bars().assign(dividend=0.1)
    signals = pd.DataFrame()
    with pytest.raises(ExecutionRuleError, match="公司行动"):
        BacktestEngine().run(bars, signals, 1000.0)


def test_as_of_excludes_future_bars_before_execution() -> None:
    signals = pd.DataFrame(
        {"signal_date": [pd.Timestamp("2024-01-04")], "code": ["600519"], "action": ["BUY"], "target_shares": [100]}
    )
    result = BacktestEngine(ExecutionRules(slippage_bps=0)).run(
        _bars(), signals, 2000.0, as_of="2024-01-04"
    )
    assert result.fills == ()
    assert result.failures[0].reason == "NO_NEXT_BAR"
    assert result.equity_curve["date"].max() == pd.Timestamp("2024-01-04")


def test_target_weight_uses_open_execution_price_not_future_close() -> None:
    """target_weight 计算不能读取执行日收盘价。"""
    bars = _bars().copy()
    bars["close"] = [10.0, 10.0, 10.0, 1000.0]
    signals = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "code": ["600519", "600519"],
            "action": ["BUY", "BUY"],
            "target_shares": [100, pd.NA],
            "target_weight": [pd.NA, 0.5],
        }
    )

    result = BacktestEngine(ExecutionRules(slippage_bps=0)).run(bars, signals, 2000.0)

    assert len(result.fills) == 1
    assert result.failures[-1].reason == "NO_ORDER"


def test_daily_volume_is_not_used_as_open_execution_capacity() -> None:
    """收盘后才完整可见的日成交量不能限制开盘成交。"""
    signals = pd.DataFrame(
        {
            "signal_date": [pd.Timestamp("2024-01-02")],
            "code": ["600519"],
            "action": ["BUY"],
            "target_shares": [200],
        }
    )
    baseline = _bars()
    changed_daily_volume = baseline.copy()
    changed_daily_volume["volume"] = 0

    first = BacktestEngine(ExecutionRules(slippage_bps=0)).run(
        baseline, signals, 5000.0
    )
    second = BacktestEngine(ExecutionRules(slippage_bps=0)).run(
        changed_daily_volume, signals, 5000.0
    )

    assert first.fills[0].filled_shares == 200
    assert second.fills[0].filled_shares == 200


def test_missing_or_unknown_trading_status_fails_closed() -> None:
    """交易状态缺失或未知时，订单必须记录为不可成交。"""
    bars = _bars().drop(columns=["limit_down"])
    bars["can_trade"] = bars["can_trade"].astype(object)
    bars.loc[1, "can_trade"] = "unknown"
    signals = pd.DataFrame(
        {
            "signal_date": [pd.Timestamp("2024-01-02")],
            "code": ["600519"],
            "action": ["BUY"],
            "target_shares": [100],
        }
    )

    result = BacktestEngine(ExecutionRules(slippage_bps=0)).run(bars, signals, 2000.0)

    assert result.fills == ()
    assert result.failures[0].reason == "UNKNOWN_TRADING_STATUS"
