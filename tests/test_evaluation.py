"""评价指标的独立小样例和不可计算反例。"""
from __future__ import annotations

import pandas as pd
import pytest

from research.backtest import BacktestEngine, ExecutionRules
from research.evaluation import calibration_table, cost_stress_matrix, evaluate_backtest, evaluate_periods, rank_ic_by_date


def test_rank_ic_skips_zero_variance_dates_instead_of_returning_zero() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 3 + ["2024-01-03"] * 3),
            "code": ["A", "B", "C", "A", "B", "C"],
            "score": [1.0, 2.0, 3.0, 1.0, 1.0, 1.0],
            "realized_return": [0.01, 0.02, 0.03, 0.01, 0.02, 0.03],
        }
    )
    result = rank_ic_by_date(frame)
    assert result.metrics["rank_ic_mean"] == 1.0
    assert result.sample_counts == {"valid_dates": 1, "skipped_dates": 1}
    assert "RANK_IC_DATES_SKIPPED" in result.reasons


def test_rank_ic_with_no_valid_date_is_not_zero() -> None:
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 2,
            "code": ["A", "B"],
            "score": [1.0, 1.0],
            "realized_return": [0.01, 0.02],
        }
    )
    result = rank_ic_by_date(frame)
    assert result.metrics["rank_ic_mean"] is None
    assert "RANK_IC_UNDEFINED" in result.reasons


def test_periods_use_unique_dates_not_rows() -> None:
    """同一交易日多证券不能虚增独立时间样本。"""
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    predictions = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "code": ["A"] * 4 + ["B"] * 4,
            "score": [1.0] * 8,
            "realized_return": [0.01, 0.02, 0.03, 0.04] * 2,
        }
    )

    result = evaluate_periods(predictions, period_days=2)

    assert result.sample_counts["valid_periods"] == 2


def test_periods_are_non_overlapping_and_backtest_metrics_are_net() -> None:
    predictions = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=6, freq="B"),
            "code": ["A"] * 6,
            "score": [1.0] * 6,
            "realized_return": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        }
    )
    periods = evaluate_periods(predictions, period_days=3)
    assert periods.sample_counts["valid_periods"] == 2
    assert periods.metrics["period_return_mean"] == pytest.approx(0.035)

    bars = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=3, freq="B"),
            "code": ["A"] * 3,
            "open": [10.0, 10.0, 11.0],
            "close": [10.0, 10.0, 11.0],
            "volume": [1000, 1000, 1000],
            "execution_volume": [1000, 1000, 1000],
            "can_trade": [True] * 3,
            "suspended": [False] * 3,
            "limit_up": [False] * 3,
            "limit_down": [False] * 3,
        }
    )
    signals = pd.DataFrame(
        {"signal_date": [bars.iloc[0]["date"]], "code": ["A"], "action": ["BUY"], "target_shares": [100]}
    )
    result = BacktestEngine(ExecutionRules(slippage_bps=0)).run(bars, signals, 2000.0)
    evaluation = evaluate_backtest(result)
    assert evaluation.metrics["net_return"] == pytest.approx(0.04985)
    assert evaluation.metrics["failed_fill_rate"] == 0.0


def test_calibration_and_cost_stress_are_explicit() -> None:
    """校准和成本压力测试输出可解释结果。"""
    predictions = pd.DataFrame(
        {
            "score": [0.1, 0.2, 0.8, 0.9],
            "realized_return": [0.0, 0.1, 0.7, 0.8],
        }
    )
    calibration = calibration_table(predictions, bins=2)
    assert calibration.sample_counts["rows"] == 4
    assert calibration.metrics["calibration_error"] >= 0

    bars = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=4),
            "code": "600519",
            "open": [10, 10, 11, 12],
            "close": [10, 10, 11, 12],
            "volume": [1000] * 4,
            "execution_volume": [1000] * 4,
            "can_trade": [True] * 4,
            "suspended": [False] * 4,
            "limit_up": [False] * 4,
            "limit_down": [False] * 4,
        }
    )
    signals = pd.DataFrame(
        {
            "signal_date": [pd.Timestamp("2024-01-01")],
            "code": ["600519"],
            "action": ["BUY"],
            "target_shares": [100],
        }
    )
    stress = cost_stress_matrix(bars, signals, initial_cash=2000)
    assert stress.sample_counts["scenarios"] == 4
