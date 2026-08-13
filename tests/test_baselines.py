"""基线比较模块测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.baselines import compare_baselines, equal_weight_returns, momentum_rank_ic, point_in_time_equal_weight_returns, random_walk_rank_ic


def _predictions() -> pd.DataFrame:
    rows = []
    for anchor_index, anchor in enumerate(("2024-01-05", "2024-01-12")):
        for stock_index, code in enumerate(("600519", "000001", "300750")):
            rows.append({"anchor_date": anchor, "execution_date": "2024-01-08", "stock_code": code, "horizon": 20, "score": float(stock_index + anchor_index), "actual_return": float(stock_index) / 100})
    return pd.DataFrame(rows)


def _prices() -> pd.DataFrame:
    index = pd.bdate_range("2023-12-01", "2024-01-20")
    data = {}
    for index_i, code in enumerate(("600519", "000001", "300750")):
        base = 100 + index_i * 10
        data[code] = base * (1 + np.linspace(0, 0.05, len(index)))
    return pd.DataFrame(data, index=index)


def test_random_walk_rank_ic_is_zero() -> None:
    frame = _predictions()
    frame["actual_return"] = frame["score"] / 100
    ic = random_walk_rank_ic(frame)
    assert ic.isna().all() or ic.abs().max() < 1e-9


def test_momentum_uses_visible_past_returns() -> None:
    frame = _predictions()
    frame["actual_return"] = frame["score"] / 100
    ic = momentum_rank_ic(frame, _prices())
    assert not ic.empty


def test_equal_weight_and_benchmark_returns() -> None:
    prices = _prices()
    equal = equal_weight_returns(prices)
    assert equal.iloc[0] == 0.0
    hs300 = pd.Series(100 * (1 + np.linspace(0, 0.1, len(prices))), index=prices.index)
    summary = compare_baselines(_predictions(), prices, hs300, False)
    assert summary["benchmark_is_total_return"] is False
    assert summary["benchmark_annualized_return"] > 0
    assert summary["point_in_time_equal_weight"]
    assert summary["point_in_time_equal_weight_observations"] == 2


def test_point_in_time_equal_weight_uses_each_anchor_membership() -> None:
    frame = pd.DataFrame([
        {"anchor_date": "2024-01-05", "stock_code": "600519", "horizon": 20, "actual_return": 0.10},
        {"anchor_date": "2024-01-05", "stock_code": "000001", "horizon": 20, "actual_return": 0.00},
        {"anchor_date": "2024-01-12", "stock_code": "300750", "horizon": 20, "actual_return": -0.02},
    ])

    returns = point_in_time_equal_weight_returns(frame)

    assert returns.to_dict() == {"2024-01-05": 0.05, "2024-01-12": -0.02}


def test_compare_baselines_summary_shape() -> None:
    frame = _predictions()
    frame["actual_return"] = frame["score"] / 100
    summary = compare_baselines(frame, _prices(), pd.Series(100 * (1 + np.linspace(0, 0.1, len(_prices()))), index=_prices().index))
    assert summary["rank_ic_mean_model"] == 1.0
    assert "rank_ic_mean_random_walk" in summary
    assert "rank_ic_mean_momentum_20d" in summary
    assert "benchmark_is_total_return" in summary


def test_baseline_rank_ic_ignores_non_primary_horizons() -> None:
    """5/60日标签不能污染20日模型与基线的横截面比较。"""
    from evaluation.baselines import model_rank_ic

    frame = _predictions()
    short = frame.assign(horizon=5, actual_return=-frame["actual_return"])
    multi = pd.concat([frame, short], ignore_index=True)

    assert (model_rank_ic(multi) > 0).all()
