"""前瞻模拟测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.forward_sim import attach_actuals, forward_report, simulate_forward


def _prices() -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", "2024-03-31")
    return pd.DataFrame({"600519": 100 * (1 + np.linspace(0, 0.2, len(index))), "000001": 100 * (1 - np.linspace(0, 0.1, len(index))), "300750": 100 * (1 + np.linspace(0, 0.05, len(index)))}, index=index)


def _predictions() -> pd.DataFrame:
    rows = []
    for anchor in ("2024-01-05", "2024-01-12"):
        for index, code in enumerate(("600519", "000001", "300750")):
            rows.append({"prediction_key": f"{anchor}-{code}", "anchor_date": anchor, "execution_date": anchor, "stock_code": code, "horizon": 20, "score": float(3 - index), "predicted_return": 0.0, "actual_return": float("nan"), "data_as_of": anchor})
    return pd.DataFrame(rows)


def test_attach_actuals_fills_future_returns() -> None:
    frame = attach_actuals(_predictions(), _prices())
    assert frame["actual_return"].notna().all()
    assert (frame["actual_return"] > -0.5).all()


def test_forward_simulation_picks_topk_per_anchor() -> None:
    simulated = simulate_forward(_predictions(), _prices(), topk=2)
    assert len(simulated) == 2
    assert (simulated["stocks"] == 2).all()
    report = forward_report(simulated)
    assert report["status"] == "completed"
    assert report["periods"] == 2
    assert "disclaimer" in report
