"""AKQuant 隔离对照测试。"""
from __future__ import annotations

import pandas as pd

from evaluation.akquant_compare import compare_engines, signal_hash


def _predictions() -> pd.DataFrame:
    rows = []
    for anchor in ("2024-01-05", "2024-01-12"):
        for code in ("600519", "000001", "300750"):
            rows.append({"anchor_date": anchor, "execution_date": anchor, "stock_code": code, "horizon": 20, "score": 0.5})
    return pd.DataFrame(rows)


def _returns(length: int = 40, scale: float = 1.0) -> pd.Series:
    return pd.Series([0.001 * scale] * length)


def test_signal_hash_is_stable_and_order_independent() -> None:
    first = signal_hash(_predictions())
    second = signal_hash(_predictions().sample(frac=1, random_state=1))
    assert first == second


def test_without_akquant_keeps_qlib() -> None:
    result = compare_engines(_predictions(), _returns())
    assert result["status"] == "AKQUANT_NOT_INSTALLED"
    assert result["decision"] == "keep_qlib"


def test_unexplained_difference_keeps_qlib() -> None:
    import numpy as np

    qlib = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, size=40))
    akquant = pd.Series(np.random.default_rng(2).normal(0.001, 0.01, size=40))
    result = compare_engines(_predictions(), qlib, akquant)
    assert result["decision"] == "keep_qlib"
    assert "相关性" in result["reason"]


def test_consistent_and_better_akquant_triggers_evaluation() -> None:
    result = compare_engines(_predictions(), _returns(scale=1.0), _returns(scale=2.0))
    assert result["decision"] == "consider_evaluation"
