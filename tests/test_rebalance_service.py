"""模拟调仓服务测试。"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from data.industry import IndustryMap
from evaluation.artifacts import EvaluationArtifacts
from evaluation.binding import EvaluationBinding
from evaluation.gates import evaluate_evidence_gate
from portfolio.service import RebalanceService
from portfolio.store import PortfolioStore


def _gate_metrics() -> dict:
    return {"data_complete": True, "quality_error": False, "coverage_80": 0.8, "up_probability_ece": 0.08, "rank_ic_mean": 0.01, "rank_ic_positive_probability": 0.8, "annualized_excess_return": 0.01, "information_ratio": 0.5, "positive_12m_window_ratio": 0.6, "drawdown_worsening": 0.05, "versions_match": True, "formal_protocol": True, "evaluated_at": datetime.now().isoformat()}


def _write_binding(root: Path) -> EvaluationBinding:
    artifacts = EvaluationArtifacts.open("rebalance-test", root)
    artifacts.write_manifest({"status": "completed", "as_of": "2026-08-01", "seed": 1, "horizons": [20], "model": "Kronos", "config_hash": "cfg", "data_hash": "data"})
    artifacts.write_gate_result(evaluate_evidence_gate(_gate_metrics()).as_dict())
    rows = []
    for index in range(20):
        code = f"{index:06d}"
        rows.append({"prediction_key": f"k-{code}", "anchor_date": "2026-08-01", "execution_date": "2026-08-04", "stock_code": code, "horizon": 20, "score": float(index), "predicted_return": 0.01, "actual_return": 0.01, "data_as_of": "2026-08-01", "model_hash": "m", "config_hash": "cfg", "data_hash": "data", "raw_up_probability": 0.6, "q05": -0.05, "q50": -0.01 + index * 0.001, "q95": 0.1})
    artifacts.write_frame("predictions", pd.DataFrame(rows))
    return EvaluationBinding.load_latest(root)


def _prices(code: str) -> pd.DataFrame:
    dates = pd.bdate_range("2025-06-01", periods=260)
    return pd.DataFrame({"date": dates, "close": 100 + np.arange(len(dates)) * 0.01})


def test_rebalance_generates_rounded_simulated_orders(tmp_path) -> None:
    binding = _write_binding(tmp_path)
    industry = IndustryMap(tmp_path / "industry.csv")
    source = tmp_path / "source.csv"
    pd.DataFrame({"stock_code": [f"{i:06d}" for i in range(20)], "industry": [f"s{i % 4}" for i in range(20)]}).to_csv(source, index=False)
    industry.import_csv(source)
    store = PortfolioStore(tmp_path / "kronos.db")
    store.replace_portfolio({"risk_profile": "balanced", "cash": 500_000, "holdings": [{"stock_code": "000000", "shares": 100, "cost_basis": 100}]})
    service = RebalanceService(store, industry)
    result = service.rebalance(binding, "default", _prices)
    assert result["status"] == "ok"
    assert result["orders"]
    assert all(order["shares"] % 100 == 0 for order in result["orders"] if order["side"] == "BUY")
    assert result["estimated_fees"] > 0
    assert all(order["estimated_fee"] > 0 for order in result["orders"])
    assert result["remaining_cash"] >= 0
    assert store.get_portfolio("default")["cash"] >= 0


def test_rebalance_without_industry_map_fails_explicitly(tmp_path) -> None:
    binding = _write_binding(tmp_path)
    store = PortfolioStore(tmp_path / "kronos.db")
    service = RebalanceService(store, IndustryMap(tmp_path / "missing.csv"))
    result = service.rebalance(binding, "default", _prices)
    assert result["status"] == "insufficient_evidence"
    assert "INDUSTRY_MAPPING_MISSING" in result["reason_codes"]


def test_rebalance_fails_closed_when_held_position_has_no_price(tmp_path) -> None:
    binding = _write_binding(tmp_path)
    industry = IndustryMap(tmp_path / "industry.csv")
    source = tmp_path / "source.csv"
    pd.DataFrame({"stock_code": [f"{i:06d}" for i in range(20)], "industry": [f"s{i % 4}" for i in range(20)]}).to_csv(source, index=False)
    industry.import_csv(source)
    store = PortfolioStore(tmp_path / "kronos.db")
    store.replace_portfolio({"risk_profile": "balanced", "cash": 500_000, "holdings": [{"stock_code": "999999", "shares": 100, "cost_basis": 100}]})

    def missing_held_price(code: str) -> pd.DataFrame:
        return pd.DataFrame() if code == "999999" else _prices(code)

    result = RebalanceService(store, industry).rebalance(binding, "default", missing_held_price)
    assert result["status"] == "insufficient_evidence"
    assert result["reason_codes"] == ("HELD_POSITION_PRICE_MISSING",)


def test_recommendation_history_roundtrip(tmp_path) -> None:
    store = PortfolioStore(tmp_path / "kronos.db")
    store.save_recommendation("default", "600519", "INSUFFICIENT_EVIDENCE", {"20": {"up_probability": None}}, "no-evidence", "2026-08-10T10:00:00", "2026-08-08", False)
    history = store.list_recommendations("default")
    assert len(history) == 1
    assert history[0]["stock_code"] == "600519"
    assert history[0]["action"] == "INSUFFICIENT_EVIDENCE"
    assert history[0]["data_as_of"] == "2026-08-08"
    assert history[0]["evidence_gate_passed"] is False
    assert history[0]["horizons"]["20"]["up_probability"] is None
    assert history[0]["evidence_version"] == "no-evidence"
