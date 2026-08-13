"""本地前瞻建议台账的成熟状态和 SQLite 迁移测试。"""
from __future__ import annotations

import json
import os
import sqlite3
import sys

import pandas as pd
import pytest

from evaluation.forward_ledger import evaluate_forward_ledger, forward_ledger_report, load_latest_forward_report
from portfolio.store import PortfolioStore


def _prices() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=22)
    return pd.DataFrame({"date": dates, "600519": range(100, 122)})


def test_forward_ledger_only_observes_matured_gate_passing_recommendations() -> None:
    recommendations = [{"id": 1, "stock_code": "600519", "action": "ADD", "data_as_of": "2024-01-02", "evidence_gate_passed": True}, {"id": 2, "stock_code": "600519", "action": "HOLD", "data_as_of": "2024-01-30", "evidence_gate_passed": True}, {"id": 3, "stock_code": "600519", "action": "INSUFFICIENT_EVIDENCE", "data_as_of": "2024-01-02", "evidence_gate_passed": False}]

    observations = evaluate_forward_ledger(recommendations, _prices())
    report = forward_ledger_report(observations)

    assert observations["status"].tolist() == ["MATURED", "PENDING_HORIZON", "EXCLUDED_EVIDENCE_GATE_FAILED"]
    assert observations.loc[0, "actual_return"] == pytest.approx(0.2)
    assert report["matured_recommendations"] == 1
    assert report["pending_recommendations"] == 1
    assert report["by_action"]["ADD"]["count"] == 1


def test_store_migrates_old_recommendation_history_schema(tmp_path) -> None:
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE recommendation_history (id INTEGER PRIMARY KEY, profile_id TEXT NOT NULL, stock_code TEXT NOT NULL, action TEXT NOT NULL, horizons_json TEXT NOT NULL, evidence_version TEXT NOT NULL, generated_at TEXT NOT NULL)")
    store = PortfolioStore(path)
    store.save_recommendation("default", "600519", "HOLD", {}, "gate-v1", "2024-01-02T16:00:00", "2024-01-02", True)

    history = store.list_recommendations()

    assert history[0]["data_as_of"] == "2024-01-02"
    assert history[0]["evidence_gate_passed"] is True


def test_forward_ledger_cli_writes_separate_csv_and_json(tmp_path, monkeypatch) -> None:
    from scripts.evaluate_forward_recommendations import main

    database = tmp_path / "kronos.db"
    prices = tmp_path / "prices.csv"
    output = tmp_path / "forward"
    PortfolioStore(database).save_recommendation("default", "600519", "ADD", {}, "gate-v1", "2024-01-02T16:00:00", "2024-01-02", True)
    _prices().to_csv(prices, index=False)
    monkeypatch.setattr(sys, "argv", ["evaluate_forward_recommendations.py", "--price-csv", str(prices), "--database", str(database), "--out-dir", str(output)])

    assert main() == 0

    reports = list(output.glob("*.json"))
    observations = list(output.glob("*.csv"))
    assert len(reports) == len(observations) == 1
    assert json.loads(reports[0].read_text(encoding="utf-8"))["matured_recommendations"] == 1


def test_latest_forward_report_skips_damaged_artifact(tmp_path) -> None:
    broken = tmp_path / "newer.json"
    valid = tmp_path / "valid.json"
    broken.write_text("{broken", encoding="utf-8")
    valid.write_text(json.dumps({"status": "awaiting_maturity"}), encoding="utf-8")
    os.utime(valid, (1, 1))
    os.utime(broken, (2, 2))

    latest = load_latest_forward_report(tmp_path)

    assert latest == {"artifact": "valid.json", "report": {"status": "awaiting_maturity"}}
