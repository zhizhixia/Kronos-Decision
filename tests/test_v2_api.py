"""v2 决策报告 API 的日期契约测试。"""
from __future__ import annotations

import pandas as pd


def test_v2_api_validates_and_forwards_as_of(monkeypatch) -> None:
    from webui.app import app

    observed = {}

    def fake_report(self, stock_code, portfolio_id, include_display_paths, as_of):
        observed.update({"stock_code": stock_code, "portfolio_id": portfolio_id, "as_of": as_of})
        return {"status": "ok", "evidence_gate": {"checks": {}}, "recommendation": {"action": "INSUFFICIENT_EVIDENCE"}, "horizons": {}, "generated_at": "2024-01-05T16:00:00"}

    class FakeStore:
        def save_recommendation(self, *args, **kwargs):
            return 1

    monkeypatch.setattr("decision.engine.DecisionEngine.decision_report_v2", fake_report)
    monkeypatch.setattr("portfolio.store.PortfolioStore", FakeStore)
    client = app.test_client()

    response = client.post("/api/v2/decision-report", json={"stock_code": "600519", "as_of": "2024-01-05"})

    assert response.status_code == 200
    assert observed == {"stock_code": "600519", "portfolio_id": "default", "as_of": pd.Timestamp("2024-01-05")}


def test_v2_api_rejects_invalid_as_of_without_running_engine(monkeypatch) -> None:
    from webui.app import app

    monkeypatch.setattr("decision.engine.DecisionEngine.decision_report_v2", lambda *args: (_ for _ in ()).throw(AssertionError("不应调用引擎")))
    response = app.test_client().post("/api/v2/decision-report", json={"stock_code": "600519", "as_of": "not-a-date"})

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "INVALID_REQUEST"


def test_recommendations_api_returns_new_ledger_fields_and_rejects_invalid_limit(monkeypatch) -> None:
    from webui.app import app

    class FakeStore:
        def list_recommendations(self, profile_id, limit):
            assert profile_id == "default"
            assert limit == 1
            return [{"stock_code": "600519", "data_as_of": "2024-01-02", "evidence_gate_passed": True}]

    monkeypatch.setattr("portfolio.store.PortfolioStore", FakeStore)
    client = app.test_client()

    ok = client.get("/api/v2/recommendations?limit=1")
    invalid = client.get("/api/v2/recommendations?limit=not-a-number")

    assert ok.status_code == 200
    assert ok.get_json()["recommendations"][0]["data_as_of"] == "2024-01-02"
    assert invalid.status_code == 400
    assert invalid.get_json()["error"]["code"] == "INVALID_REQUEST"


def test_forward_ledger_api_returns_latest_or_not_found(monkeypatch) -> None:
    from webui.app import app

    client = app.test_client()
    monkeypatch.setattr("evaluation.forward_ledger.load_latest_forward_report", lambda: {"artifact": "2024-01-30.json", "report": {"status": "awaiting_maturity"}})
    ok = client.get("/api/v2/forward-ledger/latest")
    monkeypatch.setattr("evaluation.forward_ledger.load_latest_forward_report", lambda: None)
    missing = client.get("/api/v2/forward-ledger/latest")

    assert ok.status_code == 200
    assert ok.get_json()["artifact"] == "2024-01-30.json"
    assert missing.status_code == 404
    assert missing.get_json()["error"]["code"] == "FORWARD_LEDGER_NOT_FOUND"
