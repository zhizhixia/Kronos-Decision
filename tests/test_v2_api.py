"""v2 决策报告 API 的日期契约测试。"""
from __future__ import annotations

import pandas as pd
from types import SimpleNamespace


def test_v2_api_validates_and_forwards_as_of(monkeypatch) -> None:
    from webui.app import app

    observed = {}

    def fake_report(self, stock_code, portfolio_id, include_display_paths, as_of):
        observed.update({"stock_code": stock_code, "portfolio_id": portfolio_id, "as_of": as_of})
        return {"status": "ok", "evidence_gate": {"checks": {}}, "recommendation": {"action": "INSUFFICIENT_EVIDENCE"}, "horizons": {}, "generated_at": "2024-01-05T16:00:00"}

    class FakeStore:
        def save_recommendation(self, *args, **kwargs):
            return 1

    class FakePipeline:
        """绕过真实取数，验证 API 对流水线参数的转发。"""

        def run(self, stock_code, *, portfolio_id, include_display_paths, as_of):
            report = fake_report(None, stock_code, portfolio_id, include_display_paths, as_of)
            return SimpleNamespace(
                report=report,
                report_id="decision:test:2024-01-05",
                version=1,
                snapshot_id="snapshot-test",
            )

    monkeypatch.setattr("webui.app._get_report_pipeline_class", lambda: FakePipeline)
    monkeypatch.setattr("portfolio.store.PortfolioStore", FakeStore)
    client = app.test_client()

    response = client.post(
        "/api/v2/decision-report",
        json={"stock_code": "600519", "as_of": "2024-01-05"},
        headers={"Origin": "http://127.0.0.1:7070"},
    )

    assert response.status_code == 200
    assert observed == {"stock_code": "600519", "portfolio_id": "default", "as_of": pd.Timestamp("2024-01-05")}


def test_v2_api_rejects_invalid_as_of_without_running_engine(monkeypatch) -> None:
    from webui.app import app

    monkeypatch.setattr("webui.app._get_report_pipeline_class", lambda: (_ for _ in ()).throw(AssertionError("不应调用流水线")))
    response = app.test_client().post(
        "/api/v2/decision-report",
        json={"stock_code": "600519", "as_of": "not-a-date"},
        headers={"Origin": "http://127.0.0.1:7070"},
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "INVALID_REQUEST"


def test_v2_history_api_reads_persisted_report_without_running_engine(monkeypatch) -> None:
    """历史接口只通过流水线读回，并返回快照和版本引用。"""
    from webui.app import app

    report = {
        "schema_version": "2.0",
        "status": "degraded",
        "run_status": "SUCCEEDED",
        "evidence_status": "RESEARCH_ONLY",
        "action_permission": "NONE",
        "stock": {"code": "600519", "name": "测试"},
        "data_provenance": {"snapshot_id": "snapshot-test", "as_of": "2024-01-05"},
        "model_provenance": {},
        "chart_data": {"history": [], "forecast": {"timestamps": []}},
        "horizons": {"5": {}, "10": {}, "20": {}},
        "recommendation": {"action": None, "legacy_signal": None, "reason_codes": []},
        "evidence_gate": {"passed": False, "checks": {}},
    }

    class FakePipeline:
        """提供只读历史夹具。"""

        def read_published(self, report_id):
            assert report_id == "decision:600519:2024-01-05"
            return SimpleNamespace(
                payload=report,
                report_id=report_id,
                version=1,
                snapshot_id="snapshot-test",
            )

    monkeypatch.setattr("webui.app._get_report_pipeline_class", lambda: FakePipeline)
    response = app.test_client().get("/api/v2/decision-report/decision:600519:2024-01-05")

    assert response.status_code == 200
    body = response.get_json()
    assert body["publication_status"] == "PUBLISHED"
    assert body["snapshot_id"] == "snapshot-test"
    assert body["report_version"] == 1


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
