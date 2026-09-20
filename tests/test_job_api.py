"""异步决策任务 API 的状态和取消边界。"""
from __future__ import annotations

from decision.jobs import JobManager, SynchronousExecutor
from webui.app import app
from webui.job_service import DecisionJobService


def test_decision_job_api_returns_status(monkeypatch) -> None:
    class Pipeline:
        def run_baseline(self, code: str, *, as_of: str | None = None):
            return {"code": code, "as_of": as_of}

    service = DecisionJobService(lambda: Pipeline(), manager=JobManager(max_pending=2, executor=SynchronousExecutor()))
    monkeypatch.setattr("webui.app._get_job_service", lambda: service)
    client = app.test_client()
    response = client.post(
        "/api/v2/decision-jobs",
        json={"stock_code": "600519", "as_of": "2024-01-02", "mode": "baseline"},
        headers={"Origin": "http://127.0.0.1:7070"},
    )
    assert response.status_code == 202
    job_id = response.get_json()["job_id"]
    status = client.get(f"/api/v2/decision-jobs/{job_id}")
    assert status.status_code == 200
    assert status.get_json()["status"] == "SUCCEEDED"
    service.close()
