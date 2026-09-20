"""任务 API 服务的状态、幂等和取消边界。"""
from __future__ import annotations

from decision.jobs import JobManager, SynchronousExecutor
from webui.job_service import DecisionJobService


def test_decision_job_service_returns_terminal_status_and_deduplicates() -> None:
    class Pipeline:
        def run_baseline(
            self,
            code: str,
            *,
            as_of: str | None = None,
            context=None,
        ) -> dict[str, object]:
            assert context is not None
            assert not context.is_cancelled()
            return {"code": code, "as_of": as_of}

    manager = JobManager(max_pending=2, executor=SynchronousExecutor())
    service = DecisionJobService(lambda: Pipeline(), manager=manager)
    first = service.submit("600519", as_of="2024-01-02", mode="baseline")
    second = service.submit("600519", as_of="2024-01-02", mode="baseline")
    assert first.job_id == second.job_id
    status = service.get(first.job_id)
    assert status["status"] == "SUCCEEDED"
    assert status["result"]["code"] == "600519"
    service.close()
