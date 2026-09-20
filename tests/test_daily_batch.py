"""收盘后批处理的有界、幂等和失败保留测试。"""
from __future__ import annotations

from decision.jobs import JobManager, SynchronousExecutor
from research.daily_batch import DailyBatchRunner


def test_daily_batch_saves_all_item_results_and_can_restart(tmp_path) -> None:
    runner = DailyBatchRunner(tmp_path / "batch.json", max_items=3)
    manager = JobManager(max_pending=3, executor=SynchronousExecutor())
    calls: list[str] = []

    def worker(code: str, as_of: str) -> dict[str, str]:
        calls.append(code)
        if code == "000002":
            raise RuntimeError("夹具失败")
        return {"code": code, "as_of": as_of}

    result = runner.run(["000001", "000002"], as_of="2024-01-02", worker=worker, batch_id="b1", job_manager=manager)
    manager.close()
    assert result.items[0].status == "SUCCEEDED"
    assert result.items[1].status == "FAILED"
    assert result.succeeded is False
    assert calls == ["000001", "000002"]
    assert runner.checkpoint_path.exists()


def test_daily_batch_rejects_duplicate_codes_and_unbounded_scope(tmp_path) -> None:
    runner = DailyBatchRunner(tmp_path / "batch.json", max_items=1)
    try:
        runner.run(["000001", "000001"], as_of="2024-01-02", worker=lambda *_: 1, batch_id="b")
    except ValueError as exc:
        assert "重复" in str(exc)
    else:
        raise AssertionError("重复证券未被拒绝")
