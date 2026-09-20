"""受控任务状态机的主流程测试。"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from decision.jobs import (
    DuplicateJobError,
    JobManager,
    JobStatus,
    ManagerClosedError,
    QueueFullError,
    SynchronousExecutor,
    UnknownJobError,
)


def _quick_worker(context) -> dict[str, object]:
    """执行一个带阶段报告的短任务。"""
    assert context.report("准备输入", 25.0)
    assert context.report_progress("完成计算", 75.0)
    return {"ok": True}


def _blocking_worker(started: threading.Event, release: threading.Event):
    """等待测试信号，以便观察运行中状态。"""
    def work(context):
        """执行可释放的阻塞任务。"""
        started.set()
        assert context.report("等待资源", 20.0)
        release.wait(3.0)
        context.report("晚到阶段", 90.0)
        return "late-result"

    return work


def _wait_for_thread_prefix(prefix: str, timeout: float = 2.0) -> bool:
    """等待指定前缀的执行器线程消失。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(thread.name.startswith(prefix) for thread in threading.enumerate()):
            return True
        time.sleep(0.01)
    return not any(thread.name.startswith(prefix) for thread in threading.enumerate())


def test_synchronous_executor_success_progress_and_lookup() -> None:
    """同步执行器应覆盖成功、进度、结果和未知任务错误。"""
    executor = SynchronousExecutor()
    manager = JobManager(executor=executor, max_pending=2)
    try:
        submitted = manager.submit(
            _quick_worker,
            job_id="sync-job",
            idempotency_key="sync-key",
        )
        assert submitted.status is JobStatus.SUCCEEDED
        assert submitted.result == {"ok": True}
        assert submitted.progress == 100.0
        assert submitted.stage == "completed"
        assert submitted.version >= 4
        assert manager.get("sync-job") == submitted
        with pytest.raises(UnknownJobError):
            manager.get("missing-job")
    finally:
        manager.close()
    assert executor.closed


def test_bounded_admission_and_idempotency_do_not_run_duplicate() -> None:
    """活动任务达到上限时拒绝新任务，但幂等重试仍返回原任务。"""
    started = threading.Event()
    release = threading.Event()
    calls: list[str] = []
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kronos-bounded")
    manager = JobManager(
        executor=executor,
        max_pending=1,
        thread_name_prefix="unused-prefix",
    )
    try:
        def work(context):
            """执行第一个占满准入槽的任务。"""
            calls.append("first")
            started.set()
            context.report("等待", 10.0)
            release.wait(3.0)
            return "first-result"

        first = manager.submit(work, job_id="bounded-1", idempotency_key="same-request")
        assert started.wait(1.0)
        observed = manager.get(first.job_id)
        assert observed.status is JobStatus.RUNNING
        assert observed.stage == "等待"
        assert observed.progress == 10.0

        retry = manager.submit(
            _quick_worker,
            job_id="different-id-is-ignored-by-key",
            idempotency_key="same-request",
        )
        assert retry.job_id == first.job_id
        assert len(calls) == 1
        with pytest.raises(QueueFullError):
            manager.submit(_quick_worker, job_id="overflow")
        with pytest.raises(DuplicateJobError):
            manager.submit(_quick_worker, job_id="bounded-1", idempotency_key="other-key")

        release.set()
        finished = manager.wait(first.job_id, timeout=2.0)
        assert finished.status is JobStatus.SUCCEEDED
        assert finished.result == "first-result"
        next_job = manager.submit(_quick_worker, job_id="after-release")
        assert manager.wait(next_job.job_id, timeout=2.0).status is JobStatus.SUCCEEDED
    finally:
        release.set()
        manager.close()
    assert _wait_for_thread_prefix("kronos-bounded")


def test_running_progress_and_close_leave_no_executor_thread() -> None:
    """真实线程任务完成后，close 应回收线程且不留下常驻资源。"""
    started = threading.Event()
    release = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kronos-lifecycle")
    manager = JobManager(executor=executor, max_pending=2)
    try:
        job = manager.submit(_blocking_worker(started, release), job_id="lifecycle")
        assert started.wait(1.0)
        running = manager.get(job.job_id)
        assert running.status is JobStatus.RUNNING
        assert running.progress == 20.0
        release.set()
        assert manager.wait(job.job_id, timeout=2.0).status is JobStatus.SUCCEEDED
    finally:
        release.set()
        manager.close()
    assert _wait_for_thread_prefix("kronos-lifecycle")


def test_cancelled_running_worker_cannot_commit_late_result() -> None:
    """线程任务取消后即使晚到结果，也必须保持 CANCELLED。"""
    started = threading.Event()
    release = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kronos-cancel")
    manager = JobManager(executor=executor, max_pending=1)

    def work(context):
        """等待取消请求后返回一个不应提交的结果。"""
        started.set()
        release.wait(2.0)
        assert context.is_cancelled()
        return "late-result"

    try:
        job = manager.submit(work, job_id="cancel-late")
        assert started.wait(1.0)
        assert manager.cancel(job.job_id).status is JobStatus.CANCELLING
        release.set()
        finished = manager.wait(job.job_id, timeout=2.0)
        assert finished.status is JobStatus.CANCELLED
        assert finished.result is None
    finally:
        release.set()
        manager.close()
    assert _wait_for_thread_prefix("kronos-cancel")


def test_manager_rejects_submission_after_close() -> None:
    """关闭后的管理器不能静默创建第二套执行资源。"""
    manager = JobManager(executor=SynchronousExecutor())
    manager.close()
    with pytest.raises(ManagerClosedError):
        manager.submit(_quick_worker)
