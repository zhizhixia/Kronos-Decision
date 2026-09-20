"""受控任务状态机的取消、超时和故障注入测试。"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from decision.jobs import JobManager, JobStatus, SynchronousExecutor


def _real_thread_failure(context) -> None:
    """在真实执行器线程中注入一个可观察异常。"""
    context.report("故障注入", 30.0)
    raise RuntimeError("真实线程故障")


def _late_success_worker(started: threading.Event, release: threading.Event):
    """返回一个会在取消或超时之后晚到的结果。"""
    def work(context):
        """执行可在取消或超时后晚到的任务。"""
        started.set()
        context.report("执行中", 40.0)
        release.wait(3.0)
        # 这次更新和返回值都必须被状态机拒绝。
        context.report("晚到结果", 95.0)
        return "不应提交"

    return work


def _queued_worker() -> str:
    """提供一个可排队的短任务。"""
    return "queued-result"


def test_real_thread_exception_becomes_failed() -> None:
    """真实 ThreadPoolExecutor 中的异常必须成为 FAILED，而非丢失。"""
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kronos-failure")
    manager = JobManager(executor=executor, max_pending=2)
    try:
        job = manager.submit(_real_thread_failure, job_id="thread-failure")
        result = manager.wait(job.job_id, timeout=2.0)
        assert result.status is JobStatus.FAILED
        assert result.result is None
        assert result.error_type == "RuntimeError"
        assert result.error is not None
        assert "真实线程故障" in result.error
    finally:
        manager.close()


def test_cancelled_running_result_is_rejected_by_state_and_version() -> None:
    """运行中取消后，晚到结果和进度不能改写 CANCELLED 快照。"""
    started = threading.Event()
    release = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kronos-cancel")
    manager = JobManager(executor=executor, max_pending=1)
    try:
        job = manager.submit(
            _late_success_worker(started, release),
            job_id="cancel-running",
        )
        assert started.wait(1.0)
        before_cancel = manager.get(job.job_id)
        requested = manager.cancel(job.job_id)
        assert requested.status in {JobStatus.CANCELLING, JobStatus.CANCELLED}
        assert requested.version > before_cancel.version

        release.set()
        final = manager.wait(job.job_id, timeout=2.0)
        assert final.status is JobStatus.CANCELLED
        assert final.result is None
        assert final.stage == "cancelled"
        after_late_callback = manager.get(job.job_id)
        assert after_late_callback == final
    finally:
        release.set()
        manager.close()


def test_timeout_is_terminal_before_worker_returns_and_rejects_late_result() -> None:
    """任务超时应立即终态化，不能只放弃 HTTP 等待。"""
    started = threading.Event()
    release = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kronos-timeout")
    manager = JobManager(executor=executor, max_pending=1)
    try:
        job = manager.submit(
            _late_success_worker(started, release),
            job_id="timeout-running",
            timeout=0.05,
        )
        assert started.wait(1.0)
        timed_out = manager.wait(job.job_id, timeout=2.0)
        assert timed_out.status is JobStatus.TIMED_OUT
        assert timed_out.result is None
        assert timed_out.error_type == "TimeoutError"
        version = timed_out.version

        release.set()
        assert manager.wait(job.job_id, timeout=2.0).status is JobStatus.TIMED_OUT
        after_late_callback = manager.get(job.job_id)
        assert after_late_callback.result is None
        assert after_late_callback.version == version
    finally:
        release.set()
        manager.close()


def test_queued_cancel_is_terminal_and_does_not_start_work() -> None:
    """排队任务被取消后，Future 不应执行工作函数。"""
    started = threading.Event()
    release = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kronos-queued-cancel")
    manager = JobManager(executor=executor, max_pending=2)
    try:
        blocker = manager.submit(_late_success_worker(started, release), job_id="blocker")
        assert started.wait(1.0)
        queued_calls: list[str] = []

        def queued():
            """记录排队任务是否真的被执行。"""
            queued_calls.append("started")
            return _queued_worker()

        queued_job = manager.submit(queued, job_id="queued-cancel")
        cancelled = manager.cancel(queued_job.job_id)
        assert cancelled.status is JobStatus.CANCELLED
        assert manager.wait(queued_job.job_id, timeout=2.0).status is JobStatus.CANCELLED
        assert queued_calls == []
        release.set()
        assert manager.wait(blocker.job_id, timeout=2.0).status is JobStatus.SUCCEEDED
    finally:
        release.set()
        manager.close()


def test_synchronous_worker_exception_is_failed_without_mocking_executor() -> None:
    """同步测试执行器也必须把工作异常保留为 FAILED。"""
    def failing() -> None:
        """注入一个同步 ValueError。"""
        raise ValueError("同步注入故障")

    manager = JobManager(executor=SynchronousExecutor())
    try:
        result = manager.submit(failing, job_id="sync-failure")
        assert result.status is JobStatus.FAILED
        assert result.error_type == "ValueError"
        assert result.error is not None and "同步注入故障" in result.error
    finally:
        manager.close()
