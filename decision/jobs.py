"""受控的有界任务状态机。

本模块只提供运行时原语，不导入模型、不创建全局线程或进程，也不注册计划任务。
任务执行器在 ``JobManager`` 实例化时显式创建，调用方可以注入同步执行器或
标准库执行器，从而分别测试状态机和真实短任务边界。
"""
from __future__ import annotations

import inspect
import itertools
import math
import multiprocessing
import threading
import time
import uuid
from concurrent.futures import CancelledError, Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Generic, Literal, Protocol, TypeVar, cast


T = TypeVar("T")


class JobError(RuntimeError):
    """任务状态机异常的基类。"""


class JobCancelledError(JobError):
    """工作函数在受保护副作用前发现任务已取消。"""


class InvalidJobError(JobError):
    """任务参数或任务执行器不符合约束。"""


class DuplicateJobError(JobError):
    """显式 job_id 已经被另一个任务占用。"""


class IdempotencyConflictError(JobError):
    """幂等键与请求中的显式任务标识冲突。"""


class QueueFullError(JobError):
    """有界准入已满，调用方需要稍后重试。"""


class UnknownJobError(JobError):
    """请求的任务不存在。"""


class ManagerClosedError(JobError):
    """任务管理器已经关闭，不再接受新任务。"""


class InvalidStateTransitionError(JobError):
    """内部状态迁移不符合任务状态机约束。"""


class JobStatus(str, Enum):
    """任务可观察的状态集合。"""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"


# 兼容计划和调用方可能使用的名称。
JobState = JobStatus

_TERMINAL_STATUSES = frozenset(
    {
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
        JobStatus.TIMED_OUT,
    }
)

_UNSET = object()

_ALLOWED_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.QUEUED: frozenset(
        {
            JobStatus.RUNNING,
            JobStatus.CANCELLING,
            JobStatus.CANCELLED,
            JobStatus.FAILED,
            JobStatus.TIMED_OUT,
        }
    ),
    JobStatus.RUNNING: frozenset(
        {
            JobStatus.CANCELLING,
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.TIMED_OUT,
        }
    ),
    JobStatus.CANCELLING: frozenset(
        {
            JobStatus.CANCELLED,
            JobStatus.FAILED,
            JobStatus.TIMED_OUT,
        }
    ),
    JobStatus.SUCCEEDED: frozenset(),
    JobStatus.FAILED: frozenset(),
    JobStatus.CANCELLED: frozenset(),
    JobStatus.TIMED_OUT: frozenset(),
}


class ExecutorLike(Protocol):
    """任务管理器所需的最小执行器协议。"""

    def submit(self, fn: Callable[..., object], /, *args: object, **kwargs: object) -> Future[object]:
        """提交一个可调用对象并返回 Future。"""

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """关闭执行器并按实现约定回收资源。"""


@dataclass(frozen=True, slots=True)
class JobSnapshot(Generic[T]):
    """任务状态的不可变快照。

    ``version`` 会在状态或进度更新时递增。结果只会出现在 ``SUCCEEDED`` 快照中；
    被取消、超时或失效版本的工作函数即使晚到，也不会写入该快照。
    """

    job_id: str
    idempotency_key: str | None
    status: JobStatus
    stage: str
    progress: float
    result: T | None
    error: str | None
    error_type: str | None
    version: int
    created_at: float
    started_at: float | None
    finished_at: float | None

    @property
    def state(self) -> JobStatus:
        """返回 status 的兼容别名。"""
        return self.status

    @property
    def terminal(self) -> bool:
        """判断任务是否已经进入终态。"""
        return self.status in _TERMINAL_STATUSES

    @property
    def is_terminal(self) -> bool:
        """返回 terminal 的方法式兼容别名。"""
        return self.terminal

    @property
    def elapsed_seconds(self) -> float | None:
        """返回已知的单调时钟耗时。"""
        end = self.finished_at
        if end is None:
            end = time.monotonic()
        start = self.started_at or self.created_at
        return max(0.0, end - start)

    def as_dict(self) -> dict[str, object]:
        """将快照转换成不包含内部对象的普通字典。"""
        return {
            "job_id": self.job_id,
            "idempotency_key": self.idempotency_key,
            "status": self.status.value,
            "state": self.status.value,
            "stage": self.stage,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "error_type": self.error_type,
            "version": self.version,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    def to_dict(self) -> dict[str, object]:
        """返回 as_dict 的兼容别名。"""
        return self.as_dict()


# 方案中的正式名称；保留 JobSnapshot 作为兼容名称。
JobRecord = JobSnapshot


class JobContext:
    """传给工作函数的受控上下文。

    工作函数可以接受一个 ``JobContext`` 参数，也可以是零参数函数。上下文回调
    只在任务仍拥有当前运行令牌时生效；超时或取消之后的进度更新会返回 False。
    进度范围为 0 到 100，阶段名称必须是非空文本。
    """

    def __init__(
        self,
        job_id: str,
        run_token: int,
        *,
        begin_callback: Callable[[], bool] | None = None,
        progress_callback: Callable[[str, float], bool] | None = None,
        cancel_callback: Callable[[], bool] | None = None,
        active_callback: Callable[[Callable[[], object]], object] | None = None,
        commit_callback: Callable[[], bool] | None = None,
    ) -> None:
        """创建一个绑定到单次运行令牌的上下文。"""
        self.job_id = job_id
        self.run_token = run_token
        self._begin_callback = begin_callback
        self._progress_callback = progress_callback
        self._cancel_callback = cancel_callback
        self._active_callback = active_callback
        self._commit_callback = commit_callback

    def report_progress(self, stage: str, progress: float) -> bool:
        """报告阶段和百分比；若任务版本已经失效则返回 False。"""
        if not isinstance(stage, str) or not stage.strip():
            raise InvalidJobError("任务阶段必须是非空文本。")
        if isinstance(progress, bool) or not isinstance(progress, (int, float)):
            raise InvalidJobError("任务进度必须是数值。")
        progress_value = float(progress)
        if not math.isfinite(progress_value) or not 0.0 <= progress_value <= 100.0:
            raise InvalidJobError("任务进度必须在 0 到 100 之间。")
        if self._progress_callback is None:
            # 进程执行器通过 spawn 传输上下文时不会携带管理器回调；最终结果仍会
            # 经过管理器的令牌校验，避免为了进度通道引入全局 Manager 进程。
            return False
        return bool(self._progress_callback(stage, progress_value))

    def report(self, stage: str, progress: float) -> bool:
        """报告任务进度的简短别名。"""
        return self.report_progress(stage, progress)

    def update_progress(self, stage: str, progress: float) -> bool:
        """报告任务进度的兼容别名。"""
        return self.report_progress(stage, progress)

    def is_cancelled(self) -> bool:
        """检查当前任务是否已经收到取消或超时请求。"""
        if self._cancel_callback is None:
            return False
        return bool(self._cancel_callback())

    def run_if_active(self, operation: Callable[[], T]) -> T:
        """在取消令牌保护下执行一次不可逆副作用。"""
        if not callable(operation):
            raise InvalidJobError("受保护操作必须可调用。")
        if self._active_callback is None:
            if self.is_cancelled():
                raise JobCancelledError("任务已取消，拒绝执行受保护操作。")
            return operation()
        return cast(T, self._active_callback(operation))

    def enter_commit(self) -> bool:
        """进入不可取消的最终提交边界；提交前仍会检查取消令牌。"""
        if self._commit_callback is None:
            if self.is_cancelled():
                raise JobCancelledError("任务已取消，拒绝进入提交边界。")
            return True
        return bool(self._commit_callback())

    @property
    def cancelled(self) -> bool:
        """提供属性式的取消检查。"""
        return self.is_cancelled()

    def _begin(self) -> bool:
        """在工作函数真正开始前请求进入 RUNNING。"""
        if self._begin_callback is None:
            return True
        return bool(self._begin_callback())

    def __getstate__(self) -> dict[str, object]:
        """仅序列化跨进程安全的上下文字段。"""
        return {"job_id": self.job_id, "run_token": self.run_token}

    def __setstate__(self, state: dict[str, object]) -> None:
        """从 spawn 子进程恢复上下文并清除不可序列化回调。"""
        self.job_id = cast(str, state["job_id"])
        self.run_token = cast(int, state["run_token"])
        self._begin_callback = None
        self._progress_callback = None
        self._cancel_callback = None
        self._active_callback = None
        self._commit_callback = None


class SynchronousExecutor:
    """不创建线程的同步测试执行器。

    该执行器用于验证状态机本身，不替代真实线程或进程证据。它实现了
    ``concurrent.futures`` Future 的异常传播和回调行为，适合确定性单元测试。
    """

    def __init__(self) -> None:
        """创建一个尚未关闭的同步执行器。"""
        self._closed = False
        self._lock = threading.Lock()

    def submit(self, fn: Callable[..., object], /, *args: object, **kwargs: object) -> Future[object]:
        """在当前调用线程执行任务并返回已完成的 Future。"""
        with self._lock:
            if self._closed:
                raise RuntimeError("同步执行器已经关闭。")
        future: Future[object] = Future()
        try:
            value = fn(*args, **kwargs)
        except BaseException as exc:
            future.set_exception(exc)
        else:
            future.set_result(value)
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """关闭同步执行器；参数只为兼容标准库执行器接口。"""
        del wait, cancel_futures
        with self._lock:
            self._closed = True

    @property
    def closed(self) -> bool:
        """返回执行器是否已经关闭。"""
        with self._lock:
            return self._closed


# 更短的测试名称，保留一个明确的主名称供文档使用。
InlineExecutor = SynchronousExecutor


@dataclass
class _JobEntry:
    """管理器内部的可变任务记录。"""

    job_id: str
    idempotency_key: str | None
    status: JobStatus
    stage: str
    progress: float
    version: int
    run_token: int | None
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    result: object | None = None
    error: str | None = None
    error_type: str | None = None
    future: Future[object] | None = None
    timer: threading.Timer | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    counted_as_active: bool = True


class JobManager:
    """提供有界提交、幂等去重和受控终态的任务管理器。

    ``max_pending`` 限制所有非终态任务的总数，而不是只限制执行器当前队列。
    任务管理器默认使用线程执行器；需要验证 Windows spawn 时可设置
    ``executor_kind="process"``，或直接注入实现了 ``ExecutorLike`` 的测试执行器。
    管理器不加载模型、不创建全局资源；调用 ``close`` 或使用上下文管理器会关闭
    它拥有的执行器和超时计时器。
    """

    def __init__(
        self,
        *,
        max_pending: int = 32,
        max_workers: int = 1,
        executor: ExecutorLike | None = None,
        executor_kind: Literal["thread", "process"] = "thread",
        default_timeout: float | None = None,
        timeout: float | None = None,
        thread_name_prefix: str = "kronos-job",
    ) -> None:
        """创建任务管理器，但只有真正提交任务时才会启动工作资源。"""
        self._validate_positive_integer(max_pending, "max_pending")
        self._validate_positive_integer(max_workers, "max_workers")
        if executor_kind not in {"thread", "process"}:
            raise InvalidJobError("executor_kind 必须是 thread 或 process。")
        if default_timeout is not None and timeout is not None:
            raise InvalidJobError("default_timeout 与 timeout 不能同时指定。")
        selected_timeout = default_timeout if default_timeout is not None else timeout
        self._validate_timeout(selected_timeout, "default_timeout")
        if not isinstance(thread_name_prefix, str) or not thread_name_prefix:
            raise InvalidJobError("thread_name_prefix 必须是非空文本。")

        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._jobs: dict[str, _JobEntry] = {}
        self._idempotency: dict[str, str] = {}
        self._active_count = 0
        self._max_pending = max_pending
        self._default_timeout = None if selected_timeout is None else float(selected_timeout)
        self._closed = False
        self._closing = False
        self._shutdown_complete = False
        self._run_tokens = itertools.count(1)

        if executor is None:
            if executor_kind == "process":
                spawn_context = multiprocessing.get_context("spawn")
                self._executor: ExecutorLike = ProcessPoolExecutor(
                    max_workers=max_workers,
                    mp_context=spawn_context,
                )
            else:
                self._executor = ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix=thread_name_prefix,
                )
        else:
            if not hasattr(executor, "submit") or not hasattr(executor, "shutdown"):
                raise InvalidJobError("executor 必须提供 submit 和 shutdown 方法。")
            self._executor = executor

    @property
    def max_pending(self) -> int:
        """返回非终态任务的准入上限。"""
        return self._max_pending

    @property
    def active_count(self) -> int:
        """返回当前非终态任务数量。"""
        with self._lock:
            return self._active_count

    @property
    def closed(self) -> bool:
        """返回管理器是否已经拒绝新任务。"""
        with self._lock:
            return self._closed

    def submit(
        self,
        work: Callable[..., T],
        *,
        job_id: str | None = None,
        idempotency_key: str | None = None,
        timeout: float | None = None,
    ) -> JobSnapshot[T]:
        """提交任务并返回其当前快照。

        重复的 ``job_id`` 或 ``idempotency_key`` 不会再次执行工作函数，而是返回已有
        任务快照。幂等查找在容量检查之前进行，因此重试请求不会因队列已满而产生
        第二个错误。工作函数自身的异常不会从 submit 抛出，而会记录为 FAILED。
        """
        if not callable(work):
            raise InvalidJobError("work 必须是可调用对象。")
        self._validate_identifier(job_id, "job_id")
        self._validate_identifier(idempotency_key, "idempotency_key")
        self._validate_timeout(timeout, "timeout")
        selected_timeout = self._default_timeout if timeout is None else float(timeout)

        with self._lock:
            self._ensure_open_locked()
            existing = self._find_duplicate_locked(job_id, idempotency_key)
            if existing is not None:
                return self._snapshot_locked(existing)
            if self._active_count >= self._max_pending:
                raise QueueFullError(
                    f"任务队列已满：活动任务 {self._active_count}，上限 {self._max_pending}。"
                )

            actual_job_id = job_id or uuid.uuid4().hex
            while actual_job_id in self._jobs:
                if job_id is not None:
                    raise DuplicateJobError(f"job_id 已存在：{job_id}。")
                actual_job_id = uuid.uuid4().hex
            run_token = next(self._run_tokens)
            now = time.monotonic()
            entry = _JobEntry(
                job_id=actual_job_id,
                idempotency_key=idempotency_key,
                status=JobStatus.QUEUED,
                stage="queued",
                progress=0.0,
                version=0,
                run_token=run_token,
                created_at=now,
            )
            self._jobs[actual_job_id] = entry
            if idempotency_key is not None:
                self._idempotency[idempotency_key] = actual_job_id
            self._active_count += 1
            if selected_timeout is not None:
                timer = threading.Timer(
                    selected_timeout,
                    self._on_timeout,
                    args=(actual_job_id, run_token),
                )
                timer.daemon = True
                entry.timer = timer
                timer.start()

        context = JobContext(
            actual_job_id,
            run_token,
            begin_callback=lambda: self._begin(actual_job_id, run_token),
            progress_callback=lambda stage, progress: self._report(
                actual_job_id, run_token, stage, progress
            ),
            cancel_callback=lambda: self._is_cancelled(actual_job_id, run_token),
            active_callback=lambda operation: self._run_if_active(
                actual_job_id, run_token, operation
            ),
            commit_callback=lambda: self._begin_commit(actual_job_id, run_token),
        )
        try:
            future = self._executor.submit(_invoke_worker, work, context)
        except BaseException as exc:
            self._record_submission_failure(actual_job_id, run_token, exc)
            return self.get(actual_job_id)

        with self._lock:
            entry = self._jobs.get(actual_job_id)
            if entry is not None:
                entry.future = future

        try:
            future.add_done_callback(
                lambda completed: self._future_done(actual_job_id, run_token, completed)
            )
        except BaseException as exc:
            self._record_submission_failure(actual_job_id, run_token, exc)
        return self.get(actual_job_id)

    def submit_job(
        self,
        work: Callable[..., T],
        *,
        job_id: str | None = None,
        idempotency_key: str | None = None,
        timeout: float | None = None,
    ) -> JobSnapshot[T]:
        """提交任务的显式命名别名。"""
        return self.submit(
            work,
            job_id=job_id,
            idempotency_key=idempotency_key,
            timeout=timeout,
        )

    def create_job(
        self,
        work: Callable[..., T],
        *,
        job_id: str | None = None,
        idempotency_key: str | None = None,
        timeout: float | None = None,
    ) -> JobSnapshot[T]:
        """创建任务的兼容别名；创建动作仍然立即受有界准入约束。"""
        return self.submit(
            work,
            job_id=job_id,
            idempotency_key=idempotency_key,
            timeout=timeout,
        )

    def get(self, job_id: str) -> JobSnapshot[Any]:
        """读取任务快照；未知 job_id 会抛出明确异常。"""
        self._validate_identifier(job_id, "job_id", required=True)
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None:
                raise UnknownJobError(f"任务不存在：{job_id}。")
            return self._snapshot_locked(entry)

    def get_job(self, job_id: str) -> JobSnapshot[Any]:
        """读取任务快照的显式命名别名。"""
        return self.get(job_id)

    def cancel(self, job_id: str) -> JobSnapshot[Any]:
        """请求取消任务。

        排队任务通常可以立即进入 CANCELLED；已经运行的任务先进入 CANCELLING，
        等工作函数返回后才进入 CANCELLED。无论工作函数是否合作，取消后的结果都
        不会提交到快照。
        """
        self._validate_identifier(job_id, "job_id", required=True)
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None:
                raise UnknownJobError(f"任务不存在：{job_id}。")
            if entry.status in _TERMINAL_STATUSES:
                return self._snapshot_locked(entry)
            token = entry.run_token
            if token is None:
                return self._snapshot_locked(entry)
            if entry.stage == "committing":
                return self._snapshot_locked(entry)
            if entry.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                self._transition_locked(
                    entry,
                    JobStatus.CANCELLING,
                    stage="cancelling",
                )
            entry.cancel_event.set()
            future = entry.future
            if future is not None and future.cancel():
                if entry.status == JobStatus.CANCELLING:
                    self._transition_locked(
                        entry,
                        JobStatus.CANCELLED,
                        stage="cancelled",
                        error="任务已取消。",
                        error_type="CancelledError",
                    )
            return self._snapshot_locked(entry)

    def cancel_job(self, job_id: str) -> JobSnapshot[Any]:
        """请求取消任务的显式命名别名。"""
        return self.cancel(job_id)

    def wait(self, job_id: str, timeout: float | None = None) -> JobSnapshot[Any]:
        """等待任务进入终态并返回快照。

        这里的 timeout 只限制调用方等待时长，不会修改任务的执行超时策略；任务自身
        的超时应在 submit 时传入。
        """
        self._validate_identifier(job_id, "job_id", required=True)
        self._validate_wait_timeout(timeout)
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._condition:
            while True:
                entry = self._jobs.get(job_id)
                if entry is None:
                    raise UnknownJobError(f"任务不存在：{job_id}。")
                if entry.status in _TERMINAL_STATUSES:
                    return self._snapshot_locked(entry)
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("等待任务进入终态超时。")
                self._condition.wait(remaining)

    def close(self, wait: bool = True) -> None:
        """关闭管理器、取消未完成任务并回收执行器和计时器资源。"""
        if not isinstance(wait, bool):
            raise InvalidJobError("close 的 wait 必须是布尔值。")
        with self._condition:
            if self._shutdown_complete:
                return
            if self._closing:
                if wait:
                    while not self._shutdown_complete:
                        self._condition.wait()
                return
            self._closing = True
            self._closed = True
            for entry in self._jobs.values():
                if entry.status in _TERMINAL_STATUSES:
                    continue
                if entry.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                    self._transition_locked(
                        entry,
                        JobStatus.CANCELLING,
                        stage="cancelling",
                    )
                entry.cancel_event.set()
                if entry.future is not None:
                    entry.future.cancel()
            executor = self._executor

        shutdown_error: BaseException | None = None
        try:
            try:
                executor.shutdown(wait=wait, cancel_futures=True)
            except TypeError:
                # 允许极简同步测试执行器只实现 shutdown(wait)。
                executor.shutdown(wait=wait)
        except BaseException as exc:
            shutdown_error = exc
        finally:
            timers: list[threading.Timer] = []
            with self._condition:
                if wait:
                    for entry in self._jobs.values():
                        if entry.status not in _TERMINAL_STATUSES:
                            self._transition_locked(
                                entry,
                                JobStatus.CANCELLED,
                                stage="cancelled",
                                error="任务管理器关闭，任务未完成。",
                                error_type="CancelledError",
                            )
                for entry in self._jobs.values():
                    if entry.timer is not None:
                        timers.append(entry.timer)
                self._shutdown_complete = True
                self._condition.notify_all()
            for timer in timers:
                timer.cancel()
            if wait:
                for timer in timers:
                    timer.join()
        if shutdown_error is not None:
            raise JobError(f"关闭任务执行器失败：{shutdown_error}") from shutdown_error

    def shutdown(self, wait: bool = True) -> None:
        """close 的标准库风格别名。"""
        self.close(wait=wait)

    def __enter__(self) -> JobManager:
        """进入上下文管理器并返回自身。"""
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """退出上下文管理器并确保资源被回收。"""
        del exc_type, exc_value, traceback
        self.close()

    def _find_duplicate_locked(
        self,
        job_id: str | None,
        idempotency_key: str | None,
    ) -> _JobEntry | None:
        """在持锁状态下查找 job_id 或幂等键对应的已有任务。"""
        if job_id is not None:
            existing = self._jobs.get(job_id)
            if existing is not None:
                if existing.idempotency_key == idempotency_key:
                    return existing
                raise DuplicateJobError(f"job_id 已存在且幂等键不同：{job_id}。")
        if idempotency_key is not None:
            existing_job_id = self._idempotency.get(idempotency_key)
            if existing_job_id is not None:
                existing = self._jobs.get(existing_job_id)
                if existing is None:
                    raise IdempotencyConflictError(
                        f"幂等键索引损坏，找不到任务：{idempotency_key}。"
                    )
                # 幂等重试优先返回最初请求，避免客户端重试时生成第二个 job_id。
                return existing
        return None

    def _begin(self, job_id: str, run_token: int) -> bool:
        """校验运行令牌并将排队任务迁移到 RUNNING。"""
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None or entry.run_token != run_token:
                return False
            if entry.status == JobStatus.QUEUED:
                self._transition_locked(entry, JobStatus.RUNNING, stage="running")
            return entry.status == JobStatus.RUNNING

    def _begin_commit(self, job_id: str, run_token: int) -> bool:
        """把运行任务切换到最终提交阶段，之后取消请求不再打断提交。"""
        with self._lock:
            entry = self._jobs.get(job_id)
            if (
                entry is None
                or entry.run_token != run_token
                or entry.status != JobStatus.RUNNING
                or entry.cancel_event.is_set()
            ):
                raise JobCancelledError("任务已取消，拒绝进入提交边界。")
            entry.stage = "committing"
            entry.progress = max(entry.progress, 99.0)
            entry.version += 1
            self._condition.notify_all()
            return True

    def _report(self, job_id: str, run_token: int, stage: str, progress: float) -> bool:
        """在令牌和状态均有效时写入阶段进度。"""
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None or entry.run_token != run_token:
                return False
            if entry.status != JobStatus.RUNNING:
                return False
            if progress < entry.progress:
                return False
            entry.stage = stage
            entry.progress = progress
            entry.version += 1
            self._condition.notify_all()
            return True

    def _is_cancelled(self, job_id: str, run_token: int) -> bool:
        """返回任务是否已被取消、超时或替换成了更新的运行版本。"""
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None or entry.run_token != run_token:
                return True
            return entry.cancel_event.is_set() or entry.status in {
                JobStatus.CANCELLING,
                JobStatus.CANCELLED,
                JobStatus.TIMED_OUT,
            }

    def _run_if_active(
        self,
        job_id: str,
        run_token: int,
        operation: Callable[[], object],
    ) -> object:
        """持有任务锁检查令牌，并串行化受保护副作用。"""
        with self._lock:
            entry = self._jobs.get(job_id)
            if (
                entry is None
                or entry.run_token != run_token
                or entry.status != JobStatus.RUNNING
                or entry.cancel_event.is_set()
            ):
                raise JobCancelledError("任务已取消，拒绝执行受保护操作。")
            return operation()

    def _future_done(self, job_id: str, run_token: int, future: Future[object]) -> None:
        """处理 Future 完成回调，并在提交终态前再次校验版本和状态。"""
        outcome: Literal["success", "failure", "cancelled"]
        value: object | None = None
        failure: BaseException | None = None
        try:
            value = future.result()
        except CancelledError:
            outcome = "cancelled"
        except BaseException as exc:
            outcome = "failure"
            failure = exc
        else:
            outcome = "success"

        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None or entry.run_token != run_token:
                # 任务已经被取消/超时/替换；晚到结果必须被丢弃。
                return
            if entry.status == JobStatus.CANCELLING:
                self._transition_locked(
                    entry,
                    JobStatus.CANCELLED,
                    stage="cancelled",
                    error="任务已取消，工作函数结果未提交。",
                    error_type="CancelledError",
                )
                return
            if entry.status in _TERMINAL_STATUSES:
                return
            if entry.status == JobStatus.QUEUED:
                # spawn 进程中的上下文不携带管理器回调，完成回调在这里补齐运行迁移。
                self._transition_locked(entry, JobStatus.RUNNING, stage="running")
            if outcome == "cancelled":
                self._transition_locked(
                    entry,
                    JobStatus.CANCELLED,
                    stage="cancelled",
                    error="Future 已取消。",
                    error_type="CancelledError",
                )
            elif outcome == "failure":
                assert failure is not None
                self._transition_locked(
                    entry,
                    JobStatus.FAILED,
                    stage="failed",
                    error=_format_exception(failure),
                    error_type=type(failure).__name__,
                )
            elif isinstance(value, _WorkerCancelled):
                self._transition_locked(
                    entry,
                    JobStatus.CANCELLED,
                    stage="cancelled",
                    error="任务在开始前已取消。",
                    error_type="CancelledError",
                )
            else:
                self._transition_locked(
                    entry,
                    JobStatus.SUCCEEDED,
                    stage="completed",
                    progress=100.0,
                    result=value,
                )

    def _on_timeout(self, job_id: str, run_token: int) -> None:
        """将仍未完成的任务置为 TIMED_OUT 并使工作版本失效。"""
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None or entry.run_token != run_token:
                return
            if entry.status in _TERMINAL_STATUSES:
                return
            if entry.stage == "committing":
                return
            entry.cancel_event.set()
            future = entry.future
            self._transition_locked(
                entry,
                JobStatus.TIMED_OUT,
                stage="timed_out",
                error="任务执行超时。",
                error_type="TimeoutError",
            )
            if future is not None:
                future.cancel()

    def _record_submission_failure(self, job_id: str, run_token: int, exc: BaseException) -> None:
        """把执行器拒绝提交记录为 FAILED，或保留已经请求的取消。"""
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None or entry.run_token != run_token:
                return
            if entry.status == JobStatus.CANCELLING:
                self._transition_locked(
                    entry,
                    JobStatus.CANCELLED,
                    stage="cancelled",
                    error="任务在提交前已取消。",
                    error_type="CancelledError",
                )
            elif entry.status not in _TERMINAL_STATUSES:
                self._transition_locked(
                    entry,
                    JobStatus.FAILED,
                    stage="failed",
                    error=f"提交工作函数失败：{_format_exception(exc)}",
                    error_type=type(exc).__name__,
                )

    def _transition_locked(
        self,
        entry: _JobEntry,
        status: JobStatus,
        *,
        stage: str | None = None,
        progress: float | None = None,
        result: object = _UNSET,
        error: str | None | object = _UNSET,
        error_type: str | None | object = _UNSET,
    ) -> None:
        """在持锁状态下执行一次受约束的状态迁移。"""
        if status == entry.status:
            raise InvalidStateTransitionError(
                f"任务 {entry.job_id} 重复迁移到 {status.value}。"
            )
        if status not in _ALLOWED_TRANSITIONS[entry.status]:
            raise InvalidStateTransitionError(
                f"任务 {entry.job_id} 不能从 {entry.status.value} 迁移到 {status.value}。"
            )
        entry.status = status
        entry.version += 1
        if stage is not None:
            entry.stage = stage
        if progress is not None:
            entry.progress = progress
        if result is not _UNSET:
            entry.result = result
        if error is not _UNSET:
            entry.error = cast(str | None, error)
        if error_type is not _UNSET:
            entry.error_type = cast(str | None, error_type)
        if status == JobStatus.RUNNING:
            entry.started_at = time.monotonic()
        if status in _TERMINAL_STATUSES:
            entry.finished_at = time.monotonic()
            entry.run_token = None
            if entry.counted_as_active:
                entry.counted_as_active = False
                self._active_count -= 1
            if entry.timer is not None:
                entry.timer.cancel()
            self._condition.notify_all()

    def _snapshot_locked(self, entry: _JobEntry) -> JobSnapshot[Any]:
        """在持锁状态下复制内部记录，防止调用方看到可变对象。"""
        return JobSnapshot(
            job_id=entry.job_id,
            idempotency_key=entry.idempotency_key,
            status=entry.status,
            stage=entry.stage,
            progress=entry.progress,
            result=entry.result,
            error=entry.error,
            error_type=entry.error_type,
            version=entry.version,
            created_at=entry.created_at,
            started_at=entry.started_at,
            finished_at=entry.finished_at,
        )

    def _ensure_open_locked(self) -> None:
        """在持锁状态下拒绝关闭后的新提交。"""
        if self._closed:
            raise ManagerClosedError("任务管理器已经关闭。")

    @staticmethod
    def _validate_positive_integer(value: object, name: str) -> None:
        """验证必须为正整数的参数。"""
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise InvalidJobError(f"{name} 必须是正整数。")

    @staticmethod
    def _validate_identifier(
        value: object,
        name: str,
        *,
        required: bool = False,
    ) -> None:
        """验证任务标识文本，但不修改调用方提供的原始值。"""
        if value is None and not required:
            return
        if not isinstance(value, str) or not value.strip():
            raise InvalidJobError(f"{name} 必须是非空文本。")

    @staticmethod
    def _validate_timeout(value: object, name: str) -> None:
        """验证执行超时必须是正的有限数值。"""
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidJobError(f"{name} 必须是正数或 None。")
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise InvalidJobError(f"{name} 必须是正数或 None。")

    @staticmethod
    def _validate_wait_timeout(value: object) -> None:
        """验证 wait 的等待时长允许为零但不能为负数。"""
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidJobError("wait timeout 必须是非负数或 None。")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise InvalidJobError("wait timeout 必须是非负数或 None。")


@dataclass(frozen=True, slots=True)
class _WorkerCancelled:
    """表示工作函数尚未开始就被取消的内部返回值。"""

def _invoke_worker(work: Callable[..., object], context: JobContext) -> object:
    """在执行器线程或进程中启动工作函数，并先执行令牌检查。"""
    if not context._begin():
        return _WorkerCancelled()
    if _accepts_context(work):
        return work(context)
    return work()


def _accepts_context(work: Callable[..., object]) -> bool:
    """判断工作函数是否至少能接受一个位置参数。"""
    try:
        signature = inspect.signature(work)
    except (TypeError, ValueError):
        return True
    parameters = tuple(signature.parameters.values())
    if not parameters:
        return False
    if any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
        return True
    return True


def _format_exception(exc: BaseException) -> str:
    """生成不依赖 traceback 对象的稳定中文异常摘要。"""
    message = str(exc).strip()
    if not message:
        message = repr(exc)
    return f"{type(exc).__name__}: {message}"
