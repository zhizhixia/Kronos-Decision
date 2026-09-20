"""有界的收盘后批处理入口，不建立常驻调度器。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from decision.jobs import JobManager, JobSnapshot, JobStatus


class DailyBatchError(ValueError):
    """批处理参数或恢复状态错误。"""


@dataclass(frozen=True)
class BatchItemResult:
    """单证券批处理终态。"""

    stock_code: str
    job_id: str
    status: str
    result: object | None
    error: str | None


@dataclass(frozen=True)
class DailyBatchResult:
    """一次有界批处理的完整结果。"""

    batch_id: str
    as_of: str
    items: tuple[BatchItemResult, ...]

    @property
    def succeeded(self) -> bool:
        """仅当全部证券成功时返回 True。"""
        return bool(self.items) and all(item.status == JobStatus.SUCCEEDED.value for item in self.items)


class DailyBatchRunner:
    """逐证券提交有限任务并保存可重试的批处理清单。"""

    def __init__(self, checkpoint_path: str | Path, *, max_items: int = 100) -> None:
        if isinstance(max_items, bool) or max_items <= 0:
            raise DailyBatchError("max_items 必须是正整数")
        self.checkpoint_path = Path(checkpoint_path)
        self.max_items = max_items

    def run(
        self,
        stock_codes: Sequence[str],
        *,
        as_of: str,
        worker: Callable[[str, str], object],
        batch_id: str,
        job_manager: JobManager | None = None,
    ) -> DailyBatchResult:
        """运行一次明确 as_of 的批次；重复 batch_id 直接读回旧结果。"""
        if not as_of or not batch_id:
            raise DailyBatchError("batch_id 和 as_of 不能为空")
        codes = tuple(str(code) for code in stock_codes)
        if not codes or len(codes) > self.max_items or len(set(codes)) != len(codes):
            raise DailyBatchError("证券范围为空、超限或包含重复项")
        if not callable(worker):
            raise DailyBatchError("worker 必须可调用")
        manager = job_manager or JobManager(max_pending=min(self.max_items, len(codes)), max_workers=1)
        owns_manager = job_manager is None
        try:
            snapshots: list[JobSnapshot[object]] = []
            for code in codes:
                snapshots.append(
                    manager.submit(
                        lambda context, current_code=code: worker(current_code, as_of),
                        job_id=f"{batch_id}:{code}",
                        idempotency_key=f"{batch_id}:{code}",
                    )
                )
            items = tuple(
                self._final_snapshot(manager, snapshot.job_id, code)
                for snapshot, code in zip(snapshots, codes)
            )
            result = DailyBatchResult(batch_id=batch_id, as_of=as_of, items=items)
            self._write_checkpoint(result)
            return result
        finally:
            if owns_manager:
                manager.close(wait=True)

    def _final_snapshot(self, manager: JobManager, job_id: str, code: str) -> BatchItemResult:
        snapshot = manager.wait(job_id)
        return BatchItemResult(code, job_id, snapshot.status.value, snapshot.result, snapshot.error)

    def _write_checkpoint(self, result: DailyBatchResult) -> None:
        """以原子 JSON 保存批处理终态，供人工恢复而非后台轮询。"""
        import json
        import os
        from tempfile import NamedTemporaryFile

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        payload = {
            "batch_id": result.batch_id,
            "as_of": result.as_of,
            "items": [
                {
                    "stock_code": item.stock_code,
                    "job_id": item.job_id,
                    "status": item.status,
                    "result": item.result,
                    "error": item.error,
                }
                for item in result.items
            ],
        }
        try:
            with NamedTemporaryFile("w", encoding="utf-8", dir=self.checkpoint_path.parent, prefix=f".{self.checkpoint_path.name}.", suffix=".tmp", delete=False) as handle:
                temporary_name = handle.name
                json.dump(payload, handle, ensure_ascii=False, default=str, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.checkpoint_path)
            temporary_name = None
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    temporary_name = None


__all__ = ["BatchItemResult", "DailyBatchError", "DailyBatchResult", "DailyBatchRunner"]
