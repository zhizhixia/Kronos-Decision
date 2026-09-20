"""WebUI 使用的显式、可取消决策任务服务。"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from inspect import Parameter, signature
from typing import Callable

from decision.jobs import JobCancelledError, JobContext, JobManager, JobSnapshot


_SHANGHAI = timezone(timedelta(hours=8))


class DecisionJobService:
    """封装 JobManager，避免 API 层自行创建不可控后台线程。"""

    def __init__(self, pipeline_factory: Callable[[], object], *, manager: JobManager | None = None) -> None:
        if not callable(pipeline_factory):
            raise ValueError("pipeline_factory 必须可调用")
        self.pipeline_factory = pipeline_factory
        self.manager = manager or JobManager(max_pending=4, max_workers=1, default_timeout=300.0)
        self._owns_manager = manager is None

    def submit(self, stock_code: str, *, as_of: str | None = None, mode: str = "model") -> JobSnapshot[object]:
        """提交单证券任务；模式只允许 model 或 baseline。"""
        if not stock_code or mode not in {"model", "baseline"}:
            raise ValueError("证券代码和任务模式无效")

        def work(context: JobContext) -> object:
            """执行任务并把取消令牌传到报告发布边界。"""
            if context.is_cancelled():
                raise JobCancelledError("任务已取消，拒绝启动报告流水线。")
            pipeline = self.pipeline_factory()
            if mode == "baseline":
                return self._invoke_pipeline(pipeline.run_baseline, stock_code, as_of, context)
            return self._invoke_pipeline(pipeline.run, stock_code, as_of, context)

        request_scope = as_of or datetime.now(_SHANGHAI).date().isoformat()
        return self.manager.submit(
            work,
            idempotency_key=f"{mode}:{stock_code}:{request_scope}",
        )

    @staticmethod
    def _invoke_pipeline(
        method: Callable[..., object],
        stock_code: str,
        as_of: str | None,
        context: JobContext,
    ) -> object:
        """调用流水线并兼容尚未接入 context 的旧式测试实现。"""
        try:
            parameters = signature(method).parameters.values()
            supports_context = any(
                parameter.name == "context"
                or parameter.kind == Parameter.VAR_KEYWORD
                for parameter in parameters
            )
        except (TypeError, ValueError):
            supports_context = False
        if supports_context:
            return method(stock_code, as_of=as_of, context=context)
        return method(stock_code, as_of=as_of)

    def get(self, job_id: str) -> dict[str, object]:
        """读取任务状态，终态结果只返回可序列化摘要。"""
        snapshot = self.manager.get(job_id)
        payload = snapshot.as_dict()
        if is_dataclass(payload.get("result")):
            payload["result"] = asdict(payload["result"])
        return payload

    def cancel(self, job_id: str) -> dict[str, object]:
        """请求取消并立即返回最新状态。"""
        self.manager.cancel(job_id)
        return self.get(job_id)

    def close(self) -> None:
        """关闭该服务拥有的执行器。"""
        if self._owns_manager:
            self.manager.close(wait=True)


__all__ = ["DecisionJobService"]
