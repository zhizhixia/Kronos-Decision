"""把取数、快照、决策报告和 SQLite 发布串成一条可恢复链路。"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable, TypeVar

from data.contracts import MarketDataBundle
from data.fetcher import DataFetcher
from data.snapshots import SnapshotStore
from decision.config import get_config
from decision.engine import DecisionEngine
from decision.jobs import JobCancelledError, JobContext
from decision.storage import PublishResult, ReportRecord, ReportStorage
from decision.versioning import config_hash


class ReportPipelineError(RuntimeError):
    """报告流水线无法形成完整发布时抛出的异常。"""


_T = TypeVar("_T")


@dataclass(frozen=True)
class PipelineResult:
    """一次报告发布的可审计结果。"""

    report: dict[str, Any]
    publication: PublishResult
    snapshot_id: str

    @property
    def report_id(self) -> str:
        """返回已发布报告 ID。"""
        return self.publication.record.report_id

    @property
    def version(self) -> int:
        """返回已发布报告版本。"""
        return self.publication.record.version

    @property
    def created(self) -> bool:
        """返回本次是否创建了新的报告版本。"""
        return self.publication.created


class DecisionReportPipeline:
    """协调一次取数、快照绑定、报告生成、发布和重启读回。"""

    def __init__(
        self,
        *,
        fetcher: DataFetcher | None = None,
        engine: DecisionEngine | None = None,
        snapshot_store: SnapshotStore | None = None,
        report_storage: ReportStorage | None = None,
    ) -> None:
        """创建可注入路径和依赖的流水线。"""
        cfg = get_config().data
        self.fetcher = fetcher or DataFetcher()
        snapshot_root = getattr(self.fetcher, "_snapshot_dir", cfg.snapshot_dir)
        self.snapshot_store = snapshot_store or SnapshotStore(snapshot_root)
        self.engine = engine or DecisionEngine()
        self.report_storage = report_storage or ReportStorage(cfg.report_db)

    def run(
        self,
        stock_code: str,
        *,
        portfolio_id: str = "default",
        include_display_paths: bool = False,
        as_of: str | None = None,
        context: JobContext | None = None,
    ) -> PipelineResult:
        """执行一次完整发布；任何外部引用失败都不返回成功结果。"""
        self._ensure_active(context)
        bundle = self.fetcher.fetch_daily_bundle(stock_code, as_of=as_of)
        self._ensure_active(context)
        if not bundle.snapshot_id:
            raise ReportPipelineError("取数结果没有绑定 snapshot_id，拒绝生成报告。")
        self.snapshot_store.read_manifest(bundle.snapshot_id)
        self._ensure_active(context)
        report = self.engine.decision_report_v2(
            stock_code,
            portfolio_id,
            include_display_paths,
            as_of,
            prepared_bundle=bundle,
        )
        self._ensure_active(context)
        self._validate_report_snapshot(report, bundle.snapshot_id)
        return self._publish_report(stock_code, bundle, report, context=context)

    def run_baseline(
        self,
        stock_code: str,
        *,
        as_of: str | None = None,
        stock_name: str = "",
        context: JobContext | None = None,
    ) -> PipelineResult:
        """用同一条快照链发布无模型、不可行动的事实基线。"""
        from decision.strategies import run_momentum_baseline

        self._ensure_active(context)
        bundle = self.fetcher.fetch_daily_bundle(stock_code, as_of=as_of)
        self._ensure_active(context)
        if not bundle.snapshot_id:
            raise ReportPipelineError("取数结果没有绑定 snapshot_id，拒绝生成基线报告。")
        self.snapshot_store.read_manifest(bundle.snapshot_id)
        self._ensure_active(context)
        report = run_momentum_baseline(bundle, stock_code, stock_name)
        self._ensure_active(context)
        self._validate_report_snapshot(report.report, bundle.snapshot_id)
        return self._publish_report(stock_code, bundle, report.report, context=context)

    def _publish_report(
        self,
        stock_code: str,
        bundle: MarketDataBundle,
        report: dict[str, Any],
        *,
        context: JobContext | None = None,
    ) -> PipelineResult:
        """保存报告、保护快照并独立读回验证。"""
        report_id = self._report_id(stock_code, bundle.as_of.isoformat())
        self._ensure_active(context)
        existing = self.report_storage.get_latest(report_id)
        persisted_report = report
        if existing is not None and self._reports_equivalent(existing.payload, report):
            persisted_report = existing.payload
        publication = self._run_if_active(
            context,
            lambda: self.report_storage.save_report(
                report_id=report_id,
                payload=persisted_report,
                snapshot_id=bundle.snapshot_id,
                evaluation_id=self._evaluation_id(report),
                model_revision=self._model_revision(report),
                config_hash=self._config_hash(report),
                publication_status="PENDING",
            ),
        )
        try:
            self._run_if_active(
                context,
                lambda: self.snapshot_store.mark_referenced(
                    bundle.snapshot_id,
                    reference_id=publication.record.report_id,
                ),
            )
        except ReportPipelineError:
            raise
        except Exception as exc:
            try:
                self.report_storage.mark_failed(publication.record.report_id, publication.record.version)
            except Exception as marker_exc:
                raise ReportPipelineError("报告和失败状态都无法持久化。") from marker_exc
            raise ReportPipelineError("报告已写入但快照引用保护失败，拒绝宣称发布完成。") from exc
        if context is not None:
            try:
                if not context.enter_commit():
                    raise ReportPipelineError("任务已取消，拒绝进入报告提交边界。")
            except JobCancelledError as exc:
                raise ReportPipelineError("任务已取消，拒绝进入报告提交边界。") from exc
        published_record = self._run_if_active(
            context,
            lambda: self.report_storage.mark_published(
                publication.record.report_id,
                publication.record.version,
            ),
        )
        publication = PublishResult(record=published_record, created=publication.created)
        try:
            verified = self.read_published(
                publication.record.report_id,
                version=publication.record.version,
            )
        except Exception as exc:
            raise ReportPipelineError("报告发布后读回校验失败，拒绝宣称发布完成。") from exc
        if verified is None:
            raise ReportPipelineError("报告发布后不存在，拒绝宣称发布完成。")
        return PipelineResult(report=persisted_report, publication=publication, snapshot_id=bundle.snapshot_id)

    @staticmethod
    def _ensure_active(context: JobContext | None) -> None:
        """在流水线阶段边界拒绝已取消或超时的任务。"""
        if context is not None and context.is_cancelled():
            raise ReportPipelineError("任务已取消或超时，拒绝继续报告流水线。")

    @classmethod
    def _run_if_active(
        cls,
        context: JobContext | None,
        operation: Callable[[], _T],
    ) -> _T:
        """在线程任务锁内执行发布副作用，避免取消与发布竞态。"""
        cls._ensure_active(context)
        if context is None:
            return operation()
        try:
            return context.run_if_active(operation)
        except JobCancelledError as exc:
            raise ReportPipelineError("任务已取消或超时，拒绝执行报告发布操作。") from exc

    def read_published(
        self,
        report_id: str,
        version: int | None = None,
    ) -> ReportRecord | None:
        """重启后只读恢复报告，并重新校验其快照，不重新取数或推理。"""
        record = self.report_storage.get_published(report_id, version=version)
        if record is None:
            return None
        self.snapshot_store.read_manifest(record.snapshot_id)
        self.snapshot_store.read(record.snapshot_id)
        self._validate_report_snapshot(record.payload, record.snapshot_id)
        return record

    @staticmethod
    def _report_id(stock_code: str, as_of: str) -> str:
        """生成不包含可变路径的稳定报告逻辑 ID。"""
        return f"decision:{stock_code}:{as_of[:10]}"

    @staticmethod
    def _validate_report_snapshot(payload: object, snapshot_id: str) -> None:
        """确保报告中的快照引用与流水线输入一致。"""
        if not isinstance(payload, dict):
            raise ReportPipelineError("报告载荷不是 JSON 对象。")
        provenance = payload.get("data_provenance")
        if not isinstance(provenance, dict) or provenance.get("snapshot_id") != snapshot_id:
            raise ReportPipelineError("报告没有绑定同一份数据快照。")

    @staticmethod
    def _evaluation_id(report: dict[str, Any]) -> str:
        """读取评估运行 ID；没有正式评估时保留明确的 none 标识。"""
        value = report.get("evaluation_id", "none")
        return value if isinstance(value, str) and value.strip() else "none"

    @staticmethod
    def _model_revision(report: dict[str, Any]) -> str:
        """提取模型修订；缺失时保留不可用标记，不伪造版本。"""
        model = report.get("model_provenance")
        nested = model.get("model_provenance") if isinstance(model, dict) else None
        value = nested.get("model_revision") if isinstance(nested, dict) else None
        if not value and isinstance(model, dict):
            value = model.get("model_revision")
        return value if isinstance(value, str) and value.strip() else "unresolved"

    @staticmethod
    def _config_hash(report: dict[str, Any]) -> str:
        """优先使用报告记录的配置哈希，缺失时读取当前配置哈希。"""
        model = report.get("model_provenance")
        value = model.get("config_hash") if isinstance(model, dict) else None
        if isinstance(value, str) and value.strip():
            return value
        return config_hash()

    @staticmethod
    def _reports_equivalent(left: object, right: object) -> bool:
        """比较不含运行耗时和读取路径噪声的报告语义。"""
        def canonical(value: object) -> object:
            if isinstance(value, dict):
                ignored = {"generated_at", "elapsed_seconds", "retrieval_source"}
                if "snapshot_id" in value:
                    ignored.add("source")
                return {
                    key: canonical(item)
                    for key, item in value.items()
                    if key not in ignored
                }
            if isinstance(value, list):
                return [canonical(item) for item in value]
            return value

        try:
            return json.dumps(
                canonical(left), ensure_ascii=False, sort_keys=True, allow_nan=False
            ) == json.dumps(
                canonical(right), ensure_ascii=False, sort_keys=True, allow_nan=False
            )
        except (TypeError, ValueError):
            return False


__all__ = ["DecisionReportPipeline", "PipelineResult", "ReportPipelineError"]
