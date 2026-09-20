"""验证取数快照、报告发布和重启读回的纵向连接。"""
from __future__ import annotations

import threading
from pathlib import Path

import pandas as pd
import pytest

from data.fetcher import DataFetcher
from data.snapshots import SnapshotCorruptionError, SnapshotStore
from decision.jobs import JobCancelledError, JobContext, JobManager
from decision.report_pipeline import DecisionReportPipeline, ReportPipelineError
from decision.storage import ReportStorage
from webui.job_service import DecisionJobService


class FixedCalendar:
    """为流水线测试提供固定完整交易日。"""

    version = "pipeline-calendar-v1"

    def __init__(self, complete: pd.Timestamp) -> None:
        self.complete = complete

    def latest_complete_session(self):
        """返回完整交易日和无诊断旗标。"""
        return self.complete, ()


def _bars() -> pd.DataFrame:
    """构造真实临时 CSV 链路使用的日线表。"""
    dates = pd.bdate_range("2024-01-02", periods=60)
    return pd.DataFrame(
        {
            "date": dates,
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.0,
            "volume": 1000.0,
            "amount": 10000.0,
        }
    )


def _fetcher(tmp_path: Path, monkeypatch) -> DataFetcher:
    """创建隔离目录中的真实 DataFetcher。"""
    fetcher = DataFetcher()
    fetcher._cache_dir = tmp_path / "cache"
    fetcher._snapshot_dir = tmp_path / "snapshots"
    fetcher._cache_dir.mkdir(parents=True, exist_ok=True)
    fetcher._retry_max = 1
    fetcher._calendar = FixedCalendar(_bars()["date"].max())
    monkeypatch.setattr(fetcher, "_fetch_akshare", lambda code: _bars())
    monkeypatch.setattr(
        fetcher,
        "_fetch_baostock",
        lambda code: (_ for _ in ()).throw(AssertionError("不应访问备源")),
    )
    return fetcher


class StubEngine:
    """只模拟决策层输出，验证持久化边界而不启动真实模型。"""

    def decision_report_v2(
        self,
        stock_code: str,
        portfolio_id: str,
        include_display_paths: bool,
        as_of: str | None,
        *,
        prepared_bundle,
    ) -> dict[str, object]:
        """返回带有输入快照引用的最小合法报告。"""
        del portfolio_id, include_display_paths, as_of
        return {
            "schema_version": "2.0",
            "status": "degraded",
            "run_status": "SUCCEEDED",
            "evidence_status": "INSUFFICIENT_EVIDENCE",
            "action_permission": "NONE",
            "stock": {"code": stock_code, "name": "测试股票"},
            "data_provenance": {
                "snapshot_id": prepared_bundle.snapshot_id,
                "as_of": prepared_bundle.as_of.isoformat(),
                "source": prepared_bundle.source,
            },
            "model_provenance": {
                "config_hash": "config-test",
                "model_provenance": {"model_revision": "model-test"},
            },
            "evaluation_id": "none",
            "recommendation": {"action": None, "legacy_signal": None},
        }


def test_pipeline_publishes_report_and_protects_snapshot(tmp_path: Path, monkeypatch) -> None:
    """真实临时目录中，报告保存后快照必须得到引用保护。"""
    fetcher = _fetcher(tmp_path, monkeypatch)
    snapshots = SnapshotStore(tmp_path / "snapshots")
    storage = ReportStorage(tmp_path / "reports" / "decision.sqlite")
    pipeline = DecisionReportPipeline(
        fetcher=fetcher,
        engine=StubEngine(),
        snapshot_store=snapshots,
        report_storage=storage,
    )

    result = pipeline.run("600519", as_of="2024-03-25")

    assert result.created is True
    assert result.publication.record.publication_status == "PUBLISHED"
    assert result.snapshot_id
    assert result.publication.record.snapshot_id == result.snapshot_id
    assert snapshots.is_referenced(result.snapshot_id)
    assert storage.get_latest(result.report_id) is not None


def test_pipeline_is_idempotent_and_restarts_from_sqlite_and_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    """同一报告时点遇到来源变化时生成新版本，重启后仍能读回最新报告。"""
    fetcher = _fetcher(tmp_path, monkeypatch)
    storage = ReportStorage(tmp_path / "reports.sqlite")
    pipeline = DecisionReportPipeline(
        fetcher=fetcher,
        engine=StubEngine(),
        snapshot_store=SnapshotStore(tmp_path / "snapshots"),
        report_storage=storage,
    )
    first = pipeline.run("600519", as_of="2024-03-25")
    duplicate = pipeline.run("600519", as_of="2024-03-25")

    assert duplicate.created is True
    assert duplicate.version == first.version + 1
    assert duplicate.snapshot_id != first.snapshot_id

    restarted = DecisionReportPipeline(
        fetcher=fetcher,
        engine=StubEngine(),
        snapshot_store=SnapshotStore(tmp_path / "snapshots"),
        report_storage=ReportStorage(tmp_path / "reports.sqlite"),
    )
    loaded = restarted.read_published(duplicate.report_id)

    assert loaded is not None
    assert loaded.report_id == first.report_id
    assert loaded.snapshot_id == duplicate.snapshot_id
    assert loaded.payload["data_provenance"]["snapshot_id"] == duplicate.snapshot_id


def test_pipeline_readback_rejects_corrupted_snapshot(tmp_path: Path, monkeypatch) -> None:
    """快照损坏时历史报告读回失败，不重新取数或伪造结果。"""
    fetcher = _fetcher(tmp_path, monkeypatch)
    pipeline = DecisionReportPipeline(
        fetcher=fetcher,
        engine=StubEngine(),
        snapshot_store=SnapshotStore(tmp_path / "snapshots"),
        report_storage=ReportStorage(tmp_path / "reports.sqlite"),
    )
    result = pipeline.run("600519", as_of="2024-03-25")
    csv_path = tmp_path / "snapshots" / result.snapshot_id / "data.csv"
    csv_path.write_bytes(csv_path.read_bytes() + b"tampered\n")

    with pytest.raises(SnapshotCorruptionError):
        pipeline.read_published(result.report_id)


def test_pipeline_rejects_report_bound_to_different_snapshot(tmp_path: Path, monkeypatch) -> None:
    """决策层返回错误快照引用时，流水线不写入报告。"""
    fetcher = _fetcher(tmp_path, monkeypatch)

    class WrongSnapshotEngine(StubEngine):
        """返回不匹配的快照引用。"""

        def decision_report_v2(self, *args, **kwargs):
            """将合法报告改成错误的快照编号。"""
            report = super().decision_report_v2(*args, **kwargs)
            report["data_provenance"]["snapshot_id"] = "wrong-snapshot"
            return report

    storage = ReportStorage(tmp_path / "reports.sqlite")
    pipeline = DecisionReportPipeline(
        fetcher=fetcher,
        engine=WrongSnapshotEngine(),
        snapshot_store=SnapshotStore(tmp_path / "snapshots"),
        report_storage=storage,
    )

    with pytest.raises(ReportPipelineError, match="同一份数据快照"):
        pipeline.run("600519", as_of="2024-03-25")
    assert storage.list_reports() == []


def test_pipeline_publishes_non_actionable_baseline_from_same_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    """无模型基线也必须使用同一快照并保持 NONE 动作权限。"""
    fetcher = _fetcher(tmp_path, monkeypatch)
    storage = ReportStorage(tmp_path / "reports.sqlite")
    pipeline = DecisionReportPipeline(
        fetcher=fetcher,
        engine=StubEngine(),
        snapshot_store=SnapshotStore(tmp_path / "snapshots"),
        report_storage=storage,
    )

    result = pipeline.run_baseline("600519", as_of="2024-03-25", stock_name="测试股票")

    assert result.report["model_provenance"]["model_revision"] == "baseline-momentum-v1"
    assert result.report["recommendation"]["action"] is None
    assert result.report["action_permission"] == "NONE"
    assert result.report["data_provenance"]["snapshot_id"] == result.snapshot_id
    assert pipeline.read_published(result.report_id) is not None


def test_cancelled_pipeline_cannot_mark_pending_report_published(
    tmp_path: Path, monkeypatch
) -> None:
    """取消令牌在引用保护后到达时，SQLite 报告仍不得进入 PUBLISHED。"""
    fetcher = _fetcher(tmp_path, monkeypatch)
    snapshots = SnapshotStore(tmp_path / "snapshots")
    storage = ReportStorage(tmp_path / "reports.sqlite")
    pipeline = DecisionReportPipeline(
        fetcher=fetcher,
        engine=StubEngine(),
        snapshot_store=snapshots,
        report_storage=storage,
    )
    cancelled = False

    def guard(operation):
        """模拟 JobManager 在副作用边界执行的取消令牌保护。"""
        if cancelled:
            raise JobCancelledError("测试任务已取消")
        return operation()

    context = JobContext(
        "cancelled-report",
        1,
        cancel_callback=lambda: cancelled,
        active_callback=guard,
    )
    original_mark_referenced = snapshots.mark_referenced

    def mark_then_cancel(snapshot_id: str, reference_id: str | None = None) -> None:
        """完成快照引用后模拟取消请求到达发布边界。"""
        nonlocal cancelled
        original_mark_referenced(snapshot_id, reference_id=reference_id)
        cancelled = True

    monkeypatch.setattr(snapshots, "mark_referenced", mark_then_cancel)

    with pytest.raises(ReportPipelineError, match="取消"):
        pipeline.run("600519", as_of="2024-03-25", context=context)

    records = storage.list_reports("decision:600519:2024-03-25")
    assert len(records) == 1
    assert records[0].publication_status == "PENDING"
    assert storage.get_published(records[0].report_id, version=records[0].version) is None


def test_cancel_requested_during_commit_finishes_successfully_without_partial_cancel(
    tmp_path: Path, monkeypatch
) -> None:
    """进入最终提交边界后，取消请求不能留下 CANCELLED/PUBLISHED 矛盾状态。"""
    fetcher = _fetcher(tmp_path, monkeypatch)
    storage = ReportStorage(tmp_path / "reports.sqlite")
    pipeline = DecisionReportPipeline(
        fetcher=fetcher,
        engine=StubEngine(),
        snapshot_store=SnapshotStore(tmp_path / "snapshots"),
        report_storage=storage,
    )
    entered = threading.Event()
    release = threading.Event()
    original_mark_published = storage.mark_published

    def delayed_mark_published(report_id: str, version: int):
        """在真实发布调用中暂停，制造取消请求与提交的竞态。"""
        entered.set()
        assert release.wait(5)
        return original_mark_published(report_id, version)

    monkeypatch.setattr(storage, "mark_published", delayed_mark_published)
    manager = JobManager(max_pending=1, max_workers=1, default_timeout=10)
    service = DecisionJobService(lambda: pipeline, manager=manager)
    submitted = service.submit("600519", as_of="2024-03-25")
    assert entered.wait(5)

    cancel_result: list[dict[str, object]] = []

    def request_cancel() -> None:
        """从另一个线程发送取消请求，验证提交边界的原子性。"""
        cancel_result.append(service.cancel(submitted.job_id))

    cancel_thread = threading.Thread(target=request_cancel)
    cancel_thread.start()
    release.set()
    cancel_thread.join(5)
    final = manager.wait(submitted.job_id, timeout=5)
    service.close()

    assert cancel_result
    assert final.status.value == "SUCCEEDED"
    assert final.result is not None
    assert storage.get_published("decision:600519:2024-03-25") is not None
