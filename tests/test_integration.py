"""集成测试。"""
import pytest


def test_import_all_modules():
    """所有新模块可正常导入。"""
    from decision.errors import KronosError
    from decision.config import Config, get_config
    from decision.model_manager import ModelManager
    from decision.analyzer import SignalAnalyzer
    from decision.engine import DecisionEngine
    from data.fetcher import DataFetcher
    from data.pool import get_hs300_pool
    assert True


def test_config_accessible():
    """配置可访问。"""
    from decision.config import get_config
    cfg = get_config()
    assert cfg.model.max_context == 512
    assert cfg.signal.trend_strength_buy_threshold == 0.7


@pytest.mark.network
@pytest.mark.model
def test_full_pipeline_minimal():
    """最小化端到端测试（不依赖 GPU/网络的重度测试）。"""
    from decision.engine import DecisionEngine, DecisionReport
    engine = DecisionEngine()
    report = engine.predict_and_analyze("000001")
    assert isinstance(report, DecisionReport)
    assert report.status in ("ok", "error", "degraded")


def test_baseline_api_persists_and_reads_back_after_pipeline_rebuild(tmp_path, monkeypatch) -> None:
    """在禁止联网的夹具输入上，API→存储→快照→历史 API 必须闭环。"""
    from decision.report_pipeline import DecisionReportPipeline
    from decision.storage import ReportStorage
    from data.snapshots import SnapshotStore
    from tests.test_report_pipeline import _fetcher
    from webui.app import app

    fetcher = _fetcher(tmp_path, monkeypatch)
    pipeline = DecisionReportPipeline(
        fetcher=fetcher,
        snapshot_store=SnapshotStore(tmp_path / "snapshots"),
        report_storage=ReportStorage(tmp_path / "reports.sqlite"),
    )

    class FixedPipeline:
        """让 Flask 路由使用本测试创建的真实流水线对象。"""

        def __new__(cls):
            return pipeline

    monkeypatch.setattr("webui.app._get_report_pipeline_class", lambda: FixedPipeline)
    response = app.test_client().post(
        "/api/v2/decision-report",
        json={"stock_code": "600519", "mode": "baseline", "as_of": "2024-03-25"},
        headers={"Origin": "http://127.0.0.1:7070"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["publication_status"] == "PUBLISHED"
    assert payload["action_permission"] == "NONE"
    assert payload["snapshot_id"]

    history = app.test_client().get(
        f"/api/v2/decision-report/{payload['report_id']}"
    )
    assert history.status_code == 200
    restored = history.get_json()
    assert restored["report_id"] == payload["report_id"]
    assert restored["snapshot_id"] == payload["snapshot_id"]
    assert restored["publication_status"] == "PUBLISHED"
