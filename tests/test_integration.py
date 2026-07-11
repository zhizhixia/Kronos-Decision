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


def test_full_pipeline_minimal():
    """最小化端到端测试（不依赖 GPU/网络的重度测试）。"""
    from decision.engine import DecisionEngine, DecisionReport
    engine = DecisionEngine()
    report = engine.predict_and_analyze("000001")
    assert isinstance(report, DecisionReport)
    assert report.status in ("ok", "error", "degraded")