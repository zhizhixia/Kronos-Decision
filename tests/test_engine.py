"""测试 DecisionEngine。"""
import pytest
from decision.engine import DecisionEngine, DecisionReport


@pytest.fixture
def engine():
    return DecisionEngine()


def test_engine_returns_report_on_error(engine):
    """无效股票代码应返回 error 状态而不是抛异常。"""
    report = engine.predict_and_analyze("999999")
    assert isinstance(report, DecisionReport)
    assert report.status in ("ok", "error", "degraded")
    assert report.stock_code == "999999"


def test_report_has_elapsed_time(engine):
    """报告包含耗时。"""
    report = engine.predict_and_analyze("000001")
    assert report.elapsed_seconds >= 0


def test_engine_accepts_params(engine):
    """Engine 接受自定义参数。"""
    report = engine.predict_and_analyze("000001", {"pred_len": 60})
    assert isinstance(report, DecisionReport)