"""测试 DecisionEngine。"""
from types import SimpleNamespace

import pytest
import decision.engine as engine_module
from decision.engine import DecisionEngine, DecisionReport
from decision.gates import evaluate_evidence_gate as evaluate_decision_evidence_gate
from decision.v2 import RecommendationAction
from evaluation.gates import EvidenceGateResult
from evaluation.binding import EvaluationBinding
from data.events import EventRiskProvider
from data.fundamental import FundamentalAdapter

pytestmark = [pytest.mark.network, pytest.mark.model]


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


def test_bad_evidence_values_fail_closed_without_raising():
    """字符串和 NaN 证据必须返回 INSUFFICIENT_EVIDENCE，而不是泄漏异常。"""
    gate = EvidenceGateResult(True, (), {})
    evidence = SimpleNamespace(
        short_probability="unknown",
        calibrated_up_probability=float("nan"),
        long_probability=0.7,
        short_rank=0.8,
        cross_section_rank=0.8,
        long_rank=0.8,
    )

    action, reasons = DecisionEngine._decide_with_evidence(gate, evidence)

    assert action is RecommendationAction.INSUFFICIENT_EVIDENCE
    assert reasons == ("MULTI_HORIZON_CALIBRATION_UNAVAILABLE",)


def test_mapping_evidence_is_supported_without_relaxing_probability_checks():
    """映射型证据可进入既有五态规则，但仍必须满足全部数值门槛。"""
    gate = EvidenceGateResult(True, (), {})
    evidence = {
        "short_probability": 0.8,
        "calibrated_up_probability": 0.8,
        "long_probability": 0.8,
        "short_rank": 0.8,
        "cross_section_rank": 0.8,
        "long_rank": 0.8,
    }

    action, reasons = DecisionEngine._decide_with_evidence(gate, evidence)

    assert action is RecommendationAction.ADD
    assert reasons == ("H20_TOP_QUINTILE", "H20_CALIBRATED_UP_PROBABILITY")


def test_v2_probability_strategy_passes_explicit_calibration_and_rejects_nan(monkeypatch):
    """概率策略显式要求校准；成功可参考，NaN 证据则结构化拒绝。"""
    engine = DecisionEngine()
    report = DecisionReport(
        "ok",
        signal=SimpleNamespace(signal="HOLD", analysis_valid=True),
        prediction={
            "data_provenance": {
                "source": "test",
                "as_of": "2026-09-01",
                "is_stale": False,
                "quality_flags": [],
                "has_hard_quality_error": False,
            },
            "sampling": {"paths_valid": True, "paths_finite": True, "synthetic": False},
            "horizons": {},
            "chart_data": {"history": [], "forecast": {}},
        },
        stock_code="000001",
        stock_name="测试",
    )
    evidence = SimpleNamespace(
        short_probability=0.8,
        calibrated_up_probability=0.8,
        long_probability=0.8,
        short_rank=0.8,
        cross_section_rank=0.8,
        long_rank=0.8,
    )
    research_gate = EvidenceGateResult(True, (), {})
    calibration_calls = []

    def evidence_gate_spy(*args, **kwargs):
        calibration_calls.append(kwargs.get("requires_calibration"))
        return evaluate_decision_evidence_gate(*args, **kwargs)

    monkeypatch.setattr(engine_module, "evaluate_decision_evidence_gate", evidence_gate_spy)
    monkeypatch.setattr(engine, "predict_and_analyze", lambda stock_code, params: report)
    monkeypatch.setattr(engine, "_bind_evidence", lambda binding, bound_report: (research_gate, evidence))
    monkeypatch.setattr(engine, "_eligibility_for", lambda stock_code, as_of: {"eligible": True, "reasons": []})
    monkeypatch.setattr(engine, "_is_held", lambda portfolio_id, stock_code: False)
    monkeypatch.setattr(engine, "_market_context_payload", lambda as_of=None: {"available": False})
    monkeypatch.setattr(EvaluationBinding, "load_latest", staticmethod(lambda: None))
    monkeypatch.setattr(FundamentalAdapter, "snapshot", lambda self, *args, **kwargs: {})
    monkeypatch.setattr(EventRiskProvider, "events", lambda self, *args, **kwargs: [])

    success = engine.decision_report_v2("000001")
    assert calibration_calls[-1] is True
    assert success["recommendation"]["action"] == "ADD"
    assert success["action_permission"] == "CONDITIONAL_REFERENCE"
    assert success["recommendation"]["legacy_signal"] == "BUY"

    monkeypatch.setattr(engine, "_eligibility_for", lambda stock_code, as_of: {"eligible": False, "reasons": ["RISK_BLOCKED"]})
    blocked = engine.decision_report_v2("000001")
    assert blocked["recommendation"]["action"] == "INSUFFICIENT_EVIDENCE"
    assert blocked["recommendation"]["legacy_signal"] is None
    assert blocked["action_permission"] == "NONE"
    assert blocked["portfolio_impact"]["eligible_for_rebalance"] is False

    monkeypatch.setattr(engine, "_eligibility_for", lambda stock_code, as_of: {"eligible": True, "reasons": []})

    evidence.calibrated_up_probability = float("nan")
    rejected = engine.decision_report_v2("000001")
    assert calibration_calls[-1] is True
    assert rejected["recommendation"]["action"] == "INSUFFICIENT_EVIDENCE"
    assert rejected["recommendation"]["legacy_signal"] is None
    assert rejected["action_permission"] == "NONE"
