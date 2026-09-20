"""决策门禁的严格权限与 research_gate 反例测试。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from decision.contracts import ActionPermission, EvidenceStatus
from decision.gates import (
    DecisionGateResult,
    evaluate_data_gate,
    evaluate_decision_gate,
    evaluate_eligibility_gate,
    evaluate_evidence_gate,
)


def _valid_provenance() -> dict[str, object]:
    """返回满足数据新鲜度和质量要求的元数据。"""
    return {
        "as_of": "2026-09-01",
        "is_stale": False,
        "quality_flags": [],
    }


def _valid_paths() -> list[list[list[float]]]:
    """返回最小的有限预测路径。"""
    return [[[1.0, 1.0, 1.0, 1.0]]]


def _valid_evidence() -> dict[str, float]:
    """返回三周期校准概率和排名。"""
    return {
        "short_probability": 0.7,
        "calibrated_up_probability": 0.7,
        "long_probability": 0.7,
        "short_rank": 0.7,
        "cross_section_rank": 0.7,
        "long_rank": 0.7,
    }


def test_passed_gates_only_report_success_and_keep_permission_none() -> None:
    """各层门禁通过时仍不能代替最终 ADD/HOLD/REDUCE/AVOID 决策。"""
    data_gate = evaluate_data_gate(
        _valid_provenance(),
        {"model_status": "READY"},
        _valid_paths(),
    )
    evidence_gate = evaluate_evidence_gate(
        _valid_evidence(),
        research_gate={"passed": True},
        requires_calibration=True,
    )
    eligibility_gate = evaluate_eligibility_gate({"eligible": True})

    assert data_gate.passed
    assert evidence_gate.passed
    assert eligibility_gate.passed
    assert all(gate.action_permission is ActionPermission.NONE for gate in (
        data_gate,
        evidence_gate,
        eligibility_gate,
    ))
    assert all(gate.as_dict()["action_permission"] == "NONE" for gate in (
        data_gate,
        evidence_gate,
        eligibility_gate,
    ))


def test_combined_gate_passes_without_publishing_a_final_action_permission() -> None:
    """组合门禁通过也只表示前置条件通过，不表示已有最终动作。"""
    result = evaluate_decision_gate(
        {
            "run_status": "SUCCEEDED",
            "signal_available": True,
            "data_provenance": _valid_provenance(),
            "sampling": {"model_status": "READY"},
            "prediction_paths": _valid_paths(),
            "evidence": _valid_evidence(),
            "research_gate": {"passed": True},
            "requires_calibration": True,
            "eligibility": {"eligible": True},
        }
    )

    assert result.passed
    assert result.evidence_status is EvidenceStatus.QUALIFIED
    assert result.action_permission is ActionPermission.NONE
    assert result.as_dict()["action_permission"] == "NONE"


@pytest.mark.parametrize(
    "research_gate",
    [
        None,
        {},
        {"passed": "true"},
        {"passed": 1},
        {"passed": False},
        SimpleNamespace(),
        SimpleNamespace(passed="true"),
    ],
)
def test_research_gate_requires_strict_boolean_true(research_gate: object) -> None:
    """research_gate 缺失、未知或非 bool True 时必须拒绝。"""
    result = evaluate_evidence_gate(
        _valid_evidence(),
        research_gate=research_gate,
        requires_calibration=True,
    )

    assert not result.passed
    assert result.checks["RESEARCH_GATE_PASSED"] is False
    assert "RESEARCH_GATE_PASSED" in result.failed_codes
    assert result.action_permission is ActionPermission.NONE


def test_as_dict_does_not_emit_legacy_conditional_permission_without_action() -> None:
    """旧式无动作条件权限在结果序列化时必须收敛为 NONE。"""
    result = DecisionGateResult(
        passed=True,
        failed_codes=(),
        checks={},
        evidence_status=EvidenceStatus.QUALIFIED,
        action_permission=ActionPermission.CONDITIONAL_REFERENCE,
    )

    assert result.action_permission is ActionPermission.NONE
    assert result.as_dict()["action_permission"] == "NONE"
