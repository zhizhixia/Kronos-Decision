"""SA02 最小报告契约与拒绝门禁的独立反例。"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from decision.contracts import (
    ActionPermission,
    DecisionReportContract,
    EvidenceStatus,
    RunStatus,
    is_valid_report_contract,
    safe_legacy_signal,
    validate_report_contract,
)
from decision.gates import (
    evaluate_data_gate,
    evaluate_decision_gate,
    evaluate_evidence_gate,
)
from decision.v2 import RecommendationAction


def _valid_paths() -> np.ndarray:
    """返回仅用于门禁反例的有限路径。"""
    return np.ones((2, 5, 4), dtype=float)


def test_contract_exposes_stable_status_fields_and_safe_legacy_mapping() -> None:
    """失败报告必须能序列化为明确状态，旧信号必须为空。"""
    contract = DecisionReportContract(
        RunStatus.FAILED,
        EvidenceStatus.INVALID,
        ActionPermission.NONE,
        None,
        ("MODEL_FAILED",),
    )

    payload = contract.as_dict()

    assert payload["run_status"] == "FAILED"
    assert payload["evidence_status"] == "INVALID"
    assert payload["action_permission"] == "NONE"
    assert payload["action"] is None
    assert payload["legacy_signal"] is None
    assert safe_legacy_signal("UNKNOWN") is None


def test_avoid_is_non_action_without_rebalance_permission() -> None:
    """AVOID 即使有证据也不能变成旧 SELL 或调仓权限。"""
    contract = DecisionReportContract(
        RunStatus.SUCCEEDED,
        EvidenceStatus.QUALIFIED,
        ActionPermission.NONE,
        RecommendationAction.AVOID,
    )

    payload = contract.as_dict()

    assert payload["action"] == "AVOID"
    assert payload["legacy_signal"] is None
    assert payload["action_permission"] == "NONE"
    with pytest.raises(ValueError):
        DecisionReportContract(
            RunStatus.SUCCEEDED,
            EvidenceStatus.QUALIFIED,
            ActionPermission.CONDITIONAL_REFERENCE,
            RecommendationAction.AVOID,
        )


def test_contract_rejects_unknown_state_and_unsafe_none_action() -> None:
    """未知状态和无权限动作都必须在契约层失败闭合。"""
    with pytest.raises(ValueError):
        DecisionReportContract("UNKNOWN", EvidenceStatus.INVALID, ActionPermission.NONE)
    with pytest.raises(ValueError):
        DecisionReportContract(
            RunStatus.FAILED,
            EvidenceStatus.INVALID,
            ActionPermission.NONE,
            RecommendationAction.ADD,
        )
    assert not is_valid_report_contract({"run_status": "UNKNOWN"})


def test_conditional_reference_requires_explicit_action() -> None:
    """条件参考权限不能伪装成空动作或证据不足动作。"""
    with pytest.raises(ValueError):
        DecisionReportContract(
            RunStatus.SUCCEEDED,
            EvidenceStatus.QUALIFIED,
            ActionPermission.CONDITIONAL_REFERENCE,
            None,
        )
    with pytest.raises(ValueError):
        DecisionReportContract(
            RunStatus.SUCCEEDED,
            EvidenceStatus.QUALIFIED,
            ActionPermission.CONDITIONAL_REFERENCE,
            RecommendationAction.INSUFFICIENT_EVIDENCE,
        )

    research_only = DecisionReportContract(
        RunStatus.SUCCEEDED,
        EvidenceStatus.RESEARCH_ONLY,
        ActionPermission.NONE,
        None,
    )
    assert research_only.as_dict()["action"] is None


@pytest.mark.parametrize(
    ("run_status", "evidence_status"),
    [
        (RunStatus.RUNNING, EvidenceStatus.QUALIFIED),
        (RunStatus.FAILED, EvidenceStatus.QUALIFIED),
        (RunStatus.SUCCEEDED, EvidenceStatus.RESEARCH_ONLY),
        (RunStatus.SUCCEEDED, EvidenceStatus.INSUFFICIENT),
        (RunStatus.SUCCEEDED, EvidenceStatus.STALE),
        (RunStatus.SUCCEEDED, EvidenceStatus.INVALID),
    ],
)
def test_conditional_reference_rejects_non_success_or_non_qualified(
    run_status: RunStatus,
    evidence_status: EvidenceStatus,
) -> None:
    """非成功运行或非 QUALIFIED 证据不能获得条件参考权限。"""
    with pytest.raises(ValueError):
        DecisionReportContract(
            run_status,
            evidence_status,
            ActionPermission.CONDITIONAL_REFERENCE,
            RecommendationAction.HOLD,
        )


def test_validate_report_contract_rejects_conflicting_action_sources() -> None:
    """顶层 action 与嵌套 action 冲突时拒绝歧义报告。"""
    payload = {
        "run_status": "SUCCEEDED",
        "evidence_status": "QUALIFIED",
        "action_permission": "CONDITIONAL_REFERENCE",
        "action": "ADD",
        "recommendation": {"action": "HOLD"},
    }

    with pytest.raises(ValueError, match="冲突"):
        validate_report_contract(payload)
    assert not is_valid_report_contract(payload)


def test_validate_report_contract_reads_nested_recommendation() -> None:
    """校验器支持 v2 recommendation 嵌套字段但仍严格检查状态。"""
    contract = validate_report_contract(
        {
            "run_status": "SUCCEEDED",
            "evidence_status": "QUALIFIED",
            "action_permission": "CONDITIONAL_REFERENCE",
            "recommendation": {"action": "HOLD", "reason_codes": ["NO_ACTION_THRESHOLD_MET"]},
        }
    )

    assert contract.action is RecommendationAction.HOLD
    assert contract.reason_codes == ("NO_ACTION_THRESHOLD_MET",)
    assert contract.as_dict()["legacy_signal"] == "HOLD"


def test_missing_evidence_is_default_deny() -> None:
    """缺失证据不能被默认解释为 HOLD 或任何五态动作。"""
    result = evaluate_evidence_gate()

    assert not result.passed
    assert result.action_permission is ActionPermission.NONE
    assert "EVIDENCE_MISSING" in result.failed_codes


def test_stale_or_error_data_is_default_deny_for_object_and_mapping() -> None:
    """对象型数据包、陈旧数据和 ERROR 质量都必须拒绝。"""
    stale = SimpleNamespace(as_of="2026-09-01", is_stale=True, quality_flags=())
    stale_result = evaluate_data_gate(stale, {"paths_valid": True, "paths_finite": True})
    error_result = evaluate_data_gate(
        {"as_of": "2026-09-01", "is_stale": False, "quality_flags": ["ERROR_BAD_PRICE"]},
        {"paths_valid": True, "paths_finite": True},
        _valid_paths(),
    )

    assert not stale_result.passed
    assert stale_result.evidence_status is EvidenceStatus.STALE
    assert not error_result.passed
    assert "NO_QUALITY_ERROR" in error_result.failed_codes


def test_nan_or_empty_paths_cannot_pass_data_gate() -> None:
    """空路径和 NaN 路径均拒绝，不能进入动作层。"""
    provenance = {"as_of": "2026-09-01", "is_stale": False, "quality_flags": []}
    sampling = {"model_status": "READY"}
    empty_result = evaluate_data_gate(provenance, sampling, np.empty((0, 5, 4)))
    nan_result = evaluate_data_gate(provenance, sampling, np.array([[[1.0, 1.0, 1.0, np.nan]]]))

    assert not empty_result.passed
    assert not nan_result.passed
    assert "PATHS_FINITE" in nan_result.failed_codes


def test_data_gate_accepts_only_explicit_valid_path_metadata_without_array() -> None:
    """引擎可消费已验证的路径元数据，但独立调用缺字段仍默认拒绝。"""
    provenance = {"as_of": "2026-09-01", "is_stale": False, "quality_flags": []}
    valid_metadata = evaluate_data_gate(
        provenance,
        {"paths_valid": True, "paths_finite": True},
    )
    missing_metadata = evaluate_data_gate(provenance, {})

    assert valid_metadata.passed
    assert not missing_metadata.passed
    assert "PATHS_NONEMPTY" in missing_metadata.failed_codes
    assert "PATHS_FINITE" in missing_metadata.failed_codes


def test_combined_gate_rejects_unknown_run_and_missing_signal() -> None:
    """组合门禁缺少运行状态、资格或信号时默认无权限。"""
    result = evaluate_decision_gate(
        {
            "run_status": "UNKNOWN",
            "data_provenance": {"as_of": "2026-09-01", "is_stale": False},
            "sampling": {"paths_valid": True, "paths_finite": True},
            "prediction_paths": _valid_paths(),
            "evidence": None,
            "eligibility": {"eligible": True},
            "signal_available": False,
        }
    )

    assert not result.passed
    assert result.action_permission is ActionPermission.NONE
    assert "RUN_STATUS_KNOWN" in result.failed_codes
    assert "SIGNAL_AVAILABLE" in result.failed_codes
    assert safe_legacy_signal(None) is None


def test_synthetic_evidence_never_becomes_calibrated() -> None:
    """合成证据即使带有数值也不能获得校准动作权限。"""
    evidence = {
        "calibrated_up_probability": 0.7,
        "short_probability": 0.7,
        "long_probability": 0.7,
        "cross_section_rank": 0.8,
        "short_rank": 0.8,
        "long_rank": 0.8,
    }

    result = evaluate_evidence_gate(evidence, synthetic=True)

    assert not result.passed
    assert result.action_permission is ActionPermission.NONE
    assert "SYNTHETIC_EVIDENCE_NOT_CALIBRATED" in result.failed_codes


def test_pure_ranking_strategy_may_explicitly_skip_probability_calibration() -> None:
    """明确声明纯排序依赖时只检查有限排名，不伪造校准概率。"""
    evidence = {
        "cross_section_rank": 0.8,
        "short_rank": 0.7,
        "long_rank": 0.9,
    }

    research_gate = SimpleNamespace(passed=True)
    unknown = evaluate_evidence_gate(evidence, research_gate=research_gate)
    ranking = evaluate_evidence_gate(
        evidence,
        research_gate=research_gate,
        requires_calibration=False,
    )
    bad_optional_probability = evaluate_evidence_gate(
        {**evidence, "calibrated_up_probability": np.nan},
        research_gate=research_gate,
        requires_calibration=False,
    )

    assert not unknown.passed
    assert "CALIBRATION_REQUIREMENT_UNKNOWN" in unknown.failed_codes
    assert ranking.passed
    assert ranking.checks["CALIBRATION_NOT_REQUIRED"] is True
    assert not bad_optional_probability.passed
    assert "OPTIONAL_PROBABILITIES_FINITE" in bad_optional_probability.failed_codes


def test_explicit_calibration_requirement_rejects_missing_probabilities() -> None:
    """概率策略显式要求校准时，缺失概率仍然默认拒绝。"""
    evidence = {"cross_section_rank": 0.8, "short_rank": 0.7, "long_rank": 0.9}

    result = evaluate_evidence_gate(evidence, requires_calibration=True)

    assert not result.passed
    assert "MULTI_HORIZON_CALIBRATION_UNAVAILABLE" in result.failed_codes
