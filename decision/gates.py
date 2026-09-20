"""决策后端的纯函数拒绝门禁。

门禁只读取输入并返回结构化结果，不读取文件、时间、网络或全局状态。缺失输入、未知状态、坏路径和质量错误默认拒绝。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite
from typing import Any, Iterable, Mapping

import numpy as np

from decision.contracts import ActionPermission, EvidenceStatus, RunStatus

try:
    from data.contracts import has_hard_quality_flags
except ModuleNotFoundError as exc:
    if exc.name != "pandas":
        raise

    _HARD_QUALITY_FLAGS = {
        "CALENDAR_QLIB_OUTDATED",
        "CALENDAR_WEEKDAY_FALLBACK",
    }

    def has_hard_quality_flags(flags: tuple[str, ...] | list[str]) -> bool:
        """在数据依赖不可导入时保持质量旗标门禁的纯函数行为。"""
        normalized = {str(flag) for flag in flags}
        return any(flag.startswith("ERROR_") for flag in normalized) or bool(
            normalized & _HARD_QUALITY_FLAGS
        )


@dataclass(frozen=True)
class DecisionGateResult:
    """统一门禁结果；门禁本身不决定最终四态动作。"""

    passed: bool
    failed_codes: tuple[str, ...]
    checks: dict[str, bool]
    evidence_status: EvidenceStatus
    action_permission: ActionPermission

    def __post_init__(self) -> None:
        """门禁结果没有最终动作，因此始终关闭动作权限。"""
        object.__setattr__(self, "action_permission", ActionPermission.NONE)

    @property
    def reason_codes(self) -> tuple[str, ...]:
        """返回与报告契约一致的原因码视图。"""
        return self.failed_codes

    def as_dict(self) -> dict[str, Any]:
        """转换为基础值，且不序列化无动作的条件权限。"""
        payload = asdict(self)
        payload["evidence_status"] = self.evidence_status.value
        payload["action_permission"] = ActionPermission.NONE.value
        return payload


# 语义别名指向同一结果类型，避免调用方因命名不同复制契约。
GateResult = DecisionGateResult
ActionGateResult = DecisionGateResult


def _read(source: Any, key: str, default: Any = None) -> Any:
    """从映射或对象读取字段。"""
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _status(enum_type: type[Enum], value: Any) -> tuple[Enum | None, bool]:
    """严格解析状态并返回是否为已知值。"""
    if isinstance(value, enum_type):
        return value, True
    if value is None:
        return None, False
    try:
        return enum_type(value), True
    except (TypeError, ValueError):
        return None, False


_MISSING = object()


def _strict_passed(source: Any) -> bool:
    """只有字段值的实际类型为 bool 且值为 True 才算通过。"""
    value = _read(source, "passed", _MISSING)
    return type(value) is bool and value is True


def _finite_number(value: Any, lower: float | None = None, upper: float | None = None) -> bool:
    """判断数值存在、有限并可选地落在闭区间内。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    if not isfinite(number):
        return False
    return (lower is None or number >= lower) and (upper is None or number <= upper)


def _normalise_flags(*sources: Any) -> tuple[str, ...]:
    """合并质量旗标并保留稳定顺序。"""
    result: list[str] = []
    for source in sources:
        if source is None:
            continue
        values = (source,) if isinstance(source, str) else source
        try:
            iterator = iter(values)
        except TypeError:
            iterator = iter((values,))
        for value in iterator:
            flag = str(value)
            if flag and flag not in result:
                result.append(flag)
    return tuple(result)


def _paths_are_valid(paths: Any) -> tuple[bool, bool]:
    """返回预测路径是否非空、是否满足有限数值与最小形状。"""
    if paths is None:
        return False, False
    try:
        array = np.asarray(paths, dtype=float)
    except (TypeError, ValueError):
        return False, False
    nonempty = array.size > 0 and array.ndim >= 3 and array.shape[0] > 0 and array.shape[1] > 0 and array.shape[-1] > 3
    return bool(nonempty), bool(nonempty and np.isfinite(array).all())


def _failed_codes(checks: Mapping[str, bool], extra: Iterable[str] = ()) -> tuple[str, ...]:
    """按检查声明顺序合并失败原因，避免重复原因码。"""
    codes: list[str] = [code for code, passed in checks.items() if not passed]
    for code in extra:
        if code not in codes:
            codes.append(code)
    return tuple(codes)


def evaluate_eligibility_gate(eligibility: Any = None) -> DecisionGateResult:
    """评估交易资格；缺失或未知资格默认拒绝。"""
    eligible_value = _read(eligibility, "eligible")
    eligible = eligible_value is True
    reasons = _read(eligibility, "reasons", ()) or ()
    checks = {"ELIGIBILITY_KNOWN": isinstance(eligible_value, bool), "ELIGIBLE": eligible}
    extra = tuple(str(reason) for reason in reasons if str(reason)) if not eligible else ()
    if not eligible and not extra:
        extra = ("TRADE_ELIGIBILITY_OR_RISK_BLOCK",)
    failed = _failed_codes(checks, extra)
    return DecisionGateResult(
        passed=not failed,
        failed_codes=failed,
        checks=checks,
        evidence_status=EvidenceStatus.QUALIFIED if not failed else EvidenceStatus.INSUFFICIENT,
        action_permission=ActionPermission.NONE,
    )


def evaluate_data_gate(
    data_provenance: Any = None,
    sampling: Any = None,
    prediction_paths: Any = None,
    *,
    model_failed: bool | None = None,
) -> DecisionGateResult:
    """评估数据新鲜度、质量、模型状态和预测路径；任何缺失前提均拒绝。"""
    data = data_provenance if data_provenance is not None else {}
    if prediction_paths is None and sampling is not None:
        embedded_paths = _read(sampling, "paths")
        if embedded_paths is not None:
            prediction_paths = embedded_paths
        else:
            candidate_nonempty, candidate_finite = _paths_are_valid(sampling)
            if candidate_nonempty or candidate_finite:
                prediction_paths = sampling
                sampling = None
    flags = _normalise_flags(
        _read(data_provenance, "quality_flags", ()),
        _read(sampling, "quality_flags", ()),
    )
    hard_error = bool(_read(data_provenance, "has_hard_quality_error", False)) or has_hard_quality_flags(flags) or "PATH_REPAIR_EXCESSIVE" in flags
    quality_status = _read(data_provenance, "quality_status")
    unknown_quality_status = quality_status is not None and str(quality_status).upper() not in {"OK", "WARNING", "DEGRADED"}
    if unknown_quality_status:
        hard_error = True
    nonempty, finite = _paths_are_valid(prediction_paths)
    if prediction_paths is None:
        nonempty = _read(sampling, "paths_valid") is True
        finite = _read(sampling, "paths_finite") is True and nonempty
    explicit_model_status = _read(sampling, "model_status")
    model_status_ok = explicit_model_status is None or str(explicit_model_status).upper() in {"OK", "READY", "SUCCEEDED"}
    if explicit_model_status is not None and not model_status_ok:
        model_failed = True
    if model_failed is None:
        model_failed = bool(_read(sampling, "model_failed", False))

    checks = {
        "DATA_PRESENT": _read(data, "as_of") not in (None, ""),
        "DATA_FRESH": _read(data, "is_stale") is False,
        "NO_QUALITY_ERROR": not hard_error,
        "PATHS_NONEMPTY": nonempty,
        "PATHS_FINITE": finite,
        "MODEL_OK": not bool(model_failed) and model_status_ok,
    }
    extra = ("QUALITY_STATUS_UNKNOWN",) if unknown_quality_status else ()
    failed = _failed_codes(checks, extra)
    if _read(data, "is_stale") is True:
        status = EvidenceStatus.STALE
    elif failed:
        status = EvidenceStatus.INVALID
    else:
        status = EvidenceStatus.RESEARCH_ONLY
    return DecisionGateResult(
        passed=not failed,
        failed_codes=failed,
        checks=checks,
        evidence_status=status,
        action_permission=ActionPermission.NONE,
    )


def _probabilities(evidence: Any) -> tuple[Any, Any, Any]:
    """提取主、短期和长期校准概率。"""
    main = _read(evidence, "calibrated_up_probability")
    short = _read(evidence, "short_probability")
    long = _read(evidence, "long_probability")
    horizons = _read(evidence, "horizons", {})
    if isinstance(horizons, Mapping):
        short_item = horizons.get(5, horizons.get("5"))
        main_item = horizons.get(20, horizons.get("20"))
        long_item = horizons.get(60, horizons.get("60"))
        if short is None:
            short = _read(short_item, "calibrated_up_probability")
        if main is None:
            main = _read(main_item, "calibrated_up_probability")
        if long is None:
            long = _read(long_item, "calibrated_up_probability")
    return short, main, long


def evaluate_evidence_gate(
    evidence: Any = None,
    *,
    research_gate: Any = _MISSING,
    synthetic: bool = False,
    requires_calibration: bool | None = None,
) -> DecisionGateResult:
    """评估点时证据和策略依赖；未知校准要求默认拒绝。"""
    explicit_status, status_known = _status(EvidenceStatus, _read(evidence, "evidence_status"))
    short, main, long = _probabilities(evidence)
    cross_rank = _read(evidence, "cross_section_rank")
    short_rank = _read(evidence, "short_rank")
    long_rank = _read(evidence, "long_rank")
    # 纯排序策略明确不要求校准时，省略 research_gate 表示该依赖不适用；
    # 一旦传入 research_gate，仍必须严格提供 passed=True。
    ranking_without_research_gate = research_gate is _MISSING and requires_calibration is False
    research_gate_passed = ranking_without_research_gate or _strict_passed(research_gate)
    checks: dict[str, bool] = {
        "EVIDENCE_PRESENT": evidence is not None,
        "EVIDENCE_STATUS_KNOWN": _read(evidence, "evidence_status") is None or status_known,
        "EVIDENCE_QUALIFIED": explicit_status is None or explicit_status is EvidenceStatus.QUALIFIED,
        "RESEARCH_GATE_PASSED": research_gate_passed,
        "RANKS_FINITE": _finite_number(cross_rank, 0.0, 1.0) and _finite_number(short_rank, 0.0, 1.0) and _finite_number(long_rank, 0.0, 1.0),
        "NOT_SYNTHETIC": not synthetic,
    }
    extra: list[str] = []
    if evidence is None:
        extra.append("EVIDENCE_MISSING")
    if synthetic:
        extra.append("SYNTHETIC_EVIDENCE_NOT_CALIBRATED")
    if not research_gate_passed:
        extra.append("EVIDENCE_GATE_FAILED")
    if requires_calibration is True:
        checks.update(
            {
                "CALIBRATED_SHORT": _finite_number(short, 0.0, 1.0),
                "CALIBRATED_MAIN": _finite_number(main, 0.0, 1.0),
                "CALIBRATED_LONG": _finite_number(long, 0.0, 1.0),
            }
        )
        if any(value is None for value in (short, main, long)):
            extra.append("MULTI_HORIZON_CALIBRATION_UNAVAILABLE")
    elif requires_calibration is False:
        checks["CALIBRATION_NOT_REQUIRED"] = True
        checks["OPTIONAL_PROBABILITIES_FINITE"] = all(
            value is None or _finite_number(value, 0.0, 1.0)
            for value in (short, main, long)
        )
    else:
        checks["CALIBRATION_REQUIREMENT_KNOWN"] = False
        extra.append("CALIBRATION_REQUIREMENT_UNKNOWN")
    failed = _failed_codes(checks, extra)
    status = EvidenceStatus.QUALIFIED if not failed else (explicit_status or EvidenceStatus.INSUFFICIENT)
    if status is EvidenceStatus.QUALIFIED and failed:
        status = EvidenceStatus.INSUFFICIENT
    return DecisionGateResult(
        passed=not failed,
        failed_codes=failed,
        checks=checks,
        evidence_status=status,
        action_permission=ActionPermission.NONE,
    )


def evaluate_decision_gate(context: Mapping[str, Any] | None = None) -> DecisionGateResult:
    """组合运行、资格、数据和证据门禁；缺少任何上下文时默认拒绝。"""
    values = context if isinstance(context, Mapping) else {}
    run_status, run_known = _status(RunStatus, values.get("run_status"))
    signal_available = values.get("signal_available") is True
    run_checks = {
        "RUN_STATUS_KNOWN": run_known,
        "RUN_SUCCEEDED": run_status is RunStatus.SUCCEEDED,
        "SIGNAL_AVAILABLE": signal_available,
    }
    data_gate = evaluate_data_gate(
        values.get("data_provenance"),
        values.get("sampling"),
        values.get("prediction_paths"),
        model_failed=values.get("model_failed"),
    )
    evidence_gate = evaluate_evidence_gate(
        values.get("evidence"),
        research_gate=values.get("research_gate"),
        synthetic=bool(values.get("synthetic", False)),
        requires_calibration=values.get("requires_calibration"),
    )
    eligibility_gate = evaluate_eligibility_gate(values.get("eligibility"))
    failed = list(_failed_codes(run_checks))
    for result in (data_gate, evidence_gate, eligibility_gate):
        failed.extend(code for code in result.failed_codes if code not in failed)
    checks = dict(run_checks)
    checks.update({f"DATA_{key}": value for key, value in data_gate.checks.items()})
    checks.update({f"EVIDENCE_{key}": value for key, value in evidence_gate.checks.items()})
    checks.update({f"ELIGIBILITY_{key}": value for key, value in eligibility_gate.checks.items()})
    if data_gate.evidence_status is EvidenceStatus.STALE:
        evidence_status = EvidenceStatus.STALE
    elif data_gate.evidence_status is EvidenceStatus.INVALID or evidence_gate.evidence_status is EvidenceStatus.INVALID:
        evidence_status = EvidenceStatus.INVALID
    elif not failed:
        evidence_status = EvidenceStatus.QUALIFIED
    else:
        evidence_status = evidence_gate.evidence_status
    return DecisionGateResult(not failed, tuple(failed), checks, evidence_status, ActionPermission.NONE)


# 语义别名：调用方可按“动作门禁”或“报告门禁”命名，但实现只有一份。
evaluate_qualification_gate = evaluate_eligibility_gate
evaluate_action_gate = evaluate_decision_gate
evaluate_report_gates = evaluate_decision_gate


__all__ = [
    "ActionGateResult",
    "DecisionGateResult",
    "GateResult",
    "evaluate_action_gate",
    "evaluate_data_gate",
    "evaluate_decision_gate",
    "evaluate_eligibility_gate",
    "evaluate_evidence_gate",
    "evaluate_qualification_gate",
    "evaluate_report_gates",
]
