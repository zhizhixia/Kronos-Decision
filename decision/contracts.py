"""决策报告的最小稳定状态契约。

本模块只描述报告状态和安全映射，不负责调用模型、读取数据或执行副作用。
五态动作直接复用 :mod:`decision.v2` 的 RecommendationAction，避免产生第二套动作语义。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from decision.v2 import RecommendationAction, legacy_signal


class RunStatus(str, Enum):
    """报告运行状态；运行成功不等于策略获得动作资格。"""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"


class EvidenceStatus(str, Enum):
    """报告证据状态；未知或未达到资格的状态均不得开放动作。"""

    RESEARCH_ONLY = "RESEARCH_ONLY"
    QUALIFIED = "QUALIFIED"
    INSUFFICIENT = "INSUFFICIENT"
    STALE = "STALE"
    INVALID = "INVALID"


class ActionPermission(str, Enum):
    """报告动作展示权限。"""

    NONE = "NONE"
    CONDITIONAL_REFERENCE = "CONDITIONAL_REFERENCE"


_SAFE_NON_ACTION = frozenset({
    None,
    RecommendationAction.AVOID,
    RecommendationAction.INSUFFICIENT_EVIDENCE,
})
_REFERENCEABLE_ACTIONS = frozenset({
    RecommendationAction.ADD,
    RecommendationAction.HOLD,
    RecommendationAction.REDUCE,
})


def _coerce_enum(enum_type: type[Enum], value: Any, field_name: str) -> Enum:
    """将外部状态值严格转换为枚举，未知值直接拒绝。"""
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 状态未知：{value!r}") from exc


def safe_legacy_signal(
    action: RecommendationAction | str | None,
    evidence_status: Any = None,
    action_permission: Any = None,
) -> str | None:
    """在完整证据和权限上下文中映射旧信号，否则返回无动作。"""
    return legacy_signal(action, evidence_status, action_permission)


@dataclass(frozen=True)
class DecisionReportContract:
    """后端最小报告契约。

    ``action_permission=NONE`` 时只允许没有动作、AVOID 或 INSUFFICIENT_EVIDENCE，
    这样失败、陈旧、未知和研究结果不能被消费方误读成可执行交易动作。
    """

    run_status: RunStatus
    evidence_status: EvidenceStatus
    action_permission: ActionPermission
    action: RecommendationAction | None = None
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """严格规范化并校验状态之间的安全组合。"""
        run_status = _coerce_enum(RunStatus, self.run_status, "run_status")
        evidence_status = _coerce_enum(EvidenceStatus, self.evidence_status, "evidence_status")
        action_permission = _coerce_enum(ActionPermission, self.action_permission, "action_permission")
        action = self.action
        if action is not None and not isinstance(action, RecommendationAction):
            try:
                action = RecommendationAction(action)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"action 状态未知：{action!r}") from exc
        reason_codes = tuple(str(code) for code in self.reason_codes if str(code))

        if action_permission is ActionPermission.NONE and action not in _SAFE_NON_ACTION:
            raise ValueError("action_permission=NONE 时不能携带可执行动作。")
        if run_status is not RunStatus.SUCCEEDED and action_permission is not ActionPermission.NONE:
            raise ValueError("未成功运行不能获得动作权限。")
        if evidence_status is not EvidenceStatus.QUALIFIED and action_permission is not ActionPermission.NONE:
            raise ValueError("未达 QUALIFIED 证据状态不能获得动作权限。")
        if action_permission is ActionPermission.CONDITIONAL_REFERENCE and evidence_status is not EvidenceStatus.QUALIFIED:
            raise ValueError("只有 QUALIFIED 证据可以获得 CONDITIONAL_REFERENCE 权限。")
        if action_permission is ActionPermission.CONDITIONAL_REFERENCE and action not in _REFERENCEABLE_ACTIONS:
            raise ValueError("CONDITIONAL_REFERENCE 权限只能携带 ADD、HOLD 或 REDUCE。")
        if run_status is not RunStatus.SUCCEEDED and action not in _SAFE_NON_ACTION:
            raise ValueError("未成功运行不能携带可执行动作。")
        if evidence_status is not EvidenceStatus.QUALIFIED and action not in _SAFE_NON_ACTION:
            raise ValueError("未达 QUALIFIED 证据状态不能携带可执行动作。")

        object.__setattr__(self, "run_status", run_status)
        object.__setattr__(self, "evidence_status", evidence_status)
        object.__setattr__(self, "action_permission", action_permission)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "reason_codes", reason_codes)

    def as_dict(self) -> dict[str, Any]:
        """返回只含基础值的可序列化报告状态。"""
        return {
            "run_status": self.run_status.value,
            "evidence_status": self.evidence_status.value,
            "action_permission": self.action_permission.value,
            "action": self.action.value if self.action is not None else None,
            "reason_codes": list(self.reason_codes),
            "legacy_signal": safe_legacy_signal(
                self.action,
                self.evidence_status,
                self.action_permission,
            ),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DecisionReportContract":
        """从顶层或 recommendation 嵌套字段读取并校验报告契约；重复动作冲突时拒绝。"""
        recommendation = payload.get("recommendation")
        recommendation_mapping = recommendation if isinstance(recommendation, Mapping) else {}
        top_level_has_action = "action" in payload
        nested_has_action = "action" in recommendation_mapping
        if top_level_has_action and nested_has_action:
            top_level_action = payload["action"]
            nested_action = recommendation_mapping["action"]
            if getattr(top_level_action, "value", top_level_action) != getattr(nested_action, "value", nested_action):
                raise ValueError("顶层 action 与 recommendation.action 冲突。")
        action = payload.get("action", recommendation_mapping.get("action"))
        reason_codes = payload.get("reason_codes", recommendation_mapping.get("reason_codes", ()))
        if reason_codes is None:
            reason_codes = ()
        if isinstance(reason_codes, str):
            reason_codes = (reason_codes,)
        return cls(
            run_status=payload.get("run_status"),
            evidence_status=payload.get("evidence_status"),
            action_permission=payload.get("action_permission"),
            action=action,
            reason_codes=tuple(reason_codes),
        )


# 便于调用方使用更短的描述名；两者是同一个类型，不产生第二套契约。
ReportContract = DecisionReportContract


def validate_report_contract(payload: Mapping[str, Any]) -> DecisionReportContract:
    """严格校验报告状态并返回规范化契约对象。"""
    if not isinstance(payload, Mapping):
        raise ValueError("报告契约必须是映射对象。")
    return DecisionReportContract.from_mapping(payload)


def is_valid_report_contract(payload: Mapping[str, Any]) -> bool:
    """返回报告是否满足最小状态契约。"""
    try:
        validate_report_contract(payload)
    except (TypeError, ValueError):
        return False
    return True


__all__ = [
    "ActionPermission",
    "DecisionReportContract",
    "EvidenceStatus",
    "ReportContract",
    "RunStatus",
    "is_valid_report_contract",
    "safe_legacy_signal",
    "validate_report_contract",
]
