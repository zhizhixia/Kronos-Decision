"""五态建议及 v1 映射；动作只由结构化规则决定。"""
from __future__ import annotations

from enum import Enum
from typing import Any


class RecommendationAction(str, Enum):
    """稳定的五态建议枚举。"""

    ADD = "ADD"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    AVOID = "AVOID"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


def decide_action(gate_passed: bool, ranks: dict[int, float], up_probabilities: dict[int, float], is_held: bool, risk_blocked: bool) -> tuple[RecommendationAction, tuple[str, ...]]:
    """以20日为主、5/60日否决的固定五态规则。"""
    if not gate_passed:
        return RecommendationAction.INSUFFICIENT_EVIDENCE, ("EVIDENCE_GATE_FAILED",)
    main_rank, main_prob = ranks[20], up_probabilities[20]
    short_or_long_weak = ranks[5] <= 0.2 or ranks[60] <= 0.2 or up_probabilities[5] <= 0.35 or up_probabilities[60] <= 0.35
    if risk_blocked:
        return RecommendationAction.AVOID, ("TRADE_ELIGIBILITY_OR_RISK_BLOCK",)
    if main_rank >= 0.8 and main_prob >= 0.65 and not short_or_long_weak:
        return RecommendationAction.ADD, ("H20_TOP_QUINTILE", "H20_CALIBRATED_UP_PROBABILITY")
    if main_rank <= 0.2 and main_prob <= 0.35:
        return (RecommendationAction.REDUCE if is_held else RecommendationAction.AVOID), ("H20_BOTTOM_QUINTILE", "H20_CALIBRATED_DOWN_PROBABILITY")
    return RecommendationAction.HOLD, ("NO_ACTION_THRESHOLD_MET",)


def _enum_value(value: Any) -> Any:
    """读取枚举值；映射层不依赖契约模块，避免循环导入。"""
    return getattr(value, "value", value)


def legacy_signal(
    action: RecommendationAction | str | None,
    evidence_status: Any = None,
    action_permission: Any = None,
) -> str | None:
    """仅把有资格的五态动作映射到旧三态信号。

    缺失、未知、证据不足和 AVOID 都是无动作结果，不能借默认值恢复成
    ``BUY``、``HOLD`` 或 ``SELL``。映射必须同时具备 QUALIFIED 证据和
    CONDITIONAL_REFERENCE 权限。
    """
    if _enum_value(evidence_status) != "QUALIFIED":
        return None
    if _enum_value(action_permission) != "CONDITIONAL_REFERENCE":
        return None
    try:
        normalized = action if isinstance(action, RecommendationAction) else RecommendationAction(action)
    except (TypeError, ValueError):
        return None
    return {
        RecommendationAction.ADD: "BUY",
        RecommendationAction.HOLD: "HOLD",
        RecommendationAction.REDUCE: "SELL",
    }.get(normalized)


def error_payload(code: str, message: str, retryable: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    """v2 错误响应的统一结构。"""
    return {"status": "error", "error": {"code": code, "message": message, "retryable": retryable, "details": details or {}}}
