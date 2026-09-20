"""正式投资建议的硬证据门禁。"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import math
from numbers import Real
from typing import Any


_CHECK_CODES: tuple[str, ...] = (
    "DATA_COMPLETE",
    "NO_QUALITY_ERROR",
    "INTERVAL_COVERAGE",
    "PROBABILITY_ECE",
    "RANK_IC",
    "BOOTSTRAP_RANK_IC",
    "NET_EXCESS_RETURN",
    "INFORMATION_RATIO",
    "WINDOW_STABILITY",
    "DRAWDOWN",
    "VERSION_MATCH",
    "FORMAL_PROTOCOL",
    "FRESH_ARTIFACT",
)
_MISSING = object()


@dataclass(frozen=True)
class EvidenceGateResult:
    """门禁逐项结果；任一失败均不得输出交易动作。"""

    passed: bool
    failed_codes: tuple[str, ...]
    checks: dict[str, bool]

    def as_dict(self) -> dict[str, Any]:
        """转换为 API 和工件可序列化结构。"""
        return asdict(self)


def evaluate_evidence_gate(metrics: dict[str, Any], now: datetime | None = None) -> EvidenceGateResult:
    """按锁定阈值评估正式研究；缺失、异常或非有限输入一律拒绝。"""
    checks = _empty_checks()
    if not isinstance(metrics, dict):
        return _build_result(checks)

    resolved_now = datetime.now() if now is None else now
    try:
        checks["DATA_COMPLETE"] = _is_strict_bool(metrics.get("data_complete", _MISSING), True)
        checks["NO_QUALITY_ERROR"] = _is_strict_bool(metrics.get("quality_error", _MISSING), False)
        checks["INTERVAL_COVERAGE"] = _passes_interval_coverage(metrics)
        checks["PROBABILITY_ECE"] = _passes_numeric_threshold(metrics, "up_probability_ece", lambda value: value <= 0.08)
        checks["RANK_IC"] = _passes_numeric_threshold(metrics, "rank_ic_mean", lambda value: value > 0)
        checks["BOOTSTRAP_RANK_IC"] = _passes_numeric_threshold(
            metrics, "rank_ic_positive_probability", lambda value: value >= 0.80
        )
        checks["NET_EXCESS_RETURN"] = _passes_numeric_threshold(
            metrics, "annualized_excess_return", lambda value: value > 0
        )
        checks["INFORMATION_RATIO"] = _passes_numeric_threshold(
            metrics, "information_ratio", lambda value: value >= 0.5
        )
        checks["WINDOW_STABILITY"] = _passes_numeric_threshold(
            metrics, "positive_12m_window_ratio", lambda value: value >= 0.60
        )
        checks["DRAWDOWN"] = _passes_numeric_threshold(
            metrics, "drawdown_worsening", lambda value: value <= 0.05
        )
        checks["VERSION_MATCH"] = _is_strict_bool(metrics.get("versions_match", _MISSING), True)
        checks["FORMAL_PROTOCOL"] = _is_strict_bool(metrics.get("formal_protocol", _MISSING), True)
        checks["FRESH_ARTIFACT"] = _is_fresh(metrics.get("evaluated_at", _MISSING), resolved_now)
    except Exception:
        # 自定义映射或异常值不得让门禁异常退出，也不得保留部分成功状态。
        checks = _empty_checks()
    return _build_result(checks)


def _empty_checks() -> dict[str, bool]:
    """创建默认全失败的逐项检查结果。"""
    return {code: False for code in _CHECK_CODES}


def _build_result(checks: dict[str, bool]) -> EvidenceGateResult:
    """根据逐项结果生成包含明确失败代码的门禁结果。"""
    failed = tuple(code for code, passed in checks.items() if not passed)
    return EvidenceGateResult(not failed, failed, checks)


def _is_strict_bool(value: Any, expected: bool) -> bool:
    """只接受真正的布尔值，拒绝字符串、数字和其他可转换类型。"""
    return type(value) is bool and value is expected


def _finite_number(value: Any) -> float | None:
    """把有限实数转换为浮点数；非数值、NaN 和无穷值返回空值。"""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def _passes_numeric_threshold(
    metrics: dict[str, Any], key: str, predicate: Callable[[float], bool]
) -> bool:
    """验证数值字段有限且满足对应阈值。"""
    number = _finite_number(metrics.get(key, _MISSING))
    return number is not None and bool(predicate(number))


def _passes_interval_coverage(metrics: dict[str, Any]) -> bool:
    """验证区间覆盖率为有限数值且落在锁定范围内。"""
    coverage = _finite_number(metrics.get("coverage_80", _MISSING))
    return coverage is not None and 0.75 <= coverage <= 0.85


def _is_fresh(value: Any, now: datetime) -> bool:
    """验证评估时间可解析、与当前时间同属时区体系且处于有效窗口。"""
    if not isinstance(now, datetime):
        return False
    try:
        if isinstance(value, datetime):
            evaluated_at = value
        elif isinstance(value, str):
            evaluated_at = datetime.fromisoformat(value)
        else:
            return False
        evaluated_offset = evaluated_at.utcoffset()
        now_offset = now.utcoffset()
        if (evaluated_offset is None) != (now_offset is None):
            return False
        age = now - evaluated_at
        return timedelta(0) <= age <= timedelta(days=30)
    except Exception:
        return False
