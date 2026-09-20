"""研究证据门禁 fail-closed 行为的回归测试。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
import math
from pathlib import Path
import sys
from typing import Any, cast

import pytest


_GATE_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "gates.py"
_GATE_SPEC = importlib.util.spec_from_file_location("_kronos_evaluation_gates", _GATE_PATH)
if _GATE_SPEC is None or _GATE_SPEC.loader is None:
    raise ImportError(f"无法加载门禁模块：{_GATE_PATH}")
_GATE_MODULE = importlib.util.module_from_spec(_GATE_SPEC)
sys.modules[_GATE_SPEC.name] = _GATE_MODULE
_GATE_SPEC.loader.exec_module(_GATE_MODULE)
evaluate_evidence_gate = _GATE_MODULE.evaluate_evidence_gate


_NOW = datetime(2026, 9, 19, 12, 0, 0)
_REQUIRED_CODES = (
    ("data_complete", "DATA_COMPLETE"),
    ("quality_error", "NO_QUALITY_ERROR"),
    ("coverage_80", "INTERVAL_COVERAGE"),
    ("up_probability_ece", "PROBABILITY_ECE"),
    ("rank_ic_mean", "RANK_IC"),
    ("rank_ic_positive_probability", "BOOTSTRAP_RANK_IC"),
    ("annualized_excess_return", "NET_EXCESS_RETURN"),
    ("information_ratio", "INFORMATION_RATIO"),
    ("positive_12m_window_ratio", "WINDOW_STABILITY"),
    ("drawdown_worsening", "DRAWDOWN"),
    ("versions_match", "VERSION_MATCH"),
    ("formal_protocol", "FORMAL_PROTOCOL"),
    ("evaluated_at", "FRESH_ARTIFACT"),
)
_NUMERIC_FIELDS = (
    ("coverage_80", "INTERVAL_COVERAGE"),
    ("up_probability_ece", "PROBABILITY_ECE"),
    ("rank_ic_mean", "RANK_IC"),
    ("rank_ic_positive_probability", "BOOTSTRAP_RANK_IC"),
    ("annualized_excess_return", "NET_EXCESS_RETURN"),
    ("information_ratio", "INFORMATION_RATIO"),
    ("positive_12m_window_ratio", "WINDOW_STABILITY"),
    ("drawdown_worsening", "DRAWDOWN"),
)
_BOOLEAN_FIELDS = (
    ("data_complete", "DATA_COMPLETE"),
    ("versions_match", "VERSION_MATCH"),
    ("formal_protocol", "FORMAL_PROTOCOL"),
)


def _full_metrics() -> dict[str, Any]:
    """构造满足全部锁定阈值的完整指标。"""
    return {
        "data_complete": True,
        "quality_error": False,
        "coverage_80": 0.8,
        "up_probability_ece": 0.08,
        "rank_ic_mean": 0.01,
        "rank_ic_positive_probability": 0.8,
        "annualized_excess_return": 0.01,
        "information_ratio": 0.5,
        "positive_12m_window_ratio": 0.6,
        "drawdown_worsening": 0.05,
        "versions_match": True,
        "formal_protocol": True,
        "evaluated_at": (_NOW - timedelta(days=1)).isoformat(),
    }


@pytest.mark.parametrize(("field", "code"), _REQUIRED_CODES)
def test_missing_required_field_fails_closed(field: str, code: str) -> None:
    """任一研究证据字段缺失时必须拒绝并指出对应门禁。"""
    metrics = _full_metrics()
    del metrics[field]

    result = evaluate_evidence_gate(metrics, now=_NOW)

    assert result.passed is False
    assert code in result.failed_codes
    assert result.checks[code] is False


def test_complete_metrics_pass_with_explicit_timestamp() -> None:
    """合法的全量指标仍应通过全部门禁。"""
    result = evaluate_evidence_gate(_full_metrics(), now=_NOW)

    assert result.passed is True
    assert result.failed_codes == ()
    assert all(result.checks.values())


@pytest.mark.parametrize(("field", "code"), _BOOLEAN_FIELDS)
@pytest.mark.parametrize("value", ["true", "false", 1, 0, None, object()])
def test_boolean_contract_rejects_non_bool_values(field: str, code: str, value: Any) -> None:
    """布尔门禁只接受真正的 bool，不能把字符串或数字当作布尔值。"""
    metrics = _full_metrics()
    metrics[field] = value

    result = evaluate_evidence_gate(metrics, now=_NOW)

    assert result.passed is False
    assert code in result.failed_codes


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, object()])
def test_quality_error_requires_explicit_bool_false(value: Any) -> None:
    """质量错误字段缺失或类型不对时不得被解释为无错误。"""
    metrics = _full_metrics()
    metrics["quality_error"] = value

    result = evaluate_evidence_gate(metrics, now=_NOW)

    assert result.passed is False
    assert "NO_QUALITY_ERROR" in result.failed_codes


@pytest.mark.parametrize(("field", "code"), _NUMERIC_FIELDS)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_numeric_metrics_fail_closed(field: str, code: str, value: float) -> None:
    """所有 NaN 和无穷数值都必须拒绝，不能被比较运算误放行。"""
    metrics = _full_metrics()
    metrics[field] = value

    result = evaluate_evidence_gate(metrics, now=_NOW)

    assert result.passed is False
    assert code in result.failed_codes


@pytest.mark.parametrize(("field", "code"), _NUMERIC_FIELDS)
@pytest.mark.parametrize("value", [None, "0.8", True, object(), [], {}])
def test_invalid_numeric_types_fail_closed(field: str, code: str, value: Any) -> None:
    """异常数值类型必须返回失败结果，而不是抛出转换异常。"""
    metrics = _full_metrics()
    metrics[field] = value

    result = evaluate_evidence_gate(metrics, now=_NOW)

    assert result.passed is False
    assert code in result.failed_codes


@pytest.mark.parametrize(
    "evaluated_at",
    [
        "not-a-timestamp",
        None,
        object(),
        (_NOW + timedelta(seconds=1)).isoformat(),
    ],
)
def test_unknown_or_future_evaluated_at_fails_closed(evaluated_at: Any) -> None:
    """未知或未来的评估时间必须阻断新鲜度门禁。"""
    metrics = _full_metrics()
    metrics["evaluated_at"] = evaluated_at

    result = evaluate_evidence_gate(metrics, now=_NOW)

    assert result.passed is False
    assert "FRESH_ARTIFACT" in result.failed_codes


@pytest.mark.parametrize(
    ("evaluated_at", "now"),
    [
        ((_NOW - timedelta(days=1)).isoformat(), _NOW.replace(tzinfo=timezone.utc)),
        (
            (_NOW - timedelta(days=1)).replace(tzinfo=timezone.utc).isoformat(),
            _NOW,
        ),
    ],
)
def test_mixed_naive_and_aware_timestamps_fail_closed(evaluated_at: str, now: datetime) -> None:
    """评估时间和当前时间混用有无时区时必须拒绝而不能抛 TypeError。"""
    metrics = _full_metrics()
    metrics["evaluated_at"] = evaluated_at

    result = evaluate_evidence_gate(metrics, now=now)

    assert result.passed is False
    assert "FRESH_ARTIFACT" in result.failed_codes


def test_matching_aware_timestamps_can_pass() -> None:
    """双方明确使用同一时区体系时，合法指标仍可通过。"""
    now = _NOW.replace(tzinfo=timezone.utc)
    metrics = _full_metrics()
    metrics["evaluated_at"] = (now - timedelta(days=1)).isoformat()

    result = evaluate_evidence_gate(metrics, now=now)

    assert result.passed is True


class _ExplodingMetrics(dict[str, Any]):
    """模拟读取字段时抛出异常的映射。"""

    def get(self, key: str, default: Any = None) -> Any:
        """模拟不可靠工件读取。"""
        raise RuntimeError(f"读取失败: {key}")


def test_exception_during_metric_read_fails_closed_with_codes() -> None:
    """字段读取异常也必须返回可诊断的失败结果。"""
    result = evaluate_evidence_gate(_ExplodingMetrics(), now=_NOW)

    assert result.passed is False
    assert result.failed_codes
    assert all(not value for value in result.checks.values())


def test_invalid_now_type_fails_closed_without_exception() -> None:
    """当前时间类型异常时只失败新鲜度门禁，不向外泄漏异常。"""
    result = evaluate_evidence_gate(_full_metrics(), now=cast(datetime, object()))

    assert result.passed is False
    assert "FRESH_ARTIFACT" in result.failed_codes


def test_timestamp_older_than_window_fails_closed() -> None:
    """超过三十天的评估工件必须视为过期。"""
    metrics = _full_metrics()
    metrics["evaluated_at"] = (_NOW - timedelta(days=31)).isoformat()

    result = evaluate_evidence_gate(metrics, now=_NOW)

    assert result.passed is False
    assert "FRESH_ARTIFACT" in result.failed_codes
