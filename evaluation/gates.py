"""正式投资建议的硬证据门禁。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any


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
    """按锁定阈值评估正式研究，不对缺失指标宽松放行。"""
    now = now or datetime.now()
    checks = {
        "DATA_COMPLETE": bool(metrics.get("data_complete")),
        "NO_QUALITY_ERROR": not bool(metrics.get("quality_error")),
        "INTERVAL_COVERAGE": 0.75 <= float(metrics.get("coverage_80", -1)) <= 0.85,
        "PROBABILITY_ECE": float(metrics.get("up_probability_ece", 1)) <= 0.08,
        "RANK_IC": float(metrics.get("rank_ic_mean", 0)) > 0,
        "BOOTSTRAP_RANK_IC": float(metrics.get("rank_ic_positive_probability", 0)) >= 0.80,
        "NET_EXCESS_RETURN": float(metrics.get("annualized_excess_return", 0)) > 0,
        "INFORMATION_RATIO": float(metrics.get("information_ratio", 0)) >= 0.5,
        "WINDOW_STABILITY": float(metrics.get("positive_12m_window_ratio", 0)) >= 0.60,
        "DRAWDOWN": float(metrics.get("drawdown_worsening", 1)) <= 0.05,
        "VERSION_MATCH": bool(metrics.get("versions_match")),
        "FORMAL_PROTOCOL": bool(metrics.get("formal_protocol")),
        "FRESH_ARTIFACT": _is_fresh(metrics.get("evaluated_at"), now),
    }
    failed = tuple(code for code, passed in checks.items() if not passed)
    return EvidenceGateResult(not failed, failed, checks)


def _is_fresh(value: Any, now: datetime) -> bool:
    if not value:
        return False
    try:
        return now - datetime.fromisoformat(str(value)) <= timedelta(days=30)
    except ValueError:
        return False
