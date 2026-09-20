"""不依赖模型的低成本趋势基线。

该基线只描述截至 ``as_of`` 已知的历史事实，不把观察到的过去收益
包装成可执行动作。它要求输入带有不可变 ``snapshot_id``，便于同一
份数据重放和审计。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from data.contracts import MarketDataBundle

BASELINE_MODEL_REVISION = "baseline-momentum-v1"
BASELINE_HORIZONS = (5, 10, 20)


class BaselineDataError(ValueError):
    """基线输入无法形成可重放事实分析。"""


@dataclass(frozen=True)
class BaselineResult:
    """低成本基线的只读事实结果。"""

    report: dict[str, Any]


def run_momentum_baseline(
    bundle: MarketDataBundle,
    stock_code: str,
    stock_name: str = "",
) -> BaselineResult:
    """只使用快照截止日前数据计算 5/10/20 日历史动量事实。"""
    if not isinstance(bundle.snapshot_id, str) or not bundle.snapshot_id:
        raise BaselineDataError("基线输入缺少 snapshot_id。")
    bars = _prepare_bars(bundle)
    visible = bars.loc[bars["date"] <= pd.Timestamp(bundle.as_of).normalize()].copy()
    if visible.empty:
        raise BaselineDataError("基线输入在 as_of 前没有可用日线。")
    horizons: dict[str, dict[str, Any]] = {}
    supporting: list[dict[str, Any]] = []
    opposing: list[dict[str, Any]] = []
    for horizon in BASELINE_HORIZONS:
        item = _horizon_fact(visible, horizon)
        horizons[str(horizon)] = item
        if item["status"] != "AVAILABLE":
            continue
        target = supporting if item["direction"] == "UP" else opposing
        if item["direction"] in {"UP", "DOWN"}:
            target.append(
                {
                    "horizon": horizon,
                    "direction": item["direction"],
                    "observed_return": item["observed_return"],
                    "anchor_date": item["anchor_date"],
                    "reference_date": item["reference_date"],
                }
            )
    available = sum(item["status"] == "AVAILABLE" for item in horizons.values())
    report = {
        "schema_version": "2.0",
        "status": "ok" if available else "degraded",
        "run_status": "SUCCEEDED" if available else "FAILED",
        "evidence_status": "RESEARCH_ONLY",
        "action_permission": "NONE",
        "stock": {"code": stock_code, "name": stock_name},
        "data_provenance": {
            "source": bundle.source,
            "snapshot_id": bundle.snapshot_id,
            "as_of": pd.Timestamp(bundle.as_of).normalize().isoformat(),
            "max_input_date": visible["date"].max().isoformat(),
            "used_rows": int(len(visible)),
            "future_rows_excluded": int(len(bars) - len(visible)),
            "content_hash": bundle.content_hash,
            "universe_version": bundle.universe_version,
            "quality_flags": list(bundle.quality_flags),
            "is_stale": bundle.is_stale,
        },
        "model_provenance": {
            "model_revision": BASELINE_MODEL_REVISION,
            "config_hash": "not_applicable",
            "model_type": "deterministic_trailing_return",
        },
        "horizons": horizons,
        "baseline_facts": {
            "supporting": supporting,
            "opposing": opposing,
            "method": "anchor_close / close_(h+1) - 1",
        },
        "recommendation": {
            "action": None,
            "legacy_signal": None,
            "reason_codes": ["BASELINE_NOT_ACTIONABLE"],
        },
        "evidence_gate": {
            "passed": False,
            "failed_codes": ["BASELINE_NOT_ACTIONABLE"],
            "checks": {"MODEL_CALIBRATION": False},
        },
    }
    return BaselineResult(report=report)


def _prepare_bars(bundle: MarketDataBundle) -> pd.DataFrame:
    """校验并按日期排序快照中的必要列。"""
    required = {"date", "close"}
    if not required.issubset(bundle.bars.columns):
        raise BaselineDataError("基线输入缺少 date 或 close。")
    bars = bundle.bars[["date", "close"]].copy()
    bars["date"] = pd.to_datetime(bars["date"], errors="coerce").dt.normalize()
    bars["close"] = pd.to_numeric(bars["close"], errors="coerce")
    bars = bars.dropna(subset=["date", "close"]).sort_values("date")
    if bars.empty or not np.isfinite(bars["close"].to_numpy()).all() or (bars["close"] <= 0).any():
        raise BaselineDataError("基线输入包含空值、非有限值或非正收盘价。")
    if bars["date"].duplicated().any():
        raise BaselineDataError("基线输入存在重复交易日。")
    return bars.reset_index(drop=True)


def _horizon_fact(bars: pd.DataFrame, horizon: int) -> dict[str, Any]:
    """计算单个 horizon 的历史事实，不访问未来行。"""
    if len(bars) <= horizon:
        return {
            "status": "UNAVAILABLE",
            "reason_code": "INSUFFICIENT_HISTORY",
            "observed_return": None,
            "direction": None,
            "anchor_date": bars["date"].iloc[-1].isoformat(),
            "reference_date": None,
        }
    anchor = bars.iloc[-1]
    reference = bars.iloc[-(horizon + 1)]
    observed_return = float(anchor["close"] / reference["close"] - 1.0)
    direction = "UP" if observed_return > 0 else "DOWN" if observed_return < 0 else "FLAT"
    return {
        "status": "AVAILABLE",
        "reason_code": None,
        "observed_return": observed_return,
        "direction": direction,
        "anchor_date": anchor["date"].isoformat(),
        "reference_date": reference["date"].isoformat(),
    }


__all__ = [
    "BASELINE_HORIZONS",
    "BASELINE_MODEL_REVISION",
    "BaselineDataError",
    "BaselineResult",
    "run_momentum_baseline",
]
