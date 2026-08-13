"""股票交易资格规则：ST/退市、上市时长、长期停牌与流动性门槛。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

# 计划锁定的固定门槛：上市不足 252 个交易日、日均成交额低于 2000 万元、
# 最新成交日距锚点超过 14 个自然日（约 10 个交易日）视为停牌
MIN_LISTED_SESSIONS = 252
MIN_AVG_DAILY_AMOUNT = 20_000_000.0
MAX_SUSPENSION_DAYS = 14


@dataclass(frozen=True)
class EligibilityResult:
    """交易资格结论；不合格股票不能获得增持。"""

    eligible: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"eligible": self.eligible, "reasons": list(self.reasons)}


def evaluate_stock_eligibility(
    bars: pd.DataFrame | None,
    stock_name: str = "",
    as_of: pd.Timestamp | None = None,
) -> EligibilityResult:
    """按固定规则评估交易资格；任何一条不满足都返回不可增持。"""
    reasons: list[str] = []
    name = (stock_name or "").upper()
    if "ST" in name or "退" in name:
        reasons.append("ST_OR_DELISTING")
    if bars is None or len(bars) < MIN_LISTED_SESSIONS:
        reasons.append("LISTED_LESS_THAN_252_SESSIONS")
    elif as_of is not None and pd.Timestamp(as_of) - pd.Timestamp(bars["date"].iloc[-1]) > pd.Timedelta(days=MAX_SUSPENSION_DAYS):
        reasons.append("LONG_SUSPENSION")
    if bars is not None and not bars.empty and "amount" in bars:
        mean_amount = float(pd.to_numeric(bars["amount"], errors="coerce").tail(20).mean())
        if mean_amount > 0 and mean_amount < MIN_AVG_DAILY_AMOUNT:
            reasons.append("INSUFFICIENT_LIQUIDITY")
    return EligibilityResult(not reasons, tuple(reasons))
