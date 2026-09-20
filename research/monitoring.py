"""影子样本的受控到期结算与暂停信号。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from research.shadow_ledger import ShadowDecision, ShadowLedger, ShadowLedgerError


@dataclass(frozen=True)
class SettlementSummary:
    """一次结算尝试的逐周期结果。"""

    decision_id: str
    outcomes: tuple[dict[str, object], ...]
    mature_count: int
    pending_count: int
    failed_count: int


def settle_decision(
    ledger: ShadowLedger,
    decision: ShadowDecision,
    prices: pd.DataFrame,
    *,
    current_as_of: str,
    horizons: tuple[int, ...] = (5, 10, 20),
) -> SettlementSummary:
    """只结算已成熟的交易日周期，缺价和未成熟样本不记零。"""
    frame = _prepare_prices(prices)
    current = pd.Timestamp(current_as_of)
    decision_date = pd.Timestamp(decision.decision_as_of)
    visible = frame.loc[frame["date"] <= current]
    code_prices = visible.loc[visible["code"] == decision.stock_code].sort_values("date")
    outcomes: list[dict[str, object]] = []
    mature = pending = failed = 0
    for horizon in horizons:
        dates = code_prices.loc[code_prices["date"] >= decision_date, "date"].drop_duplicates().tolist()
        if not dates or dates[0] > decision_date:
            status = "FAILED"
            failed += 1
            outcomes.append(ledger.record_outcome(decision.decision_id, horizon=horizon, matured_as_of=current_as_of, realized_return=None, status=status, settled_at=current_as_of))
            continue
        maturity_index = horizon
        if len(dates) <= maturity_index:
            pending += 1
            outcomes.append(ledger.record_outcome(decision.decision_id, horizon=horizon, matured_as_of=current_as_of, realized_return=None, status="PENDING", settled_at=current_as_of))
            continue
        maturity_date = dates[maturity_index]
        if maturity_date > current:
            pending += 1
            outcomes.append(ledger.record_outcome(decision.decision_id, horizon=horizon, matured_as_of=current_as_of, realized_return=None, status="PENDING", settled_at=current_as_of))
            continue
        entry = code_prices.loc[code_prices["date"] == dates[0], "close"]
        exit_value = code_prices.loc[code_prices["date"] == maturity_date, "close"]
        if entry.empty or exit_value.empty or float(entry.iloc[0]) <= 0 or float(exit_value.iloc[0]) <= 0:
            failed += 1
            outcomes.append(ledger.record_outcome(decision.decision_id, horizon=horizon, matured_as_of=current_as_of, realized_return=None, status="FAILED", settled_at=current_as_of))
            continue
        realized = float(exit_value.iloc[0] / entry.iloc[0] - 1.0)
        mature += 1
        outcomes.append(ledger.record_outcome(decision.decision_id, horizon=horizon, matured_as_of=str(maturity_date.date()), realized_return=realized, status="SETTLED", settled_at=current_as_of))
    return SettlementSummary(decision.decision_id, tuple(outcomes), mature, pending, failed)


def _prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """校验结算只使用日期、证券和收盘价。"""
    required = {"date", "code", "close"}
    if not isinstance(prices, pd.DataFrame) or not required.issubset(prices.columns):
        raise ValueError("结算价格缺少 date/code/close")
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    if frame["date"].isna().any() or frame["close"].isna().any():
        raise ValueError("结算价格含无效值")
    return frame


__all__ = ["SettlementSummary", "settle_decision"]
