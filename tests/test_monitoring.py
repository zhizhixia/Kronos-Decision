"""影子结算的未成熟、成熟和缺价路径。"""
from __future__ import annotations

import pandas as pd

from research.monitoring import settle_decision
from research.shadow_ledger import ShadowDecision, ShadowLedger


def _decision() -> ShadowDecision:
    return ShadowDecision(
        decision_id="d-monitor",
        stock_code="600519",
        decision_as_of="2024-01-02",
        snapshot_id="s1",
        report_id="r1",
        strategy_id="baseline",
        strategy_version="1",
        protocol_version="p1",
        action="INSUFFICIENT_EVIDENCE",
        action_permission="NONE",
        selected=False,
        status="EXCLUDED",
        reason="仅研究",
        created_at="2024-01-02T16:00:00+08:00",
    )


def test_monitor_leaves_unmatured_result_pending_then_settles(tmp_path) -> None:
    ledger = ShadowLedger(tmp_path / "ledger.json")
    decision = ledger.append(_decision())
    dates = pd.bdate_range("2024-01-02", periods=22)
    prices = pd.DataFrame({"date": dates, "code": "600519", "close": range(100, 122)})

    pending = settle_decision(ledger, decision, prices, current_as_of="2024-01-03")
    assert pending.pending_count == 3
    settled = settle_decision(ledger, decision, prices, current_as_of="2024-02-01")
    assert settled.mature_count == 3
    assert ledger.outcomes("d-monitor")[0]["status"] == "SETTLED"
    assert ledger.outcomes("d-monitor")[0]["realized_return"] > 0
    settled_again = settle_decision(ledger, decision, prices, current_as_of="2024-02-02")
    assert settled_again.mature_count == 3
    assert ledger.outcomes("d-monitor")[0]["settled_at"] == "2024-02-01"
