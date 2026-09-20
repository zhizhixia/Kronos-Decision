"""影子账本追加、幂等和未成熟结算测试。"""
from __future__ import annotations

import pytest

from research.shadow_ledger import ShadowDecision, ShadowLedger, ShadowLedgerError


def _decision() -> ShadowDecision:
    return ShadowDecision(
        decision_id="d1",
        stock_code="600519",
        decision_as_of="2024-01-02",
        snapshot_id="s1",
        report_id="r1",
        strategy_id="baseline",
        strategy_version="v1",
        protocol_version="p1",
        action="INSUFFICIENT_EVIDENCE",
        action_permission="NONE",
        selected=False,
        status="EXCLUDED",
        reason="无正式资格",
        created_at="2024-01-02T16:00:00+08:00",
    )


def test_shadow_decision_is_idempotent_and_keeps_excluded_records(tmp_path) -> None:
    ledger = ShadowLedger(tmp_path / "shadow.json")
    first = ledger.append(_decision())
    second = ledger.append(_decision())
    assert first == second
    assert [item.decision_id for item in ledger.list_decisions()] == ["d1"]
    with pytest.raises(ShadowLedgerError, match="不可覆盖"):
        payload = _decision().to_dict()
        payload.pop("content_hash")
        payload["reason"] = "被篡改"
        ledger.append(ShadowDecision(**payload))


def test_outcome_requires_maturity_and_does_not_turn_missing_into_zero(tmp_path) -> None:
    ledger = ShadowLedger(tmp_path / "shadow.json")
    ledger.append(_decision())
    with pytest.raises(ShadowLedgerError, match="不能早于"):
        ledger.record_outcome(
            "d1", horizon=5, matured_as_of="2024-01-02", realized_return=None,
            status="PENDING", settled_at="2024-01-02T16:00:00+08:00"
        )
    pending = ledger.record_outcome(
        "d1", horizon=5, matured_as_of="2024-01-10", realized_return=None,
        status="PENDING", settled_at="2024-01-10T16:00:00+08:00"
    )
    assert pending["realized_return"] is None
    with pytest.raises(ShadowLedgerError, match="不能用零"):
        ledger.record_outcome(
            "d1", horizon=5, matured_as_of="2024-01-10", realized_return=None,
            status="SETTLED", settled_at="2024-01-10T16:00:00+08:00"
        )


def test_shadow_ledger_rejects_action_permission() -> None:
    with pytest.raises(ShadowLedgerError, match="不得保存"):
        ShadowDecision(**{**_decision().to_dict(), "action_permission": "CONDITIONAL_REFERENCE"})
