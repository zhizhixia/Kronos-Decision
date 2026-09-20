"""持仓快照、预览确认和风险约束的反例测试。"""
from __future__ import annotations

import json

import pytest

from decision.portfolio import (
    PortfolioCorruptionError,
    PortfolioError,
    PortfolioLedger,
    PortfolioSnapshot,
    Position,
    RiskConstraints,
)


def _csv() -> str:
    return "code,quantity,available_quantity,avg_cost,as_of,source,name\n600519,1000,900,100.5,2024-01-02,manual,贵州茅台\n"


def test_preview_does_not_write_until_explicit_confirmation(tmp_path) -> None:
    ledger = PortfolioLedger(tmp_path / "portfolio.json")
    preview = ledger.preview_csv(_csv())
    assert preview.valid is True
    assert ledger.latest() is None
    snapshot = ledger.confirm_preview(
        preview,
        cash=10000.0,
        as_of="2024-01-02",
        available_at="2024-01-02T16:00:00+08:00",
        source="manual-csv",
        confirmed=True,
    )
    assert ledger.latest() == snapshot
    assert ledger.sellable_quantity(snapshot, "600519", "2024-01-03") == 900


def test_invalid_preview_and_unconfirmed_import_cannot_pollute_old_snapshot(tmp_path) -> None:
    ledger = PortfolioLedger(tmp_path / "portfolio.json")
    preview = ledger.preview_csv("code,quantity\n600519,100\n")
    assert not preview.valid
    with pytest.raises(PortfolioError):
        ledger.confirm_preview(
            preview,
            cash=1000,
            as_of="2024-01-02",
            available_at="2024-01-02T16:00:00+08:00",
            source="manual",
            confirmed=True,
        )
    with pytest.raises(PortfolioError, match="未确认"):
        ledger.confirm_preview(
            PortfolioLedger(tmp_path / "other.json").preview_csv(_csv()),
            cash=1000,
            as_of="2024-01-02",
            available_at="2024-01-02T16:00:00+08:00",
            source="manual",
            confirmed=False,
        )
    assert ledger.latest() is None


def test_missing_cost_and_price_do_not_become_zero_or_pass_risk(tmp_path) -> None:
    position = Position("000001", 100, 0, None, "2024-01-02", "manual")
    from decision.portfolio import PortfolioSnapshot

    snapshot = PortfolioSnapshot(
        "p1", "2024-01-02", "2024-01-02T16:00:00+08:00", 1000, (position,), "manual"
    )
    ledger = PortfolioLedger(tmp_path / "portfolio-test.json")
    errors = ledger.validate_risk(snapshot, {}, RiskConstraints(max_position_weight=0.5))
    assert "缺少价格：000001" in errors


def test_corrupted_hash_is_rejected_on_restart(tmp_path) -> None:
    path = tmp_path / "portfolio.json"
    ledger = PortfolioLedger(path)
    snapshot = ledger.confirm_preview(
        ledger.preview_csv(_csv()),
        cash=1000,
        as_of="2024-01-02",
        available_at="2024-01-02T16:00:00+08:00",
        source="manual",
        confirmed=True,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["snapshots"][snapshot.snapshot_id]["cash"] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PortfolioCorruptionError):
        PortfolioLedger(path).get(snapshot.snapshot_id)


def test_stale_ledger_instance_cannot_drop_another_snapshot(tmp_path) -> None:
    """多个账本实例保存时检测版本冲突，不能整文件覆盖已有快照。"""
    path = tmp_path / "portfolio.json"
    first = PortfolioLedger(path)
    second = PortfolioLedger(path)
    position = Position("600519", 100, 100, 100.0, "2024-01-02", "test")
    first_snapshot = PortfolioSnapshot(
        "first", "2024-01-02", "2024-01-02T16:00:00+08:00", 1000.0, (position,), "test"
    )
    second_snapshot = PortfolioSnapshot(
        "second", "2024-01-03", "2024-01-03T16:00:00+08:00", 900.0, (position,), "test"
    )

    first.save_snapshot(first_snapshot)
    with pytest.raises(PortfolioError, match="其他实例修改"):
        second.save_snapshot(second_snapshot)

    second.save_snapshot(second_snapshot)
    reopened = PortfolioLedger(path)
    assert reopened.get("first") == first_snapshot
    assert reopened.get("second") == second_snapshot
