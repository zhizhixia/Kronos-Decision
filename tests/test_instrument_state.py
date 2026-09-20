"""验证股票池和证券状态的时点边界。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.pool import get_current_universe_snapshot, get_universe_snapshot


_HISTORY_COLUMNS = [
    "code",
    "name",
    "effective_date",
    "available_at",
    "security_status",
    "listed_date",
    "delisted_date",
    "source",
    "source_revision",
    "corporate_action_version",
]


def _write_history(path: Path, rows: list[dict[str, object]]) -> None:
    """写入真实临时 CSV 历史档案。"""
    pd.DataFrame(rows, columns=_HISTORY_COLUMNS).to_csv(path, index=False)


def _row(
    code: str,
    name: str,
    *,
    effective_date: str = "2023-01-03",
    available_at: str = "2023-01-04",
    security_status: str = "NORMAL",
    listed_date: str = "2020-01-01",
    delisted_date: str = "",
    source: str = "approved-history.csv",
    source_revision: str = "rev-1",
) -> dict[str, object]:
    """构造一条完整的历史证券状态记录。"""
    return {
        "code": code,
        "name": name,
        "effective_date": effective_date,
        "available_at": available_at,
        "security_status": security_status,
        "listed_date": listed_date,
        "delisted_date": delisted_date,
        "source": source,
        "source_revision": source_revision,
        "corporate_action_version": "ca-1",
    }


def test_current_pool_without_status_is_explicitly_not_usable(tmp_path, monkeypatch) -> None:
    """旧版当前池只有名称时不能猜测为正常交易。"""
    import data.pool as pool_module

    cache_path = tmp_path / "pool.csv"
    metadata_path = tmp_path / "pool.meta.json"
    monkeypatch.setattr(pool_module, "_CACHE_PATH", cache_path)
    monkeypatch.setattr(pool_module, "_META_PATH", metadata_path)
    pool_module._save_cache({"600519": "贵州茅台"})

    snapshot = get_current_universe_snapshot("2024-01-05")

    assert snapshot.scope == "current"
    assert snapshot.history_available is False
    assert snapshot.usable is False
    assert snapshot.active_codes == ()
    assert "UNKNOWN_SECURITY_STATUS" in snapshot.reasons


def test_history_uses_only_records_visible_at_as_of(tmp_path) -> None:
    """历史查询不能使用生效日或公开日位于未来的记录。"""
    path = tmp_path / "pool_history.csv"
    _write_history(
        path,
        [
            _row("600519", "贵州茅台"),
            _row(
                "600519",
                "贵州茅台新版本",
                effective_date="2024-01-01",
                available_at="2024-02-01",
                source_revision="rev-2",
            ),
            _row("601318", "中国平安", effective_date="2025-01-01"),
        ],
    )

    snapshot = get_universe_snapshot("2024-01-31", history_path=path)

    assert snapshot.history_available is True
    assert snapshot.reasons == ()
    assert snapshot.usable is True
    assert snapshot.active_codes == ("600519",)
    assert snapshot.instruments[0].name == "贵州茅台"
    assert snapshot.instruments[0].source_revision == "rev-1"


def test_unknown_or_suspended_state_blocks_history_snapshot(tmp_path) -> None:
    """未知和停牌状态不能被股票池层自动放行。"""
    path = tmp_path / "pool_history.csv"
    _write_history(
        path,
        [
            _row("600519", "贵州茅台", security_status=""),
            _row("000001", "平安银行", security_status="SUSPENDED"),
        ],
    )

    snapshot = get_universe_snapshot("2024-01-31", history_path=path)

    assert snapshot.history_available is True
    assert snapshot.usable is False
    assert "UNKNOWN_SECURITY_STATUS" in snapshot.reasons
    assert "SECURITY_NOT_TRADABLE" in snapshot.reasons


def test_missing_history_does_not_fallback_to_current_pool(tmp_path) -> None:
    """缺少历史档案时必须明确失败，不能借用今天的股票池。"""
    snapshot = get_universe_snapshot(
        "2017-01-03", history_path=tmp_path / "does-not-exist.csv"
    )

    assert snapshot.history_available is False
    assert snapshot.instruments == ()
    assert snapshot.usable is False
    assert snapshot.reasons == ("HISTORY_SOURCE_UNAVAILABLE",)


def test_delisted_security_is_not_visible_after_delisted_date(tmp_path) -> None:
    """退市日期已到的证券不能继续出现在历史股票池。"""
    path = tmp_path / "pool_history.csv"
    _write_history(
        path,
        [
            _row(
                "600519",
                "贵州茅台",
                delisted_date="2024-01-15",
            )
        ],
    )

    snapshot = get_universe_snapshot("2024-02-01", history_path=path)

    assert snapshot.history_available is False
    assert snapshot.instruments == ()
    assert "HISTORY_NO_MEMBERSHIP_AT_AS_OF" in snapshot.reasons
