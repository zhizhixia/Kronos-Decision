"""验证取数、不可变快照与离线恢复之间的真实连接。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from data.contracts import MarketDataBundle
from data.fetcher import DataFetcher
from data.snapshots import SnapshotCorruptionError, SnapshotStore


class FixedCalendar:
    """为离线测试提供确定的最后完整交易日。"""

    version = "fixed-test-v1"

    def __init__(self, complete: pd.Timestamp) -> None:
        self.complete = complete

    def latest_complete_session(self) -> tuple[pd.Timestamp, tuple[str, ...]]:
        """返回固定的完整交易日。"""
        return self.complete, ()


def _bars(close: float = 10.0) -> pd.DataFrame:
    """构造满足取数校验的本地日线样本。"""
    dates = pd.bdate_range("2024-01-02", periods=60)
    return pd.DataFrame(
        {
            "date": dates,
            "open": 10.0,
            "high": max(11.0, close),
            "low": 9.0,
            "close": close,
            "volume": 1000.0,
            "amount": 10000.0,
        }
    )


def _fetcher(tmp_path: Path) -> DataFetcher:
    """创建只使用测试临时目录的取数器。"""
    fetcher = DataFetcher()
    fetcher._cache_dir = tmp_path / "cache"
    fetcher._snapshot_dir = tmp_path / "snapshots"
    fetcher._cache_dir.mkdir(parents=True, exist_ok=True)
    fetcher._retry_max = 1
    fetcher._calendar = FixedCalendar(_bars()["date"].max())
    return fetcher


def _install_source(fetcher: DataFetcher, bars: pd.DataFrame, monkeypatch) -> None:
    """把一个离线数据源接入真实 fetch 调用路径。"""
    monkeypatch.setattr(fetcher, "_fetch_akshare", lambda code: bars.copy())
    monkeypatch.setattr(
        fetcher,
        "_fetch_baostock",
        lambda code: (_ for _ in ()).throw(AssertionError("不应调用备源")),
    )


def test_fetch_binds_snapshot_and_keeps_cache_separate(tmp_path: Path, monkeypatch) -> None:
    """实际 fetch 结果必须有快照 ID，缓存文件不能代替快照。"""
    fetcher = _fetcher(tmp_path)
    bars = _bars()
    _install_source(fetcher, bars, monkeypatch)

    bundle = fetcher.fetch_daily_bundle("600519")

    assert isinstance(bundle, MarketDataBundle)
    assert len(bundle.snapshot_id) == 64
    assert bundle.snapshot_id == bundle.snapshot_id.lower()
    assert (tmp_path / "cache" / "600519.csv").is_file()
    assert (tmp_path / "cache" / "600519.meta.json").is_file()
    assert (tmp_path / "snapshots" / bundle.snapshot_id / "data.csv").is_file()
    assert (tmp_path / "snapshots" / bundle.snapshot_id / "manifest.json").is_file()

    metadata = json.loads(
        (tmp_path / "cache" / "600519.meta.json").read_text(encoding="utf-8")
    )
    assert metadata["snapshot_id"] == bundle.snapshot_id
    manifest = SnapshotStore(tmp_path / "snapshots").read_manifest(bundle.snapshot_id)
    assert manifest.source == "akshare"
    assert manifest.as_of == bundle.as_of.isoformat()


def test_cache_refresh_creates_new_snapshot_without_mutating_old_one(
    tmp_path: Path, monkeypatch
) -> None:
    """缓存刷新后的修订必须生成新快照，旧快照仍可读。"""
    fetcher = _fetcher(tmp_path)
    first = _bars(10.0)
    _install_source(fetcher, first, monkeypatch)
    original = fetcher.fetch_daily_bundle("600519")

    revised = _bars(10.5)
    monkeypatch.setattr(fetcher, "_fetch_akshare", lambda code: revised.copy())
    fetcher._ttl_hours = -1
    refreshed = fetcher.fetch_daily_bundle("600519")

    assert refreshed.snapshot_id != original.snapshot_id
    store = SnapshotStore(tmp_path / "snapshots")
    old_bars = store.read(original.snapshot_id)
    old_bars["date"] = pd.to_datetime(old_bars["date"])
    pd.testing.assert_frame_equal(old_bars, original.bars)
    assert float(refreshed.bars["close"].iloc[-1]) == 10.5


def test_snapshot_bundle_can_be_read_offline_without_network(
    tmp_path: Path, monkeypatch
) -> None:
    """重建取数器后可只从快照恢复，不重新访问数据源。"""
    fetcher = _fetcher(tmp_path)
    _install_source(fetcher, _bars(), monkeypatch)
    published = fetcher.fetch_daily_bundle("600519")

    restored_fetcher = _fetcher(tmp_path)
    monkeypatch.setattr(
        restored_fetcher,
        "_source_candidates",
        lambda: (_ for _ in ()).throw(AssertionError("离线读快照不应访问数据源")),
    )
    restored = restored_fetcher.load_snapshot_bundle(published.snapshot_id)

    assert restored.snapshot_id == published.snapshot_id
    assert restored.source == published.source
    assert restored.as_of == published.as_of
    pd.testing.assert_frame_equal(restored.bars, published.bars)


def test_corrupted_bound_snapshot_is_rejected_during_offline_restore(
    tmp_path: Path, monkeypatch
) -> None:
    """已绑定快照被篡改时必须拒绝恢复，不能退回可变缓存。"""
    fetcher = _fetcher(tmp_path)
    _install_source(fetcher, _bars(), monkeypatch)
    published = fetcher.fetch_daily_bundle("600519")
    csv_path = tmp_path / "snapshots" / published.snapshot_id / "data.csv"
    csv_path.write_bytes(csv_path.read_bytes() + b"tampered\n")

    with pytest.raises(SnapshotCorruptionError):
        fetcher.load_snapshot_bundle(published.snapshot_id)
