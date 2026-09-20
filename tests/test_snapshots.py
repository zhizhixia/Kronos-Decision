"""不可变 CSV 快照存储的真实临时目录测试。"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pandas as pd
import pytest

import data.snapshots as snapshots
from data.snapshots import (
    SnapshotCorruptionError,
    SnapshotStore,
    SnapshotWriteError,
    UnsupportedSnapshotVersionError,
)


def _frame(value: float = 1.0) -> pd.DataFrame:
    """返回不依赖外部数据源的普通表格样例。"""
    return pd.DataFrame(
        {
            "field": ["alpha", "beta"],
            "value": [value, value + 1.0],
            "state": ["ready", "ready"],
        }
    )


def test_save_read_manifest_and_duplicate_reuse(tmp_path: Path) -> None:
    """相同字段和行稳定复用 ID，并能从真实文件读回。"""
    store = SnapshotStore(tmp_path)
    frame = _frame()

    first_id = store.save(
        frame,
        source="offline-fixture",
        as_of="2026-09-18",
        quality={"complete": True, "flags": []},
    )
    second_id = store.save(
        frame,
        source="offline-fixture",
        as_of="2026-09-18",
        quality={"complete": True, "flags": []},
    )

    assert first_id == second_id
    manifest = store.read_manifest(first_id)
    assert manifest.snapshot_id == first_id
    assert manifest.source == "offline-fixture"
    assert manifest.as_of == "2026-09-18"
    assert manifest.quality == {"complete": True, "flags": []}
    assert manifest.columns == ("field", "value", "state")
    assert manifest.row_count == len(frame)
    expected_content_hash = hashlib.sha256(
        frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()
    assert manifest.content_hash == expected_content_hash
    assert (tmp_path / first_id / "data.csv").is_file()
    assert (tmp_path / first_id / "manifest.json").is_file()
    pd.testing.assert_frame_equal(store.read(first_id), frame)


def test_same_content_with_different_provenance_gets_new_snapshot_id(
    tmp_path: Path,
) -> None:
    """来源、时点或质量变化不能复用旧快照版本。"""
    store = SnapshotStore(tmp_path)
    frame = _frame()
    first_id = store.save(
        frame,
        source="offline-fixture",
        as_of="2026-09-18",
        quality={"complete": True},
    )
    second_id = store.save(
        frame,
        source="another-source",
        as_of="2026-09-19",
        quality={"complete": False},
    )

    assert second_id != first_id
    assert store.read_manifest(first_id).source == "offline-fixture"
    assert store.read_manifest(second_id).source == "another-source"


def test_revision_gets_new_id_and_keeps_old_snapshot(tmp_path: Path) -> None:
    """内容修订获得新 ID，旧 CSV 不被覆盖。"""
    store = SnapshotStore(tmp_path)
    original = _frame(1.0)
    revised = _frame(9.0)

    original_id = store.save(original, source="offline-fixture", as_of="2026-09-18")
    revised_id = store.save(revised, source="offline-fixture", as_of="2026-09-19")

    assert revised_id != original_id
    pd.testing.assert_frame_equal(store.read(original_id), original)
    pd.testing.assert_frame_equal(store.read(revised_id), revised)
    assert (tmp_path / original_id / "manifest.json").is_file()
    assert (tmp_path / revised_id / "manifest.json").is_file()


def test_corrupt_csv_fails_closed(tmp_path: Path) -> None:
    """CSV 被真实修改后必须在读取前拒绝。"""
    store = SnapshotStore(tmp_path)
    snapshot_id = store.save(_frame(), source="offline-fixture", as_of="2026-09-18")
    csv_path = tmp_path / snapshot_id / "data.csv"
    csv_path.write_bytes(csv_path.read_bytes() + b"tampered\n")

    with pytest.raises(SnapshotCorruptionError, match="大小|哈希|解析"):
        store.read(snapshot_id)


def test_unsupported_manifest_version_is_explicit(tmp_path: Path) -> None:
    """清单版本改变时抛出明确的中文版本异常。"""
    store = SnapshotStore(tmp_path)
    snapshot_id = store.save(_frame(), source="offline-fixture", as_of="2026-09-18")
    manifest_path = tmp_path / snapshot_id / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["schema_version"] = 999
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(UnsupportedSnapshotVersionError, match="不支持的快照清单版本"):
        store.read(snapshot_id)


def test_reference_protection_and_root_scoped_cleanup(tmp_path: Path) -> None:
    """清理删除未引用快照，保留已引用快照且不触碰根目录外文件。"""
    store = SnapshotStore(tmp_path / "snapshots")
    protected_id = store.save(_frame(1.0), source="offline-fixture")
    removable_id = store.save(_frame(2.0), source="offline-fixture")
    outside = tmp_path / "outside.txt"
    outside.write_text("不得删除", encoding="utf-8")

    store.mark_referenced(protected_id, "report-001")
    assert store.is_referenced(protected_id)
    assert not store.is_referenced(removable_id)

    deleted = store.cleanup()

    assert deleted == (removable_id,)
    assert (store.root / protected_id).is_dir()
    assert not (store.root / removable_id).exists()
    assert outside.read_text(encoding="utf-8") == "不得删除"
    pd.testing.assert_frame_equal(store.read(protected_id), _frame(1.0))


def test_interrupted_atomic_write_keeps_existing_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """原子发布中断时不留下新快照，也不损坏旧快照。"""
    store = SnapshotStore(tmp_path)
    original_id = store.save(_frame(1.0), source="offline-fixture")

    def fail_replace(source: object, destination: object) -> None:
        """模拟真实原子替换调用失败。"""
        raise OSError("模拟写入中断")

    monkeypatch.setattr(snapshots.os, "replace", fail_replace)
    with pytest.raises(SnapshotWriteError, match="原子写入|写入快照"):
        store.save(_frame(3.0), source="offline-fixture")

    pd.testing.assert_frame_equal(store.read(original_id), _frame(1.0))
    assert len([child for child in store.root.iterdir() if child.is_dir()]) == 1
