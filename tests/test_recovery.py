"""运行时备份、损坏检测和恢复边界。"""
from __future__ import annotations

import json

import pytest

from decision.recovery import RecoveryError, RuntimeRecovery


def test_backup_verifies_and_restores_all_declared_artifacts(tmp_path) -> None:
    source_file = tmp_path / "source" / "reports.sqlite"
    source_dir = tmp_path / "source" / "snapshots"
    source_file.parent.mkdir(parents=True)
    source_dir.mkdir()
    source_file.write_text("sqlite-placeholder", encoding="utf-8")
    (source_dir / "manifest.json").write_text("{}", encoding="utf-8")
    backup = tmp_path / "backup"
    manager = RuntimeRecovery({"reports": source_file, "snapshots": source_dir})
    manifest = manager.create_backup(backup, created_at="2024-01-02T16:00:00+08:00")
    assert manifest.missing == ()
    manager.verify_backup(backup)
    target_file = tmp_path / "restored" / "reports.sqlite"
    target_dir = tmp_path / "restored" / "snapshots"
    RuntimeRecovery({"reports": target_file, "snapshots": target_dir}).restore(backup)
    assert target_file.read_text(encoding="utf-8") == "sqlite-placeholder"


def test_backup_does_not_silently_accept_corruption_or_overwrite(tmp_path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("original", encoding="utf-8")
    backup = tmp_path / "backup"
    RuntimeRecovery({"source": source}).create_backup(backup, created_at="2024-01-02")
    (backup / "source").write_text("changed", encoding="utf-8")
    with pytest.raises(RecoveryError, match="校验失败"):
        RuntimeRecovery({"source": tmp_path / "target"}).verify_backup(backup)
