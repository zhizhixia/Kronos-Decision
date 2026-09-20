"""SA35 运行时恢复验收入口。"""
from __future__ import annotations

from decision.recovery import RuntimeRecovery


def test_runtime_recovery_clean_directory_roundtrip(tmp_path) -> None:
    source = tmp_path / "source.sqlite"
    source.write_text("脱敏 sqlite", encoding="utf-8")
    backup = tmp_path / "backup"
    RuntimeRecovery({"reports": source}).create_backup(backup, created_at="2024-01-02T16:00:00+08:00")
    target = tmp_path / "restored.sqlite"
    restored = RuntimeRecovery({"reports": target}).restore(backup)
    assert restored.files["reports"]
    assert target.read_text(encoding="utf-8") == "脱敏 sqlite"
