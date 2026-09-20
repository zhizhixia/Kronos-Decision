"""报告 SQLite 存储核心的独立行为测试。"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from decision.storage import (
    CorruptReportError,
    REPORT_SCHEMA_VERSION,
    PublishResult,
    ReportStorage,
    UnsupportedSchemaError,
)


def _payload(marker: str) -> dict[str, str]:
    """构造不含凭据和持仓明文的最小报告负载。"""
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "research",
        "marker": marker,
    }


def _save(
    storage: ReportStorage,
    payload: dict[str, str],
    report_id: str | None = None,
) -> PublishResult:
    """用固定追溯引用保存测试报告。"""
    return storage.save_report(
        report_id=report_id,
        payload=payload,
        snapshot_id="snapshot-fixture",
        evaluation_id="evaluation-fixture",
        model_revision="model-fixture",
        config_hash="config-fixture",
        schema_version=REPORT_SCHEMA_VERSION,
    )


def test_content_key_and_duplicate_publish_are_stable(tmp_path: Path) -> None:
    """相同内容在重复发布和数据库重启后都只产生一个版本。"""
    database = tmp_path / "reports.sqlite3"
    storage = ReportStorage(database)
    payload = _payload("same-content")

    first = _save(storage, payload)
    reordered_payload = {
        "marker": "same-content",
        "status": "research",
        "schema_version": REPORT_SCHEMA_VERSION,
    }
    duplicate = _save(storage, reordered_payload)
    restarted_duplicate = _save(ReportStorage(database), reordered_payload)

    assert first.created
    assert duplicate.is_duplicate
    assert restarted_duplicate.is_duplicate
    assert first.report_id == duplicate.report_id == restarted_duplicate.report_id
    assert first.version == duplicate.version == restarted_duplicate.version == 1
    assert [item.version for item in storage.list_reports()] == [1]


def test_changed_payload_retains_old_version_and_latest_is_read_only(tmp_path: Path) -> None:
    """同一报告 ID 的新负载递增版本，旧版本仍可读取且不被覆盖。"""
    storage = ReportStorage(tmp_path / "reports.sqlite3")

    first = _save(storage, _payload("old"), report_id="report-immutable")
    second = _save(storage, _payload("new"), report_id="report-immutable")

    assert first.created and second.created
    assert first.version == 1
    assert second.version == 2
    assert storage.get_report("report-immutable", version=1).payload == _payload("old")
    assert storage.get_latest("report-immutable").payload == _payload("new")
    assert [item.version for item in storage.list_reports("report-immutable")] == [1, 2]


def test_corrupt_json_is_identifiable_without_hiding_old_versions(tmp_path: Path) -> None:
    """损坏最新 JSON 时读取和核验失败，但历史版本仍保持只读可读。"""
    database = tmp_path / "reports.sqlite3"
    storage = ReportStorage(database)
    _save(storage, _payload("old"), report_id="report-corrupt")
    _save(storage, _payload("new"), report_id="report-corrupt")

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE decision_report_payloads SET payload_json = ? "
            "WHERE report_id = ? AND version = ?",
            ("{not-json", "report-corrupt", 2),
        )

    with pytest.raises(CorruptReportError, match="JSON 负载损坏"):
        storage.get_latest("report-corrupt")
    assert storage.get_report("report-corrupt", version=1).payload == _payload("old")
    verification = storage.verify()
    assert not verification.ok
    assert verification.report_count == 2
    assert any("JSON 负载损坏" in error for error in verification.errors)


def test_unknown_report_schema_is_identifiable_and_does_not_replace_history(
    tmp_path: Path,
) -> None:
    """未知报告 schema 拒绝新发布，篡改后的历史版本也必须显式失败。"""
    database = tmp_path / "reports.sqlite3"
    storage = ReportStorage(database)
    _save(storage, _payload("old"), report_id="report-schema")
    _save(storage, _payload("new"), report_id="report-schema")

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE decision_report_metadata SET schema_version = ? "
            "WHERE report_id = ? AND version = ?",
            ("99.0", "report-schema", 2),
        )

    with pytest.raises(UnsupportedSchemaError, match="不支持的报告 schema 版本"):
        storage.get_latest("report-schema")
    assert storage.get_report("report-schema", version=1).payload == _payload("old")
    assert not storage.verify().ok

    with pytest.raises(UnsupportedSchemaError, match="不支持的报告 schema 版本"):
        storage.save_report(
            report_id="report-schema",
            payload=_payload("never-written"),
            snapshot_id="snapshot-fixture",
            evaluation_id="evaluation-fixture",
            model_revision="model-fixture",
            config_hash="config-fixture",
            schema_version="99.0",
        )
    with sqlite3.connect(database) as connection:
        versions = connection.execute(
            "SELECT version FROM decision_report_metadata "
            "WHERE report_id = ? ORDER BY version",
            ("report-schema",),
        ).fetchall()
    assert versions == [(1,), (2,)]


def test_restart_reads_metadata_and_payload_from_real_sqlite(tmp_path: Path) -> None:
    """关闭并重新打开真实 SQLite 文件后，报告和追溯引用保持一致。"""
    database = tmp_path / "reports.sqlite3"
    original = ReportStorage(database)
    saved = _save(original, _payload("restart"), report_id="report-restart")

    reopened = ReportStorage(database)
    loaded = reopened.get_report("report-restart")

    assert loaded is not None
    assert loaded == saved.record
    assert loaded.snapshot_id == "snapshot-fixture"
    assert loaded.evaluation_id == "evaluation-fixture"
    assert loaded.model_revision == "model-fixture"
    assert loaded.config_hash == "config-fixture"
    assert reopened.storage_schema_version == 1
    assert reopened.verify().ok


def test_backup_creates_verifiable_temporary_sqlite_copy(tmp_path: Path) -> None:
    """备份副本可被独立打开、核验并读取，不依赖原连接生命周期。"""
    database = tmp_path / "reports.sqlite3"
    backup_path = tmp_path / "reports.backup.sqlite3"
    storage = ReportStorage(database)
    _save(storage, _payload("backup"), report_id="report-backup")

    returned_path = storage.backup(backup_path)
    backup_storage = ReportStorage(returned_path)
    loaded = backup_storage.get_latest("report-backup")

    assert returned_path == backup_path
    assert backup_path.exists()
    assert backup_storage.verify().ok
    assert loaded is not None
    assert loaded.payload == _payload("backup")
    assert storage.verify().ok


def test_invalid_json_input_leaves_database_without_a_partial_publish(
    tmp_path: Path,
) -> None:
    """严格 JSON 序列化失败发生在事务前，不留下半条报告。"""
    storage = ReportStorage(tmp_path / "reports.sqlite3")

    with pytest.raises(ValueError, match="不可严格序列化"):
        _save(storage, {"schema_version": REPORT_SCHEMA_VERSION, "value": float("nan")})

    assert storage.list_reports() == []
    assert storage.verify().ok
