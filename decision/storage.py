"""不可变决策报告的 SQLite 存储核心。

SQLite 元数据表只负责存储报告 ID、版本、快照/评估/模型/配置引用、
报告 schema 版本、发布时间和发布指纹；负载表保存规范化后的 JSON 文本及其
哈希。两张表在同一个 SQLite 事务中提交，读取端会同时验证 JSON、哈希和
元数据指纹。该模块不负责外部 CSV、模型文件或其他目录的发布，因此 SQLite
事务成功不代表外部文件与报告系统已经整体成功。
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final


STORAGE_SCHEMA_VERSION: Final[int] = 1
REPORT_SCHEMA_VERSION: Final[str] = "2.0"
SUPPORTED_REPORT_SCHEMA_VERSIONS: Final[frozenset[str]] = frozenset({REPORT_SCHEMA_VERSION})

_META_TABLE: Final[str] = "decision_storage_meta"
_REPORT_TABLE: Final[str] = "decision_report_metadata"
_PAYLOAD_TABLE: Final[str] = "decision_report_payloads"

_META_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset({"key", "value"})
_REPORT_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "report_id",
        "version",
        "snapshot_id",
        "evaluation_id",
        "model_revision",
        "config_hash",
        "schema_version",
        "publication_key",
        "created_at",
        "publication_status",
    }
)
_PAYLOAD_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {"report_id", "version", "payload_json", "payload_sha256"}
)


class ReportStorageError(RuntimeError):
    """报告存储或读取失败。"""


class StorageSchemaError(ReportStorageError):
    """SQLite 元数据或表结构损坏。"""


class UnsupportedSchemaError(ReportStorageError):
    """报告或 SQLite 存储 schema 版本不受当前代码支持。"""


class CorruptReportError(ReportStorageError):
    """报告 JSON、哈希或关联元数据损坏。"""


# 为调用方提供更明确的兼容名称；异常语义与 UnsupportedSchemaError 相同。
StorageError = ReportStorageError
StorageCorruptionError = CorruptReportError
UnsupportedReportSchemaError = UnsupportedSchemaError
UnknownSchemaError = UnsupportedSchemaError
CorruptJSONError = CorruptReportError


@dataclass(frozen=True)
class ReportRecord:
    """一条只读报告版本及其可追溯引用。"""

    report_id: str
    version: int
    payload: Any
    snapshot_id: str
    evaluation_id: str
    model_revision: str
    config_hash: str
    schema_version: str
    payload_sha256: str
    created_at: str
    publication_status: str = "PUBLISHED"

    @property
    def content_hash(self) -> str:
        """返回规范化 JSON 负载的 SHA-256。"""
        return self.payload_sha256


@dataclass(frozen=True)
class PublishResult:
    """发布结果；created=False 表示发现完全相同的既有版本。"""

    record: ReportRecord
    created: bool

    @property
    def is_duplicate(self) -> bool:
        """返回本次发布是否为幂等重复发布。"""
        return not self.created

    @property
    def duplicate(self) -> bool:
        """返回本次发布是否为幂等重复发布。"""
        return self.is_duplicate

    @property
    def report_id(self) -> str:
        """转发已发布报告 ID，便于调用方直接使用结果。"""
        return self.record.report_id

    @property
    def version(self) -> int:
        """转发已发布版本号，便于调用方直接使用结果。"""
        return self.record.version

    @property
    def payload(self) -> Any:
        """转发已发布 JSON 负载，便于调用方直接使用结果。"""
        return self.record.payload


@dataclass(frozen=True)
class VerificationResult:
    """数据库验证结果，不把异常状态伪装成空报告。"""

    ok: bool
    report_count: int
    errors: tuple[str, ...]

    @property
    def checked_reports(self) -> int:
        """返回实际检查过的报告元数据行数。"""
        return self.report_count

    def __bool__(self) -> bool:
        """允许调用方用布尔上下文判断验证是否通过。"""
        return self.ok


class ReportStorage:
    """以 SQLite 保存不可变报告版本，并提供只读恢复核验接口。

    每次发布只会 INSERT 新版本，完全相同的发布指纹返回旧版本而不新增行；
    相同报告 ID 的不同内容会递增版本。报告 JSON 与其元数据在同一事务提交，
    但本类不替代快照文件、评估目录或模型文件的事务协调器。
    """

    def __init__(self, path: str | Path) -> None:
        """打开或创建 SQLite 文件，并初始化受支持的存储 schema。"""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @property
    def storage_schema_version(self) -> int:
        """返回当前代码支持的 SQLite 存储 schema 版本。"""
        return STORAGE_SCHEMA_VERSION

    def save_report(
        self,
        report_id: str | None = None,
        payload: Any = None,
        snapshot_id: str = "",
        evaluation_id: str = "",
        model_revision: str = "",
        config_hash: str = "",
        schema_version: str = REPORT_SCHEMA_VERSION,
        publication_status: str = "PUBLISHED",
    ) -> PublishResult:
        """原子发布一份报告，重复发布返回既有版本而不覆盖历史。

        ``report_id`` 为 ``None`` 时使用规范化 JSON 负载的 SHA-256 生成稳定
        ID。相同报告 ID 的不同负载会保留为下一个版本；完全相同的负载和全部
        追溯字段则视为幂等重复发布。JSON 负载必须可被严格序列化，不能含
        ``NaN``、凭据或调用方不应持久化的敏感数据。
        """
        canonical_json = _canonical_json(payload)
        payload_sha256 = _sha256(canonical_json)
        resolved_report_id = (
            _require_text(report_id, "report_id")
            if report_id is not None
            else f"content-{payload_sha256}"
        )
        resolved_snapshot_id = _require_text(snapshot_id, "snapshot_id")
        resolved_evaluation_id = _require_text(evaluation_id, "evaluation_id")
        resolved_model_revision = _require_text(model_revision, "model_revision")
        resolved_config_hash = _require_text(config_hash, "config_hash")
        resolved_schema_version = _validate_report_schema_version(schema_version)
        resolved_publication_status = _validate_publication_status(publication_status)
        try:
            normalized_payload = json.loads(canonical_json)
        except json.JSONDecodeError as exc:
            raise ValueError("报告 JSON 负载不可读取。") from exc
        _validate_embedded_schema_version(normalized_payload, resolved_schema_version)
        publication_key = _publication_key(
            resolved_report_id,
            payload_sha256,
            resolved_snapshot_id,
            resolved_evaluation_id,
            resolved_model_revision,
            resolved_config_hash,
            resolved_schema_version,
        )
        created_at = _now_utc()

        self._require_database()
        conn = self._connect(self.path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            history_rows = conn.execute(
                _SELECT_REPORT_SQL
                + " WHERE m.report_id = ? ORDER BY m.version ASC",
                (resolved_report_id,),
            ).fetchall()
            # 先验证同一报告的全部历史，避免新版本把损坏或未知记录掩盖掉。
            for history_row in history_rows:
                _row_to_record(history_row)
            duplicate_row = conn.execute(
                _SELECT_REPORT_SQL + " WHERE m.report_id = ? AND m.publication_key = ?",
                (resolved_report_id, publication_key),
            ).fetchone()
            if duplicate_row is not None:
                record = _row_to_record(duplicate_row)
                conn.commit()
                return PublishResult(record=record, created=False)

            next_version_row = conn.execute(
                f"SELECT COALESCE(MAX(version), 0) + 1 FROM {_REPORT_TABLE} WHERE report_id = ?",
                (resolved_report_id,),
            ).fetchone()
            next_version = int(next_version_row[0])
            conn.execute(
                f"""
                INSERT INTO {_REPORT_TABLE} (
                    report_id, version, snapshot_id, evaluation_id, model_revision,
                    config_hash, schema_version, publication_key, created_at, publication_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved_report_id,
                    next_version,
                    resolved_snapshot_id,
                    resolved_evaluation_id,
                    resolved_model_revision,
                    resolved_config_hash,
                    resolved_schema_version,
                    publication_key,
                    created_at,
                    resolved_publication_status,
                ),
            )
            conn.execute(
                f"""
                INSERT INTO {_PAYLOAD_TABLE} (
                    report_id, version, payload_json, payload_sha256
                ) VALUES (?, ?, ?, ?)
                """,
                (resolved_report_id, next_version, canonical_json, payload_sha256),
            )
            row = conn.execute(
                _SELECT_REPORT_SQL + " WHERE m.report_id = ? AND m.version = ?",
                (resolved_report_id, next_version),
            ).fetchone()
            if row is None:
                raise StorageSchemaError("报告写入后无法读取刚创建的版本。")
            record = _row_to_record(row)
            conn.commit()
            return PublishResult(record=record, created=True)
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            conn.close()

    def get_report(self, report_id: str, version: int | None = None) -> ReportRecord | None:
        """只读读取指定报告版本；未找到时返回 ``None``。"""
        resolved_report_id = _require_text(report_id, "report_id")
        resolved_version = _validate_version(version) if version is not None else None
        self._require_database()
        conn = self._connect(self.path)
        try:
            if resolved_version is None:
                row = conn.execute(
                    _SELECT_REPORT_SQL
                    + " WHERE m.report_id = ? ORDER BY m.version DESC LIMIT 1",
                    (resolved_report_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    _SELECT_REPORT_SQL + " WHERE m.report_id = ? AND m.version = ?",
                    (resolved_report_id, resolved_version),
                ).fetchone()
            if row is None:
                return None
            return _row_to_record(row)
        finally:
            conn.close()

    def get_latest(self, report_id: str) -> ReportRecord | None:
        """只读返回报告的最高版本，不重新运行或修改报告。"""
        return self.get_report(report_id, version=None)

    def mark_published(self, report_id: str, version: int) -> ReportRecord:
        """将待发布报告原子标记为 PUBLISHED 并读回校验。"""
        return self._set_publication_status(report_id, version, "PUBLISHED")

    def mark_failed(self, report_id: str, version: int) -> ReportRecord:
        """将无法完成跨文件引用的报告标记为 FAILED。"""
        return self._set_publication_status(report_id, version, "FAILED")

    def get_published(self, report_id: str, version: int | None = None) -> ReportRecord | None:
        """只读读取已完成跨文件发布的报告。"""
        record = self.get_report(report_id, version=version)
        if record is None or record.publication_status != "PUBLISHED":
            return None
        return record

    def _set_publication_status(self, report_id: str, version: int, status: str) -> ReportRecord:
        """执行受限的 PENDING 到终态迁移。"""
        resolved_report_id = _require_text(report_id, "report_id")
        resolved_version = _validate_version(version)
        resolved_status = _validate_publication_status(status)
        self._require_database()
        conn = self._connect(self.path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                _SELECT_REPORT_SQL + " WHERE m.report_id = ? AND m.version = ?",
                (resolved_report_id, resolved_version),
            ).fetchone()
            if row is None:
                raise ReportStorageError("待发布报告不存在。")
            current_status = row["publication_status"]
            if current_status == resolved_status:
                conn.commit()
                return _row_to_record(row)
            if current_status != "PENDING":
                raise ReportStorageError("已完成报告不能回写发布状态。")
            conn.execute(
                f"UPDATE {_REPORT_TABLE} SET publication_status = ? WHERE report_id = ? AND version = ?",
                (resolved_status, resolved_report_id, resolved_version),
            )
            updated = conn.execute(
                _SELECT_REPORT_SQL + " WHERE m.report_id = ? AND m.version = ?",
                (resolved_report_id, resolved_version),
            ).fetchone()
            if updated is None:
                raise StorageSchemaError("发布状态更新后无法读回报告。")
            record = _row_to_record(updated)
            conn.commit()
            return record
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            conn.close()

    def list_reports(
        self,
        report_id: str | None = None,
        limit: int | None = None,
    ) -> list[ReportRecord]:
        """只读列出报告历史，按报告 ID 和版本升序返回全部可解析版本。"""
        resolved_report_id = (
            _require_text(report_id, "report_id") if report_id is not None else None
        )
        resolved_limit = _validate_limit(limit) if limit is not None else None
        self._require_database()
        conn = self._connect(self.path)
        try:
            sql = _SELECT_REPORT_SQL
            params: list[Any] = []
            if resolved_report_id is not None:
                sql += " WHERE m.report_id = ?"
                params.append(resolved_report_id)
            sql += " ORDER BY m.report_id ASC, m.version ASC"
            if resolved_limit is not None:
                sql += " LIMIT ?"
                params.append(resolved_limit)
            rows = conn.execute(sql, tuple(params)).fetchall()
            return [_row_to_record(row) for row in rows]
        finally:
            conn.close()

    def backup(self, destination: str | Path) -> Path:
        """把 SQLite 数据库原子复制到临时副本，不包含外部报告或快照文件。

        目标文件先写入同目录临时文件，复制连接关闭后再替换目标；调用方应
        对返回路径调用 ``verify``，以区分 SQLite 副本成功和报告内容校验成功。
        """
        self._require_database()
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path.resolve() == self.path.resolve():
            raise ValueError("备份目标不能与源 SQLite 文件相同。")

        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination_path.name}.",
            suffix=".tmp",
            dir=destination_path.parent,
        )
        os.close(fd)
        temporary_path = Path(temporary_name)
        source: sqlite3.Connection | None = None
        target: sqlite3.Connection | None = None
        try:
            source = self._connect(self.path)
            target = self._connect(temporary_path)
            source.backup(target)
            target.commit()
            target.close()
            target = None
            source.close()
            source = None
            os.replace(temporary_path, destination_path)
            return destination_path
        finally:
            if target is not None:
                target.close()
            if source is not None:
                source.close()
            if temporary_path.exists():
                temporary_path.unlink()

    def verify(self, path: str | Path | None = None) -> VerificationResult:
        """只读核验 SQLite 完整性、孤儿记录、schema、JSON 和内容哈希。"""
        target_path = Path(path) if path is not None else self.path
        if not target_path.exists():
            return VerificationResult(
                ok=False,
                report_count=0,
                errors=(f"数据库文件不存在：{target_path}",),
            )

        errors: list[str] = []
        report_count = 0
        conn: sqlite3.Connection | None = None
        try:
            conn = self._connect(target_path)
            integrity_row = conn.execute("PRAGMA integrity_check").fetchone()
            if integrity_row is None or integrity_row[0] != "ok":
                errors.append(f"SQLite integrity_check 失败：{integrity_row[0] if integrity_row else '无结果'}")
            try:
                self._check_schema_without_writing(conn)
            except (ReportStorageError, sqlite3.Error) as exc:
                errors.append(f"schema 校验失败：{exc}")

            try:
                meta_row = conn.execute(
                    f"SELECT value FROM {_META_TABLE} WHERE key = ?",
                    ("storage_schema_version",),
                ).fetchone()
                if meta_row is None:
                    errors.append("缺少 SQLite storage schema 版本元数据。")
                else:
                    try:
                        _validate_storage_schema_version(meta_row[0])
                    except ReportStorageError as exc:
                        errors.append(str(exc))
            except sqlite3.Error as exc:
                errors.append(f"无法读取 SQLite schema 元数据：{exc}")

            try:
                orphan_metadata = conn.execute(
                    f"""
                    SELECT m.report_id, m.version
                    FROM {_REPORT_TABLE} AS m
                    LEFT JOIN {_PAYLOAD_TABLE} AS p
                      ON p.report_id = m.report_id AND p.version = m.version
                    WHERE p.report_id IS NULL
                    """
                ).fetchall()
                for row in orphan_metadata:
                    errors.append(f"报告元数据缺少 JSON 负载：{row[0]}@v{row[1]}")
                orphan_payloads = conn.execute(
                    f"""
                    SELECT p.report_id, p.version
                    FROM {_PAYLOAD_TABLE} AS p
                    LEFT JOIN {_REPORT_TABLE} AS m
                      ON m.report_id = p.report_id AND m.version = p.version
                    WHERE m.report_id IS NULL
                    """
                ).fetchall()
                for row in orphan_payloads:
                    errors.append(f"JSON 负载缺少报告元数据：{row[0]}@v{row[1]}")
            except sqlite3.Error as exc:
                errors.append(f"无法核验报告关联：{exc}")

            try:
                rows = conn.execute(
                    _SELECT_REPORT_SQL + " ORDER BY m.report_id ASC, m.version ASC"
                ).fetchall()
                report_count = len(rows)
                for row in rows:
                    try:
                        _row_to_record(row)
                    except ReportStorageError as exc:
                        errors.append(str(exc))
            except sqlite3.Error as exc:
                errors.append(f"无法读取报告索引：{exc}")

            try:
                foreign_key_rows = conn.execute("PRAGMA foreign_key_check").fetchall()
                for row in foreign_key_rows:
                    errors.append(f"SQLite 外键检查失败：{tuple(row)}")
            except sqlite3.Error as exc:
                errors.append(f"无法执行 SQLite 外键检查：{exc}")
        except (OSError, sqlite3.Error) as exc:
            errors.append(f"无法打开或读取 SQLite 数据库：{exc}")
        finally:
            if conn is not None:
                conn.close()
        return VerificationResult(ok=not errors, report_count=report_count, errors=tuple(errors))

    def _initialize(self) -> None:
        """在一次事务中创建或验证 SQLite 元数据和报告表。"""
        conn = self._connect(self.path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            tables = self._table_names(conn)
            if _META_TABLE not in tables:
                if _REPORT_TABLE in tables or _PAYLOAD_TABLE in tables:
                    raise StorageSchemaError("报告表存在但缺少 SQLite storage schema 元数据。")
                conn.execute(
                    f"CREATE TABLE {_META_TABLE} (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                conn.execute(
                    f"INSERT INTO {_META_TABLE} (key, value) VALUES (?, ?)",
                    ("storage_schema_version", str(STORAGE_SCHEMA_VERSION)),
                )
                conn.execute(
                    f"INSERT INTO {_META_TABLE} (key, value) VALUES (?, ?)",
                    ("created_at", _now_utc()),
                )
            else:
                try:
                    row = conn.execute(
                        f"SELECT value FROM {_META_TABLE} WHERE key = ?",
                        ("storage_schema_version",),
                    ).fetchone()
                except sqlite3.Error as exc:
                    raise StorageSchemaError("SQLite storage 元数据表结构不可读。") from exc
                if row is None:
                    raise StorageSchemaError("SQLite storage 元数据缺少 schema 版本。")
                _validate_storage_schema_version(row[0])

            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {_REPORT_TABLE} (
                    report_id TEXT NOT NULL,
                    version INTEGER NOT NULL CHECK (version > 0),
                    snapshot_id TEXT NOT NULL,
                    evaluation_id TEXT NOT NULL,
                    model_revision TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    publication_key TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    publication_status TEXT NOT NULL DEFAULT 'PUBLISHED',
                    PRIMARY KEY (report_id, version),
                    UNIQUE (report_id, publication_key)
                )
                """
            )
            report_columns = {
                str(row[1])
                for row in conn.execute(f"PRAGMA table_info({_REPORT_TABLE})").fetchall()
            }
            if "publication_status" not in report_columns:
                conn.execute(
                    f"ALTER TABLE {_REPORT_TABLE} ADD COLUMN publication_status TEXT NOT NULL DEFAULT 'PUBLISHED'"
                )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {_PAYLOAD_TABLE} (
                    report_id TEXT NOT NULL,
                    version INTEGER NOT NULL CHECK (version > 0),
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    PRIMARY KEY (report_id, version),
                    FOREIGN KEY (report_id, version)
                      REFERENCES {_REPORT_TABLE} (report_id, version)
                      ON DELETE RESTRICT
                )
                """
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{_REPORT_TABLE}_created_at "
                f"ON {_REPORT_TABLE} (created_at DESC)"
            )
            self._check_schema_without_writing(conn)
            conn.commit()
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            conn.close()

    def _require_database(self) -> None:
        """确认只读或备份操作不会因 SQLite 自动建空文件而改变状态。"""
        if not self.path.exists():
            raise FileNotFoundError(f"数据库文件不存在：{self.path}")

    @staticmethod
    def _connect(path: Path) -> sqlite3.Connection:
        """打开一个启用外键和忙等待的 SQLite 连接。"""
        conn = sqlite3.connect(str(path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    @staticmethod
    def _table_names(conn: sqlite3.Connection) -> set[str]:
        """读取 SQLite 中已有的表名，避免静默接管残缺数据库。"""
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        return {str(row[0]) for row in rows}

    @staticmethod
    def _check_schema_without_writing(conn: sqlite3.Connection) -> None:
        """只读检查存储表是否包含当前版本要求的列。"""
        required_tables = {
            _META_TABLE: _META_REQUIRED_COLUMNS,
            _REPORT_TABLE: _REPORT_REQUIRED_COLUMNS,
            _PAYLOAD_TABLE: _PAYLOAD_REQUIRED_COLUMNS,
        }
        for table, required_columns in required_tables.items():
            rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
            actual_columns = {str(row[1]) for row in rows}
            if not rows:
                raise StorageSchemaError(f"缺少 SQLite 表：{table}")
            missing = sorted(required_columns - actual_columns)
            if missing:
                raise StorageSchemaError(
                    f"SQLite 表 {table} 缺少列：{missing}"
                )


def _canonical_json(payload: Any) -> str:
    """生成排序、紧凑且禁止非有限数的 JSON 文本。"""
    if payload is None:
        raise ValueError("报告 JSON 负载不能为 None。")
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("报告 JSON 负载不可严格序列化。") from exc


def _sha256(value: str) -> str:
    """返回 UTF-8 文本的 SHA-256 十六进制摘要。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _now_utc() -> str:
    """返回带时区的 UTC 发布时间。"""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _require_text(value: str, field_name: str) -> str:
    """校验引用字段为非空字符串，并保留调用方原始值。"""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} 必须是非空字符串。")
    return value


def _validate_version(version: int) -> int:
    """校验单个报告版本号为正整数。"""
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("报告版本必须是正整数。")
    return version


def _validate_limit(limit: int) -> int:
    """校验查询上限为正整数。"""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("报告查询 limit 必须是正整数。")
    return limit


def _validate_report_schema_version(schema_version: str) -> str:
    """校验报告 schema 版本，并拒绝当前代码未知的版本。"""
    if not isinstance(schema_version, str) or not schema_version:
        raise ValueError("报告 schema_version 必须是非空字符串。")
    if schema_version not in SUPPORTED_REPORT_SCHEMA_VERSIONS:
        raise UnsupportedSchemaError(
            f"不支持的报告 schema 版本：{schema_version}。"
        )
    return schema_version


def _validate_publication_status(status: str) -> str:
    """校验跨文件发布状态。"""
    if status not in {"PENDING", "PUBLISHED", "FAILED"}:
        raise ValueError("publication_status 必须是 PENDING、PUBLISHED 或 FAILED。")
    return status


def _validate_storage_schema_version(value: Any) -> int:
    """校验 SQLite 元数据中的 storage schema 版本。"""
    try:
        version = int(value)
    except (TypeError, ValueError) as exc:
        raise StorageSchemaError(f"SQLite storage schema 版本无效：{value!r}") from exc
    if version != STORAGE_SCHEMA_VERSION:
        raise UnsupportedSchemaError(
            f"不支持的 SQLite storage schema 版本：{version}。"
        )
    return version


def _validate_embedded_schema_version(
    payload: Any,
    expected_schema_version: str,
    context: str = "报告 JSON",
) -> None:
    """校验 JSON 负载若声明 schema 时与元数据一致且受支持。"""
    if not isinstance(payload, dict) or "schema_version" not in payload:
        return
    embedded_schema_version = payload["schema_version"]
    if not isinstance(embedded_schema_version, str) or not embedded_schema_version:
        raise CorruptReportError(f"{context}中的 schema_version 损坏。")
    try:
        _validate_report_schema_version(embedded_schema_version)
    except UnsupportedSchemaError as exc:
        raise UnsupportedSchemaError(
            f"{context}包含不支持的报告 schema 版本：{embedded_schema_version}。"
        ) from exc
    if embedded_schema_version != expected_schema_version:
        raise CorruptReportError(
            f"{context}的 schema_version 与 SQLite 元数据不一致。"
        )


def _publication_key(
    report_id: str,
    payload_sha256: str,
    snapshot_id: str,
    evaluation_id: str,
    model_revision: str,
    config_hash: str,
    schema_version: str,
) -> str:
    """对一次完整发布输入生成稳定幂等指纹。"""
    value = json.dumps(
        {
            "config_hash": config_hash,
            "evaluation_id": evaluation_id,
            "model_revision": model_revision,
            "payload_sha256": payload_sha256,
            "report_id": report_id,
            "schema_version": schema_version,
            "snapshot_id": snapshot_id,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256(value)


def _row_to_record(row: sqlite3.Row) -> ReportRecord:
    """解析并校验一行报告；损坏或未知 schema 必须显式失败。"""
    report_id = row["report_id"]
    version = row["version"]
    schema_version = row["schema_version"]
    payload_json = row["payload_json"]
    stored_payload_sha256 = row["payload_sha256"]
    required_metadata = {
        "report_id": report_id,
        "snapshot_id": row["snapshot_id"],
        "evaluation_id": row["evaluation_id"],
        "model_revision": row["model_revision"],
        "config_hash": row["config_hash"],
        "created_at": row["created_at"],
    }
    if not isinstance(report_id, str) or not report_id:
        raise CorruptReportError("报告元数据中的 report_id 损坏。")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise CorruptReportError(f"报告 {report_id} 的版本号损坏：{version!r}")
    if not isinstance(schema_version, str) or not schema_version:
        raise CorruptReportError(f"报告 {report_id}@v{version} 的 schema 版本损坏。")
    try:
        _validate_report_schema_version(schema_version)
    except UnsupportedSchemaError:
        raise
    for field_name, value in required_metadata.items():
        if not isinstance(value, str) or not value:
            raise CorruptReportError(
                f"报告 {report_id}@v{version} 的元数据字段损坏：{field_name}。"
            )
    publication_status = row["publication_status"]
    try:
        _validate_publication_status(publication_status)
    except ValueError as exc:
        raise CorruptReportError(
            f"报告 {report_id}@v{version} 的发布状态损坏。"
        ) from exc
    if not isinstance(payload_json, str):
        raise CorruptReportError(f"报告 {report_id}@v{version} 的 JSON 负载缺失。")
    try:
        payload = json.loads(payload_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise CorruptReportError(
            f"报告 {report_id}@v{version} 的 JSON 负载损坏。"
        ) from exc
    _validate_embedded_schema_version(payload, schema_version, f"报告 {report_id}@v{version}")
    try:
        canonical_json = _canonical_json(payload)
    except ValueError as exc:
        raise CorruptReportError(
            f"报告 {report_id}@v{version} 的 JSON 负载不可规范化。"
        ) from exc
    actual_payload_sha256 = _sha256(canonical_json)
    if stored_payload_sha256 != actual_payload_sha256:
        raise CorruptReportError(
            f"报告 {report_id}@v{version} 的 JSON 哈希不匹配。"
        )
    expected_publication_key = _publication_key(
        report_id,
        actual_payload_sha256,
        required_metadata["snapshot_id"],
        required_metadata["evaluation_id"],
        required_metadata["model_revision"],
        required_metadata["config_hash"],
        schema_version,
    )
    if row["publication_key"] != expected_publication_key:
        raise CorruptReportError(
            f"报告 {report_id}@v{version} 的发布指纹不匹配。"
        )
    return ReportRecord(
        report_id=report_id,
        version=version,
        payload=payload,
        snapshot_id=required_metadata["snapshot_id"],
        evaluation_id=required_metadata["evaluation_id"],
        model_revision=required_metadata["model_revision"],
        config_hash=required_metadata["config_hash"],
        schema_version=schema_version,
        payload_sha256=actual_payload_sha256,
        created_at=required_metadata["created_at"],
        publication_status=publication_status,
    )


_SELECT_REPORT_SQL: Final[str] = f"""
SELECT
    m.report_id,
    m.version,
    m.snapshot_id,
    m.evaluation_id,
    m.model_revision,
    m.config_hash,
    m.schema_version,
    m.publication_key,
    m.created_at,
    m.publication_status,
    p.payload_json,
    p.payload_sha256
FROM {_REPORT_TABLE} AS m
LEFT JOIN {_PAYLOAD_TABLE} AS p
  ON p.report_id = m.report_id AND p.version = m.version
"""


__all__ = [
    "CorruptJSONError",
    "CorruptReportError",
    "PublishResult",
    "REPORT_SCHEMA_VERSION",
    "ReportRecord",
    "ReportStorage",
    "ReportStorageError",
    "STORAGE_SCHEMA_VERSION",
    "SUPPORTED_REPORT_SCHEMA_VERSIONS",
    "StorageError",
    "StorageSchemaError",
    "StorageCorruptionError",
    "UnknownSchemaError",
    "UnsupportedReportSchemaError",
    "UnsupportedSchemaError",
    "VerificationResult",
]
