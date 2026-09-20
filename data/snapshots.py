"""不可变 CSV 快照的独立文件存储原语。"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TypeAlias, cast

import pandas as pd


JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

SCHEMA_VERSION = 1
REFERENCES_SCHEMA_VERSION = 1
CSV_FILENAME = "data.csv"
MANIFEST_FILENAME = "manifest.json"
REFERENCES_FILENAME = "references.json"
_SNAPSHOT_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_STORE_LOCK = threading.RLock()


class SnapshotError(Exception):
    """快照存储操作失败的基类。"""


class SnapshotNotFoundError(SnapshotError):
    """请求的快照不存在。"""


class SnapshotCorruptionError(SnapshotError):
    """快照清单、文件或内容校验失败。"""


class UnsupportedSnapshotVersionError(SnapshotCorruptionError):
    """快照清单版本无法由当前实现读取。"""


class SnapshotWriteError(SnapshotError):
    """快照无法安全写入或发布。"""


@dataclass(frozen=True)
class SnapshotManifest:
    """描述一份 CSV 快照及其来源、时点和内容边界。"""

    snapshot_id: str
    schema_version: int
    source: str
    as_of: str | None
    quality: JsonValue
    content_hash: str
    columns: tuple[str, ...]
    row_count: int
    csv_size: int
    created_at: str
    csv_file: str = CSV_FILENAME

    def to_dict(self) -> dict[str, object]:
        """将清单转换为可写入 JSON 的字典。"""
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "format": "csv",
            "encoding": "utf-8",
            "index": False,
            "csv_file": self.csv_file,
            "source": self.source,
            "as_of": self.as_of,
            "quality": self.quality,
            "content_hash_algorithm": "sha256",
            "content_hash": self.content_hash,
            "columns": list(self.columns),
            "row_count": self.row_count,
            "csv_size": self.csv_size,
            "created_at": self.created_at,
        }


class SnapshotStore:
    """在指定根目录中发布、校验、引用和清理不可变 CSV 快照。"""

    def __init__(self, root: str | Path) -> None:
        """创建快照存储并确保根目录存在。"""
        self._root = Path(root).expanduser().resolve()
        self._ensure_root()

    @property
    def root(self) -> Path:
        """返回已经解析的快照根目录。"""
        return self._root

    def save(
        self,
        frame: pd.DataFrame,
        source: str = "unknown",
        as_of: str | date | datetime | pd.Timestamp | None = None,
        quality: object | None = None,
    ) -> str:
        """保存 DataFrame，重复内容复用 ID，内容修订生成新 ID。"""
        with _STORE_LOCK:
            self._ensure_root()
            csv_bytes, columns, row_count = _serialize_frame(frame)
            normalized_source = _normalize_source(source)
            normalized_as_of = _normalize_as_of(as_of)
            normalized_quality = _normalize_json_value(quality if quality is not None else {}, "quality")
            content_hash = hashlib.sha256(csv_bytes).hexdigest()
            snapshot_id = _snapshot_id_for(
                content_hash,
                normalized_source,
                normalized_as_of,
                normalized_quality,
                columns,
                row_count,
            )
            target = self._snapshot_directory(snapshot_id)

            if target.is_symlink():
                raise SnapshotCorruptionError(f"快照目录 {snapshot_id} 是符号链接，拒绝访问。")
            if target.exists():
                self._read_snapshot(snapshot_id)
                return snapshot_id

            manifest = SnapshotManifest(
                snapshot_id=snapshot_id,
                schema_version=SCHEMA_VERSION,
                source=normalized_source,
                as_of=normalized_as_of,
                quality=normalized_quality,
                content_hash=content_hash,
                columns=columns,
                row_count=row_count,
                csv_size=len(csv_bytes),
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            stage = self._make_stage_directory(snapshot_id)
            try:
                _atomic_write_bytes(stage / CSV_FILENAME, csv_bytes)
                _atomic_write_bytes(stage / MANIFEST_FILENAME, _json_bytes(manifest.to_dict()))
                _atomic_write_bytes(stage / REFERENCES_FILENAME, _json_bytes(_empty_references()))
                _fsync_directory(stage)
                try:
                    os.rename(stage, target)
                except FileExistsError:
                    self._remove_stage(stage)
                    stage = None
                    self._read_snapshot(snapshot_id)
                    return snapshot_id
                except OSError as exc:
                    if target.exists() and not target.is_symlink():
                        self._remove_stage(stage)
                        stage = None
                        self._read_snapshot(snapshot_id)
                        return snapshot_id
                    raise SnapshotWriteError(f"发布快照 {snapshot_id} 失败：{exc}") from exc
                stage = None
                _fsync_directory(self._root)
                return snapshot_id
            except SnapshotError:
                if stage is not None:
                    self._remove_stage(stage)
                raise
            except OSError as exc:
                if stage is not None:
                    self._remove_stage(stage)
                raise SnapshotWriteError(f"写入快照 {snapshot_id} 失败：{exc}") from exc
            except Exception as exc:
                if stage is not None:
                    self._remove_stage(stage)
                raise SnapshotWriteError(f"写入快照 {snapshot_id} 失败：{exc}") from exc

    def read(self, snapshot_id: str) -> pd.DataFrame:
        """校验清单、哈希、字段和行数后读取快照 CSV。"""
        with _STORE_LOCK:
            frame, _ = self._read_snapshot(snapshot_id)
            return frame

    def read_manifest(self, snapshot_id: str) -> SnapshotManifest:
        """完整校验快照后返回其不可变清单对象。"""
        with _STORE_LOCK:
            _, manifest = self._read_snapshot(snapshot_id)
            return manifest

    def mark_referenced(self, snapshot_id: str, reference_id: str | None = None) -> None:
        """记录一个引用，使清理不会删除该快照。"""
        reference = _normalize_reference_id(reference_id)
        with _STORE_LOCK:
            self._read_snapshot(snapshot_id)
            directory = self._snapshot_directory(snapshot_id)
            references = self._read_references(directory)
            if reference in references:
                return
            references.add(reference)
            _atomic_write_bytes(
                directory / REFERENCES_FILENAME,
                _json_bytes(
                    {
                        "schema_version": REFERENCES_SCHEMA_VERSION,
                        "references": sorted(references),
                    }
                ),
            )

    def is_referenced(self, snapshot_id: str) -> bool:
        """返回快照是否存在至少一个保护引用。"""
        with _STORE_LOCK:
            self._read_snapshot(snapshot_id)
            return bool(self._read_references(self._snapshot_directory(snapshot_id)))

    def unmark_referenced(self, snapshot_id: str, reference_id: str | None = None) -> None:
        """移除一个引用；默认移除手工引用标记。"""
        reference = _normalize_reference_id(reference_id)
        with _STORE_LOCK:
            self._read_snapshot(snapshot_id)
            directory = self._snapshot_directory(snapshot_id)
            references = self._read_references(directory)
            if reference not in references:
                return
            references.remove(reference)
            _atomic_write_bytes(
                directory / REFERENCES_FILENAME,
                _json_bytes(
                    {
                        "schema_version": REFERENCES_SCHEMA_VERSION,
                        "references": sorted(references),
                    }
                ),
            )

    def cleanup(self) -> tuple[str, ...]:
        """只删除根目录内校验完整且未被引用的快照目录。"""
        with _STORE_LOCK:
            self._ensure_root()
            deleted: list[str] = []
            for child in sorted(self._root.iterdir(), key=lambda item: item.name):
                if not _is_safe_snapshot_child(child, self._root):
                    continue
                snapshot_id = child.name
                try:
                    self._read_snapshot(snapshot_id)
                    references = self._read_references(child)
                except SnapshotError:
                    continue
                if references:
                    continue
                try:
                    shutil.rmtree(child)
                except OSError as exc:
                    raise SnapshotWriteError(f"清理快照 {snapshot_id} 失败：{exc}") from exc
                deleted.append(snapshot_id)
            return tuple(deleted)

    def _ensure_root(self) -> None:
        """确保根路径是可用的真实目录。"""
        try:
            self._root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SnapshotWriteError(f"无法创建快照根目录 {self._root}：{exc}") from exc
        if not self._root.is_dir():
            raise SnapshotWriteError(f"快照根路径不是目录：{self._root}。")

    def _snapshot_directory(self, snapshot_id: str) -> Path:
        """校验 ID 后构造根目录内的快照路径。"""
        if not isinstance(snapshot_id, str) or _SNAPSHOT_ID_PATTERN.fullmatch(snapshot_id) is None:
            raise SnapshotError("快照 ID 必须是 64 位小写十六进制字符串。")
        return self._root / snapshot_id

    def _make_stage_directory(self, snapshot_id: str) -> Path:
        """在快照根目录下创建本次发布专用的临时目录。"""
        try:
            return Path(tempfile.mkdtemp(prefix=f".snapshot-{snapshot_id[:12]}-", dir=str(self._root)))
        except OSError as exc:
            raise SnapshotWriteError(f"无法创建快照临时目录：{exc}") from exc

    def _read_snapshot(self, snapshot_id: str) -> tuple[pd.DataFrame, SnapshotManifest]:
        """读取并验证一份快照；调用方负责持有存储锁。"""
        directory = self._require_snapshot_directory(snapshot_id)
        manifest = self._read_manifest(snapshot_id, directory)
        csv_path = _safe_file_path(directory, manifest.csv_file)
        if not csv_path.exists():
            raise SnapshotCorruptionError(f"快照 {snapshot_id} 缺少 CSV 文件。")
        try:
            csv_bytes = csv_path.read_bytes()
        except OSError as exc:
            raise SnapshotCorruptionError(f"读取快照 {snapshot_id} 的 CSV 失败：{exc}") from exc
        if len(csv_bytes) != manifest.csv_size:
            raise SnapshotCorruptionError(f"快照 {snapshot_id} 的 CSV 大小不匹配。")
        actual_hash = hashlib.sha256(csv_bytes).hexdigest()
        if actual_hash != manifest.content_hash:
            raise SnapshotCorruptionError(f"快照 {snapshot_id} 的内容哈希不匹配。")
        try:
            frame = pd.read_csv(io.BytesIO(csv_bytes), encoding="utf-8")
        except Exception as exc:
            raise SnapshotCorruptionError(f"快照 {snapshot_id} 的 CSV 无法完整解析：{exc}") from exc
        actual_columns = tuple(str(column) for column in frame.columns)
        if actual_columns != manifest.columns:
            raise SnapshotCorruptionError(f"快照 {snapshot_id} 的字段不匹配清单。")
        if len(frame) != manifest.row_count:
            raise SnapshotCorruptionError(f"快照 {snapshot_id} 的行数不匹配清单。")
        return frame, manifest

    def _require_snapshot_directory(self, snapshot_id: str) -> Path:
        """定位并确认快照目录没有越过根目录边界。"""
        directory = self._snapshot_directory(snapshot_id)
        if directory.is_symlink():
            raise SnapshotCorruptionError(f"快照目录 {snapshot_id} 是符号链接，拒绝访问。")
        if not directory.exists():
            raise SnapshotNotFoundError(f"快照不存在：{snapshot_id}。")
        if not directory.is_dir() or not _is_safe_snapshot_child(directory, self._root):
            raise SnapshotCorruptionError(f"快照目录 {snapshot_id} 无效。")
        return directory

    def _read_manifest(self, snapshot_id: str, directory: Path) -> SnapshotManifest:
        """读取并严格解析 JSON 清单。"""
        manifest_path = _safe_file_path(directory, MANIFEST_FILENAME)
        if not manifest_path.exists():
            raise SnapshotCorruptionError(f"快照 {snapshot_id} 缺少 JSON 清单。")
        try:
            payload = json.loads(
                manifest_path.read_text(encoding="utf-8"),
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            )
        except Exception as exc:
            raise SnapshotCorruptionError(f"快照 {snapshot_id} 的 JSON 清单损坏：{exc}") from exc
        return _parse_manifest(payload, snapshot_id)

    def _read_references(self, directory: Path) -> set[str]:
        """读取引用保护清单；缺失清单按未引用处理。"""
        reference_path = _safe_file_path(directory, REFERENCES_FILENAME)
        if not reference_path.exists():
            return set()
        try:
            payload = json.loads(
                reference_path.read_text(encoding="utf-8"),
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            )
        except Exception as exc:
            raise SnapshotCorruptionError(f"快照引用清单损坏：{exc}") from exc
        if not isinstance(payload, dict):
            raise SnapshotCorruptionError("快照引用清单必须是 JSON 对象。")
        version = payload.get("schema_version")
        if not isinstance(version, int) or isinstance(version, bool):
            raise SnapshotCorruptionError("快照引用清单缺少有效版本。")
        if version != REFERENCES_SCHEMA_VERSION:
            raise UnsupportedSnapshotVersionError(f"不支持的快照引用清单版本：{version}。")
        values = payload.get("references")
        if not isinstance(values, list) or any(not isinstance(item, str) or not item for item in values):
            raise SnapshotCorruptionError("快照引用清单的引用标识无效。")
        if len(set(values)) != len(values):
            raise SnapshotCorruptionError("快照引用清单包含重复引用。")
        return set(values)

    @staticmethod
    def _remove_stage(stage: Path) -> None:
        """尽力移除本次失败写入留下的临时目录。"""
        try:
            if stage.exists():
                shutil.rmtree(stage)
        except OSError:
            # 临时目录被占用时保留它，不能覆盖原始写入异常或误删其他路径。
            return


def save_snapshot(
    root: str | Path,
    frame: pd.DataFrame,
    source: str = "unknown",
    as_of: str | date | datetime | pd.Timestamp | None = None,
    quality: object | None = None,
) -> str:
    """使用一次性存储实例保存 DataFrame 快照。"""
    return SnapshotStore(root).save(frame, source, as_of, quality)


def load_snapshot(root: str | Path, snapshot_id: str) -> pd.DataFrame:
    """使用一次性存储实例读取并验证 CSV 快照。"""
    return SnapshotStore(root).read(snapshot_id)


def cleanup_snapshots(root: str | Path) -> tuple[str, ...]:
    """清理指定根目录内未引用且完整的快照。"""
    return SnapshotStore(root).cleanup()


def _serialize_frame(frame: pd.DataFrame) -> tuple[bytes, tuple[str, ...], int]:
    """把 DataFrame 序列化为稳定的 UTF-8 CSV 字节。"""
    if not isinstance(frame, pd.DataFrame):
        raise SnapshotError("快照输入必须是 pandas DataFrame。")
    columns = tuple(frame.columns)
    if not columns or any(not isinstance(column, str) or not column for column in columns):
        raise SnapshotError("快照字段必须是非空字符串，且至少包含一个字段。")
    if len(set(columns)) != len(columns):
        raise SnapshotError("快照字段不能重复。")
    try:
        csv_text = frame.to_csv(index=False, lineterminator="\n")
    except Exception as exc:
        raise SnapshotError(f"DataFrame 无法序列化为 CSV：{exc}") from exc
    return csv_text.encode("utf-8"), columns, len(frame)


def _normalize_source(source: str) -> str:
    """验证并保留来源标识。"""
    if not isinstance(source, str) or not source.strip():
        raise SnapshotError("快照来源必须是非空字符串。")
    return source


def _normalize_as_of(value: str | date | datetime | pd.Timestamp | None) -> str | None:
    """将快照时点转换为可审计的字符串。"""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            raise SnapshotError("快照 as_of 不能是空时间。")
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str) and value.strip():
        return value
    raise SnapshotError("快照 as_of 必须是日期、时间或非空字符串。")


def _normalize_reference_id(reference_id: str | None) -> str:
    """验证引用保护标识。"""
    value = "manual" if reference_id is None else reference_id
    if not isinstance(value, str) or not value.strip():
        raise SnapshotError("快照引用标识必须是非空字符串。")
    return value


def _normalize_json_value(value: object, field_name: str) -> JsonValue:
    """验证元数据可被严格编码为 JSON。"""
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        normalized = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise SnapshotError(f"{field_name} 必须是可编码的 JSON 值。") from exc
    return cast(JsonValue, normalized)


def _snapshot_id_for(
    content_hash: str,
    source: str,
    as_of: str | None,
    quality: JsonValue,
    columns: tuple[str, ...],
    row_count: int,
) -> str:
    """按内容及不可变来源元数据生成快照版本 ID。

    ``retrieved_at`` 仅用于审计，不参与身份；同一份数据的重复读取仍应幂等。
    """
    identity_quality = quality
    if isinstance(quality, dict) and "retrieved_at" in quality:
        identity_quality = {
            key: value for key, value in quality.items() if key != "retrieved_at"
        }
    identity = {
        "content_hash": content_hash,
        "source": source,
        "as_of": as_of,
        "quality": identity_quality,
        "columns": list(columns),
        "row_count": row_count,
    }
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_manifest(payload: object, expected_snapshot_id: str) -> SnapshotManifest:
    """把不可信 JSON 清单解析为严格的清单对象。"""
    if not isinstance(payload, dict):
        raise SnapshotCorruptionError("快照清单必须是 JSON 对象。")
    version = payload.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise SnapshotCorruptionError("快照清单缺少有效版本。")
    if version != SCHEMA_VERSION:
        raise UnsupportedSnapshotVersionError(f"不支持的快照清单版本：{version}。")
    if payload.get("snapshot_id") != expected_snapshot_id:
        raise SnapshotCorruptionError("快照清单的 snapshot_id 不匹配目录。")
    if payload.get("format") != "csv" or payload.get("encoding") != "utf-8" or payload.get("index") is not False:
        raise SnapshotCorruptionError("快照清单的 CSV 格式声明无效。")
    if payload.get("content_hash_algorithm") != "sha256":
        raise SnapshotCorruptionError("快照清单的内容哈希算法声明无效。")
    if "csv_file" not in payload or "source" not in payload or "as_of" not in payload or "quality" not in payload:
        raise SnapshotCorruptionError("快照清单缺少来源、时点或质量字段。")
    csv_file = payload.get("csv_file")
    if csv_file != CSV_FILENAME:
        raise SnapshotCorruptionError("快照清单的 CSV 路径无效。")
    source = payload.get("source")
    if not isinstance(source, str) or not source.strip():
        raise SnapshotCorruptionError("快照清单缺少有效来源。")
    as_of = payload.get("as_of")
    if as_of is not None and not isinstance(as_of, str):
        raise SnapshotCorruptionError("快照清单的 as_of 无效。")
    quality = payload.get("quality")
    try:
        _normalize_json_value(quality, "quality")
    except SnapshotError as exc:
        raise SnapshotCorruptionError("快照清单的 quality 无效。") from exc
    content_hash = payload.get("content_hash")
    if not isinstance(content_hash, str) or re.fullmatch(r"[0-9a-f]{64}", content_hash) is None:
        raise SnapshotCorruptionError("快照清单的内容哈希无效。")
    columns = payload.get("columns")
    if not isinstance(columns, list) or not columns or any(not isinstance(column, str) or not column for column in columns):
        raise SnapshotCorruptionError("快照清单的字段列表无效。")
    if len(set(columns)) != len(columns):
        raise SnapshotCorruptionError("快照清单包含重复字段。")
    row_count = payload.get("row_count")
    if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 0:
        raise SnapshotCorruptionError("快照清单的行数无效。")
    csv_size = payload.get("csv_size")
    if not isinstance(csv_size, int) or isinstance(csv_size, bool) or csv_size < 1:
        raise SnapshotCorruptionError("快照清单的 CSV 大小无效。")
    created_at = payload.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise SnapshotCorruptionError("快照清单缺少创建时间。")
    manifest = SnapshotManifest(
        snapshot_id=expected_snapshot_id,
        schema_version=version,
        source=source,
        as_of=as_of,
        quality=cast(JsonValue, quality),
        content_hash=content_hash,
        columns=tuple(columns),
        row_count=row_count,
        csv_size=csv_size,
        created_at=created_at,
        csv_file=csv_file,
    )
    expected_id = _snapshot_id_for(
        manifest.content_hash,
        manifest.source,
        manifest.as_of,
        manifest.quality,
        manifest.columns,
        manifest.row_count,
    )
    if expected_id != expected_snapshot_id:
        raise SnapshotCorruptionError("快照清单的来源或质量元数据与 snapshot_id 不匹配。")
    return manifest


def _empty_references() -> dict[str, object]:
    """返回空的引用清单。"""
    return {"schema_version": REFERENCES_SCHEMA_VERSION, "references": []}


def _json_bytes(payload: object) -> bytes:
    """将已验证 JSON 对象编码为稳定、可读的 UTF-8 字节。"""
    try:
        return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SnapshotWriteError(f"快照 JSON 元数据无法编码：{exc}") from exc


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """用同目录临时文件和 os.replace 原子写入单个文件。"""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except OSError as exc:
        _remove_temporary_file(temporary)
        raise SnapshotWriteError(f"原子写入 {path.name} 失败：{exc}") from exc


def _fsync_directory(directory: Path) -> None:
    """在支持的平台上刷新目录项；Windows 不支持时安全跳过。"""
    try:
        descriptor = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            # Windows 等平台可能不支持目录 fsync；文件已完成原子替换。
            return
    finally:
        os.close(descriptor)


def _remove_temporary_file(path: Path) -> bool:
    """尽力删除临时文件，并以布尔值保留清理失败的明确结果。"""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return False
    return True


def _safe_file_path(directory: Path, filename: str) -> Path:
    """构造目录内的普通文件路径，拒绝符号链接和越界路径。"""
    path = directory / filename
    if path.is_symlink():
        raise SnapshotCorruptionError(f"快照文件 {filename} 是符号链接，拒绝访问。")
    try:
        resolved_parent = path.resolve(strict=False).parent
        resolved_directory = directory.resolve(strict=False)
    except OSError as exc:
        raise SnapshotCorruptionError(f"无法确认快照文件路径：{exc}") from exc
    if os.path.normcase(str(resolved_parent)) != os.path.normcase(str(resolved_directory)):
        raise SnapshotCorruptionError(f"快照文件 {filename} 越过目录边界。")
    return path


def _is_safe_snapshot_child(path: Path, root: Path) -> bool:
    """判断路径是否为根目录内的真实一级子目录。"""
    if not path.is_dir() or path.is_symlink() or _SNAPSHOT_ID_PATTERN.fullmatch(path.name) is None:
        return False
    try:
        resolved_path = path.resolve(strict=False)
        resolved_root = root.resolve(strict=False)
    except OSError:
        return False
    return os.path.normcase(str(resolved_path.parent)) == os.path.normcase(str(resolved_root))


__all__ = [
    "SnapshotCorruptionError",
    "SnapshotError",
    "SnapshotManifest",
    "SnapshotNotFoundError",
    "SnapshotStore",
    "SnapshotWriteError",
    "UnsupportedSnapshotVersionError",
    "cleanup_snapshots",
    "load_snapshot",
    "save_snapshot",
]
