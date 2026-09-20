"""配置、报告、快照和持仓的离线备份与校验。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Iterable, Mapping


class RecoveryError(RuntimeError):
    """备份或恢复的完整性错误。"""


@dataclass(frozen=True)
class BackupManifest:
    """可重放的备份清单。"""

    schema_version: int
    created_at: str
    files: Mapping[str, str]
    missing: tuple[str, ...]


class RuntimeRecovery:
    """对指定运行时文件和目录执行不联网的备份校验。"""

    schema_version = 1

    def __init__(self, artifacts: Mapping[str, str | Path]) -> None:
        if not artifacts:
            raise RecoveryError("至少需要一个备份对象")
        self.artifacts = {str(name): Path(path) for name, path in artifacts.items()}
        if any(not name or "/" in name or "\\" in name or name in {".", ".."} for name in self.artifacts):
            raise RecoveryError("备份名称不能包含路径分隔符")

    def create_backup(self, destination: str | Path, *, created_at: str) -> BackupManifest:
        """复制运行时对象并写入哈希清单，不删除源数据。"""
        target = Path(destination)
        if target.exists() and any(target.iterdir()):
            raise RecoveryError("备份目录必须为空")
        target.mkdir(parents=True, exist_ok=True)
        files: dict[str, str] = {}
        missing: list[str] = []
        for name, source in self.artifacts.items():
            if not source.exists():
                missing.append(name)
                continue
            destination_path = target / name
            if source.is_dir():
                shutil.copytree(source, destination_path)
            else:
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination_path)
            files[name] = _digest(destination_path)
        manifest = BackupManifest(self.schema_version, created_at, files, tuple(missing))
        _write_json(target / "manifest.json", {
            "schema_version": manifest.schema_version,
            "created_at": manifest.created_at,
            "files": dict(manifest.files),
            "missing": list(manifest.missing),
        })
        return manifest

    def verify_backup(self, destination: str | Path) -> BackupManifest:
        """重新计算备份哈希；任一文件损坏即拒绝。"""
        target = Path(destination)
        try:
            payload = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RecoveryError("备份清单不存在或损坏") from exc
        if payload.get("schema_version") != self.schema_version or not isinstance(payload.get("files"), dict):
            raise RecoveryError("备份清单版本或结构无效")
        for name, expected in payload["files"].items():
            path = target / name
            if not path.exists() or _digest(path) != expected:
                raise RecoveryError(f"备份对象校验失败：{name}")
        return BackupManifest(
            schema_version=self.schema_version,
            created_at=str(payload.get("created_at", "")),
            files=dict(payload["files"]),
            missing=tuple(str(item) for item in payload.get("missing", [])),
        )

    def restore(self, source: str | Path, *, allow_overwrite: bool = False) -> BackupManifest:
        """校验后恢复到构造时声明的目标路径；默认不覆盖现有数据。"""
        manifest = self.verify_backup(source)
        source_root = Path(source)
        for name in manifest.files:
            target = self.artifacts[name]
            if target.exists() and not allow_overwrite:
                raise RecoveryError(f"恢复目标已存在，拒绝覆盖：{name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            origin = source_root / name
            if origin.is_dir():
                shutil.copytree(origin, target)
            else:
                shutil.copy2(origin, target)
        return manifest


def _digest(path: Path) -> str:
    """计算文件或目录的稳定 SHA-256。"""
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    """以临时文件和替换写入清单。"""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


__all__ = ["BackupManifest", "RecoveryError", "RuntimeRecovery"]
