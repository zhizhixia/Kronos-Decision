"""可审计的手工持仓快照和风险约束，不连接券商。"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import csv
from contextlib import contextmanager
from hashlib import sha256
from io import StringIO
import json
import math
from pathlib import Path
import os
import threading
import time
from tempfile import NamedTemporaryFile
from typing import Any, Iterator, Mapping, Sequence


class PortfolioError(ValueError):
    """持仓输入、风险约束或账本状态错误。"""


class PortfolioCorruptionError(PortfolioError):
    """持仓账本缺失、损坏或哈希不一致。"""


@dataclass(frozen=True)
class Position:
    """单证券持仓及可卖数量。"""

    code: str
    quantity: int
    available_quantity: int
    avg_cost: float | None
    as_of: str
    source: str
    name: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code.strip():
            raise PortfolioError("持仓 code 不能为空")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int) or self.quantity < 0:
            raise PortfolioError(f"持仓数量无效：{self.code}")
        if isinstance(self.available_quantity, bool) or not isinstance(self.available_quantity, int):
            raise PortfolioError(f"可卖数量无效：{self.code}")
        if self.available_quantity < 0 or self.available_quantity > self.quantity:
            raise PortfolioError(f"可卖数量超过持仓：{self.code}")
        if self.avg_cost is not None and (not math.isfinite(float(self.avg_cost)) or float(self.avg_cost) < 0):
            raise PortfolioError(f"持仓成本无效：{self.code}")
        if not isinstance(self.as_of, str) or not self.as_of.strip():
            raise PortfolioError("持仓 as_of 不能为空")
        if not isinstance(self.source, str) or not self.source.strip():
            raise PortfolioError("持仓来源不能为空")

    def to_dict(self) -> dict[str, Any]:
        """返回 JSON 兼容字典。"""
        return {
            "code": self.code,
            "quantity": self.quantity,
            "available_quantity": self.available_quantity,
            "avg_cost": self.avg_cost,
            "as_of": self.as_of,
            "source": self.source,
            "name": self.name,
        }


@dataclass(frozen=True)
class PortfolioSnapshot:
    """不可变持仓快照。"""

    snapshot_id: str
    as_of: str
    available_at: str
    cash: float
    positions: tuple[Position, ...]
    source: str
    schema_version: int = 1
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.snapshot_id or not self.as_of or not self.available_at:
            raise PortfolioError("持仓快照标识和时点不能为空")
        if not math.isfinite(float(self.cash)) or float(self.cash) < 0:
            raise PortfolioError("现金必须是非负有限数")
        object.__setattr__(self, "cash", float(self.cash))
        if not isinstance(self.source, str) or not self.source.strip():
            raise PortfolioError("持仓快照来源不能为空")
        if self.schema_version != 1:
            raise PortfolioError("不支持的持仓快照 schema_version")
        codes = [position.code for position in self.positions]
        if len(codes) != len(set(codes)):
            raise PortfolioError("持仓快照不能包含重复证券")
        calculated = _hash_payload(self._payload_without_hash())
        if self.content_hash and self.content_hash != calculated:
            raise PortfolioCorruptionError("持仓快照 content_hash 不匹配")
        object.__setattr__(self, "content_hash", calculated)

    def _payload_without_hash(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "as_of": self.as_of,
            "available_at": self.available_at,
            "cash": self.cash,
            "positions": [position.to_dict() for position in self.positions],
            "source": self.source,
        }

    def to_dict(self) -> dict[str, Any]:
        """返回带内容哈希的持仓快照。"""
        payload = self._payload_without_hash()
        payload["content_hash"] = self.content_hash
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PortfolioSnapshot":
        """从 JSON 字典重建并校验快照。"""
        if not isinstance(payload, Mapping):
            raise PortfolioCorruptionError("持仓快照必须是对象")
        positions = tuple(Position(**item) for item in payload.get("positions", []))
        return cls(
            snapshot_id=str(payload.get("snapshot_id", "")),
            as_of=str(payload.get("as_of", "")),
            available_at=str(payload.get("available_at", "")),
            cash=float(payload.get("cash", -1)),
            positions=positions,
            source=str(payload.get("source", "")),
            schema_version=int(payload.get("schema_version", -1)),
            content_hash=str(payload.get("content_hash", "")),
        )


@dataclass(frozen=True)
class RiskConstraints:
    """组合级风险约束。"""

    max_position_weight: float = 0.2
    max_total_positions: int = 20

    def __post_init__(self) -> None:
        if not 0 < self.max_position_weight <= 1:
            raise PortfolioError("max_position_weight 必须在 0 到 1 之间")
        if isinstance(self.max_total_positions, bool) or self.max_total_positions <= 0:
            raise PortfolioError("max_total_positions 必须是正整数")


@dataclass(frozen=True)
class PortfolioPreview:
    """CSV 预览结果；预览本身不会写入正式账本。"""

    preview_id: str
    positions: tuple[Position, ...]
    errors: tuple[str, ...]
    content_hash: str

    @property
    def valid(self) -> bool:
        """是否可以由用户显式确认导入。"""
        return not self.errors


_LEDGER_LOCKS: dict[str, Any] = {}
_LEDGER_LOCKS_GUARD = threading.Lock()


def _ledger_lock(path: Path) -> threading.RLock:
    """返回同一路径账本实例共享的读改写锁。"""
    key = str(path.resolve())
    with _LEDGER_LOCKS_GUARD:
        lock = _LEDGER_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _LEDGER_LOCKS[key] = lock
        return lock


@contextmanager
def _ledger_process_lock(path: Path) -> Iterator[None]:
    """用操作系统文件锁串行化跨进程账本读改写。"""
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class PortfolioLedger:
    """使用原子 JSON 文件保存持仓快照。"""

    schema_version = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = _ledger_lock(self.path)
        self._data = self._read()
        self._revision = _hash_payload(self._data)

    def save_snapshot(self, snapshot: PortfolioSnapshot) -> PortfolioSnapshot:
        """幂等保存快照；相同 ID 的不同内容拒绝覆盖。"""
        with self._lock:
            with _ledger_process_lock(self.path):
                current = self._read()
                current_revision = _hash_payload(current)
                if current_revision != self._revision:
                    self._data = current
                    self._revision = current_revision
                    raise PortfolioError("持仓账本已被其他实例修改，请重新读取后重试")
                self._data = current
                existing = self._data["snapshots"].get(snapshot.snapshot_id)
                if existing is not None:
                    if existing != snapshot.to_dict():
                        raise PortfolioError("相同 snapshot_id 的持仓内容冲突")
                    return PortfolioSnapshot.from_dict(existing)
                self._data["snapshots"][snapshot.snapshot_id] = snapshot.to_dict()
                self._atomic_write()
                self._revision = _hash_payload(self._data)
                restored = self.get(snapshot.snapshot_id)
                if restored is None:
                    raise PortfolioCorruptionError("持仓快照写入后无法读回")
                return restored

    def get(self, snapshot_id: str) -> PortfolioSnapshot | None:
        """按 ID 读取并校验快照。"""
        payload = self._data["snapshots"].get(snapshot_id)
        return None if payload is None else PortfolioSnapshot.from_dict(payload)

    def latest(self) -> PortfolioSnapshot | None:
        """读取按 available_at 排序的最新快照。"""
        snapshots = [self.get(identifier) for identifier in self._data["snapshots"]]
        valid = [item for item in snapshots if item is not None]
        return max(valid, key=lambda item: item.available_at) if valid else None

    def preview_csv(self, content: str, *, preview_id: str = "") -> PortfolioPreview:
        """解析 CSV 预览，不修改账本。"""
        if not isinstance(content, str) or len(content.encode("utf-8")) > 2_000_000:
            raise PortfolioError("持仓 CSV 为空或超过大小限制")
        reader = csv.DictReader(StringIO(content))
        required = {"code", "quantity", "available_quantity", "avg_cost", "as_of", "source"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            missing = sorted(required.difference(set(reader.fieldnames or [])))
            return PortfolioPreview(preview_id or "preview-invalid", (), (f"缺少字段：{missing}",), "")
        positions: list[Position] = []
        errors: list[str] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                avg_cost = None if not str(row.get("avg_cost", "")).strip() else float(row["avg_cost"])
                positions.append(
                    Position(
                        code=str(row["code"]),
                        quantity=int(row["quantity"]),
                        available_quantity=int(row["available_quantity"]),
                        avg_cost=avg_cost,
                        as_of=str(row["as_of"]),
                        source=str(row["source"]),
                        name=str(row.get("name", "")),
                    )
                )
            except (TypeError, ValueError, PortfolioError) as exc:
                errors.append(f"第 {line_number} 行：{exc}")
        if len({item.code for item in positions}) != len(positions):
            errors.append("CSV 包含重复证券")
        digest = sha256(content.encode("utf-8")).hexdigest()
        return PortfolioPreview(preview_id or f"preview-{digest[:16]}", tuple(positions), tuple(errors), digest)

    def confirm_preview(
        self,
        preview: PortfolioPreview,
        *,
        cash: float,
        as_of: str,
        available_at: str,
        source: str,
        confirmed: bool,
    ) -> PortfolioSnapshot:
        """只有显式 confirmed=True 且预览无错时才建立正式快照。"""
        if not confirmed:
            raise PortfolioError("未确认持仓预览，拒绝写入")
        if not preview.valid:
            raise PortfolioError("持仓预览存在错误，拒绝写入")
        snapshot_id = f"portfolio-{preview.content_hash[:16]}-{as_of}"
        snapshot = PortfolioSnapshot(
            snapshot_id=snapshot_id,
            as_of=as_of,
            available_at=available_at,
            cash=cash,
            positions=preview.positions,
            source=source,
        )
        return self.save_snapshot(snapshot)

    def validate_risk(
        self,
        snapshot: PortfolioSnapshot,
        prices: Mapping[str, float],
        constraints: RiskConstraints | None = None,
    ) -> tuple[str, ...]:
        """用显式当前价格检查集中度；缺价格不会被当成零。"""
        rules = constraints or RiskConstraints()
        errors: list[str] = []
        values: dict[str, float] = {}
        for position in snapshot.positions:
            if position.code not in prices:
                errors.append(f"缺少价格：{position.code}")
                continue
            price = float(prices[position.code])
            if not math.isfinite(price) or price <= 0:
                errors.append(f"价格无效：{position.code}")
                continue
            values[position.code] = position.quantity * price
        total = snapshot.cash + sum(values.values())
        if total <= 0:
            errors.append("组合总资产无效")
            return tuple(errors)
        if len([value for value in values.values() if value > 0]) > rules.max_total_positions:
            errors.append("持仓数量超过上限")
        for code, value in values.items():
            if value / total > rules.max_position_weight:
                errors.append(f"单证券集中度超限：{code}")
        return tuple(errors)

    def sellable_quantity(self, snapshot: PortfolioSnapshot, code: str, as_of: str) -> int:
        """返回指定时点可卖数量，不以总持仓替代 T+1 可卖数量。"""
        if snapshot.as_of > as_of:
            raise PortfolioError("不能用未来持仓快照处理历史时点")
        for position in snapshot.positions:
            if position.code == code:
                return position.available_quantity
        return 0

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": self.schema_version, "snapshots": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PortfolioCorruptionError("无法读取持仓账本") from exc
        if payload.get("schema_version") != self.schema_version or not isinstance(payload.get("snapshots"), dict):
            raise PortfolioCorruptionError("持仓账本 schema 无效")
        return payload

    def _atomic_write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp", delete=False) as handle:
                temporary_name = handle.name
                json.dump(self._data, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
            temporary_name = None
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    temporary_name = None


def _hash_payload(payload: Mapping[str, Any]) -> str:
    """计算持仓 JSON 规范哈希。"""
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "PortfolioCorruptionError",
    "PortfolioError",
    "PortfolioLedger",
    "PortfolioPreview",
    "PortfolioSnapshot",
    "Position",
    "RiskConstraints",
]
