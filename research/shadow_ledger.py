"""不可回写的前瞻/影子决策账本。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
from hashlib import sha256
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping


class ShadowLedgerError(ValueError):
    """影子记录或期后结算违反冻结边界。"""


@dataclass(frozen=True)
class ShadowDecision:
    """原始决策记录；后续结果不能覆盖其核心判断。"""

    decision_id: str
    stock_code: str
    decision_as_of: str
    snapshot_id: str
    report_id: str
    strategy_id: str
    strategy_version: str
    protocol_version: str
    action: str
    action_permission: str
    selected: bool
    status: str
    reason: str
    created_at: str
    content_hash: str = ""

    def __post_init__(self) -> None:
        required = {
            "decision_id": self.decision_id,
            "stock_code": self.stock_code,
            "decision_as_of": self.decision_as_of,
            "snapshot_id": self.snapshot_id,
            "report_id": self.report_id,
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "protocol_version": self.protocol_version,
            "reason": self.reason,
            "created_at": self.created_at,
        }
        if any(not isinstance(value, str) or not value for value in required.values()):
            raise ShadowLedgerError("影子记录的标识和原因不能为空")
        if self.action_permission != "NONE":
            raise ShadowLedgerError("影子账本不得保存可执行动作权限")
        if self.status not in {"SELECTED", "EXCLUDED", "FAILED"}:
            raise ShadowLedgerError("影子记录状态无效")
        calculated = _hash(self._payload_without_hash())
        if self.content_hash and self.content_hash != calculated:
            raise ShadowLedgerError("影子记录哈希不匹配")
        object.__setattr__(self, "content_hash", calculated)

    def _payload_without_hash(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "stock_code": self.stock_code,
            "decision_as_of": self.decision_as_of,
            "snapshot_id": self.snapshot_id,
            "report_id": self.report_id,
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "protocol_version": self.protocol_version,
            "action": self.action,
            "action_permission": self.action_permission,
            "selected": self.selected,
            "status": self.status,
            "reason": self.reason,
            "created_at": self.created_at,
        }

    def to_dict(self) -> dict[str, Any]:
        """返回带哈希的 JSON 字典。"""
        payload = self._payload_without_hash()
        payload["content_hash"] = self.content_hash
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ShadowDecision":
        """从字典重建影子记录。"""
        return cls(**{key: payload.get(key, "") for key in cls.__dataclass_fields__})


class ShadowLedger:
    """追加式 JSON 账本，原始决策和期后结果分开保存。"""

    schema_version = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._data = self._read()

    def append(self, decision: ShadowDecision) -> ShadowDecision:
        """追加决策；相同内容重复提交返回原记录。"""
        existing = self._data["decisions"].get(decision.decision_id)
        if existing is not None:
            if existing != decision.to_dict():
                raise ShadowLedgerError("相同 decision_id 的原始判断不可覆盖")
            return ShadowDecision.from_dict(existing)
        self._data["decisions"][decision.decision_id] = decision.to_dict()
        self._atomic_write()
        return self.get(decision.decision_id)

    def get(self, decision_id: str) -> ShadowDecision:
        """读取原始决策并重新校验哈希。"""
        payload = self._data["decisions"].get(decision_id)
        if payload is None:
            raise ShadowLedgerError("影子决策不存在")
        return ShadowDecision.from_dict(payload)

    def list_decisions(self) -> list[ShadowDecision]:
        """按创建顺序读取全部决策，包括失败和排除记录。"""
        return [self.get(identifier) for identifier in self._data["decisions"]]

    def record_outcome(
        self,
        decision_id: str,
        *,
        horizon: int,
        matured_as_of: str,
        realized_return: float | None,
        status: str,
        settled_at: str,
    ) -> dict[str, Any]:
        """追加期后结果；不成熟或缺失结果保持待定而非记零。"""
        decision = self.get(decision_id)
        if status not in {"SETTLED", "PENDING", "FAILED"}:
            raise ShadowLedgerError("结算状态无效")
        if not isinstance(horizon, int) or horizon not in {5, 10, 20}:
            raise ShadowLedgerError("只支持 5/10/20 日结算")
        if matured_as_of <= decision.decision_as_of:
            raise ShadowLedgerError("结算时点不能早于决策时点")
        if status == "SETTLED":
            if realized_return is None:
                raise ShadowLedgerError("已结算记录必须包含实际收益，不能用零代替缺失")
            if not isinstance(realized_return, (int, float)):
                raise ShadowLedgerError("实际收益必须是数值")
        elif realized_return is not None:
            raise ShadowLedgerError("未结算/失败记录不能附带实际收益")
        outcome = {
            "decision_id": decision_id,
            "horizon": horizon,
            "matured_as_of": matured_as_of,
            "realized_return": realized_return,
            "status": status,
            "settled_at": settled_at,
        }
        key = f"{decision_id}:{horizon}"
        previous = self._data["outcomes"].get(key)
        if previous is not None:
            if previous == outcome:
                return dict(previous)
            if previous.get("status") != "PENDING":
                stable_fields = ("status", "matured_as_of", "realized_return")
                if all(previous.get(field) == outcome.get(field) for field in stable_fields):
                    return dict(previous)
                if previous.get("status") == "FAILED" and outcome.get("status") == "FAILED":
                    return dict(previous)
                raise ShadowLedgerError("已结算决策周期不可覆盖")
            self._data.setdefault("outcome_history", []).append(dict(previous))
        self._data["outcomes"][key] = outcome
        self._atomic_write()
        return dict(outcome)

    def outcomes(self, decision_id: str) -> list[dict[str, Any]]:
        """读取指定决策的全部结算结果。"""
        self.get(decision_id)
        return [dict(item) for key, item in self._data["outcomes"].items() if key.startswith(f"{decision_id}:")]

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": self.schema_version, "decisions": {}, "outcomes": {}, "outcome_history": []}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ShadowLedgerError("无法读取影子账本") from exc
        if payload.get("schema_version") != self.schema_version or not isinstance(payload.get("decisions"), dict) or not isinstance(payload.get("outcomes"), dict):
            raise ShadowLedgerError("影子账本结构无效")
        if not isinstance(payload.get("outcome_history", []), list):
            raise ShadowLedgerError("影子结算历史结构无效")
        payload.setdefault("outcome_history", [])
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


def _hash(payload: Mapping[str, Any]) -> str:
    """计算追加式记录内容哈希。"""
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


__all__ = ["ShadowDecision", "ShadowLedger", "ShadowLedgerError"]
