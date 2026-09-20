"""研究层不可变契约：策略、预测工件和验证证据。"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping


class ResearchContractError(ValueError):
    """研究契约不满足冻结边界。"""


def _hash(payload: Mapping[str, Any]) -> str:
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StrategySpec:
    """研究和盘后应用共享的策略定义。"""

    strategy_id: str
    version: str
    horizons: tuple[int, ...]
    universe_version: str
    decision_time: str
    holding_semantics: str
    risk_constraints: Mapping[str, Any]
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.strategy_id or not self.version or not self.universe_version:
            raise ResearchContractError("策略 ID、版本和股票池版本不能为空")
        if tuple(sorted(set(self.horizons))) != self.horizons or not set(self.horizons).issubset({5, 10, 20}):
            raise ResearchContractError("策略周期只能是有序的 5/10/20 日子集")
        payload = self.to_dict(include_hash=False)
        calculated = _hash(payload)
        if self.content_hash and self.content_hash != calculated:
            raise ResearchContractError("策略定义哈希不匹配")
        object.__setattr__(self, "content_hash", calculated)

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        """返回可持久化策略定义。"""
        payload = {
            "strategy_id": self.strategy_id,
            "version": self.version,
            "horizons": list(self.horizons),
            "universe_version": self.universe_version,
            "decision_time": self.decision_time,
            "holding_semantics": self.holding_semantics,
            "risk_constraints": dict(self.risk_constraints),
        }
        if include_hash:
            payload["content_hash"] = self.content_hash
        return payload


@dataclass(frozen=True)
class ForecastArtifact:
    """绑定输入快照、模型版本和采样参数的预测工件。"""

    artifact_id: str
    snapshot_id: str
    target_as_of: str
    model_revision: str
    tokenizer_revision: str
    parameters: Mapping[str, Any]
    seed: int
    samples: tuple[Mapping[str, Any], ...]
    validity_status: str
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value for value in (self.artifact_id, self.snapshot_id, self.target_as_of, self.model_revision, self.tokenizer_revision)):
            raise ResearchContractError("预测工件的追溯字段不能为空")
        if self.validity_status not in {"VALID", "INVALID", "RESEARCH_ONLY"}:
            raise ResearchContractError("预测工件有效性状态无效")
        payload = self.to_dict(include_hash=False)
        calculated = _hash(payload)
        if self.content_hash and self.content_hash != calculated:
            raise ResearchContractError("预测工件哈希不匹配")
        object.__setattr__(self, "content_hash", calculated)

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        """返回可持久化预测工件。"""
        payload = {
            "artifact_id": self.artifact_id,
            "snapshot_id": self.snapshot_id,
            "target_as_of": self.target_as_of,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "parameters": dict(self.parameters),
            "seed": self.seed,
            "samples": [dict(sample) for sample in self.samples],
            "validity_status": self.validity_status,
        }
        if include_hash:
            payload["content_hash"] = self.content_hash
        return payload


@dataclass(frozen=True)
class ValidationEvidence:
    """将协议、数据、模型和样本外结果绑定在一起的证据。"""

    evidence_id: str
    protocol_hash: str
    data_snapshot_id: str
    model_revision: str
    out_of_sample_metrics: Mapping[str, Any]
    scope: str
    limitations: tuple[str, ...]
    valid_until: str | None
    status: str
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value for value in (self.evidence_id, self.protocol_hash, self.data_snapshot_id, self.model_revision, self.scope)):
            raise ResearchContractError("验证证据的追溯字段不能为空")
        if self.status not in {"QUALIFIED", "RESEARCH_ONLY", "INVALID", "STALE"}:
            raise ResearchContractError("验证证据状态无效")
        calculated = _hash(self.to_dict(include_hash=False))
        if self.content_hash and self.content_hash != calculated:
            raise ResearchContractError("验证证据哈希不匹配")
        object.__setattr__(self, "content_hash", calculated)

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        """返回可持久化验证证据。"""
        payload = {
            "evidence_id": self.evidence_id,
            "protocol_hash": self.protocol_hash,
            "data_snapshot_id": self.data_snapshot_id,
            "model_revision": self.model_revision,
            "out_of_sample_metrics": dict(self.out_of_sample_metrics),
            "scope": self.scope,
            "limitations": list(self.limitations),
            "valid_until": self.valid_until,
            "status": self.status,
        }
        if include_hash:
            payload["content_hash"] = self.content_hash
        return payload


__all__ = ["ForecastArtifact", "ResearchContractError", "StrategySpec", "ValidationEvidence"]
