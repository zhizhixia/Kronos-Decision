"""研究实验注册与策略生命周期的最小治理实现。

本模块独立于 ``evaluation``，只保存实验、失败、证据绑定和生命周期事件；
它不运行回测、不计算资格门槛，也不替用户决定资金规模。注册表采用标准库
JSON 文件，同目录临时文件加 ``os.replace`` 实现原子写入，所有更正和状态
变化都追加历史快照而不是覆盖审计记录。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid
from tempfile import NamedTemporaryFile
from types import MappingProxyType
from typing import Any, Mapping


RESEARCH = "RESEARCH"
SHADOW = "SHADOW"
ELIGIBLE = "ELIGIBLE"
SUSPENDED = "SUSPENDED"
RETIRED = "RETIRED"
LIFECYCLE_STATES = frozenset({RESEARCH, SHADOW, ELIGIBLE, SUSPENDED, RETIRED})

RUN_SUCCEEDED = "SUCCEEDED"
RUN_FAILED = "FAILED"
RUN_CANCELLED = "CANCELLED"
RUN_STATES = frozenset({RUN_SUCCEEDED, RUN_FAILED, RUN_CANCELLED})

_BINDING_FIELDS = (
    "strategy_id",
    "strategy_version",
    "model_version",
    "data_version",
    "period",
    "stock_pool_version",
    "protocol_version",
)


class RegistryError(ValueError):
    """注册表操作或数据格式错误。"""


class RegistryCorruptionError(RegistryError):
    """注册表 JSON 缺失、损坏或写入后校验失败。"""


class DuplicateExperimentError(RegistryError):
    """实验 ID 已经存在。"""


class ExperimentNotFoundError(RegistryError):
    """请求的实验 ID 不存在。"""


class InvalidTransitionError(RegistryError):
    """生命周期状态转换不被允许。"""


class ApprovalRequiredError(RegistryError):
    """进入或恢复 ELIGIBLE 缺少显式审批凭据。"""


class BindingMismatchError(RegistryError):
    """证据绑定与实验的完整版本或适用范围不一致。"""


class ImmutableExperimentError(RegistryError):
    """已具备资格或已退役的实验不能用普通更新覆盖。"""


def _now() -> str:
    """生成可排序的 UTC 审计时间。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    """将内部映射转换为 JSON 标准库可编码的值。"""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise RegistryError(f"注册表包含不可序列化类型：{type(value).__name__}")


def _immutable_json(value: Any) -> Any:
    """递归冻结 JSON 兼容值，避免记录被调用方原地改写。"""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _immutable_json(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_immutable_json(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise RegistryError(f"注册表包含不可序列化类型：{type(value).__name__}")


def _canonical_json(value: Any) -> str:
    """生成注册表内部稳定的 JSON 文本。"""
    try:
        return json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RegistryError("注册表内容无法规范化为 JSON") from exc


def _copy_json(value: Any) -> Any:
    """复制 JSON 兼容值，确保一次变更不会修改内存中的旧快照。"""
    return json.loads(_canonical_json(value))


def _require_text(value: Any, field_name: str) -> str:
    """校验必需标识符文本，并保留调用方提供的字面值。"""
    if value is None or not isinstance(value, (str, int)) or not str(value):
        raise RegistryError(f"{field_name} 不能为空")
    return str(value)


@dataclass(frozen=True)
class EvidenceBinding:
    """资格证据必须精确绑定的策略、模型、数据、周期、池和协议版本。"""

    strategy_id: str
    strategy_version: str
    model_version: str
    data_version: str
    period: str
    stock_pool_version: str
    protocol_version: str

    def __post_init__(self) -> None:
        """校验绑定字段均为非空标识符。"""
        for field_name in _BINDING_FIELDS:
            value = _require_text(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, value)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "EvidenceBinding") -> "EvidenceBinding":
        """从映射读取绑定，并兼容常见的股票池和周期字段名称。"""
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise BindingMismatchError("evidence_binding 必须是对象")
        aliases = {
            "strategy": "strategy_id",
            "strategy_revision": "strategy_version",
            "model": "model_version",
            "model_revision": "model_version",
            "data": "data_version",
            "data_snapshot_version": "data_version",
            "data_snapshot": "data_version",
            "horizon": "period",
            "horizon_days": "period",
            "evaluation_period": "period",
            "universe_version": "stock_pool_version",
            "stock_pool": "stock_pool_version",
            "pool_version": "stock_pool_version",
            "protocol": "protocol_version",
        }
        values: dict[str, Any] = {}
        for field_name in _BINDING_FIELDS:
            if field_name in value:
                values[field_name] = value[field_name]
        for alias, field_name in aliases.items():
            if alias not in value:
                continue
            if field_name in values and str(values[field_name]) != str(value[alias]):
                raise BindingMismatchError(f"绑定字段 {field_name} 与别名冲突")
            values[field_name] = value[alias]
        missing = [field_name for field_name in _BINDING_FIELDS if field_name not in values]
        if missing:
            raise BindingMismatchError(f"evidence_binding 缺少字段：{', '.join(missing)}")
        return cls(**values)

    def to_dict(self) -> dict[str, str]:
        """返回绑定的稳定字典表示。"""
        return {field_name: getattr(self, field_name) for field_name in _BINDING_FIELDS}

    @property
    def key(self) -> tuple[str, ...]:
        """返回用于严格比较的绑定键。"""
        return tuple(getattr(self, field_name) for field_name in _BINDING_FIELDS)

    def matches(self, other: Mapping[str, Any] | "EvidenceBinding") -> bool:
        """判断另一个绑定是否逐字段完全相同。"""
        try:
            return self.key == type(self).from_mapping(other).key
        except BindingMismatchError:
            return False


@dataclass(frozen=True)
class ExperimentRecord:
    """一个实验尝试的不可变快照，成功和失败都使用同一记录结构。"""

    experiment_id: str
    status: str = RESEARCH
    run_status: str = RUN_SUCCEEDED
    strategy_id: str = ""
    strategy_version: str = ""
    model_version: str = ""
    data_version: str = ""
    period: str = ""
    stock_pool_version: str = ""
    protocol_version: str = ""
    evidence_binding: EvidenceBinding | None = None
    result: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    approval: Mapping[str, Any] | None = None
    revision: int = 1
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        """规范化记录并阻断不支持的生命周期或运行状态。"""
        identifier = _require_text(self.experiment_id, "experiment_id")
        status = str(self.status).upper()
        run_status = str(self.run_status).upper()
        if status not in LIFECYCLE_STATES:
            raise RegistryError(f"未知生命周期状态：{self.status}")
        if run_status not in RUN_STATES:
            raise RegistryError(f"未知实验运行状态：{self.run_status}")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 1:
            raise RegistryError("revision 必须是正整数")
        object.__setattr__(self, "experiment_id", identifier)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "run_status", run_status)

        direct_values = {
            field_name: str(getattr(self, field_name))
            for field_name in _BINDING_FIELDS
            if getattr(self, field_name) not in (None, "")
        }
        binding = self.evidence_binding
        if binding is not None:
            binding = EvidenceBinding.from_mapping(binding)
            for field_name in _BINDING_FIELDS:
                direct_value = direct_values.get(field_name)
                bound_value = getattr(binding, field_name)
                if direct_value is not None and direct_value != bound_value:
                    raise BindingMismatchError(
                        f"记录字段 {field_name} 与 evidence_binding 不一致"
                    )
                object.__setattr__(self, field_name, bound_value)
        elif len(direct_values) == len(_BINDING_FIELDS):
            binding = EvidenceBinding(**direct_values)
        object.__setattr__(self, "evidence_binding", binding)
        if binding is None:
            for field_name in _BINDING_FIELDS:
                object.__setattr__(self, field_name, direct_values.get(field_name, ""))

        object.__setattr__(self, "result", _immutable_json(self.result or {}))
        object.__setattr__(self, "metadata", _immutable_json(self.metadata or {}))
        if self.approval is not None:
            object.__setattr__(self, "approval", _immutable_json(self.approval))
        if self.error is not None:
            object.__setattr__(self, "error", str(self.error))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentRecord":
        """从 JSON 字典重建实验记录。"""
        if not isinstance(payload, Mapping):
            raise RegistryError("实验记录必须是对象")
        values = dict(payload)
        if "experiment_id" not in values and "id" in values:
            values["experiment_id"] = values.pop("id")
        if "status" not in values:
            values["status"] = values.pop(
                "lifecycle_status", values.pop("qualification_status", RESEARCH)
            )
        binding = values.get("evidence_binding")
        if binding is not None and not isinstance(binding, EvidenceBinding):
            values["evidence_binding"] = EvidenceBinding.from_mapping(binding)
        allowed = {
            "experiment_id",
            "status",
            "run_status",
            "strategy_id",
            "strategy_version",
            "model_version",
            "data_version",
            "period",
            "stock_pool_version",
            "protocol_version",
            "evidence_binding",
            "result",
            "error",
            "metadata",
            "approval",
            "revision",
            "created_at",
            "updated_at",
        }
        unknown = set(values) - allowed
        if unknown:
            extra = dict(values.get("metadata") or {})
            for key in sorted(unknown):
                extra[key] = values[key]
            values["metadata"] = extra
        return cls(**{key: values[key] for key in values if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        """返回实验记录的 JSON 字典表示。"""
        payload: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "run_status": self.run_status,
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "model_version": self.model_version,
            "data_version": self.data_version,
            "period": self.period,
            "stock_pool_version": self.stock_pool_version,
            "protocol_version": self.protocol_version,
            "evidence_binding": (
                None if self.evidence_binding is None else self.evidence_binding.to_dict()
            ),
            "result": _jsonable(self.result),
            "error": self.error,
            "metadata": _jsonable(self.metadata),
            "approval": None if self.approval is None else _jsonable(self.approval),
            "revision": self.revision,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        return payload

    as_dict = to_dict

    @property
    def lifecycle_status(self) -> str:
        """返回 lifecycle status 的兼容名称。"""
        return self.status

    @property
    def qualification_status(self) -> str:
        """返回资格状态的兼容名称。"""
        return self.status

    @property
    def binding(self) -> EvidenceBinding | None:
        """返回证据绑定对象。"""
        return self.evidence_binding

    def __getitem__(self, key: str) -> Any:
        """允许调用方以字典下标读取记录字段。"""
        return self.to_dict()[key]


class ResearchRegistry:
    """以 JSON 文件保存实验记录和不可变生命周期历史。"""

    schema_version = 1

    def __init__(self, path: str | Path) -> None:
        """打开注册表；首次使用时创建最小 JSON 文档。"""
        self.path = Path(path)
        self._data = self._read_or_create()

    def _empty_data(self) -> dict[str, Any]:
        """返回新注册表的空结构。"""
        return {"schema_version": self.schema_version, "experiments": {}, "events": []}

    def _read_or_create(self) -> dict[str, Any]:
        """读取注册表或以原子方式创建空文件。"""
        if not self.path.exists():
            data = self._empty_data()
            self._atomic_write(data)
            return data
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise RegistryCorruptionError(f"无法读取注册表：{self.path}") from exc
        self._validate_store(data)
        return data

    def _validate_store(self, data: Any) -> None:
        """校验注册表顶层结构和所有当前记录。"""
        if not isinstance(data, Mapping):
            raise RegistryCorruptionError("注册表根节点必须是对象")
        if data.get("schema_version") != self.schema_version:
            raise RegistryCorruptionError(
                f"不支持的注册表 schema_version：{data.get('schema_version')}"
            )
        if not isinstance(data.get("experiments"), Mapping):
            raise RegistryCorruptionError("注册表 experiments 必须是对象")
        if not isinstance(data.get("events"), list):
            raise RegistryCorruptionError("注册表 events 必须是数组")
        for experiment_id, entry in data["experiments"].items():
            if not isinstance(entry, Mapping):
                raise RegistryCorruptionError(f"实验 {experiment_id} 的条目损坏")
            if "current" not in entry or "history" not in entry or "events" not in entry:
                raise RegistryCorruptionError(f"实验 {experiment_id} 缺少历史结构")
            ExperimentRecord.from_dict(entry["current"])
            if not isinstance(entry["history"], list) or not isinstance(entry["events"], list):
                raise RegistryCorruptionError(f"实验 {experiment_id} 的历史结构损坏")
            for snapshot in entry["history"]:
                ExperimentRecord.from_dict(snapshot)

    def _atomic_write(self, data: Mapping[str, Any]) -> None:
        """以临时文件、flush、fsync 和原子替换持久化注册表。"""
        self._validate_store(data)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_name = handle.name
                handle.write(_canonical_json(data))
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

        try:
            with self.path.open("r", encoding="utf-8") as handle:
                written = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise RegistryCorruptionError("注册表写入后无法读回") from exc
        if written != data:
            raise RegistryCorruptionError("注册表写入后读回内容不一致")

    def refresh(self) -> None:
        """从磁盘重新读取注册表，不修改任何记录。"""
        self._data = self._read_or_create()

    @staticmethod
    def _clone(data: Mapping[str, Any]) -> dict[str, Any]:
        """复制当前 JSON 文档用于事务式变更。"""
        return _copy_json(data)

    @staticmethod
    def _coerce_record(
        experiment: ExperimentRecord | Mapping[str, Any] | str | None,
        values: Mapping[str, Any],
    ) -> ExperimentRecord:
        """把多种注册入口统一为实验记录。"""
        if isinstance(experiment, ExperimentRecord):
            payload = experiment.to_dict()
        elif isinstance(experiment, Mapping):
            payload = dict(experiment)
        elif isinstance(experiment, str):
            payload = {"experiment_id": experiment}
        elif experiment is None:
            payload = {}
        else:
            raise RegistryError("experiment 必须是记录、映射、ID 或 None")
        payload.update(values)
        if "experiment_id" not in payload and "id" not in payload:
            payload["experiment_id"] = uuid.uuid4().hex
        return ExperimentRecord.from_dict(payload)

    @staticmethod
    def _entry(record: ExperimentRecord, event: Mapping[str, Any]) -> dict[str, Any]:
        """构造包含当前快照和历史数组的新实验条目。"""
        return {"current": record.to_dict(), "history": [], "events": [dict(event)]}

    @staticmethod
    def _binding_key(record: ExperimentRecord) -> tuple[str, ...] | None:
        """返回记录绑定键；未完整绑定时返回空值。"""
        return None if record.evidence_binding is None else record.evidence_binding.key

    def _get_entry(self, experiment_id: str) -> dict[str, Any]:
        """读取实验条目，不暴露内部可变对象。"""
        key = _require_text(experiment_id, "experiment_id")
        try:
            entry = self._data["experiments"][key]
        except KeyError as exc:
            raise ExperimentNotFoundError(f"实验不存在：{key}") from exc
        return entry

    def _transition(
        self,
        experiment_id: str,
        *,
        patch: Mapping[str, Any],
        event_name: str,
        reason: str = "",
        details: Mapping[str, Any] | None = None,
    ) -> ExperimentRecord:
        """追加一个新快照并原子提交状态变更。"""
        entry = self._get_entry(experiment_id)
        current = ExperimentRecord.from_dict(entry["current"])
        values = current.to_dict()
        values.update(_copy_json(patch))
        values["experiment_id"] = current.experiment_id
        values["revision"] = current.revision + 1
        values["created_at"] = current.created_at
        values["updated_at"] = _now()
        next_record = ExperimentRecord.from_dict(values)
        event = {
            "event": event_name,
            "experiment_id": current.experiment_id,
            "at": next_record.updated_at,
            "from_status": current.status,
            "to_status": next_record.status,
            "revision": next_record.revision,
            "reason": str(reason),
            "details": {} if details is None else _jsonable(details),
        }
        new_data = self._clone(self._data)
        new_entry = new_data["experiments"][current.experiment_id]
        new_entry["history"].append(current.to_dict())
        new_entry["current"] = next_record.to_dict()
        new_entry["events"].append(event)
        new_data["events"].append(event)
        self._atomic_write(new_data)
        self._data = new_data
        return next_record

    def register(
        self,
        experiment: ExperimentRecord | Mapping[str, Any] | str | None = None,
        **values: Any,
    ) -> ExperimentRecord:
        """登记实验；失败实验也必须用 run_status=FAILED 保存。"""
        record = self._coerce_record(experiment, values)
        if record.experiment_id in self._data["experiments"]:
            raise DuplicateExperimentError(f"实验 ID 已存在：{record.experiment_id}")
        if record.status == ELIGIBLE:
            raise ApprovalRequiredError(
                "普通 register 不能直接创建 ELIGIBLE，必须调用显式审批方法"
            )
        if record.status not in {RESEARCH, SHADOW}:
            raise InvalidTransitionError("新实验只能从 RESEARCH 或明确的 SHADOW 状态登记")
        event = {
            "event": "REGISTERED",
            "experiment_id": record.experiment_id,
            "at": record.created_at,
            "from_status": None,
            "to_status": record.status,
            "revision": record.revision,
            "reason": "",
            "details": {"run_status": record.run_status},
        }
        new_data = self._clone(self._data)
        new_data["experiments"][record.experiment_id] = self._entry(record, event)
        new_data["events"].append(event)
        self._atomic_write(new_data)
        self._data = new_data
        return record

    def register_experiment(
        self,
        experiment: ExperimentRecord | Mapping[str, Any] | str | None = None,
        **values: Any,
    ) -> ExperimentRecord:
        """登记实验的显式别名。"""
        return self.register(experiment, **values)

    def update(
        self,
        experiment_id: str,
        patch: Mapping[str, Any] | None = None,
        **changes: Any,
    ) -> ExperimentRecord:
        """追加实验修订；普通更新永远不能把研究状态改为 ELIGIBLE。"""
        entry = self._get_entry(experiment_id)
        current = ExperimentRecord.from_dict(entry["current"])
        values: dict[str, Any] = {}
        if patch is not None:
            if not isinstance(patch, Mapping):
                raise RegistryError("patch 必须是对象")
            values.update(patch)
        values.update(changes)
        if not values:
            raise RegistryError("update 必须包含至少一个字段")
        if "lifecycle_status" in values and "status" not in values:
            values["status"] = values.pop("lifecycle_status")
        if "qualification_status" in values and "status" not in values:
            values["status"] = values.pop("qualification_status")
        if isinstance(values.get("evidence_binding"), EvidenceBinding):
            values["evidence_binding"] = values["evidence_binding"].to_dict()
        requested_status = str(values.get("status", current.status)).upper()
        if current.status == ELIGIBLE:
            raise ImmutableExperimentError(
                "ELIGIBLE 记录不可通过普通 update 覆盖；请登记新的实验版本"
            )
        if requested_status == ELIGIBLE:
            raise ApprovalRequiredError(
                "普通 update 不能直接进入 ELIGIBLE，必须调用 approve_eligibility"
            )
        if requested_status in {SUSPENDED, RETIRED}:
            raise InvalidTransitionError("请使用 suspend 或 retire 记录生命周期事件")
        if current.status in {SUSPENDED, RETIRED}:
            raise ImmutableExperimentError("SUSPENDED/RETIRED 记录不可通过普通 update 修改")

        binding_fields = set(_BINDING_FIELDS)
        if binding_fields.intersection(values):
            values.setdefault("evidence_binding", None)
        merged = current.to_dict()
        merged.update(_copy_json(values))
        merged["status"] = requested_status
        candidate = ExperimentRecord.from_dict(merged)
        if self._binding_key(candidate) != self._binding_key(current):
            candidate_payload = candidate.to_dict()
            candidate_payload["status"] = RESEARCH
            candidate_payload["approval"] = None
            values = candidate_payload
            values.pop("revision", None)
            values.pop("created_at", None)
            values.pop("updated_at", None)
            return self._transition(
                experiment_id,
                patch=values,
                event_name="BINDING_REVISED",
                reason="版本或适用范围变化，不能继承原资格",
            )
        return self._transition(
            experiment_id,
            patch=values,
            event_name="UPDATED",
            reason="实验结果或元数据修订",
        )

    def enter_shadow(self, experiment_id: str, reason: str = "") -> ExperimentRecord:
        """将研究实验显式移入 SHADOW，不产生交易资格。"""
        current = self.get(experiment_id)
        if current.status != RESEARCH:
            raise InvalidTransitionError("只有 RESEARCH 可以进入 SHADOW")
        return self._transition(
            experiment_id,
            patch={"status": SHADOW},
            event_name="ENTERED_SHADOW",
            reason=reason,
        )

    promote_to_shadow = enter_shadow

    def approve_eligibility(
        self,
        experiment_id: str,
        approval_token: str | None = None,
        approval_method: str | None = None,
        evidence_binding: EvidenceBinding | Mapping[str, Any] | None = None,
        *,
        token: str | None = None,
        method: str | None = None,
        binding: EvidenceBinding | Mapping[str, Any] | None = None,
    ) -> ExperimentRecord:
        """用显式 token、方法和精确绑定审批进入 ELIGIBLE。"""
        current = self.get(experiment_id)
        if current.status not in {RESEARCH, SHADOW}:
            raise InvalidTransitionError("只有 RESEARCH/SHADOW 可以申请 ELIGIBLE")
        approval_token = approval_token if approval_token is not None else token
        approval_method = approval_method if approval_method is not None else method
        if evidence_binding is None:
            evidence_binding = binding
        if not isinstance(approval_token, str) or not approval_token.strip():
            raise ApprovalRequiredError("进入 ELIGIBLE 必须提供非空 approval token")
        if not isinstance(approval_method, str) or not approval_method.strip():
            raise ApprovalRequiredError("进入 ELIGIBLE 必须提供非空 approval method")
        if current.run_status != RUN_SUCCEEDED:
            raise InvalidTransitionError("失败或取消实验不能进入 ELIGIBLE")
        if current.evidence_binding is None:
            raise BindingMismatchError("实验没有完整 evidence binding，不能进入 ELIGIBLE")
        supplied = (
            current.evidence_binding
            if evidence_binding is None
            else EvidenceBinding.from_mapping(evidence_binding)
        )
        if not current.evidence_binding.matches(supplied):
            raise BindingMismatchError("审批 evidence binding 与实验版本或范围不一致")
        approval = {
            "approval_token": approval_token,
            "approval_method": approval_method,
            "approved_at": _now(),
            "evidence_binding": supplied.to_dict(),
        }
        return self._transition(
            experiment_id,
            patch={"status": ELIGIBLE, "approval": approval},
            event_name="APPROVED_ELIGIBLE",
            reason="显式资格审批",
            details={"approval_method": approval_method, "evidence_binding": supplied.to_dict()},
        )

    def approve(
        self,
        experiment_id: str,
        approval_token: str | None = None,
        approval_method: str | None = None,
        evidence_binding: EvidenceBinding | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> ExperimentRecord:
        """approve_eligibility 的简短别名。"""
        return self.approve_eligibility(
            experiment_id,
            approval_token=approval_token,
            approval_method=approval_method,
            evidence_binding=evidence_binding,
            **kwargs,
        )

    qualify = approve

    def suspend(self, experiment_id: str, reason: str = "") -> ExperimentRecord:
        """暂停实验资格并保存暂停前状态和原因。"""
        current = self.get(experiment_id)
        if current.status in {SUSPENDED, RETIRED}:
            raise InvalidTransitionError("当前实验不能重复暂停")
        if not isinstance(reason, str):
            raise RegistryError("suspend reason 必须是字符串")
        metadata = dict(current.metadata)
        metadata["_registry_previous_status"] = current.status
        metadata["suspension_reason"] = reason
        return self._transition(
            experiment_id,
            patch={"status": SUSPENDED, "metadata": metadata},
            event_name="SUSPENDED",
            reason=reason,
        )

    def reinstate(
        self,
        experiment_id: str,
        reason: str = "",
        approval_token: str | None = None,
        approval_method: str | None = None,
        evidence_binding: EvidenceBinding | Mapping[str, Any] | None = None,
        *,
        token: str | None = None,
        method: str | None = None,
        binding: EvidenceBinding | Mapping[str, Any] | None = None,
    ) -> ExperimentRecord:
        """恢复暂停记录；恢复到 ELIGIBLE 仍需重新显式审批。"""
        current = self.get(experiment_id)
        if current.status != SUSPENDED:
            raise InvalidTransitionError("只有 SUSPENDED 可以 reinstate")
        metadata = dict(current.metadata)
        previous_status = str(metadata.pop("_registry_previous_status", RESEARCH))
        metadata.pop("suspension_reason", None)
        if previous_status == ELIGIBLE:
            return self.approve_suspended(
                experiment_id,
                metadata=metadata,
                reason=reason,
                approval_token=approval_token if approval_token is not None else token,
                approval_method=approval_method if approval_method is not None else method,
                evidence_binding=evidence_binding if evidence_binding is not None else binding,
            )
        if previous_status not in {RESEARCH, SHADOW}:
            previous_status = RESEARCH
        return self._transition(
            experiment_id,
            patch={"status": previous_status, "metadata": metadata, "approval": None},
            event_name="REINSTATED",
            reason=reason,
        )

    def approve_suspended(
        self,
        experiment_id: str,
        *,
        metadata: Mapping[str, Any],
        reason: str,
        approval_token: str | None,
        approval_method: str | None,
        evidence_binding: EvidenceBinding | Mapping[str, Any] | None,
    ) -> ExperimentRecord:
        """在暂停记录上完成重新审批并恢复 ELIGIBLE。"""
        current = self.get(experiment_id)
        if current.status != SUSPENDED:
            raise InvalidTransitionError("只有 SUSPENDED 可以重新审批")
        if not isinstance(approval_token, str) or not approval_token.strip():
            raise ApprovalRequiredError("恢复 ELIGIBLE 必须提供非空 approval token")
        if not isinstance(approval_method, str) or not approval_method.strip():
            raise ApprovalRequiredError("恢复 ELIGIBLE 必须提供非空 approval method")
        if current.run_status != RUN_SUCCEEDED or current.evidence_binding is None:
            raise BindingMismatchError("暂停记录缺少可恢复的完整证据绑定")
        supplied = (
            current.evidence_binding
            if evidence_binding is None
            else EvidenceBinding.from_mapping(evidence_binding)
        )
        if not current.evidence_binding.matches(supplied):
            raise BindingMismatchError("恢复审批的 evidence binding 不匹配")
        approval = {
            "approval_token": approval_token,
            "approval_method": approval_method,
            "approved_at": _now(),
            "evidence_binding": supplied.to_dict(),
        }
        return self._transition(
            experiment_id,
            patch={"status": ELIGIBLE, "metadata": dict(metadata), "approval": approval},
            event_name="REINSTATED_ELIGIBLE",
            reason=reason,
        )

    def retire(self, experiment_id: str, reason: str = "") -> ExperimentRecord:
        """永久退役实验，保留退役前快照和原因。"""
        current = self.get(experiment_id)
        if current.status == RETIRED:
            raise InvalidTransitionError("实验已经 RETIRED")
        if not isinstance(reason, str):
            raise RegistryError("retire reason 必须是字符串")
        metadata = dict(current.metadata)
        metadata["retirement_reason"] = reason
        return self._transition(
            experiment_id,
            patch={"status": RETIRED, "metadata": metadata},
            event_name="RETIRED",
            reason=reason,
        )

    approve_eligible = approve_eligibility
    suspend_strategy = suspend
    reinstate_strategy = reinstate
    retire_strategy = retire

    def get(self, experiment_id: str) -> ExperimentRecord:
        """读取实验当前快照。"""
        entry = self._get_entry(experiment_id)
        return ExperimentRecord.from_dict(entry["current"])

    def get_experiment(self, experiment_id: str) -> ExperimentRecord:
        """读取实验当前快照的显式别名。"""
        return self.get(experiment_id)

    def list_experiments(
        self,
        *,
        status: str | None = None,
        run_status: str | None = None,
    ) -> list[ExperimentRecord]:
        """列出当前快照，可按生命周期或运行结果筛选。"""
        expected_status = None if status is None else str(status).upper()
        expected_run = None if run_status is None else str(run_status).upper()
        records = [
            ExperimentRecord.from_dict(entry["current"])
            for entry in self._data["experiments"].values()
        ]
        if expected_status is not None:
            records = [record for record in records if record.status == expected_status]
        if expected_run is not None:
            records = [record for record in records if record.run_status == expected_run]
        return records

    def list(self, **filters: Any) -> list[ExperimentRecord]:
        """list_experiments 的简短别名。"""
        return self.list_experiments(**filters)

    def history(self, experiment_id: str) -> list[ExperimentRecord]:
        """返回实验的全部版本快照，包含当前版本且不覆盖旧记录。"""
        entry = self._get_entry(experiment_id)
        snapshots = list(entry["history"]) + [entry["current"]]
        return [ExperimentRecord.from_dict(snapshot) for snapshot in snapshots]

    def history_events(self, experiment_id: str) -> list[dict[str, Any]]:
        """返回实验生命周期事件的副本。"""
        entry = self._get_entry(experiment_id)
        return _copy_json(entry["events"])

    def events(self, experiment_id: str | None = None) -> list[dict[str, Any]]:
        """返回全局事件或指定实验事件的副本。"""
        if experiment_id is None:
            return _copy_json(self._data["events"])
        return self.history_events(experiment_id)

    def qualification(
        self,
        experiment_id: str,
        evidence_binding: EvidenceBinding | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """读取资格快照；版本或范围不匹配时明确返回不可用。"""
        record = self.get(experiment_id)
        binding_matches = (
            evidence_binding is None
            or (
                record.evidence_binding is not None
                and record.evidence_binding.matches(evidence_binding)
            )
        )
        eligible = record.status == ELIGIBLE and binding_matches
        reason = "" if eligible else "状态或 evidence binding 不满足"
        return {
            "experiment_id": record.experiment_id,
            "status": record.status,
            "eligible": eligible,
            "binding_matches": binding_matches,
            "evidence_binding": (
                None if record.evidence_binding is None else record.evidence_binding.to_dict()
            ),
            "revision": record.revision,
            "reason": reason,
        }

    def is_eligible(
        self,
        experiment_id: str,
        evidence_binding: EvidenceBinding | Mapping[str, Any] | None = None,
    ) -> bool:
        """判断当前记录是否在给定精确绑定下处于 ELIGIBLE。"""
        return bool(self.qualification(experiment_id, evidence_binding)["eligible"])

    qualification_snapshot = qualification


Registry = ResearchRegistry
"""注册表类别的简短兼容名称。"""

QualificationBinding = EvidenceBinding
"""证据绑定类别的兼容名称。"""

Experiment = ExperimentRecord
"""实验记录类别的简短兼容名称。"""

__all__ = [
    "ApprovalRequiredError",
    "BindingMismatchError",
    "DuplicateExperimentError",
    "ELIGIBLE",
    "EvidenceBinding",
    "Experiment",
    "ExperimentNotFoundError",
    "ExperimentRecord",
    "ImmutableExperimentError",
    "InvalidTransitionError",
    "LIFECYCLE_STATES",
    "QualificationBinding",
    "RESEARCH",
    "RETIRED",
    "RUN_CANCELLED",
    "RUN_FAILED",
    "RUN_STATES",
    "RUN_SUCCEEDED",
    "Registry",
    "RegistryCorruptionError",
    "RegistryError",
    "ResearchRegistry",
    "SHADOW",
    "SUSPENDED",
]
