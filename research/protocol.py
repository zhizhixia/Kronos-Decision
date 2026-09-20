"""研究协议的最小治理原语。

本模块只负责协议的时间、期限、成本、预算和版本完整性校验，不执行回测、
资格判断或投资门槛判断。协议对象以不可变数据保存，冻结协议的修订必须产生
新的版本；协议文件使用标准库 JSON 保存，并携带规范化内容哈希。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import os
from tempfile import NamedTemporaryFile
from types import MappingProxyType
from typing import Any, Mapping, Sequence


SUPPORTED_HORIZONS = frozenset({5, 10, 20})
"""当前研究协议允许的交易日周期。"""

_CANONICAL_JSON_KWARGS = {
    "ensure_ascii": False,
    "sort_keys": True,
    "separators": (",", ":"),
    "allow_nan": False,
}


class ProtocolError(ValueError):
    """研究协议无效、损坏或修订方式不受支持。"""


class ProtocolValidationError(ProtocolError):
    """研究协议字段未通过约束校验。"""


class ProtocolIntegrityError(ProtocolError):
    """研究协议的保存哈希与内容不一致。"""


class FrozenProtocolError(ProtocolError):
    """尝试原地修改或非法替换已冻结协议。"""


def _jsonable(value: Any) -> Any:
    """把内部不可变值转换为可由 JSON 标准库编码的值。"""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ProtocolValidationError("JSON 字段不能包含非有限浮点数")
        return value
    raise ProtocolValidationError(f"字段包含不可序列化类型：{type(value).__name__}")


def _immutable_json(value: Any) -> Any:
    """递归冻结 JSON 兼容值，避免冻结对象内部的映射被原地修改。"""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _immutable_json(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_immutable_json(item) for item in value)
    if isinstance(value, float) and not isfinite(value):
        raise ProtocolValidationError("JSON 字段不能包含非有限浮点数")
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise ProtocolValidationError(f"字段包含不可序列化类型：{type(value).__name__}")


def _normalise_time(value: Any, field_name: str) -> str:
    """将日期或时间值规范化为可比较的 ISO 字符串。"""
    if isinstance(value, datetime):
        parsed = value
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        return parsed.isoformat(timespec="seconds")
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str) or not value.strip():
        raise ProtocolValidationError(f"{field_name} 必须是日期、时间或 ISO 字符串")

    text = value.strip()
    if "T" not in text and " " not in text:
        try:
            return date.fromisoformat(text).isoformat()
        except ValueError as exc:
            raise ProtocolValidationError(f"{field_name} 不是有效日期：{value}") from exc

    candidate = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ProtocolValidationError(f"{field_name} 不是有效时间：{value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat(timespec="seconds")


def _time_key(value: str) -> datetime:
    """把规范化时间转换为统一时区的比较值。"""
    if "T" not in value:
        return datetime.combine(
            date.fromisoformat(value), datetime.min.time(), tzinfo=timezone.utc
        )
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _day_key(value: Any, field_name: str) -> date:
    """把输入时点转换为日级边界，避免日线数据被时分误判。"""
    return _time_key(_normalise_time(value, field_name)).date()


def _normalise_number(value: Any, field_name: str) -> float:
    """校验并规范化非负有限数值。"""
    if isinstance(value, bool):
        raise ProtocolValidationError(f"{field_name} 必须是非负数，不能是布尔值")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolValidationError(f"{field_name} 必须是数值") from exc
    if not isfinite(number) or number < 0:
        raise ProtocolValidationError(f"{field_name} 必须是非负有限数值")
    return number


def _normalise_text(value: Any, field_name: str) -> str:
    """校验兼容字段中的非空文本。"""
    if not isinstance(value, str) or not value:
        raise ProtocolValidationError(f"{field_name} 必须是非空字符串")
    return value


def _normalise_horizons(value: Any) -> tuple[int, ...]:
    """校验研究周期，并以有序不可变元组保存。"""
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
        value = pieces
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise ProtocolValidationError("horizons 必须是 5/10/20 的序列")
    result: list[int] = []
    for horizon in value:
        if isinstance(horizon, bool) or not isinstance(horizon, int):
            raise ProtocolValidationError("horizons 只能包含整数 5、10、20")
        if horizon not in SUPPORTED_HORIZONS:
            raise ProtocolValidationError(
                f"不支持的 horizon={horizon}，仅支持 5、10、20"
            )
        if horizon in result:
            raise ProtocolValidationError(f"horizons 不能重复：{horizon}")
        result.append(horizon)
    if not result:
        raise ProtocolValidationError("horizons 不能为空")
    return tuple(sorted(result))


def _normalise_scope(value: Any, field_name: str) -> tuple[str, ...]:
    """将证券范围或股票池范围规范化为不可变字符串元组。"""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise ProtocolValidationError(f"{field_name} 必须是字符串或字符串序列")
    values = tuple(str(item) for item in value)
    if any(not item for item in values):
        raise ProtocolValidationError(f"{field_name} 不能包含空值")
    return values


def _choose_alias(
    primary: Any,
    alias: Any,
    field_name: str,
    normaliser: Any,
) -> Any:
    """在两个兼容字段名之间选择一个值，并拒绝冲突输入。"""
    if primary is None:
        return alias
    if alias is None:
        return primary
    first = normaliser(primary, field_name)
    second = normaliser(alias, field_name)
    if first != second:
        raise ProtocolValidationError(f"{field_name} 的兼容字段值冲突")
    return primary


def canonical_json(payload: Mapping[str, Any]) -> str:
    """生成不含 canonical_hash 字段的规范 JSON 文本。"""
    if not isinstance(payload, Mapping):
        raise ProtocolValidationError("规范化内容必须是映射")
    clean = {key: value for key, value in payload.items() if key != "canonical_hash"}
    try:
        return json.dumps(_jsonable(clean), **_CANONICAL_JSON_KWARGS)
    except (TypeError, ValueError) as exc:
        raise ProtocolValidationError("协议内容无法规范化为 JSON") from exc


def canonical_hash(payload: Mapping[str, Any]) -> str:
    """计算协议内容的 SHA-256 规范哈希。"""
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """以同目录临时文件和原子替换写入 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(_jsonable(payload), handle, **_CANONICAL_JSON_KWARGS)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                temporary_name = None


@dataclass(frozen=True, init=False)
class ResearchProtocol:
    """不可变的研究协议，冻结后只能通过修订生成新版本。"""

    protocol_id: str
    version: str
    decision_as_of: str
    data_visible_boundary: str
    training_start: str
    training_end: str
    calibration_start: str | None
    calibration_end: str | None
    test_start: str
    test_end: str
    horizons: tuple[int, ...]
    costs: Mapping[str, float]
    budget: float
    security_scope: tuple[str, ...]
    stock_pool_version: str
    benchmark: str
    primary_metric: str
    strategy_id: str
    strategy_version: str
    model_version: str
    data_version: str
    qualification_criteria: Mapping[str, Any]
    metadata: Mapping[str, Any]
    frozen: bool
    parent_version: str | None
    schema_version: int
    canonical_hash: str

    def __init__(
        self,
        protocol_id: str = "protocol",
        version: str = "1",
        decision_as_of: Any = None,
        data_visible_boundary: Any = None,
        training_start: Any = None,
        training_end: Any = None,
        calibration_start: Any = None,
        calibration_end: Any = None,
        test_start: Any = None,
        test_end: Any = None,
        horizons: Any = (5, 10, 20),
        costs: Any = None,
        budget: Any = 0.0,
        security_scope: Any = (),
        stock_pool_version: str = "",
        benchmark: str = "",
        primary_metric: str = "",
        strategy_id: str = "",
        strategy_version: str = "",
        model_version: str = "",
        data_version: str = "",
        qualification_criteria: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        frozen: bool = False,
        parent_version: str | None = None,
        schema_version: int = 1,
        *,
        data_visible_until: Any = None,
        train_start: Any = None,
        train_end: Any = None,
        experiment_budget: Any = None,
        protocol_version: str | None = None,
        **aliases: Any,
    ) -> None:
        """创建并校验研究协议；兼容少量常用字段别名。"""
        if "id" in aliases:
            identifier = aliases.pop("id")
            if protocol_id == "protocol":
                protocol_id = identifier
            else:
                protocol_id = _choose_alias(protocol_id, identifier, "protocol_id", _normalise_text)
        if protocol_version is not None:
            if version != "1" and version != protocol_version:
                raise ProtocolValidationError("version 与 protocol_version 冲突")
            version = protocol_version
        if data_visible_until is not None:
            data_visible_boundary = _choose_alias(
                data_visible_boundary,
                data_visible_until,
                "data_visible_boundary",
                _normalise_time,
            )
        if train_start is not None:
            training_start = _choose_alias(
                training_start, train_start, "training_start", _normalise_time
            )
        if train_end is not None:
            training_end = _choose_alias(training_end, train_end, "training_end", _normalise_time)
        if experiment_budget is not None:
            budget = (
                experiment_budget
                if budget == 0.0
                else _choose_alias(budget, experiment_budget, "budget", _normalise_number)
            )

        alias_time_fields = {
            "visible_until": "data_visible_boundary",
            "data_as_of": "data_visible_boundary",
            "training_from": "training_start",
            "training_to": "training_end",
            "calibration_from": "calibration_start",
            "calibration_to": "calibration_end",
            "test_from": "test_start",
            "test_to": "test_end",
        }
        for alias_name, target_name in alias_time_fields.items():
            if alias_name not in aliases:
                continue
            alias_value = aliases.pop(alias_name)
            current = locals()[target_name]
            selected = _choose_alias(
                current, alias_value, target_name, _normalise_time
            )
            if target_name == "data_visible_boundary":
                data_visible_boundary = selected
            elif target_name == "training_start":
                training_start = selected
            elif target_name == "training_end":
                training_end = selected
            elif target_name == "calibration_start":
                calibration_start = selected
            elif target_name == "calibration_end":
                calibration_end = selected
            elif target_name == "test_start":
                test_start = selected
            elif target_name == "test_end":
                test_end = selected

        if "cost" in aliases:
            if costs is not None:
                raise ProtocolValidationError("cost 与 costs 不能同时提供")
            costs = {"total": aliases.pop("cost")}
        if "transaction_cost" in aliases:
            transaction_cost = aliases.pop("transaction_cost")
            if costs is None:
                costs = {"transaction_cost": transaction_cost}
            elif isinstance(costs, Mapping):
                costs = dict(costs)
                costs["transaction_cost"] = transaction_cost
            else:
                raise ProtocolValidationError("transaction_cost 不能与非映射 costs 合并")
        if "trial_budget" in aliases:
            trial_budget = aliases.pop("trial_budget")
            budget = (
                trial_budget
                if budget == 0.0
                else _choose_alias(budget, trial_budget, "budget", _normalise_number)
            )
        if "pool_version" in aliases:
            pool_version = aliases.pop("pool_version")
            stock_pool_version = (
                pool_version
                if stock_pool_version == ""
                else _choose_alias(
                    stock_pool_version,
                    pool_version,
                    "stock_pool_version",
                    _normalise_text,
                )
            )
        if "universe_version" in aliases:
            universe_version = aliases.pop("universe_version")
            stock_pool_version = (
                universe_version
                if stock_pool_version == ""
                else _choose_alias(
                    stock_pool_version,
                    universe_version,
                    "stock_pool_version",
                    _normalise_text,
                )
            )
        if aliases:
            unknown = ", ".join(sorted(aliases))
            raise TypeError(f"未知研究协议字段：{unknown}")

        if not isinstance(protocol_id, str) or not protocol_id:
            raise ProtocolValidationError("protocol_id 不能为空")
        if not isinstance(version, str) or not version:
            raise ProtocolValidationError("version 不能为空")
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            raise ProtocolValidationError("schema_version 必须是整数")
        if schema_version != 1:
            raise ProtocolValidationError(f"不支持的协议 schema_version：{schema_version}")

        normalised_decision = _normalise_time(decision_as_of, "decision_as_of")
        normalised_boundary = _normalise_time(
            data_visible_boundary, "data_visible_boundary"
        )
        normalised_training_start = _normalise_time(training_start, "training_start")
        normalised_training_end = _normalise_time(training_end, "training_end")
        normalised_calibration_start = (
            None
            if calibration_start is None
            else _normalise_time(calibration_start, "calibration_start")
        )
        normalised_calibration_end = (
            None
            if calibration_end is None
            else _normalise_time(calibration_end, "calibration_end")
        )
        normalised_test_start = _normalise_time(test_start, "test_start")
        normalised_test_end = _normalise_time(test_end, "test_end")

        if costs is None:
            costs = {}
        elif isinstance(costs, Mapping):
            costs = {
                str(key): _normalise_number(value, f"costs[{key!r}]")
                for key, value in costs.items()
            }
        else:
            costs = {"total": _normalise_number(costs, "costs")}
        if any(not key for key in costs):
            raise ProtocolValidationError("成本名称不能为空")

        normalised_budget = _normalise_number(budget, "budget")
        normalised_horizons = _normalise_horizons(horizons)
        normalised_scope = _normalise_scope(security_scope, "security_scope")

        object.__setattr__(self, "protocol_id", protocol_id)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "decision_as_of", normalised_decision)
        object.__setattr__(self, "data_visible_boundary", normalised_boundary)
        object.__setattr__(self, "training_start", normalised_training_start)
        object.__setattr__(self, "training_end", normalised_training_end)
        object.__setattr__(self, "calibration_start", normalised_calibration_start)
        object.__setattr__(self, "calibration_end", normalised_calibration_end)
        object.__setattr__(self, "test_start", normalised_test_start)
        object.__setattr__(self, "test_end", normalised_test_end)
        object.__setattr__(self, "horizons", normalised_horizons)
        object.__setattr__(self, "costs", MappingProxyType(dict(sorted(costs.items()))))
        object.__setattr__(self, "budget", normalised_budget)
        object.__setattr__(self, "security_scope", normalised_scope)
        object.__setattr__(self, "stock_pool_version", str(stock_pool_version))
        object.__setattr__(self, "benchmark", str(benchmark))
        object.__setattr__(self, "primary_metric", str(primary_metric))
        object.__setattr__(self, "strategy_id", str(strategy_id))
        object.__setattr__(self, "strategy_version", str(strategy_version))
        object.__setattr__(self, "model_version", str(model_version))
        object.__setattr__(self, "data_version", str(data_version))
        object.__setattr__(
            self,
            "qualification_criteria",
            _immutable_json(qualification_criteria or {}),
        )
        object.__setattr__(self, "metadata", _immutable_json(metadata or {}))
        object.__setattr__(self, "frozen", bool(frozen))
        object.__setattr__(self, "parent_version", parent_version)
        object.__setattr__(self, "schema_version", schema_version)
        self._validate_values()
        object.__setattr__(self, "canonical_hash", canonical_hash(self.canonical_payload))

    def _validate_values(self) -> None:
        """校验时间隔离、数据可见性和协议字段的内部不变量。"""
        decision = _time_key(self.decision_as_of)
        visible = _time_key(self.data_visible_boundary)
        if decision > visible:
            raise ProtocolValidationError(
                "decision_as_of 不能晚于 data_visible_boundary"
            )

        windows: dict[str, tuple[datetime, datetime]] = {
            "training": (_time_key(self.training_start), _time_key(self.training_end)),
            "test": (_time_key(self.test_start), _time_key(self.test_end)),
        }
        if windows["training"][0] > windows["training"][1]:
            raise ProtocolValidationError("training_start 不能晚于 training_end")
        if self.calibration_start is None and self.calibration_end is not None:
            raise ProtocolValidationError("calibration_start 缺失")
        if self.calibration_start is not None and self.calibration_end is None:
            raise ProtocolValidationError("calibration_end 缺失")
        if self.calibration_start is not None and self.calibration_end is not None:
            windows["calibration"] = (
                _time_key(self.calibration_start),
                _time_key(self.calibration_end),
            )
            if windows["calibration"][0] > windows["calibration"][1]:
                raise ProtocolValidationError(
                    "calibration_start 不能晚于 calibration_end"
                )
        if windows["test"][0] > windows["test"][1]:
            raise ProtocolValidationError("test_start 不能晚于 test_end")

        names = tuple(windows)
        for index, left_name in enumerate(names):
            for right_name in names[index + 1 :]:
                left_start, left_end = windows[left_name]
                right_start, right_end = windows[right_name]
                if left_start <= right_end and right_start <= left_end:
                    raise ProtocolValidationError(
                        f"{left_name} 与 {right_name} 时间窗口重叠"
                    )

    @property
    def data_visible_until(self) -> str:
        """返回数据可见边界的兼容名称。"""
        return self.data_visible_boundary

    @property
    def train_start(self) -> str:
        """返回训练窗口起点的兼容名称。"""
        return self.training_start

    @property
    def train_end(self) -> str:
        """返回训练窗口终点的兼容名称。"""
        return self.training_end

    @property
    def experiment_budget(self) -> float:
        """返回试验预算的兼容名称。"""
        return self.budget

    @property
    def protocol_version(self) -> str:
        """返回协议版本的兼容名称。"""
        return self.version

    @property
    def canonical_payload(self) -> dict[str, Any]:
        """返回不含哈希字段、可用于审计的规范内容。"""
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "version": self.version,
            "decision_as_of": self.decision_as_of,
            "data_visible_boundary": self.data_visible_boundary,
            "training_start": self.training_start,
            "training_end": self.training_end,
            "calibration_start": self.calibration_start,
            "calibration_end": self.calibration_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "horizons": list(self.horizons),
            "costs": dict(self.costs),
            "budget": self.budget,
            "security_scope": list(self.security_scope),
            "stock_pool_version": self.stock_pool_version,
            "benchmark": self.benchmark,
            "primary_metric": self.primary_metric,
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "model_version": self.model_version,
            "data_version": self.data_version,
            "qualification_criteria": _jsonable(self.qualification_criteria),
            "metadata": _jsonable(self.metadata),
            "frozen": self.frozen,
            "parent_version": self.parent_version,
        }

    def to_dict(self, include_hash: bool = True) -> dict[str, Any]:
        """返回适合 JSON 保存或跨模块传递的协议字典。"""
        payload = dict(self.canonical_payload)
        if include_hash:
            payload["canonical_hash"] = self.canonical_hash
        return payload

    def validate(self) -> None:
        """重新校验协议字段和已保存的规范哈希。"""
        self._validate_values()
        expected = canonical_hash(self.canonical_payload)
        if expected != self.canonical_hash:
            raise ProtocolIntegrityError("协议 canonical_hash 与内容不一致")

    def validate_evaluation_inputs(
        self,
        *,
        as_of: Any,
        price_dates: Sequence[Any],
        signal_dates: Sequence[Any],
        security_codes: Sequence[Any],
        stock_pool_versions: Sequence[Any] = (),
    ) -> str:
        """校验评价输入必须严格落在冻结协议的边界、窗口和证券范围内。"""
        self.validate()
        if not self.frozen:
            raise FrozenProtocolError("研究评价只能使用已冻结协议")

        actual_day = _day_key(as_of, "as_of")
        visible_day = _day_key(self.data_visible_boundary, "data_visible_boundary")
        if actual_day != visible_day:
            raise ProtocolValidationError(
                "实际 as_of 必须等于冻结协议的 data_visible_boundary"
            )
        test_start = _day_key(self.test_start, "test_start")
        test_end = _day_key(self.test_end, "test_end")
        if not test_start <= actual_day <= test_end:
            raise ProtocolValidationError("实际 as_of 超出冻结协议测试窗口")

        for field_name, values in (
            ("prices.date", price_dates),
            ("signals.signal_date", signal_dates),
        ):
            for value in values:
                day = _day_key(value, field_name)
                if day > visible_day:
                    raise ProtocolValidationError(
                        f"{field_name} 超出 data_visible_boundary：{value}"
                    )
                if day < test_start or day > test_end:
                    raise ProtocolValidationError(
                        f"{field_name} 超出冻结协议测试窗口：{value}"
                    )

        for code_value in security_codes:
            code = str(code_value).strip()
            if not code or not self._code_in_scope(code):
                raise ProtocolValidationError(f"证券不在冻结协议范围内：{code_value}")

        if stock_pool_versions:
            expected_pool = self.stock_pool_version
            if not expected_pool or any(str(value) != expected_pool for value in stock_pool_versions):
                raise ProtocolValidationError("输入股票池版本与冻结协议不一致")
        return self.data_visible_boundary

    def _code_in_scope(self, code: str) -> bool:
        """判断证券代码是否符合协议声明的证券范围。"""
        scopes = {str(value).strip() for value in self.security_scope}
        upper_scopes = {value.upper() for value in scopes}
        if code in scopes or code.upper() in upper_scopes or "*" in scopes or "ALL" in upper_scopes:
            return True
        a_share_scopes = {"A_SHARE", "A-SHARE", "CN_A_SHARE", "ASHARE"}
        return bool(upper_scopes & a_share_scopes) and code.isdigit() and len(code) == 6

    @property
    def is_frozen(self) -> bool:
        """返回协议是否已冻结。"""
        return self.frozen

    def freeze(self) -> "ResearchProtocol":
        """冻结当前协议；冻结只改变新对象，不原地修改当前实例。"""
        self.validate()
        if self.frozen:
            return self
        payload = self.to_dict(include_hash=False)
        payload["frozen"] = True
        return type(self).from_dict(payload, verify_hash=False)

    @staticmethod
    def _next_version(version: str) -> str:
        """根据常见版本格式生成下一修订版本。"""
        pieces = version.split(".")
        if pieces and all(piece.isdigit() for piece in pieces):
            pieces[-1] = str(int(pieces[-1]) + 1)
            return ".".join(pieces)
        if version.startswith("v") and version[1:].isdigit():
            return f"v{int(version[1:]) + 1}"
        return f"{version}.1"

    def revise(self, **changes: Any) -> "ResearchProtocol":
        """以新版本保存修订，原协议和原哈希保持不变。"""
        self.validate()
        if not changes:
            raise ProtocolError("修订必须至少包含一个字段")
        if "canonical_hash" in changes:
            raise FrozenProtocolError("canonical_hash 由内容自动生成，不能手工修改")
        requested_version = changes.pop("version", changes.pop("protocol_version", None))
        new_version = requested_version or self._next_version(self.version)
        if new_version == self.version:
            raise FrozenProtocolError("修订必须使用新版本")
        aliases = {
            "data_visible_until": "data_visible_boundary",
            "visible_until": "data_visible_boundary",
            "data_as_of": "data_visible_boundary",
            "train_start": "training_start",
            "train_end": "training_end",
            "training_from": "training_start",
            "training_to": "training_end",
            "calibration_from": "calibration_start",
            "calibration_to": "calibration_end",
            "test_from": "test_start",
            "test_to": "test_end",
            "pool_version": "stock_pool_version",
            "universe_version": "stock_pool_version",
            "experiment_budget": "budget",
            "trial_budget": "budget",
        }
        for alias, target in aliases.items():
            if alias not in changes:
                continue
            if target in changes:
                raise ProtocolValidationError(f"修订字段 {alias} 与 {target} 冲突")
            changes[target] = changes.pop(alias)
        if "cost" in changes:
            if "costs" in changes:
                raise ProtocolValidationError("修订字段 cost 与 costs 冲突")
            changes["costs"] = {"total": changes.pop("cost")}
        if "transaction_cost" in changes:
            transaction_cost = changes.pop("transaction_cost")
            revised_costs = dict(self.costs)
            revised_costs["transaction_cost"] = transaction_cost
            if "costs" in changes:
                if not isinstance(changes["costs"], Mapping):
                    raise ProtocolValidationError("修订 costs 必须是映射")
                revised_costs.update(changes["costs"])
            changes["costs"] = revised_costs
        payload = self.to_dict(include_hash=False)
        payload.update(changes)
        payload["version"] = new_version
        payload["parent_version"] = self.version
        payload["frozen"] = False
        return type(self).from_dict(payload, verify_hash=False)

    def with_changes(self, **changes: Any) -> "ResearchProtocol":
        """返回 revise 的语义别名，便于调用方明确这是版本化修改。"""
        return self.revise(**changes)

    def update(self, **changes: Any) -> "ResearchProtocol":
        """拒绝原地更新，并返回一个版本化修订对象。"""
        return self.revise(**changes)

    def save(self, path: str | Path) -> Path:
        """原子保存协议 JSON，并持久化 canonical_hash。"""
        self.validate()
        target = Path(path)
        if target.exists():
            existing = type(self).load(target)
            if existing.canonical_hash != self.canonical_hash:
                raise FrozenProtocolError(
                    "目标协议文件已有不同版本；请使用新路径保存修订，不能覆盖历史"
                )
        _atomic_write_json(target, self.to_dict())
        with target.open("r", encoding="utf-8") as handle:
            saved = json.load(handle)
        if saved != self.to_dict():
            raise ProtocolIntegrityError("协议写入后读回内容不一致")
        return target

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        verify_hash: bool = True,
    ) -> "ResearchProtocol":
        """从字典重建协议，并在存在哈希时核验内容完整性。"""
        if not isinstance(payload, Mapping):
            raise ProtocolValidationError("协议 JSON 根节点必须是对象")
        values = dict(payload)
        stored_hash = values.pop("canonical_hash", None)
        protocol = cls(**values)
        if verify_hash and stored_hash is not None and stored_hash != protocol.canonical_hash:
            raise ProtocolIntegrityError("协议 JSON 的 canonical_hash 不匹配")
        return protocol

    @classmethod
    def load(cls, path: str | Path) -> "ResearchProtocol":
        """从 JSON 文件读取并核验研究协议。"""
        target = Path(path)
        try:
            with target.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ProtocolError(f"无法读取协议文件：{target}") from exc
        return cls.from_dict(payload)


def validate_protocol(protocol: ResearchProtocol | Mapping[str, Any]) -> ResearchProtocol:
    """校验并返回研究协议对象。"""
    value = (
        protocol
        if isinstance(protocol, ResearchProtocol)
        else ResearchProtocol.from_dict(protocol)
    )
    value.validate()
    return value


def save_protocol(protocol: ResearchProtocol, path: str | Path) -> Path:
    """保存研究协议的函数式入口。"""
    return protocol.save(path)


def load_protocol(path: str | Path) -> ResearchProtocol:
    """读取研究协议的函数式入口。"""
    return ResearchProtocol.load(path)


EvaluationProtocol = ResearchProtocol
"""与计划文档名称兼容的协议类别名。"""

Protocol = ResearchProtocol
"""简短协议类别名。"""

__all__ = [
    "EvaluationProtocol",
    "FrozenProtocolError",
    "Protocol",
    "ProtocolError",
    "ProtocolIntegrityError",
    "ProtocolValidationError",
    "ResearchProtocol",
    "SUPPORTED_HORIZONS",
    "canonical_hash",
    "canonical_json",
    "load_protocol",
    "save_protocol",
    "validate_protocol",
]
