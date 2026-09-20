"""配置加载、快照更新与原子持久化。

所有模块通过 ``get_config()`` 获取配置单例。更新时先复制完整配置，
只有校验和落盘都成功后才发布新的单例，避免失败请求污染运行中的配置。
"""
from __future__ import annotations

import dataclasses
import math
import os
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, TypeVar, get_type_hints

import yaml

from decision.errors import ConfigError

_CONFIG_DIR = Path(__file__).parent
_CONFIG_PATH = _CONFIG_DIR / "config.yaml"
_lock = Lock()
_config_instance: Config | None = None
ConfigType = TypeVar("ConfigType")


@dataclass
class ModelConfig:
    tokenizer: str = "NeoQuasar/Kronos-Tokenizer-base"
    predictor: str = "NeoQuasar/Kronos-small"
    tokenizer_revision: str = "0e0117387f39004a9016484a186a908917e22426"
    model_revision: str = "901c26c1332695a2a8f243eb2f37243a37bea320"
    max_context: int = 512
    device: str = "auto"
    idle_timeout_minutes: int = 10
    allow_model_download: bool = False


@dataclass
class PredictionConfig:
    default_pred_len: int = 60
    default_sample_count: int = 100
    default_temperature: float = 0.6
    default_top_p: float = 0.9
    timeout_seconds: int = 60
    fallback_sample_count: int = 10


@dataclass
class SignalConfig:
    trend_strength_buy_threshold: float = 0.7
    trend_strength_sell_threshold: float = 0.4
    var_risk_high_threshold: float = 0.05
    consistency_std_threshold: float = 0.15
    reversal_rsi_oversold: int = 30
    reversal_rsi_overbought: int = 70


@dataclass
class DataConfig:
    cache_dir: str = "data/cache"
    snapshot_dir: str = "data/snapshots"
    report_db: str = "data/reports/decision.sqlite"
    portfolio_ledger: str = "data/user/portfolio_snapshots.json"
    cache_ttl_hours: int = 24
    retry_max: int = 3
    retry_backoff_seconds: list[int] = field(default_factory=lambda: [2, 4, 8])
    primary_source: str = "akshare"
    backup_source: str = "baostock"
    optional_sources: list[str] = field(default_factory=list)
    tushare_token_env: str = "TUSHARE_TOKEN"


@dataclass
class WebUIConfig:
    host: str = "127.0.0.1"
    port: int = 7070
    stock_pool: str = "沪深300"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "logs/kronos.log"
    max_size_mb: int = 50
    backup_count: int = 5


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    data: DataConfig = field(default_factory=DataConfig)
    webui: WebUIConfig = field(default_factory=WebUIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _field_names(cls: type[Any]) -> set[str]:
    """返回 dataclass 声明的字段名集合。"""
    return {item.name for item in dataclasses.fields(cls)}


def _field_path(prefix: str, name: str) -> str:
    """拼接配置字段路径，不携带字段值。"""
    return f"{prefix}.{name}" if prefix else name


def _dict_to_dataclass(
    cls: type[ConfigType], data: object, path: str = ""
) -> ConfigType:
    """严格将映射递归转换为 dataclass，拒绝未知字段和错误分组。"""
    if not isinstance(data, Mapping):
        location = path or "配置根节点"
        raise ConfigError(f"{location}必须为映射对象。")

    field_types = get_type_hints(cls)
    names = _field_names(cls)
    kwargs: dict[str, object] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise ConfigError("配置字段名必须为文本。")
        if key not in names:
            raise ConfigError(f"未知配置项：{_field_path(path, key)}。")
        expected_type = field_types[key]
        child_path = _field_path(path, key)
        if isinstance(expected_type, type) and dataclasses.is_dataclass(expected_type):
            kwargs[key] = _dict_to_dataclass(expected_type, value, child_path)
        else:
            try:
                kwargs[key] = deepcopy(value)
            except Exception as exc:
                raise ConfigError("配置快照创建失败。") from exc

    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ConfigError("配置结构无效。") from exc


def _build_config(raw: object) -> Config:
    """将 YAML 内容构造成完整配置对象并执行全部校验。"""
    config = _dict_to_dataclass(Config, raw)
    validate_config(config)
    return config


def _load_raw_config() -> object:
    """从 YAML 文件加载原始配置，屏蔽可能泄露内容的解析异常。"""
    if not _CONFIG_PATH.exists():
        return {}
    try:
        with _CONFIG_PATH.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError("配置文件读取失败。") from exc


def _get_config_locked() -> Config:
    """在已持有锁时获取或初始化配置单例。"""
    global _config_instance
    if _config_instance is None:
        _config_instance = _build_config(_load_raw_config())
    return _config_instance


def get_config() -> Config:
    """获取全局配置单例，首次读取和初始化过程线程安全。"""
    with _lock:
        return _get_config_locked()


def reload_config() -> Config:
    """强制从磁盘重新加载配置，失败时保留原内存单例。"""
    global _config_instance
    with _lock:
        candidate = _build_config(_load_raw_config())
        _config_instance = candidate
        return candidate


def _clone_config(config: Config) -> Config:
    """创建待校验、待发布的完整配置快照。"""
    if not isinstance(config, Config):
        raise ConfigError("配置对象类型无效。")
    try:
        return deepcopy(config)
    except Exception as exc:
        raise ConfigError("配置快照创建失败。") from exc


def _apply_updates(config: Config, updates: Mapping[str, Mapping[str, object]]) -> None:
    """把分组式部分更新应用到配置副本，拒绝未知路径和值容器。"""
    if not isinstance(updates, Mapping):
        raise ConfigError("配置更新必须是映射对象。")

    sections = _field_names(Config)
    for section, values in updates.items():
        if not isinstance(section, str):
            raise ConfigError("配置分组名必须为文本。")
        if section not in sections:
            raise ConfigError(f"未知配置分组：{section}。")
        if not isinstance(values, Mapping):
            raise ConfigError(f"配置分组必须为映射对象：{section}。")

        section_config = getattr(config, section)
        fields = _field_names(type(section_config))
        for key, value in values.items():
            if not isinstance(key, str):
                raise ConfigError(f"配置字段名必须为文本：{section}。")
            if key not in fields:
                raise ConfigError(f"未知配置项：{section}.{key}。")
            try:
                setattr(section_config, key, deepcopy(value))
            except Exception as exc:
                raise ConfigError("配置快照创建失败。") from exc


def update_config(updates: Mapping[str, Mapping[str, object]]) -> Config:
    """基于当前配置快照更新并原子发布，失败时不改变当前配置。"""
    with _lock:
        candidate = _clone_config(_get_config_locked())
        _apply_updates(candidate, updates)
        _save_config_locked(candidate)
        return candidate


def _atomic_write(payload: str) -> None:
    """使用同目录临时文件、刷新和原子替换写入配置。"""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=_CONFIG_PATH.parent,
            prefix=f".{_CONFIG_PATH.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, _CONFIG_PATH)
        temporary = None
    except ConfigError:
        raise
    except OSError:
        raise
    except Exception as exc:
        raise ConfigError("配置保存失败，原配置未改变。") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                # 临时路径已不存在，清空清理状态即可。
                temporary = None
            except OSError:
                # 清理异常不能覆盖原始保存结果，也不回显路径或配置内容。
                temporary = None


def _save_config_locked(config: Config) -> None:
    """在已持有锁时校验、序列化并发布完整配置。"""
    validate_config(config)
    try:
        data = dataclasses.asdict(config)
        payload = yaml.safe_dump(data, allow_unicode=True, default_flow_style=False)
    except Exception as exc:
        raise ConfigError("配置序列化失败，原配置未改变。") from exc

    _atomic_write(payload)
    global _config_instance
    _config_instance = config


def save_config(config: Config) -> None:
    """校验配置快照并原子保存，任何失败都保留旧配置。"""
    with _lock:
        candidate = _clone_config(config)
        _save_config_locked(candidate)


def _validate_dataclass_shape(
    value: object, expected_cls: type[Any], path: str = ""
) -> None:
    """校验 dataclass 的嵌套类型、字段完整性和动态未知属性。"""
    if not isinstance(value, expected_cls):
        raise ConfigError(f"{path or '配置'}结构无效。")

    names = _field_names(expected_cls)
    attributes = vars(value)
    unknown = sorted(set(attributes) - names)
    if unknown:
        raise ConfigError(f"未知配置项：{_field_path(path, unknown[0])}。")

    for name in names:
        if not hasattr(value, name):
            raise ConfigError(f"配置项缺失：{_field_path(path, name)}。")

    field_types = get_type_hints(expected_cls)
    for name, field_type in field_types.items():
        if isinstance(field_type, type) and dataclasses.is_dataclass(field_type):
            _validate_dataclass_shape(
                getattr(value, name), field_type, _field_path(path, name)
            )


def validate_config(config: Config) -> None:
    """验证完整配置的字段、类型、范围和跨字段约束。"""
    if not isinstance(config, Config):
        raise ConfigError("配置对象类型无效。")
    _validate_dataclass_shape(config, Config)
    _validate_model(config.model)
    _validate_prediction(config.prediction)
    _validate_signal(config.signal)
    _validate_data(config.data)
    _validate_webui(config.webui)
    _validate_logging(config.logging)


def _validate_model(config: ModelConfig) -> None:
    """验证模型配置。"""
    _non_empty_text(config.tokenizer, "model.tokenizer")
    _non_empty_text(config.predictor, "model.predictor")
    _non_empty_text(config.tokenizer_revision, "model.tokenizer_revision")
    _non_empty_text(config.model_revision, "model.model_revision")
    _integer_in_range(config.max_context, "model.max_context", 60, 4096)
    _integer_in_range(config.idle_timeout_minutes, "model.idle_timeout_minutes", 1, 1440)
    if type(config.allow_model_download) is not bool:
        raise ConfigError("model.allow_model_download 必须是布尔值。")
    if not isinstance(config.device, str) or (
        config.device not in {"auto", "cpu"}
        and not config.device.startswith("cuda")
    ):
        raise ConfigError("model.device 必须为 auto、cpu 或 cuda 设备标识。")


def _validate_prediction(config: PredictionConfig) -> None:
    """验证推理参数及采样参数。"""
    _integer_in_range(config.default_pred_len, "prediction.default_pred_len", 60, 252)
    _integer_in_range(config.default_sample_count, "prediction.default_sample_count", 1, 1000)
    _integer_in_range(
        config.fallback_sample_count,
        "prediction.fallback_sample_count",
        1,
        config.default_sample_count,
    )
    _number_in_range(config.default_temperature, "prediction.default_temperature", 0.01, 2.0)
    _number_in_range(config.default_top_p, "prediction.default_top_p", 0.01, 1.0)
    _integer_in_range(config.timeout_seconds, "prediction.timeout_seconds", 1, 600)


def _validate_signal(config: SignalConfig) -> None:
    """验证信号阈值及 RSI 交叉约束。"""
    for name in (
        "trend_strength_buy_threshold",
        "trend_strength_sell_threshold",
        "var_risk_high_threshold",
        "consistency_std_threshold",
    ):
        _number_in_range(getattr(config, name), f"signal.{name}", 0.0, 1.0)
    _integer_in_range(config.reversal_rsi_oversold, "signal.reversal_rsi_oversold", 0, 100)
    _integer_in_range(config.reversal_rsi_overbought, "signal.reversal_rsi_overbought", 0, 100)
    if config.trend_strength_buy_threshold <= config.trend_strength_sell_threshold:
        raise ConfigError(
            "signal.trend_strength_buy_threshold 必须大于 trend_strength_sell_threshold。"
        )
    if config.reversal_rsi_oversold >= config.reversal_rsi_overbought:
        raise ConfigError("signal.reversal_rsi_oversold 必须小于 reversal_rsi_overbought。")


def _validate_data(config: DataConfig) -> None:
    """验证数据源、重试参数和可选数据源类型。"""
    _non_empty_text(config.cache_dir, "data.cache_dir")
    _non_empty_text(config.snapshot_dir, "data.snapshot_dir")
    _non_empty_text(config.report_db, "data.report_db")
    _non_empty_text(config.portfolio_ledger, "data.portfolio_ledger")
    _integer_in_range(config.cache_ttl_hours, "data.cache_ttl_hours", 1, 720)
    _integer_in_range(config.retry_max, "data.retry_max", 1, 10)
    if not isinstance(config.retry_backoff_seconds, list) or len(
        config.retry_backoff_seconds
    ) < config.retry_max - 1:
        raise ConfigError("data.retry_backoff_seconds 必须覆盖所有重试间隔。")
    for delay in config.retry_backoff_seconds:
        _integer_in_range(delay, "data.retry_backoff_seconds", 0, 300)
    if not isinstance(config.primary_source, str) or config.primary_source != "akshare":
        raise ConfigError("首版数据源顺序固定为 akshare → baostock。")
    if not isinstance(config.backup_source, str) or config.backup_source != "baostock":
        raise ConfigError("首版数据源顺序固定为 akshare → baostock。")
    if not isinstance(config.optional_sources, list):
        raise ConfigError("data.optional_sources 必须为文本列表。")
    for source in config.optional_sources:
        if not isinstance(source, str) or source != "tushare":
            raise ConfigError("data.optional_sources 仅支持 tushare。")
    _non_empty_text(config.tushare_token_env, "data.tushare_token_env")


def _validate_webui(config: WebUIConfig) -> None:
    """验证 Web UI 的本机监听地址和端口。"""
    _non_empty_text(config.host, "webui.host")
    if config.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ConfigError("webui.host 仅允许本机回环地址。")
    _integer_in_range(config.port, "webui.port", 1, 65535)
    _non_empty_text(config.stock_pool, "webui.stock_pool")


def _validate_logging(config: LoggingConfig) -> None:
    """验证日志配置的类型和值域。"""
    _non_empty_text(config.level, "logging.level")
    _non_empty_text(config.file, "logging.file")
    _integer_in_range(config.max_size_mb, "logging.max_size_mb", 1, 4096)
    _integer_in_range(config.backup_count, "logging.backup_count", 0, 100)


def _non_empty_text(value: object, name: str) -> None:
    """验证字段为非空文本且不在错误中回显内容。"""
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} 必须为非空文本。")


def _integer_in_range(value: object, name: str, minimum: int, maximum: int) -> None:
    """验证字段为非布尔整数并处于闭区间。"""
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ConfigError(f"{name} 必须为 {minimum} 到 {maximum} 的整数。")


def _number_in_range(value: object, name: str, minimum: float, maximum: float) -> None:
    """验证字段为有限数值并处于闭区间。"""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{name} 必须为 {minimum} 到 {maximum} 的数值。")
    numeric = float(value)
    if not math.isfinite(numeric) or not minimum <= numeric <= maximum:
        raise ConfigError(f"{name} 必须为 {minimum} 到 {maximum} 的数值。")
