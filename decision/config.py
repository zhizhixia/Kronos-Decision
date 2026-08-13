"""配置文件加载与热更新。

所有模块通过 get_config() 获取配置单例，避免重复读取 YAML。
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import yaml

from decision.errors import ConfigError

_CONFIG_DIR = Path(__file__).parent
_CONFIG_PATH = _CONFIG_DIR / "config.yaml"
_lock = Lock()
_config_instance: Config | None = None


@dataclass
class ModelConfig:
    tokenizer: str = "NeoQuasar/Kronos-Tokenizer-base"
    predictor: str = "NeoQuasar/Kronos-small"
    tokenizer_revision: str = "0e0117387f39004a9016484a186a908917e22426"
    model_revision: str = "901c26c1332695a2a8f243eb2f37243a37bea320"
    max_context: int = 512
    device: str = "auto"
    idle_timeout_minutes: int = 10


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


def _dict_to_dataclass(cls, data: dict):
    """递归将字典转换为 dataclass 实例。"""
    import typing
    field_types = typing.get_type_hints(cls)
    kwargs = {}
    for key, val in data.items():
        if key in field_types and hasattr(field_types[key], "__dataclass_fields__"):
            kwargs[key] = _dict_to_dataclass(field_types[key], val)
        else:
            kwargs[key] = val
    return cls(**kwargs)


def _build_config(raw: dict) -> Config:
    """将 YAML 内容构造成配置对象并执行所有边界校验。"""
    if not isinstance(raw, dict):
        raise ConfigError("配置文件根节点必须是映射对象。")
    config = _dict_to_dataclass(Config, raw)
    validate_config(config)
    return config


def _load_raw_config() -> dict:
    """从 YAML 文件加载原始配置字典。"""
    if not _CONFIG_PATH.exists():
        return {}
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_config() -> Config:
    """获取全局配置单例（线程安全）。

    首次调用时从 config.yaml 加载，后续调用返回缓存实例。
    """
    global _config_instance
    if _config_instance is None:
        with _lock:
            if _config_instance is None:
                _config_instance = _build_config(_load_raw_config())
    return _config_instance


def reload_config() -> Config:
    """强制重新加载配置（用于热更新）。"""
    global _config_instance
    with _lock:
        _config_instance = _build_config(_load_raw_config())
    return _config_instance


def save_config(config: Config) -> None:
    """将配置对象写回 YAML 文件。

    用于 Web 设置页面的持久化需求。
    """
    import dataclasses

    validate_config(config)
    with _lock:
        data = dataclasses.asdict(config)
        payload = yaml.safe_dump(data, allow_unicode=True, default_flow_style=False)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=_CONFIG_PATH.parent, suffix=".tmp") as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        os.replace(temporary, _CONFIG_PATH)
        global _config_instance
        _config_instance = config


def validate_config(config: Config) -> None:
    """验证配置类型、范围及跨字段约束，避免无效参数进入推理或数据链。"""
    if not isinstance(config, Config):
        raise ConfigError("配置对象类型无效。")
    _validate_model(config.model)
    _validate_prediction(config.prediction)
    _validate_signal(config.signal)
    _validate_data(config.data)
    _validate_webui(config.webui)
    _validate_logging(config.logging)


def _validate_model(config: ModelConfig) -> None:
    _non_empty_text(config.tokenizer, "model.tokenizer")
    _non_empty_text(config.predictor, "model.predictor")
    _non_empty_text(config.tokenizer_revision, "model.tokenizer_revision")
    _non_empty_text(config.model_revision, "model.model_revision")
    _integer_in_range(config.max_context, "model.max_context", 60, 4096)
    _integer_in_range(config.idle_timeout_minutes, "model.idle_timeout_minutes", 1, 1440)
    if not isinstance(config.device, str) or config.device not in {"auto", "cpu"} and not config.device.startswith("cuda"):
        raise ConfigError("model.device 必须为 auto、cpu 或 cuda 设备标识。")


def _validate_prediction(config: PredictionConfig) -> None:
    _integer_in_range(config.default_pred_len, "prediction.default_pred_len", 60, 252)
    _integer_in_range(config.default_sample_count, "prediction.default_sample_count", 1, 1000)
    _integer_in_range(config.fallback_sample_count, "prediction.fallback_sample_count", 1, config.default_sample_count)
    _number_in_range(config.default_temperature, "prediction.default_temperature", 0.01, 2.0)
    _number_in_range(config.default_top_p, "prediction.default_top_p", 0.01, 1.0)
    _integer_in_range(config.timeout_seconds, "prediction.timeout_seconds", 1, 600)


def _validate_signal(config: SignalConfig) -> None:
    for name in ("trend_strength_buy_threshold", "trend_strength_sell_threshold", "var_risk_high_threshold", "consistency_std_threshold"):
        _number_in_range(getattr(config, name), f"signal.{name}", 0.0, 1.0)
    _integer_in_range(config.reversal_rsi_oversold, "signal.reversal_rsi_oversold", 0, 100)
    _integer_in_range(config.reversal_rsi_overbought, "signal.reversal_rsi_overbought", 0, 100)
    if config.reversal_rsi_oversold >= config.reversal_rsi_overbought:
        raise ConfigError("signal.reversal_rsi_oversold 必须小于 reversal_rsi_overbought。")


def _validate_data(config: DataConfig) -> None:
    _non_empty_text(config.cache_dir, "data.cache_dir")
    _integer_in_range(config.cache_ttl_hours, "data.cache_ttl_hours", 1, 720)
    _integer_in_range(config.retry_max, "data.retry_max", 1, 10)
    if not isinstance(config.retry_backoff_seconds, list) or len(config.retry_backoff_seconds) < config.retry_max - 1:
        raise ConfigError("data.retry_backoff_seconds 必须覆盖所有重试间隔。")
    for delay in config.retry_backoff_seconds:
        _integer_in_range(delay, "data.retry_backoff_seconds", 0, 300)
    if config.primary_source != "akshare" or config.backup_source != "baostock":
        raise ConfigError("首版数据源顺序固定为 akshare → baostock。")
    if not isinstance(config.optional_sources, list) or set(config.optional_sources) - {"tushare"}:
        raise ConfigError("data.optional_sources 仅支持 tushare。")
    _non_empty_text(config.tushare_token_env, "data.tushare_token_env")


def _validate_webui(config: WebUIConfig) -> None:
    _non_empty_text(config.host, "webui.host")
    if config.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ConfigError("webui.host 仅允许本机回环地址。")
    _integer_in_range(config.port, "webui.port", 1, 65535)
    _non_empty_text(config.stock_pool, "webui.stock_pool")


def _validate_logging(config: LoggingConfig) -> None:
    _non_empty_text(config.level, "logging.level")
    _non_empty_text(config.file, "logging.file")
    _integer_in_range(config.max_size_mb, "logging.max_size_mb", 1, 4096)
    _integer_in_range(config.backup_count, "logging.backup_count", 0, 100)


def _non_empty_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} 必须为非空文本。")


def _integer_in_range(value: object, name: str, minimum: int, maximum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ConfigError(f"{name} 必须为 {minimum} 到 {maximum} 的整数。")


def _number_in_range(value: object, name: str, minimum: float, maximum: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not minimum <= float(value) <= maximum:
        raise ConfigError(f"{name} 必须为 {minimum} 到 {maximum} 的数值。")
