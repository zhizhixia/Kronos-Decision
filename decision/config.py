"""配置文件加载与热更新。

所有模块通过 get_config() 获取配置单例，避免重复读取 YAML。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import yaml

_CONFIG_DIR = Path(__file__).parent
_CONFIG_PATH = _CONFIG_DIR / "config.yaml"
_lock = Lock()
_config_instance: Config | None = None


@dataclass
class ModelConfig:
    tokenizer: str = "NeoQuasar/Kronos-Tokenizer-base"
    predictor: str = "NeoQuasar/Kronos-small"
    max_context: int = 512
    device: str = "cuda"
    idle_timeout_minutes: int = 10


@dataclass
class PredictionConfig:
    default_pred_len: int = 120
    default_sample_count: int = 50
    default_temperature: float = 1.0
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


@dataclass
class WebUIConfig:
    host: str = "0.0.0.0"
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
                raw = _load_raw_config()
                _config_instance = _dict_to_dataclass(Config, raw)
    return _config_instance


def reload_config() -> Config:
    """强制重新加载配置（用于热更新）。"""
    global _config_instance
    with _lock:
        raw = _load_raw_config()
        _config_instance = _dict_to_dataclass(Config, raw)
    return _config_instance


def save_config(config: Config) -> None:
    """将配置对象写回 YAML 文件。

    用于 Web 设置页面的持久化需求。
    """
    import dataclasses
    with _lock:
        data = dataclasses.asdict(config)
        with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, default_flow_style=False)
        global _config_instance
        _config_instance = config