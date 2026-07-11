"""测试配置加载与热更新。"""
import os
import tempfile
import pytest
from decision.config import Config, get_config, reload_config, save_config


def test_config_defaults():
    """默认配置值符合设计规格。"""
    cfg = Config()
    assert cfg.signal.trend_strength_buy_threshold == 0.7
    assert cfg.signal.trend_strength_sell_threshold == 0.4
    assert cfg.prediction.default_pred_len == 120
    assert cfg.model.idle_timeout_minutes == 10


def test_config_dataclass_structure():
    """配置对象包含所有必需的子配置。"""
    cfg = Config()
    assert hasattr(cfg, "model")
    assert hasattr(cfg, "prediction")
    assert hasattr(cfg, "signal")
    assert hasattr(cfg, "data")
    assert hasattr(cfg, "webui")
    assert hasattr(cfg, "logging")


def test_config_save_and_reload():
    """保存后重新加载，值保持一致。"""
    cfg = Config()
    cfg.signal.trend_strength_buy_threshold = 0.75
    save_config(cfg)
    reloaded = reload_config()
    assert reloaded.signal.trend_strength_buy_threshold == 0.75
    # 恢复默认值
    cfg.signal.trend_strength_buy_threshold = 0.7
    save_config(cfg)