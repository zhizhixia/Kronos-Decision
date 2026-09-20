"""测试配置加载与热更新。"""
import os
import tempfile
import pytest
from decision.config import (
    Config,
    get_config,
    reload_config,
    save_config,
    update_config,
    validate_config,
)
from decision.errors import ConfigError


def test_config_defaults():
    """默认配置值符合设计规格。"""
    cfg = Config()
    assert cfg.signal.trend_strength_buy_threshold == 0.7
    assert cfg.signal.trend_strength_sell_threshold == 0.4
    assert cfg.prediction.default_pred_len == 60
    assert cfg.prediction.default_sample_count == 100
    assert cfg.prediction.default_temperature == 0.6
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


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda cfg: setattr(cfg.prediction, "default_sample_count", 0), "default_sample_count"),
        (lambda cfg: setattr(cfg.prediction, "fallback_sample_count", 101), "fallback_sample_count"),
        (lambda cfg: setattr(cfg.prediction, "default_top_p", 1.1), "default_top_p"),
        (lambda cfg: setattr(cfg.data, "optional_sources", ["unknown"]), "optional_sources"),
        (lambda cfg: setattr(cfg.webui, "port", 70000), "webui.port"),
        (lambda cfg: setattr(cfg.webui, "host", "0.0.0.0"), "webui.host"),
    ],
)
def test_config_validation_rejects_invalid_ranges_and_sources(mutate, message) -> None:
    cfg = Config()
    mutate(cfg)

    with pytest.raises(ConfigError, match=message):
        validate_config(cfg)


def test_config_validation_rejects_invalid_cross_field_constraint() -> None:
    cfg = Config()
    cfg.prediction.fallback_sample_count = cfg.prediction.default_sample_count + 1

    with pytest.raises(ConfigError, match="fallback_sample_count"):
        validate_config(cfg)


def test_config_validation_rejects_unknown_dynamic_fields_and_wrong_types() -> None:
    """完整校验拒绝动态未知字段、布尔整数和敏感值回显。"""
    cfg = Config()
    cfg.prediction.unexpected = "super-secret-value"

    with pytest.raises(ConfigError, match="unexpected") as error:
        validate_config(cfg)

    assert "super-secret-value" not in str(error.value)

    cfg = Config()
    cfg.prediction.default_sample_count = True
    with pytest.raises(ConfigError, match="default_sample_count"):
        validate_config(cfg)


def test_update_config_validates_and_publishes_a_snapshot() -> None:
    """更新先修改完整副本，失败不会替换运行中的单例。"""
    original = get_config()

    with pytest.raises(ConfigError, match="default_sample_count"):
        update_config({"prediction": {"default_sample_count": 0}})

    assert get_config() is original
    assert get_config().prediction.default_sample_count == 100

    updated = update_config({"prediction": {"timeout_seconds": 90}})
    assert updated is get_config()
    assert updated.prediction.timeout_seconds == 90
