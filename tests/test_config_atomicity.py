"""配置持久化的原子性回归测试。"""
from __future__ import annotations

from pathlib import Path

import pytest

from decision.config import Config, get_config, reload_config, save_config


def test_save_config_fsyncs_before_atomic_replace(monkeypatch, tmp_path: Path) -> None:
    """保存必须刷新临时文件后再执行同目录原子替换。"""
    import decision.config as config_module

    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(config_module, "_CONFIG_PATH", config_path)
    monkeypatch.setattr(config_module, "_config_instance", Config())
    events: list[str] = []
    real_fsync = config_module.os.fsync
    real_replace = config_module.os.replace

    def record_fsync(file_descriptor: int) -> None:
        events.append("fsync")
        real_fsync(file_descriptor)

    def record_replace(source: object, destination: object) -> None:
        events.append("replace")
        real_replace(source, destination)

    monkeypatch.setattr(config_module.os, "fsync", record_fsync)
    monkeypatch.setattr(config_module.os, "replace", record_replace)

    save_config(Config())

    assert events == ["fsync", "replace"]
    assert config_path.exists()


def test_replace_failure_keeps_previous_config_readable(monkeypatch, tmp_path: Path) -> None:
    """替换失败时临时文件清理，旧磁盘和内存配置均保持可读。"""
    import decision.config as config_module

    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(config_module, "_CONFIG_PATH", config_path)
    monkeypatch.setattr(config_module, "_config_instance", Config())
    save_config(Config())
    previous_bytes = config_path.read_bytes()

    candidate = Config()
    candidate.prediction.timeout_seconds = 90

    def fail_replace(source: object, destination: object) -> None:
        del source, destination
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(config_module.os, "replace", fail_replace)
    with pytest.raises(OSError):
        save_config(candidate)

    assert get_config().prediction.timeout_seconds == 60
    assert config_path.read_bytes() == previous_bytes
    assert reload_config().prediction.timeout_seconds == 60
    assert not list(tmp_path.glob("*.tmp"))
