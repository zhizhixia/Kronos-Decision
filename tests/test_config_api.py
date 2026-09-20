"""配置 API 必须先验证副本，失败时不得污染内存配置。"""
from __future__ import annotations

from decision.config import Config


def test_invalid_config_api_does_not_mutate_live_config(monkeypatch, tmp_path) -> None:
    import decision.config as config_module
    from webui.app import app

    original = Config()
    monkeypatch.setattr(config_module, "_CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setattr(config_module, "_config_instance", original)
    client = app.test_client()

    response = client.put(
        "/api/config",
        json={"prediction": {"default_sample_count": 0}},
        headers={"Origin": "http://127.0.0.1:7070"},
    )

    assert response.status_code == 400
    assert config_module.get_config().prediction.default_sample_count == 100
    assert not (tmp_path / "config.yaml").exists()


def test_valid_config_api_atomically_replaces_config(monkeypatch, tmp_path) -> None:
    import decision.config as config_module
    from webui.app import app

    monkeypatch.setattr(config_module, "_CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setattr(config_module, "_config_instance", Config())
    client = app.test_client()

    response = client.put(
        "/api/config",
        json={"prediction": {"timeout_seconds": 90}},
        headers={"Origin": "http://127.0.0.1:7070"},
    )

    assert response.status_code == 200
    assert config_module.get_config().prediction.timeout_seconds == 90
    assert (tmp_path / "config.yaml").exists()
