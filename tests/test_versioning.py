"""版本标识与固定模型修订测试。"""
from __future__ import annotations

from decision.config import ModelConfig
from decision.versioning import config_hash, model_provenance, rules_hash


class _ComponentConfig:
    _name_or_path = "fixture/repository"
    _commit_hash = "commit-123"


class _Component:
    config = _ComponentConfig()


def test_model_config_pins_revisions_and_uses_auto_device() -> None:
    config = ModelConfig()
    assert len(config.model_revision) == 40
    assert len(config.tokenizer_revision) == 40
    assert config.device == "auto"


def test_model_provenance_uses_loaded_component_identity() -> None:
    provenance = model_provenance(_Component(), _Component())
    assert provenance["model_id"] == "fixture/repository"
    assert provenance["model_revision"] == "commit-123"
    assert provenance["revisions_pinned"]
    assert len(str(provenance["pipeline_hash"])) == 64


def test_config_and_rules_hashes_are_available() -> None:
    assert len(config_hash()) == 64
    assert len(rules_hash()) == 64
