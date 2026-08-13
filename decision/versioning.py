"""模型、配置与规则的可审计版本标识。"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from decision.config import get_config


def config_hash() -> str:
    """返回当前决策配置文件的完整内容哈希。"""
    return _file_hash(Path(__file__).with_name("config.yaml"))


def rules_hash() -> str:
    """返回五态动作规则实现的完整内容哈希。"""
    return _file_hash(Path(__file__).with_name("v2.py"))


def model_provenance(model: Any | None = None, tokenizer: Any | None = None) -> dict[str, str | bool]:
    """构建含固定修订的模型、Tokenizer 与组合管线标识。"""
    cfg = get_config().model
    model_id = _component_id(model, cfg.predictor)
    tokenizer_id = _component_id(tokenizer, cfg.tokenizer)
    model_revision = _component_revision(model, cfg.model_revision)
    tokenizer_revision = _component_revision(tokenizer, cfg.tokenizer_revision)
    model_hash = _text_hash(f"{model_id}|{model_revision}")
    tokenizer_hash = _text_hash(f"{tokenizer_id}|{tokenizer_revision}")
    return {
        "model_id": model_id,
        "model_revision": model_revision,
        "model_identifier_hash": model_hash,
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_identifier_hash": tokenizer_hash,
        "pipeline_hash": _text_hash(f"{model_hash}|{tokenizer_hash}"),
        "revisions_pinned": model_revision != "unresolved" and tokenizer_revision != "unresolved",
    }


def derive_sampling_seed(model_hash: str, stock_code: str, last_visible: Any, config_digest: str, salt: int | None = None) -> int:
    """按模型、股票、截止日和配置派生跨入口一致的采样种子。"""
    date_value = last_visible.date() if hasattr(last_visible, "date") else str(last_visible)[:10]
    parts = [str(model_hash), str(stock_code).zfill(6), str(date_value), str(config_digest)]
    if salt is not None:
        parts.append(f"salt={int(salt)}")
    return int.from_bytes(hashlib.sha256("|".join(parts).encode("utf-8")).digest()[:8], "big") % (2**63 - 1)


def _component_id(component: Any | None, fallback: str) -> str:
    """从组件或配置获取稳定的仓库标识。"""
    config = getattr(component, "config", None)
    return str(getattr(component, "name_or_path", "") or getattr(config, "_name_or_path", "") or fallback)


def _component_revision(component: Any | None, fallback: str | None) -> str:
    """优先读取加载组件的提交哈希，缺失时使用固定配置修订。"""
    config = getattr(component, "config", None)
    return str(getattr(component, "_commit_hash", "") or getattr(config, "_commit_hash", "") or fallback or "unresolved")


def _file_hash(path: Path) -> str:
    """返回文件内容哈希；文件缺失必须显式可见。"""
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"


def _text_hash(value: str) -> str:
    """对版本描述生成完整 SHA-256 哈希。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
