"""Kronos 预测路径的数据契约。"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def sampling_params_hash(
    pred_len: int,
    sample_count: int,
    sample_batch_size: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    """返回绑定全部路径生成参数的稳定审计哈希。"""
    payload = {
        "pred_len": int(pred_len),
        "sample_count": int(sample_count),
        "sample_batch_size": int(sample_batch_size),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def ensure_before_deadline(deadline: float | None) -> None:
    """在可协作中断点检查推理截止时间。"""
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("PREDICTION_TIMEOUT：模型推理超过配置的时间预算。")


@dataclass(frozen=True)
class PredictionPaths:
    """一次真实采样产生的、可审计的预测路径。"""

    paths: np.ndarray
    mean_df: pd.DataFrame
    timestamps: pd.DatetimeIndex
    sample_count: int
    seed: int
    model_id: str
    tokenizer_id: str
    sampling_params_hash: str
    repaired_values: int = 0
    quality_flags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def repair_ratio(self) -> float:
        """返回被修复数值占全部路径元素的比例。"""
        return self.repaired_values / max(1, int(self.paths.size))
