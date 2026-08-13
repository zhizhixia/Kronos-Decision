"""样本外预测与正式回测的数据合同。"""
from __future__ import annotations

import hashlib

import pandas as pd


REQUIRED_PREDICTION_COLUMNS = [
    "prediction_key",
    "anchor_date",
    "execution_date",
    "stock_code",
    "horizon",
    "score",
    "predicted_return",
    "actual_return",
    "data_as_of",
    "model_hash",
    "config_hash",
    "data_hash",
]

OPTIONAL_PREDICTION_COLUMNS = ["raw_up_probability", "q05", "q95", "q50", "sampling_seed", "sampling_params_hash"]


def validate_prediction_frame(frame: pd.DataFrame, horizon: int | None = None) -> pd.DataFrame:
    """校验点时预测；任何时间穿越或重复键均立即失败。"""
    missing = [column for column in REQUIRED_PREDICTION_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"预测文件缺少字段：{missing}")
    keep = REQUIRED_PREDICTION_COLUMNS + [column for column in OPTIONAL_PREDICTION_COLUMNS if column in frame]
    result = frame[keep].copy()
    for column in ("anchor_date", "execution_date", "data_as_of"):
        result[column] = pd.to_datetime(result[column], errors="raise").dt.normalize()
    result["stock_code"] = result["stock_code"].astype(str).str.zfill(6)
    if horizon is not None and set(result["horizon"].astype(int)) != {horizon}:
        raise ValueError(f"预测文件必须只包含 {horizon} 日期限。")
    if (result["data_as_of"] > result["anchor_date"]).any():
        raise ValueError("检测到未来数据：data_as_of 晚于 anchor_date。")
    if (result["execution_date"] <= result["anchor_date"]).any():
        raise ValueError("execution_date 必须是 anchor_date 之后的交易日。")
    if result.duplicated(subset=["prediction_key", "horizon"]).any():
        raise ValueError("同一 prediction_key 和 horizon 组合必须唯一。")
    return result.sort_values(["execution_date", "stock_code"]).reset_index(drop=True)


def prediction_key(
    stock_code: str,
    anchor_date: str,
    model_hash: str,
    config_hash: str,
    sampling_hash: str = "",
) -> str:
    """生成可用于断点续跑的稳定预测键。"""
    payload = f"{stock_code}|{anchor_date}|{model_hash}|{config_hash}|{sampling_hash}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
