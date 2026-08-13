"""AKQuant 隔离对照：同一固定信号文件、两个引擎分开回测、结果差异必须可解释。"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def signal_hash(predictions: pd.DataFrame) -> str:
    """对固定信号文件内容做稳定哈希，确保两引擎输入一致。"""
    canonical = predictions[["anchor_date", "execution_date", "stock_code", "horizon", "score"]].copy()
    canonical = canonical.sort_values(["anchor_date", "stock_code"]).reset_index(drop=True)
    return hashlib.sha256(canonical.to_csv(index=False).encode("utf-8")).hexdigest()


def compare_engines(predictions: pd.DataFrame, qlib_returns: pd.Series, akquant_returns: pd.Series | None = None) -> dict:
    """对比两引擎日度收益；差异无法解释或 AKQuant 无实际价值时保持 Qlib。"""
    digest = signal_hash(predictions)
    if akquant_returns is None:
        return {"status": "AKQUANT_NOT_INSTALLED", "signal_hash": digest, "decision": "keep_qlib", "reason": "AKQuant 未安装；不在首版依赖中，不讨论替换。"}
    aligned = pd.concat([qlib_returns.rename("qlib"), akquant_returns.rename("akquant")], axis=1).dropna()
    if aligned.empty:
        return {"status": "NO_OVERLAP", "signal_hash": digest, "decision": "keep_qlib"}
    correlation = float(aligned["qlib"].corr(aligned["akquant"]))
    qlib_annualized = _annualized(aligned["qlib"])
    akquant_annualized = _annualized(aligned["akquant"])
    difference = akquant_annualized - qlib_annualized
    if correlation < 0.90:
        decision = "keep_qlib"
        reason = f"两引擎日度收益相关性仅 {correlation:.2f}，差异无法解释，保留 Qlib。"
    elif difference > 0 and akquant_annualized > qlib_annualized:
        decision = "consider_evaluation"
        reason = "AKQuant 年化收益更高且相关性足够；需进一步解释差异来源后才讨论替换。"
    else:
        decision = "keep_qlib"
        reason = "AKQuant 未带来明确价值，保留 Qlib 作为唯一正式研究底座。"
    return {"status": "compared", "signal_hash": digest, "decision": decision, "reason": reason, "daily_correlation": round(correlation, 4), "qlib_annualized": round(qlib_annualized, 4), "akquant_annualized": round(akquant_annualized, 4)}


def _annualized(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty or (clean <= -1).any():
        return 0.0
    return float((1 + clean).prod() ** (252 / len(clean)) - 1)
