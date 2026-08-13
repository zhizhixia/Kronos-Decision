"""校准、横截面预测和样本外组合的门禁指标。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rank_ic_by_anchor(predictions: pd.DataFrame, horizon: int = 20) -> pd.Series:
    """计算指定期限（默认20日主决策）的每个锚点 Spearman RankIC。"""
    required = {"anchor_date", "score", "actual_return"}
    if not required.issubset(predictions):
        raise ValueError(f"RankIC 缺少字段：{sorted(required - set(predictions))}")
    frame = predictions
    if "horizon" in frame:
        frame = frame.loc[frame["horizon"].astype(int) == horizon]
    if frame.empty:
        return pd.Series(dtype=float)
    return frame.groupby("anchor_date", sort=True).apply(
        lambda group: group["score"].rank().corr(group["actual_return"].rank()),
        include_groups=False,
    ).dropna()


def block_bootstrap_positive_probability(values: pd.Series, block_size: int = 8, samples: int = 2000, seed: int = 0) -> float:
    """用连续锚点区块 Bootstrap 估计均值为正的概率。"""
    array = values.dropna().to_numpy(dtype=float)
    if len(array) < block_size or samples <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    starts = np.arange(0, len(array) - block_size + 1)
    blocks_needed = int(np.ceil(len(array) / block_size))
    means = np.empty(samples)
    for index in range(samples):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([array[start : start + block_size] for start in chosen])[: len(array)]
        means[index] = sample.mean()
    return float(np.mean(means > 0))


def expected_calibration_error(probabilities: pd.Series, outcomes: pd.Series, bins: int = 10) -> float:
    """计算等宽分箱的上涨概率 ECE。"""
    frame = pd.DataFrame({"probability": probabilities, "outcome": outcomes}).dropna()
    if frame.empty:
        return 1.0
    frame["bin"] = pd.cut(frame["probability"].clip(0, 1), np.linspace(0, 1, bins + 1), include_lowest=True, labels=False)
    grouped = frame.groupby("bin", observed=True)
    errors = grouped.apply(lambda group: abs(group["probability"].mean() - group["outcome"].mean()), include_groups=False)
    weights = grouped.size() / len(frame)
    return float((errors * weights).sum())


def interval_coverage(lower: pd.Series, upper: pd.Series, actual: pd.Series) -> float:
    """计算实际收益落在预测区间内的比例。"""
    frame = pd.DataFrame({"lower": lower, "upper": upper, "actual": actual}).dropna()
    if frame.empty or (frame["lower"] > frame["upper"]).any():
        return 0.0
    return float(frame["actual"].between(frame["lower"], frame["upper"], inclusive="both").mean())


def portfolio_metrics(report: pd.DataFrame) -> dict[str, float]:
    """从 Qlib 日度报告计算扣费超额、IR、稳定性和回撤恶化。"""
    required = {"return", "bench", "cost"}
    if not required.issubset(report):
        raise ValueError(f"回测报告缺少字段：{sorted(required - set(report))}")
    strategy = report["return"].fillna(0) - report["cost"].fillna(0)
    benchmark = report["bench"].fillna(0)
    excess = strategy - benchmark
    annualized_excess = _annualized_return(excess)
    information_ratio = float(excess.mean() / excess.std(ddof=1) * np.sqrt(252)) if excess.std(ddof=1) > 0 else 0.0
    rolling = excess.rolling(252, min_periods=126).sum().dropna()
    stability = float((rolling > 0).mean()) if not rolling.empty else 0.0
    worsening = max(0.0, abs(_max_drawdown(strategy)) - abs(_max_drawdown(benchmark)))
    return {"annualized_excess_return": annualized_excess, "information_ratio": information_ratio, "positive_12m_window_ratio": stability, "drawdown_worsening": worsening}


def _annualized_return(returns: pd.Series) -> float:
    if returns.empty or (returns <= -1).any():
        return -1.0
    return float((1 + returns).prod() ** (252 / len(returns)) - 1)


def _max_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns.fillna(0)).cumprod()
    if wealth.empty:
        return 0.0
    return float((wealth / wealth.cummax() - 1).min())
