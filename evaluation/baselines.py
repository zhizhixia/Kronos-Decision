"""正式研究的基线比较：随机游走、20日动量、等权与沪深300基准。"""
from __future__ import annotations

import pandas as pd


def model_rank_ic(predictions: pd.DataFrame) -> pd.Series:
    """模型预测的每锚点 RankIC。"""
    return _rank_ic_by_anchor(_main_horizon(predictions), "score")


def random_walk_rank_ic(predictions: pd.DataFrame) -> pd.Series:
    """随机游走基线：预测收益恒为 0，RankIC 应接近 0。"""
    frame = _main_horizon(predictions)
    frame["score"] = 0.0
    return _rank_ic_by_anchor(frame, "score")


def momentum_rank_ic(predictions: pd.DataFrame, price_history: pd.DataFrame) -> pd.Series:
    """20 日横截面动量基线：用锚点前 20 日收益作为信号。"""
    frame = _main_horizon(predictions)
    frame["anchor_date"] = pd.to_datetime(frame["anchor_date"])
    frame["stock_code"] = frame["stock_code"].astype(str).str.zfill(6)
    closes = price_history.copy()
    closes.columns = [str(code).zfill(6) for code in closes.columns]
    signals = []
    for (anchor, code), row in frame.set_index(["anchor_date", "stock_code"]).iterrows():
        series = closes[code].loc[:anchor]
        if len(series) < 21:
            signals.append(float("nan"))
            continue
        signals.append(float(series.iloc[-1] / series.iloc[-21] - 1))
    frame["score"] = signals
    return _rank_ic_by_anchor(frame, "score")


def equal_weight_returns(price_history: pd.DataFrame) -> pd.Series:
    """静态价格表的等权日收益；仅用于独立工具调用，不作正式点时基线。"""
    returns = price_history.pct_change(fill_method=None).fillna(0.0)
    return returns.mean(axis=1)


def point_in_time_equal_weight_returns(predictions: pd.DataFrame) -> pd.Series:
    """按每个预测锚点的有效股票集合计算20日点时等权实际收益。"""
    frame = _main_horizon(predictions)
    required = {"anchor_date", "stock_code", "actual_return"}
    if not required.issubset(frame):
        raise ValueError(f"点时等权基线缺少字段：{sorted(required - set(frame))}")
    clean = frame.dropna(subset=["actual_return"]).copy()
    clean["actual_return"] = clean["actual_return"].astype(float)
    return clean.groupby("anchor_date", sort=True)["actual_return"].mean()


def benchmark_returns(hs300_close: pd.Series, is_total_return: bool = False) -> tuple[pd.Series, bool]:
    """沪深300基准日收益；仅价格指数时标记不能通过正式门禁。"""
    series = hs300_close.dropna()
    returns = series.pct_change(fill_method=None).fillna(0.0)
    return returns, is_total_return


def compare_baselines(predictions: pd.DataFrame, price_history: pd.DataFrame, hs300_close: pd.Series, is_total_return: bool = False) -> dict:
    """汇总模型与三个基线的 RankIC 及组合级指标。"""
    rank_ic = {
        "model": model_rank_ic(predictions).dropna(),
        "random_walk": random_walk_rank_ic(predictions).dropna(),
        "momentum_20d": momentum_rank_ic(predictions, price_history).dropna(),
    }
    summary = {}
    for name, series in rank_ic.items():
        summary[f"rank_ic_mean_{name}"] = float(series.mean()) if not series.empty else float("nan")
        summary[f"rank_ic_positive_ratio_{name}"] = float((series > 0).mean()) if not series.empty else float("nan")
    equal_weight = point_in_time_equal_weight_returns(predictions)
    benchmark, total_return = benchmark_returns(hs300_close, is_total_return)
    summary["equal_weight_annualized_return"] = float(_annualized_horizon(equal_weight, 20))
    summary["point_in_time_equal_weight"] = True
    summary["point_in_time_equal_weight_observations"] = int(len(equal_weight))
    summary["benchmark_annualized_return"] = float(_annualized(benchmark))
    summary["benchmark_is_total_return"] = bool(total_return)
    summary["baseline_windows_overlap"] = bool(set(rank_ic["model"].index).issubset(set(rank_ic["momentum_20d"].index)))
    return summary


def _rank_ic_by_anchor(frame: pd.DataFrame, score_column: str) -> pd.Series:
    required = {"anchor_date", "stock_code", score_column, "actual_return"}
    if not required.issubset(frame):
        raise ValueError(f"基线比较缺少字段：{sorted(required - set(frame))}")
    return frame.groupby("anchor_date", sort=True).apply(
        lambda group: group[score_column].rank().corr(group["actual_return"].rank()),
        include_groups=False,
    ).dropna()


def _main_horizon(predictions: pd.DataFrame) -> pd.DataFrame:
    """基线比较只消费20日主决策信号，兼容旧单期限工件。"""
    if "horizon" not in predictions:
        return predictions.copy()
    return predictions.loc[predictions["horizon"].astype(int) == 20].copy()


def _annualized(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty or (clean <= -1).any():
        return 0.0
    return float((1 + clean).prod() ** (252 / len(clean)) - 1)


def _annualized_horizon(returns: pd.Series, horizon: int) -> float:
    clean = returns.dropna()
    if clean.empty or (clean <= -1).any():
        return 0.0
    return float((1 + clean).prod() ** (252 / (len(clean) * horizon)) - 1)
