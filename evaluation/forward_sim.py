"""前瞻模拟：用固定信号文件按周期模拟，与历史回测分开报告。"""
from __future__ import annotations

import pandas as pd


def attach_actuals(predictions: pd.DataFrame, price_history: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """把每个锚点预测后的实际 horizon 日收益填入 actual_return（期末可见序列）。"""
    frame = predictions.copy()
    frame["anchor_date"] = pd.to_datetime(frame["anchor_date"])
    frame["stock_code"] = frame["stock_code"].astype(str).str.zfill(6)
    closes = price_history.copy()
    closes.columns = [str(code).zfill(6) for code in closes.columns]
    actuals = []
    for _, row in frame.iterrows():
        series = closes[row["stock_code"]].dropna()
        if series.empty:
            actuals.append(float("nan"))
            continue
        start_loc = series.index.searchsorted(row["anchor_date"])
        end_loc = start_loc + horizon
        if end_loc >= len(series):
            actuals.append(float("nan"))
            continue
        actuals.append(float(series.iloc[end_loc] / series.iloc[start_loc] - 1))
    frame["actual_return"] = actuals
    return frame


def simulate_forward(predictions: pd.DataFrame, price_history: pd.DataFrame, topk: int = 5, initial_capital: float = 1_000_000.0, horizon: int = 20) -> pd.DataFrame:
    """按锚点横截面 topk 建仓持有 horizon 日，输出每个锚点周期的组合表现。"""
    frame = attach_actuals(predictions, price_history, horizon).dropna(subset=["actual_return"])
    rows = []
    for anchor, group in frame.groupby("anchor_date", sort=True):
        ranked = group.sort_values("score", ascending=False).head(topk)
        if ranked.empty:
            continue
        mean_return = float(ranked["actual_return"].mean())
        rows.append({"anchor_date": anchor, "stocks": len(ranked), "period_return": mean_return, "positive_stocks": int((ranked["actual_return"] > 0).sum()), "model_mean_rank_ic": float(ranked["score"].rank().corr(ranked["actual_return"].rank())) if len(ranked) > 1 else float("nan")})
    return pd.DataFrame(rows)


def forward_report(simulated: pd.DataFrame, initial_capital: float = 1_000_000.0) -> dict:
    """前瞻模拟汇总；与 Qlib 历史回测分开报告，不计入正式门禁。"""
    if simulated.empty:
        return {"status": "no_observations"}
    period_returns = simulated["period_return"]
    cumulative = float((1 + period_returns).prod() - 1)
    return {
        "status": "completed",
        "periods": int(len(simulated)),
        "cumulative_return": cumulative,
        "period_win_rate": float((period_returns > 0).mean()),
        "mean_period_return": float(period_returns.mean()),
        "max_period_drawdown": float((period_returns + 1).cumprod().div((period_returns + 1).cumprod().cummax()) .sub(1).min()) if not period_returns.empty else 0.0,
        "disclaimer": "前瞻模拟使用期末可见价格序列，结果与历史回测分开报告，不进入正式证据门禁。",
    }
