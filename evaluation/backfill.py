"""把锚点后的实际收益回填到预测工件（RankIC 与校准的前置）。"""
from __future__ import annotations

import pandas as pd


def backfill_actuals(predictions: pd.DataFrame, bars_at, horizon: int | None = None) -> pd.DataFrame:
    """按预测期限回填实际收益；显式 ``horizon`` 保留旧的统一期限行为。"""
    frame = predictions.copy()
    frame["anchor_date"] = pd.to_datetime(frame["anchor_date"])
    frame["stock_code"] = frame["stock_code"].astype(str).str.zfill(6)
    actuals = pd.Series(float("nan"), index=frame.index, dtype=float)
    for (code, anchor), group in frame.groupby(["stock_code", "anchor_date"], sort=False):
        horizons = _target_horizons(group, horizon)
        bars = bars_at(code, anchor + pd.Timedelta(days=int(max(horizons) * 1.8) + 10))
        returns = _actual_returns(bars, anchor, horizons)
        for index, row in group.iterrows():
            target_horizon = int(horizon) if horizon is not None else int(row["horizon"])
            actuals.at[index] = returns.get(target_horizon, float("nan"))
    frame["actual_return"] = actuals.to_numpy()
    return frame


def _target_horizons(group: pd.DataFrame, forced_horizon: int | None) -> list[int]:
    """返回一个股票锚点组需要回填的正整数期限。"""
    horizons = [int(forced_horizon)] if forced_horizon is not None else sorted({int(value) for value in group["horizon"]})
    if not horizons or min(horizons) <= 0:
        raise ValueError("预测期限必须为正整数。")
    return horizons


def _actual_returns(bars: pd.DataFrame | None, anchor: pd.Timestamp, horizons: list[int]) -> dict[int, float]:
    """用同一份可见价格数据计算多个预测期限的实际收益。"""
    if bars is None or bars.empty:
        return {}
    anchor_close = _close_at_or_before(bars, anchor)
    if anchor_close is None:
        return {}
    values = {}
    for horizon in horizons:
        target = _nth_close_after(bars, anchor, horizon)
        if target is not None:
            values[horizon] = float(target / anchor_close - 1)
    return values


def _close_at_or_before(bars: pd.DataFrame, anchor: pd.Timestamp) -> float | None:
    before = bars[pd.to_datetime(bars["date"]).dt.normalize() <= anchor]
    return float(before["close"].iloc[-1]) if not before.empty else None


def _nth_close_after(bars: pd.DataFrame, anchor: pd.Timestamp, horizon: int) -> float | None:
    after = bars[pd.to_datetime(bars["date"]).dt.normalize() > anchor].reset_index(drop=True)
    if len(after) < horizon:
        return None
    return float(after["close"].iloc[horizon - 1])
