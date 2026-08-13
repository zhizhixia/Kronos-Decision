"""实际收益回填测试。"""
from __future__ import annotations

import pandas as pd

from evaluation.backfill import backfill_actuals


def _bars(code: str, anchor: pd.Timestamp) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", anchor)
    closes = [10.0 + index * 0.1 for index in range(len(dates))]
    return pd.DataFrame({"date": dates, "open": closes, "high": closes, "low": closes, "close": closes, "volume": 1000.0, "amount": 10000.0})


def _predictions() -> pd.DataFrame:
    return pd.DataFrame([{"prediction_key": "k1", "anchor_date": "2024-02-05", "execution_date": "2024-02-06", "stock_code": "600519", "horizon": 20, "score": 0.5, "predicted_return": 0.5, "actual_return": float("nan"), "data_as_of": "2024-02-05"}])


def test_backfill_fills_actual_return_from_future_closes() -> None:
    frame = backfill_actuals(_predictions(), _bars)
    actual = frame["actual_return"].iloc[0]
    assert not pd.isna(actual)
    assert actual > 0  # 序列单调上涨


def test_backfill_returns_nan_without_enough_future_data() -> None:
    def short_bars(code, anchor):
        return _bars(code, pd.Timestamp("2024-02-05") + pd.Timedelta(days=18))  # 锚点后不足 20 根

    frame = backfill_actuals(_predictions(), short_bars)
    assert pd.isna(frame["actual_return"].iloc[0])


def test_backfill_uses_each_horizon_with_one_source_read() -> None:
    """5/20/60 日标签应分别回填，但同股票锚点只读取一次日线。"""
    predictions = pd.concat([
        _predictions().assign(prediction_key=f"k-{horizon}", horizon=horizon)
        for horizon in (5, 20, 60)
    ], ignore_index=True)
    calls: list[tuple[str, pd.Timestamp]] = []

    def counting_bars(code: str, anchor: pd.Timestamp) -> pd.DataFrame:
        calls.append((code, anchor))
        return _bars(code, anchor)

    frame = backfill_actuals(predictions, counting_bars)
    actuals = frame.set_index("horizon")["actual_return"]

    assert len(calls) == 1
    assert actuals.notna().all()
    assert actuals[5] < actuals[20] < actuals[60]
