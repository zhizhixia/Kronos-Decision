"""研究预测器：从 Qlib 点数据或实时源生成时点预测工件。"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from decision.versioning import derive_sampling_seed
from evaluation.contracts import prediction_key
from model.prediction import sampling_params_hash

HORIZONS = (5, 20, 60)


class ResearchPredictor:
    """按锚点生成多期限时点预测；同一键命中缓存时跳过重复推理。"""

    def __init__(self, predict_paths: Callable, fetch_bars: Callable, calendar: Callable, model_hash: str, config_hash: str, data_hash: str, sample_count: int = 100, sample_batch_size: int = 20, seed: int | None = None) -> None:
        self.predict_paths = predict_paths
        self.fetch_bars = fetch_bars
        self.calendar = calendar
        self.model_hash = model_hash
        self.config_hash = config_hash
        self.data_hash = data_hash
        self.sample_count = sample_count
        self.sample_batch_size = min(sample_batch_size, sample_count)
        self.seed = seed
        self.sampling_params_hash = sampling_params_hash(
            max(HORIZONS), sample_count, self.sample_batch_size, 0.6, 0.9, 0
        )

    def run(self, stock_codes: list[str], anchor_dates: list[str | pd.Timestamp], skip_existing: Callable[[str], bool] | None = None) -> pd.DataFrame:
        """为每只股票、每个锚点生成预测行。"""
        rows: list[dict] = []
        for anchor in anchor_dates:
            anchor_ts = pd.Timestamp(anchor).normalize()
            for code in stock_codes:
                key = prediction_key(
                    code,
                    anchor_ts.date().isoformat(),
                    self.model_hash,
                    self.config_hash,
                    self.sampling_params_hash,
                )
                if skip_existing is not None and skip_existing(key):
                    continue
                paths, current, last_visible = self._predict_stock(code, anchor_ts)
                rows.extend(self._to_rows(code, anchor_ts, key, paths, current, last_visible))
        return pd.DataFrame(rows)

    def _predict_stock(self, code: str, anchor: pd.Timestamp) -> tuple[pd.DataFrame, float, pd.Timestamp]:
        bars = self.fetch_bars(code, anchor)
        if bars is None or bars.empty:
            raise RuntimeError(f"{code} 在 {anchor.date()} 没有可见数据。")
        bars = bars.loc[pd.to_datetime(bars["date"]).dt.normalize() <= anchor].sort_values("date")
        if len(bars) < 60:
            raise RuntimeError(f"{code} 在 {anchor.date()} 的数据不足 60 根。")
        last_visible = pd.Timestamp(bars["date"].iloc[-1])
        timestamps = self.calendar(last_visible, max(HORIZONS))
        pred_len = int(len(timestamps))
        if pred_len <= 0:
            raise RuntimeError(f"{code} 在 {anchor.date()} 之后没有可预测的交易日。")
        sampling_seed = self._sampling_seed(code, last_visible)
        paths = self.predict_paths(bars, pd.Series(bars["date"].values), timestamps, pred_len, self.sample_count, self.sample_batch_size, sampling_seed)
        actual_hash = str(getattr(paths, "sampling_params_hash", ""))
        if actual_hash != self.sampling_params_hash:
            raise RuntimeError("SAMPLING_PARAMS_HASH_MISMATCH")
        return paths, float(bars["close"].iloc[-1]), last_visible

    def _sampling_seed(self, code: str, last_visible: pd.Timestamp) -> int:
        """派生逐股票、逐截止日的稳定真实采样种子。"""
        return derive_sampling_seed(self.model_hash, code, pd.Timestamp(last_visible), self.config_hash, self.seed)

    def _to_rows(self, code: str, anchor: pd.Timestamp, key: str, paths, current: float, last_visible: pd.Timestamp) -> list[dict]:
        rows = []
        close_paths = paths.paths[:, :, 3]
        execution = self.calendar(last_visible, 1)
        execution_date = execution.iloc[0].date().isoformat()
        for horizon in HORIZONS:
            if paths.paths.shape[1] < horizon:
                continue
            returns = close_paths[:, horizon - 1] / current - 1
            q05, q50, q95 = (float(pd.Series(returns).quantile(0.05)), float(pd.Series(returns).quantile(0.5)), float(pd.Series(returns).quantile(0.95)))
            rows.append({
                "prediction_key": key,
                "anchor_date": anchor.date().isoformat(),
                "execution_date": execution_date,
                "stock_code": code,
                "horizon": horizon,
                "score": float(q50),
                "predicted_return": float(q50),
                "actual_return": float("nan"),
                "data_as_of": anchor.date().isoformat(),
                "model_hash": self.model_hash,
                "config_hash": self.config_hash,
                "data_hash": self.data_hash,
                "raw_up_probability": float((returns > 0).mean()),
                "q05": q05,
                "q50": q50,
                "q95": q95,
                "sampling_seed": int(getattr(paths, "seed", self._sampling_seed(code, last_visible))),
                "sampling_params_hash": str(getattr(paths, "sampling_params_hash", "")),
            })
        return rows
