"""研究预测器测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.predictor import ResearchPredictor
from model.prediction import sampling_params_hash


def _fake_predict(df, x_timestamp, y_timestamp, pred_len, sample_count, sample_batch_size, seed):
    current = float(df["close"].iloc[-1])
    paths = np.zeros((sample_count, pred_len, 6), dtype=float)
    paths[:, :, 3] = current * (1 + np.linspace(0, 0.2, pred_len))
    timestamps = pd.DatetimeIndex(pd.to_datetime(y_timestamp))
    mean_df = pd.DataFrame(paths.mean(axis=0), columns=["open", "high", "low", "close", "volume", "amount"], index=timestamps)
    params_hash = sampling_params_hash(pred_len, sample_count, sample_batch_size, 0.6, 0.9, 0)
    return type("Paths", (), {"paths": paths, "mean_df": mean_df, "sample_count": sample_count, "seed": seed, "model_id": "m", "tokenizer_id": "t", "sampling_params_hash": params_hash, "repaired_values": 0, "quality_flags": ()})()


def _fake_bars(code: str, anchor: pd.Timestamp) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=100)
    return pd.DataFrame({"date": dates, "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1000.0, "amount": 10000.0})


def _fake_calendar(last_date: pd.Timestamp, count: int) -> pd.Series:
    return pd.Series(pd.bdate_range(last_date + pd.Timedelta(days=1), periods=count), name="date")


def test_research_predictor_generates_point_in_time_rows() -> None:
    research = ResearchPredictor(_fake_predict, _fake_bars, _fake_calendar, "m", "c", "d", sample_count=10, seed=7)
    frame = research.run(["600519"], ["2024-05-01"])
    assert {"5", "20", "60"}.issubset(set(frame["horizon"].astype(str)))
    assert (frame["data_as_of"] <= frame["anchor_date"]).all()
    assert (frame["execution_date"] > frame["anchor_date"]).all()
    assert frame["raw_up_probability"].between(0, 1).all()
    assert frame["score"].is_monotonic_increasing


def test_research_predictor_skips_existing_keys() -> None:
    seen = set()

    def skip(key: str) -> bool:
        return key in seen

    research = ResearchPredictor(_fake_predict, _fake_bars, _fake_calendar, "m", "c", "d", sample_count=10, seed=7)
    first = research.run(["600519"], ["2024-05-01"], skip)
    seen.update(first["prediction_key"])
    second = research.run(["600519"], ["2024-05-01"], skip)
    assert second.empty


def test_research_predictor_is_deterministic_across_runs() -> None:
    """同一输入两次运行必须产生逐行一致的结果（可重放门禁）。"""
    first = ResearchPredictor(_fake_predict, _fake_bars, _fake_calendar, "m", "c", "d", sample_count=10, seed=7).run(["600519", "000001"], ["2024-05-01"])
    second = ResearchPredictor(_fake_predict, _fake_bars, _fake_calendar, "m", "c", "d", sample_count=10, seed=7).run(["600519", "000001"], ["2024-05-01"])
    assert first.equals(second)
    assert len(first) == 6


def test_research_predictor_derives_seed_per_stock_and_cutoff() -> None:
    """默认采样种子必须绑定模型、股票、截止日和配置，而非整轮运行的固定值。"""
    first = ResearchPredictor(_fake_predict, _fake_bars, _fake_calendar, "m", "c", "d", sample_count=10).run(["600519", "000001"], ["2024-05-01", "2024-05-02"])
    second = ResearchPredictor(_fake_predict, _fake_bars, _fake_calendar, "m", "c", "d", sample_count=10).run(["600519", "000001"], ["2024-05-01", "2024-05-02"])

    pd.testing.assert_frame_equal(first, second)
    seeds = first.groupby(["stock_code", "anchor_date"])["sampling_seed"].first()
    assert seeds.nunique() == 4
    expected = sampling_params_hash(60, 10, 10, 0.6, 0.9, 0)
    assert first["sampling_params_hash"].eq(expected).all()


def test_research_predictor_rejects_mismatched_sampling_identity() -> None:
    def mismatched(*args, **kwargs):
        paths = _fake_predict(*args, **kwargs)
        paths.sampling_params_hash = "wrong"
        return paths

    research = ResearchPredictor(mismatched, _fake_bars, _fake_calendar, "m", "c", "d", sample_count=10)

    try:
        research.run(["600519"], ["2024-05-01"])
    except RuntimeError as exc:
        assert str(exc) == "SAMPLING_PARAMS_HASH_MISMATCH"
    else:
        raise AssertionError("不兼容的采样身份必须被拒绝")
