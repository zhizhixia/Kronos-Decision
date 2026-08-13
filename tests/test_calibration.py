"""滚动校准档案测试。"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from evaluation.calibration import apply_calibration_metrics, apply_multi_horizon_calibration_metrics, assess_calibration, fit_calibration_profile


def _calibration_frame() -> pd.DataFrame:
    anchors = pd.date_range("2022-01-07", periods=25, freq="4W")
    rows = []
    for anchor_index, anchor in enumerate(anchors):
        for stock_index in range(100):
            raw = stock_index / 99
            actual = (raw - 0.5) / 10 + (anchor_index % 3 - 1) / 1000
            rows.append({"anchor_date": anchor, "horizon": 20, "q05": actual - 0.05, "q95": actual + 0.05, "raw_up_probability": raw, "actual_return": actual})
    return pd.DataFrame(rows)


def test_calibration_requires_minimum_independent_evidence() -> None:
    with pytest.raises(ValueError, match="24个独立锚点"):
        fit_calibration_profile(_calibration_frame().head(100), 20, "m", "d", "c")


def test_profile_is_hash_bound_expires_and_calibrates() -> None:
    fitted = datetime(2025, 1, 1)
    profile = fit_calibration_profile(_calibration_frame(), 20, "m", "d", "c", fitted)
    calibrated = profile.calibrate_probability(np.array([0.1, 0.9]))
    assert calibrated[0] <= calibrated[1]
    assert profile.is_compatible("m", "d", "c", fitted + timedelta(days=30))
    assert not profile.is_compatible("other", "d", "c", fitted)
    assert not profile.is_compatible("m", "d", "c", fitted + timedelta(days=31))


def test_conformal_interval_never_shrinks() -> None:
    profile = fit_calibration_profile(_calibration_frame(), 20, "m", "d", "c")
    lower, upper = profile.adjust_interval(pd.Series([-0.1]), pd.Series([0.1]))
    assert lower.iloc[0] <= -0.1
    assert upper.iloc[0] >= 0.1


def test_apply_calibration_metrics_fails_closed_when_evidence_insufficient() -> None:
    coverage, ece, profile = apply_calibration_metrics(_calibration_frame().head(100), "m", "d", "c")
    assert coverage == 0.0
    assert ece == 1.0
    assert profile is None


def test_apply_calibration_metrics_computes_coverage_and_ece() -> None:
    frame = _calibration_frame()
    coverage, ece, profile = apply_calibration_metrics(frame, "m", "d", "c", datetime(2025, 1, 1))
    assert profile is not None
    assert coverage == 1.0
    assert ece < 0.1


def test_rolling_profile_excludes_target_anchor_labels() -> None:
    """待评估锚点的标签变化不能改变此前窗口拟合的校准档案。"""
    frame = _calibration_frame()
    target = frame["anchor_date"].max()
    profile = fit_calibration_profile(frame, 20, "m", "d", "c", before_anchor=target)
    changed = frame.copy()
    changed.loc[changed["anchor_date"] == target, "actual_return"] *= -10
    comparison = fit_calibration_profile(changed, 20, "m", "d", "c", before_anchor=target)
    assert profile.conformal_adjustment == comparison.conformal_adjustment
    assert profile.isotonic_x == comparison.isotonic_x
    assert profile.isotonic_y == comparison.isotonic_y


def test_default_profile_excludes_latest_anchor_labels() -> None:
    """在线默认档案不得把最新预测锚点的真实标签用于自身校准。"""
    frame = _calibration_frame()
    profile = fit_calibration_profile(frame, 20, "m", "d", "c")
    changed = frame.copy()
    changed.loc[changed["anchor_date"] == changed["anchor_date"].max(), "actual_return"] *= -10
    comparison = fit_calibration_profile(changed, 20, "m", "d", "c")

    assert profile.anchor_count == 24
    assert profile.conformal_adjustment == comparison.conformal_adjustment
    assert profile.isotonic_y == comparison.isotonic_y


def test_assessment_reports_only_strictly_out_of_sample_observations() -> None:
    assessment = assess_calibration(_calibration_frame(), "m", "d", "c", datetime(2025, 1, 1))
    assert assessment.profile is not None
    assert assessment.oos_anchor_count == 1
    assert assessment.oos_observation_count == 100


def test_multi_horizon_calibration_keeps_independent_profiles() -> None:
    """5/20/60 日期限必须各自拟合，而不能复用20日档案。"""
    frame = pd.concat([_calibration_frame().assign(horizon=horizon) for horizon in (5, 20, 60)], ignore_index=True)
    result = apply_multi_horizon_calibration_metrics(frame, "m", "d", "c", datetime(2025, 1, 1))

    assert set(result.profiles) == {5, 20, 60}
    assert {profile.horizon for profile in result.profiles.values()} == {5, 20, 60}
    assert all(result.metrics[horizon]["available"] for horizon in (5, 20, 60))
