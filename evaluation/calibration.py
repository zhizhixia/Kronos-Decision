"""三年滚动 Conformal 区间与 Isotonic 上涨概率校准。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from evaluation.metrics import expected_calibration_error, interval_coverage

CALIBRATION_HORIZONS = (5, 20, 60)


@dataclass(frozen=True)
class CalibrationProfile:
    """与期限、模型、数据和配置绑定的校准档案。"""

    horizon: int
    fitted_at: str
    expires_at: str
    anchor_count: int
    observation_count: int
    conformal_adjustment: float
    isotonic_x: tuple[float, ...]
    isotonic_y: tuple[float, ...]
    model_hash: str
    data_hash: str
    config_hash: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def calibrate_probability(self, values: np.ndarray | pd.Series) -> np.ndarray:
        """使用持久化阈值校准原始上涨概率。"""
        return np.interp(np.asarray(values, dtype=float), self.isotonic_x, self.isotonic_y)

    def adjust_interval(self, lower: pd.Series, upper: pd.Series) -> tuple[pd.Series, pd.Series]:
        """以滚动 Conformal 残差扩张80%区间。"""
        return lower - self.conformal_adjustment, upper + self.conformal_adjustment

    def is_compatible(self, model_hash: str, data_hash: str, config_hash: str, now: datetime | None = None) -> bool:
        """检查三类哈希和30天有效期。"""
        current = now or datetime.now()
        return (model_hash, data_hash, config_hash) == (self.model_hash, self.data_hash, self.config_hash) and current <= datetime.fromisoformat(self.expires_at)


@dataclass(frozen=True)
class MultiHorizonCalibration:
    """5/20/60 日期限各自独立的校准档案和质量指标。"""

    metrics: dict[int, dict[str, float | bool | int]]
    profiles: dict[int, CalibrationProfile]

    def as_dict(self) -> dict[str, Any]:
        """转换为版本化工件，未达到样本门槛的期限也必须可见。"""
        return {
            "schema_version": "2.0",
            "metrics": {str(horizon): values for horizon, values in self.metrics.items()},
            "profiles": {str(horizon): profile.as_dict() for horizon, profile in self.profiles.items()},
        }


@dataclass(frozen=True)
class CalibrationAssessment:
    """严格滚动样本外校准的门禁统计与最新在线档案。"""

    coverage_80: float
    up_probability_ece: float
    profile: CalibrationProfile | None
    oos_anchor_count: int
    oos_observation_count: int


def fit_calibration_profile(
    frame: pd.DataFrame,
    horizon: int,
    model_hash: str,
    data_hash: str,
    config_hash: str,
    fitted_at: datetime | None = None,
    before_anchor: pd.Timestamp | str | None = None,
) -> CalibrationProfile:
    """使用最新预测锚点之前的三年样本外记录拟合独立期限档案。"""
    required = {"anchor_date", "horizon", "q05", "q95", "raw_up_probability", "actual_return"}
    if not required.issubset(frame):
        raise ValueError(f"校准数据缺少字段：{sorted(required - set(frame))}")
    reference = before_anchor
    if reference is None:
        latest = _rolling_window(frame, horizon)
        reference = latest["anchor_date"].max()
    selected = _rolling_window(frame, horizon, reference)
    anchor_count = selected["anchor_date"].nunique()
    if anchor_count < 24 or len(selected) < 2000:
        raise ValueError("校准至少需要24个独立锚点和2000个有效股票观测。")
    adjustment = _conformal_adjustment(selected)
    isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    isotonic.fit(selected["raw_up_probability"].astype(float), (selected["actual_return"] > 0).astype(float))
    fitted = fitted_at or datetime.now()
    return CalibrationProfile(horizon, fitted.isoformat(), (fitted + timedelta(days=30)).isoformat(), anchor_count, len(selected), adjustment, tuple(map(float, isotonic.X_thresholds_)), tuple(map(float, isotonic.y_thresholds_)), model_hash, data_hash, config_hash)


def _rolling_window(
    frame: pd.DataFrame,
    horizon: int,
    before_anchor: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """取期限内最近三年有效标签；指定锚点时严格排除该锚点及之后数据。"""
    selected = frame.loc[frame["horizon"].astype(int) == horizon].copy()
    selected["anchor_date"] = pd.to_datetime(selected["anchor_date"], errors="raise")
    selected = selected.dropna(subset=["q05", "q95", "raw_up_probability", "actual_return"])
    if selected.empty:
        raise ValueError(f"期限 {horizon} 没有有效校准观测。")
    if before_anchor is None:
        cutoff = selected["anchor_date"].max() - pd.DateOffset(years=3)
        return selected.loc[selected["anchor_date"] >= cutoff]
    reference = pd.Timestamp(before_anchor).normalize()
    cutoff = reference - pd.DateOffset(years=3)
    return selected.loc[(selected["anchor_date"] >= cutoff) & (selected["anchor_date"] < reference)]


def _conformal_adjustment(frame: pd.DataFrame) -> float:
    residual = np.maximum.reduce([(frame["q05"] - frame["actual_return"]).to_numpy(), (frame["actual_return"] - frame["q95"]).to_numpy(), np.zeros(len(frame))])
    quantile = min(1.0, np.ceil((len(residual) + 1) * 0.8) / len(residual))
    return float(np.quantile(residual, quantile, method="higher"))


def _rolling_oos_observations(
    frame: pd.DataFrame,
    horizon: int,
    model_hash: str,
    data_hash: str,
    config_hash: str,
    fitted_at: datetime | None,
) -> pd.DataFrame:
    """逐锚点拟合此前样本并返回严格样本外的校准预测与标签。"""
    selected = _rolling_window(frame, horizon)
    observations: list[pd.DataFrame] = []
    for anchor in sorted(selected["anchor_date"].unique()):
        try:
            profile = fit_calibration_profile(frame, horizon, model_hash, data_hash, config_hash, fitted_at, anchor)
        except ValueError:
            continue
        current = selected.loc[selected["anchor_date"] == anchor]
        lower, upper = profile.adjust_interval(current["q05"].astype(float), current["q95"].astype(float))
        observations.append(pd.DataFrame({"anchor_date": current["anchor_date"], "actual_return": current["actual_return"].astype(float), "lower": lower, "upper": upper, "probability": profile.calibrate_probability(current["raw_up_probability"].astype(float))}))
    if not observations:
        return pd.DataFrame(columns=["anchor_date", "actual_return", "lower", "upper", "probability"])
    return pd.concat(observations, ignore_index=True)


def assess_calibration(
    frame: pd.DataFrame,
    model_hash: str,
    data_hash: str,
    config_hash: str,
    fitted_at: datetime | None = None,
    horizon: int = 20,
) -> CalibrationAssessment:
    """拟合在线档案，并只以滚动样本外观测计算覆盖率与 ECE。"""
    try:
        profile = fit_calibration_profile(frame, horizon, model_hash, data_hash, config_hash, fitted_at)
    except ValueError:
        return CalibrationAssessment(0.0, 1.0, None, 0, 0)
    observations = _rolling_oos_observations(frame, horizon, model_hash, data_hash, config_hash, fitted_at)
    if observations.empty:
        return CalibrationAssessment(0.0, 1.0, profile, 0, 0)
    coverage = interval_coverage(observations["lower"], observations["upper"], observations["actual_return"])
    ece = expected_calibration_error(observations["probability"], observations["actual_return"] > 0)
    return CalibrationAssessment(float(coverage), float(ece), profile, int(observations["anchor_date"].nunique()), len(observations))


def apply_calibration_metrics(
    frame: pd.DataFrame,
    model_hash: str,
    data_hash: str,
    config_hash: str,
    fitted_at: datetime | None = None,
    horizon: int = 20,
) -> tuple[float, float, CalibrationProfile | None]:
    """兼容入口：返回严格样本外覆盖率、ECE 与最新在线档案。"""
    assessment = assess_calibration(frame, model_hash, data_hash, config_hash, fitted_at, horizon)
    return assessment.coverage_80, assessment.up_probability_ece, assessment.profile


def apply_multi_horizon_calibration_metrics(
    frame: pd.DataFrame,
    model_hash: str,
    data_hash: str,
    config_hash: str,
    fitted_at: datetime | None = None,
) -> MultiHorizonCalibration:
    """独立拟合 5/20/60 日档案；任一期不足时保留失败闭合指标。"""
    metrics: dict[int, dict[str, float | bool | int]] = {}
    profiles: dict[int, CalibrationProfile] = {}
    for horizon in CALIBRATION_HORIZONS:
        assessment = assess_calibration(frame, model_hash, data_hash, config_hash, fitted_at, horizon)
        metrics[horizon] = {"coverage_80": assessment.coverage_80, "up_probability_ece": assessment.up_probability_ece, "available": assessment.profile is not None and assessment.oos_observation_count > 0, "oos_anchor_count": assessment.oos_anchor_count, "oos_observation_count": assessment.oos_observation_count}
        if assessment.profile is not None:
            profiles[horizon] = assessment.profile
    return MultiHorizonCalibration(metrics, profiles)
