"""评估工件与在线单股报告的绑定。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.calibration import CalibrationProfile


@dataclass(frozen=True)
class HorizonEvidence:
    """单一预测期限的横截面、区间与概率校准证据。"""

    horizon: int
    cross_section_rank: float
    raw_up_probability: float
    calibrated_up_probability: float | None
    q05: float
    q50: float
    q95: float

    def as_dict(self) -> dict[str, float | int | None]:
        """转换为报告可用的稳定期限证据。"""
        return {
            "horizon": self.horizon,
            "cross_section_rank": self.cross_section_rank,
            "raw_up_probability": self.raw_up_probability,
            "calibrated_up_probability": self.calibrated_up_probability,
            "q05": self.q05,
            "q50": self.q50,
            "q95": self.q95,
        }


@dataclass(frozen=True)
class StockEvidence:
    """从评估运行中提取的单股点时证据。"""

    stock_code: str
    anchor_date: pd.Timestamp
    horizon: int
    cross_section_rank: float
    raw_up_probability: float
    q05: float
    q50: float
    q95: float
    calibrated_up_probability: float | None
    short_rank: float | None
    short_probability: float | None
    long_rank: float | None
    long_probability: float | None
    model_hash: str
    data_hash: str
    config_hash: str
    sampling_seed: int | None
    sampling_params_hash: str | None
    horizons: dict[int, HorizonEvidence] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """转换为 API 可序列化的单股证据。"""
        return {
            "stock_code": self.stock_code,
            "anchor_date": self.anchor_date.isoformat(),
            "horizon": self.horizon,
            "cross_section_rank": self.cross_section_rank,
            "raw_up_probability": self.raw_up_probability,
            "calibrated_up_probability": self.calibrated_up_probability,
            "short_rank": self.short_rank,
            "short_probability": self.short_probability,
            "long_rank": self.long_rank,
            "long_probability": self.long_probability,
            "q05": self.q05,
            "q50": self.q50,
            "q95": self.q95,
            "model_hash": self.model_hash,
            "data_hash": self.data_hash,
            "config_hash": self.config_hash,
            "sampling_seed": self.sampling_seed,
            "sampling_params_hash": self.sampling_params_hash,
            "horizons": {str(horizon): item.as_dict() for horizon, item in self.horizons.items()},
        }


@dataclass(frozen=True)
class EvaluationBinding:
    """最新评估运行的可读视图。"""

    run_id: str
    manifest: dict[str, Any]
    gate_result: dict[str, Any]
    calibration: CalibrationProfile | None
    predictions: pd.DataFrame
    calibrations: dict[int, CalibrationProfile] = field(default_factory=dict)

    @classmethod
    def load_latest(cls, root: str | Path = "artifacts/evaluations", now: datetime | None = None) -> "EvaluationBinding | None":
        """优先读取最新正式运行；没有正式运行时才回退最新诊断工件。"""
        manifests = sorted(Path(root).glob("*/manifest.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        selected = _select_latest_run(manifests)
        if selected is None:
            return None
        manifest_path, manifest, gate_result = selected
        run_dir = manifest_path.parent
        calibrations = _read_calibrations(run_dir)
        return cls(
            run_dir.name,
            manifest,
            gate_result,
            calibrations.get(20),
            _read_predictions(run_dir),
            calibrations,
        )

    def evidence_for(self, stock_code: str, horizon: int = 20, now: datetime | None = None) -> StockEvidence | None:
        """提取指定股票最近锚点的5/20/60日独立校准证据。"""
        required = {"prediction_key", "anchor_date", "stock_code", "horizon", "score", "raw_up_probability", "q05", "q50", "q95", "model_hash", "config_hash", "data_hash"}
        if not required.issubset(self.predictions):
            return None
        frame = self.predictions.copy()
        frame["anchor_date"] = pd.to_datetime(frame["anchor_date"])
        code = str(stock_code).zfill(6)
        selected = frame[(frame["stock_code"].astype(str).str.zfill(6) == code) & (frame["horizon"].astype(int) == horizon)]
        if selected.empty:
            return None
        latest = selected.sort_values("anchor_date").iloc[-1]
        anchor = pd.Timestamp(latest["anchor_date"])
        main = _build_horizon_evidence(latest, frame, self._profile_for(horizon), now)
        if main is None:
            return None
        horizons: dict[int, HorizonEvidence] = {horizon: main}
        for item_horizon in (5, 20, 60):
            row = _horizon_row(frame, code, item_horizon, anchor)
            item = _build_horizon_evidence(row, frame, self._profile_for(item_horizon), now)
            if item is not None:
                horizons[item_horizon] = item
        short = horizons.get(5)
        long = horizons.get(60)
        return StockEvidence(
            code, anchor, horizon, main.cross_section_rank, main.raw_up_probability, main.q05, main.q50, main.q95,
            main.calibrated_up_probability, short.cross_section_rank if short else None, short.calibrated_up_probability if short else None,
            long.cross_section_rank if long else None, long.calibrated_up_probability if long else None,
            str(latest["model_hash"]), str(latest["data_hash"]), str(latest["config_hash"]), _optional_int(latest.get("sampling_seed")), _optional_text(latest.get("sampling_params_hash")), horizons,
        )

    def _profile_for(self, horizon: int) -> CalibrationProfile | None:
        """返回指定期限档案，并兼容既有的单一20日档案。"""
        if horizon in self.calibrations:
            return self.calibrations[horizon]
        return self.calibration if horizon == 20 else None


def _horizon_row(frame: pd.DataFrame, stock_code: str, horizon: int, anchor: pd.Timestamp) -> pd.Series | None:
    rows = frame[(frame["stock_code"].astype(str).str.zfill(6) == stock_code) & (frame["horizon"].astype(int) == horizon) & (frame["anchor_date"] == anchor)]
    return rows.iloc[-1] if not rows.empty else None


def _optional_int(value: Any) -> int | None:
    """把 CSV 可选采样种子转换为整数，缺失保持不可用。"""
    return int(value) if value is not None and pd.notna(value) else None


def _optional_text(value: Any) -> str | None:
    """把 CSV 可选字符串字段转换为稳定值，缺失保持不可用。"""
    return str(value) if value is not None and pd.notna(value) else None


def _build_horizon_evidence(row, frame: pd.DataFrame, profile: CalibrationProfile | None, now: datetime | None) -> HorizonEvidence | None:
    """提取单一期限证据，仅在兼容档案存在时校准概率与区间。"""
    if row is None or "score" not in row or "raw_up_probability" not in row:
        return None
    anchor = pd.Timestamp(row["anchor_date"])
    horizon = int(row["horizon"])
    comparable = frame[(frame["anchor_date"] == anchor) & (frame["horizon"].astype(int) == horizon)]
    rank = float((comparable["score"].astype(float) <= float(row["score"])).mean())
    q05, q50, q95 = float(row["q05"]), float(row["q50"]), float(row["q95"])
    calibrated = None
    if profile is not None and profile.is_compatible(str(row["model_hash"]), str(row["data_hash"]), str(row["config_hash"]), now):
        calibrated = float(profile.calibrate_probability([float(row["raw_up_probability"])])[0])
        lower, upper = profile.adjust_interval(pd.Series([q05]), pd.Series([q95]))
        q05, q95 = float(lower.iloc[0]), float(upper.iloc[0])
    return HorizonEvidence(horizon, rank, float(row["raw_up_probability"]), calibrated, q05, q50, q95)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _select_latest_run(manifests: list[Path]) -> tuple[Path, dict[str, Any], dict[str, Any]] | None:
    """按修改时间挑选最新正式运行；损坏 manifest 不得阻断其他工件。"""
    candidates = []
    for path in manifests:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        gate = _read_json(path.with_name("gate_result.json")) or {}
        candidates.append((path, manifest, gate))
    for candidate in candidates:
        _, manifest, gate = candidate
        if manifest.get("formal_protocol") and (gate.get("checks") or {}).get("FORMAL_PROTOCOL"):
            return candidate
    return candidates[0] if candidates else None


def _read_calibrations(run_dir: Path) -> dict[int, CalibrationProfile]:
    """读取 v2 多期限档案，并兼容既有单期限 calibration.json。"""
    payload = _read_json(run_dir / "calibration.json")
    if not payload:
        return {}
    raw_profiles = payload.get("profiles") if isinstance(payload, dict) else None
    if isinstance(raw_profiles, dict):
        result: dict[int, CalibrationProfile] = {}
        for horizon, profile_payload in raw_profiles.items():
            profile = _profile_from_payload(profile_payload)
            if profile is not None:
                result[int(horizon)] = profile
        return result
    profile = _profile_from_payload(payload)
    return {profile.horizon: profile} if profile is not None else {}


def _profile_from_payload(payload: dict[str, Any]) -> CalibrationProfile | None:
    """把单个 JSON 校准档案转换为强类型对象。"""
    try:
        return CalibrationProfile(
            int(payload["horizon"]), payload["fitted_at"], payload["expires_at"], int(payload["anchor_count"]),
            int(payload["observation_count"]), float(payload["conformal_adjustment"]), tuple(float(x) for x in payload["isotonic_x"]),
            tuple(float(y) for y in payload["isotonic_y"]), payload["model_hash"], payload["data_hash"], payload["config_hash"],
        )
    except (KeyError, TypeError, ValueError):
        return None


def _read_predictions(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"stock_code": str})
