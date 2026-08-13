"""评估运行工件的原子写入与断点续跑。"""
from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import pandas as pd

PREDICTION_PROVENANCE_COLUMNS = {"prediction_key", "sampling_seed", "sampling_params_hash"}


@dataclass(frozen=True)
class EvaluationArtifacts:
    """一个可重放评估运行的目录。"""

    run_id: str
    root: Path

    @classmethod
    def open(cls, run_id: str, root: str | Path = "artifacts/evaluations") -> "EvaluationArtifacts":
        path = Path(root) / run_id
        path.mkdir(parents=True, exist_ok=True)
        return cls(run_id, path)

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        """写入带完成状态的清单，避免把中断运行伪装为成功。"""
        required = {"status", "as_of", "seed", "horizons", "model", "config_hash", "data_hash"}
        missing = required - set(manifest)
        if missing:
            raise ValueError(f"manifest 缺少字段：{sorted(missing)}")
        self._write_json("manifest.json", manifest)

    def write_frame(self, name: str, frame: pd.DataFrame) -> None:
        """仅允许计划内 CSV 工件，保持无 pyarrow 的可移植性。"""
        if name not in {"predictions", "horizon_metrics", "backtest_daily", "positions", "trades"}:
            raise ValueError("不支持的评估 CSV 工件。")
        self._write_csv(f"{name}.csv", frame)

    def write_gate_result(self, result: dict[str, Any]) -> None:
        """写入完整门禁结果，供在线报告绑定。"""
        self._write_json("gate_result.json", result)

    def write_calibration(self, profile: Any) -> None:
        """写入校准档案（CalibrationProfile 或可序列化 dict）。"""
        content = profile.as_dict() if hasattr(profile, "as_dict") else profile
        self._write_json("calibration.json", content)

    def has_prediction(self, key: str) -> bool:
        """仅复用带当前采样审计字段的完整预测检查点。"""
        path = self.root / "predictions.csv"
        if not path.exists() or path.stat().st_size < 20:
            return False
        try:
            header = set(pd.read_csv(path, nrows=0).columns)
            if not PREDICTION_PROVENANCE_COLUMNS.issubset(header):
                return False
            return key in set(pd.read_csv(path, usecols=["prediction_key"])["prediction_key"].astype(str))
        except (ValueError, KeyError):
            return False

    def _write_json(self, name: str, content: dict[str, Any]) -> None:
        text = json.dumps(_json_safe(content), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        self._atomic_text(self.root / name, text)

    def _write_csv(self, name: str, frame: pd.DataFrame) -> None:
        path = self.root / name
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=self.root, suffix=".tmp") as handle:
            frame.to_csv(handle, index=False)
            temp_path = Path(handle.name)
        os.replace(temp_path, path)

    @staticmethod
    def _atomic_text(path: Path, text: str) -> None:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as handle:
            handle.write(text)
            temp_path = Path(handle.name)
        os.replace(temp_path, path)


def _json_safe(value: Any) -> Any:
    """把评估中的非有限数转为 JSON null，保证工件可被严格解析。"""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
