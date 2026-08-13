"""点时股票池：每个锚点只能查询当时有效的成分股。"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


UNIVERSE_COLUMNS = ["stock_code", "start_date", "end_date", "source", "snapshot_version"]


@dataclass(frozen=True)
class UniverseMembership:
    """一条股票池有效区间记录。"""

    stock_code: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    source: str
    snapshot_version: str


class HistoricalUniverse:
    """CSV 存储的沪深300历史成员表。"""

    def __init__(self, path: str | Path = "data/cache/csi300_membership.csv") -> None:
        self.path = Path(path)

    def import_csv(self, source_path: str | Path) -> int:
        """导入标准化成员表，拒绝缺失有效期的非点时数据。"""
        data = pd.read_csv(source_path, dtype={"stock_code": str})
        self._validate(data)
        self._atomic_write(data[UNIVERSE_COLUMNS])
        return len(data)

    def members_at(self, anchor_date: str | pd.Timestamp) -> set[str]:
        """返回锚点当天有效的成员；无历史有效期时返回空集合。"""
        if not self.path.exists():
            return set()
        data = pd.read_csv(self.path, dtype={"stock_code": str}, parse_dates=["start_date", "end_date"])
        anchor = pd.Timestamp(anchor_date).normalize()
        active = data[(data["start_date"] <= anchor) & (data["end_date"] >= anchor)]
        return set(active["stock_code"].str.zfill(6))

    @staticmethod
    def _validate(data: pd.DataFrame) -> None:
        if list(data.columns.intersection(UNIVERSE_COLUMNS)) != UNIVERSE_COLUMNS:
            raise ValueError("历史股票池必须包含标准有效期字段。")
        starts = pd.to_datetime(data["start_date"], errors="coerce")
        ends = pd.to_datetime(data["end_date"], errors="coerce")
        if starts.isna().any() or ends.isna().any() or (starts > ends).any():
            raise ValueError("历史股票池存在无效有效期。")

    def _atomic_write(self, data: pd.DataFrame) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=self.path.parent, suffix=".tmp") as handle:
            data.to_csv(handle, index=False)
            temp_path = Path(handle.name)
        os.replace(temp_path, self.path)
