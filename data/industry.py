"""A 股行业映射（CSV 导入，缺失时明确失败）。"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd


class IndustryMap:
    """股票代码到行业名称的本地映射。"""

    def __init__(self, path: str | Path = "data/cache/industry_map.csv") -> None:
        self.path = Path(path)

    def load(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        data = pd.read_csv(self.path, dtype=str)
        return {str(row["stock_code"]).zfill(6): str(row["industry"]) for _, row in data.iterrows()}

    def import_csv(self, source_path: str | Path) -> int:
        data = pd.read_csv(source_path, dtype=str)
        if not {"stock_code", "industry"}.issubset(data.columns):
            raise ValueError("行业映射必须包含 stock_code 和 industry 列。")
        data = data[["stock_code", "industry"]].dropna()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=self.path.parent, suffix=".tmp") as handle:
            data.to_csv(handle, index=False)
            temp_path = Path(handle.name)
        os.replace(temp_path, self.path)
        return len(data)
