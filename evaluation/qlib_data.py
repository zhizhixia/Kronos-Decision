"""Qlib 点时数据读取器：历史锚点只能看到当时可见的数据。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class QlibPointData:
    """封装 Qlib 中国日频数据的点时访问；所有导入均惰性。"""

    def __init__(self, provider_uri: str | Path | None = None) -> None:
        self.provider_uri = str(provider_uri or (Path.home() / ".qlib" / "qlib_data" / "cn_data"))
        self._D: Any = None

    def init(self) -> None:
        if self._D is not None:
            return
        try:
            import qlib
            from qlib.data import D
        except ImportError as exc:
            raise RuntimeError("未安装 pyqlib，无法读取 Qlib 点时数据。") from exc
        if not Path(self.provider_uri).exists():
            raise RuntimeError(f"Qlib 中国市场数据目录不存在：{self.provider_uri}")
        qlib.init(provider_uri=self.provider_uri, region="cn")
        try:
            from qlib.config import C
            C["joblib_backend"] = "sequential"
            C["n_jobs"] = 1
        except Exception:
            pass
        self._D = D

    def weekly_anchors(self, start: str, end: str) -> list[pd.Timestamp]:
        """返回 [start, end] 内每周最后一个交易日的锚点列表。"""
        self.init()
        calendar = pd.DatetimeIndex(self._D.calendar(start_time=start, end_time=end)).sort_values()
        anchors: list[pd.Timestamp] = []
        current_week = None
        for day in calendar:
            week = (day.isocalendar().year, day.isocalendar().week)
            if current_week is None or week != current_week:
                if current_week is not None:
                    anchors.append(current_anchor)
                current_week = week
            current_anchor = day
        if current_week is not None:
            anchors.append(current_anchor)
        return anchors

    def monthly_anchors(self, start: str, end: str) -> list[pd.Timestamp]:
        """返回 [start, end] 内每月最后一个交易日的锚点列表。"""
        self.init()
        calendar = pd.DatetimeIndex(self._D.calendar(start_time=start, end_time=end)).sort_values()
        anchors: list[pd.Timestamp] = []
        current_month = None
        for day in calendar:
            month = (day.year, day.month)
            if current_month is None or month != current_month:
                if current_month is not None:
                    anchors.append(current_anchor)
                current_month = month
            current_anchor = day
        if current_month is not None:
            anchors.append(current_anchor)
        return anchors

    def future_sessions(self, last_date: pd.Timestamp, count: int) -> pd.Series:
        """返回给定日期之后的下一个交易日开始的 count 个交易日。"""
        self.init()
        calendar = pd.DatetimeIndex(self._D.calendar(start_time=str(last_date)))
        future = calendar[calendar > pd.Timestamp(last_date)][:count]
        return pd.Series(future, name="date")

    def latest_session(self) -> pd.Timestamp:
        """返回日历中最后一个交易日（用于期末可见序列的研究读取）。"""
        self.init()
        calendar = pd.DatetimeIndex(self._D.calendar()).sort_values()
        if calendar.empty:
            raise RuntimeError("Qlib 日历为空，无法确定最新交易日。")
        return calendar.max()

    def universe_at(self, anchor: str | pd.Timestamp) -> list[str]:
        """返回锚点当时有效且数据充足的沪深300成分（点时列表）。"""
        self.init()
        instruments = self._D.instruments(market="csi300")
        codes = self._D.list_instruments(instruments=instruments, start_time=str(anchor), end_time=str(anchor), as_list=True)
        return [_to_local_code(code) for code in codes if _to_local_code(code) != "000300"]

    def bars_at(self, code: str, anchor: str | pd.Timestamp, lookback_days: int = 760) -> pd.DataFrame:
        """返回截至锚点可见的日线；OHLCV 任一为空即停牌/缺口日，整行丢弃。amount 优先 vwap×volume，缺失时用 OHLC 均值兜底。"""
        return self._bars_for(_to_qlib_code(code), anchor, lookback_days)

    def benchmark_bars(self, anchor: str | pd.Timestamp, lookback_days: int = 1400, benchmark: str = "SH000300") -> pd.DataFrame:
        """返回沪深300基准日线（mini 数据为合成 SH000300），显式代码避免映射成 SZ000300。"""
        return self._bars_for(benchmark, anchor, lookback_days)

    def _bars_for(self, qlib_code: str, anchor: str | pd.Timestamp, lookback_days: int) -> pd.DataFrame:
        self.init()
        start = (pd.Timestamp(anchor) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        end = str(anchor)
        fields = ["$open", "$high", "$low", "$close", "$volume", "$factor", "$vwap"]
        features = self._D.features([qlib_code], fields, start_time=start, end_time=end, freq="day")
        if features is None or features.empty:
            return pd.DataFrame()
        frame = features.droplevel(0).rename(columns={f"${name}": name for name in ("open", "high", "low", "close", "volume", "factor", "vwap")})
        frame.index = pd.to_datetime(frame.index)
        frame = frame.dropna(subset=["open", "high", "low", "close", "volume"]).sort_index()
        if frame.empty:
            return pd.DataFrame()
        frame = frame.reset_index().rename(columns={"datetime": "date"})
        amount_from_vwap = frame["vwap"] * frame["volume"]
        amount_fallback = frame["volume"] * frame[["open", "high", "low", "close"]].mean(axis=1)
        frame["amount"] = amount_from_vwap.where(amount_from_vwap.notna(), amount_fallback)
        return frame[["date", "open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)


def _to_local_code(code: str) -> str:
    """SH600519 -> 600519。"""
    cleaned = str(code).replace("SH", "").replace("SZ", "")
    return cleaned.zfill(6)


def _to_qlib_code(code: str) -> str:
    """600519 -> SH600519。"""
    cleaned = str(code).zfill(6)
    market = "SH" if cleaned.startswith(("5", "6", "9")) else "SZ"
    return f"{market}{cleaned}"
