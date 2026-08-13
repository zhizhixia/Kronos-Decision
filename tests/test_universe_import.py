"""历史沪深300成员导入器测试。"""
from __future__ import annotations

import pandas as pd

from data.universe_import import _import_memberships, _instrument_bounds


class _FakeInstrument:
    def __init__(self, code, start, end) -> None:
        self.code = code
        self.start_time = start
        self.end_time = end


class _FakeD:
    """按月份变化的点时成员：1月含600519/000001，2月换入SH000300，3月只剩000001。"""

    def __init__(self) -> None:
        self._instruments = [_FakeInstrument("SH600519", "2010-01-01", "2025-01-01"), _FakeInstrument("SZ000001", "1991-04-03", "2999-12-31")]

    def instruments(self, market=None):
        return self._instruments

    def calendar(self, start_time=None, end_time=None):
        return pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"]).tolist()

    def list_instruments(self, instruments=None, start_time=None, end_time=None, as_list=True):
        month = pd.Timestamp(start_time).month
        if month == 1:
            return ["SH600519", "SZ000001", "market"]
        if month == 2:
            return ["SH600519", "SH000300"]
        return ["SZ000001"]


def test_instrument_bounds_normalize_market_prefix() -> None:
    code, start, end = _instrument_bounds(_FakeInstrument("SH600519", "2010-01-01", "2025-01-01"))
    assert code == "600519"
    assert start == "2010-01-01"
    assert end == "2025-01-01"


def test_import_memberships_writes_membership_csv(tmp_path) -> None:
    output = tmp_path / "members.csv"
    count = _import_memberships(_FakeD(), output)
    assert count == 3
    frame = pd.read_csv(output, dtype={"stock_code": str})
    assert list(frame["stock_code"]) == ["000001", "000001", "600519"]
    assert frame.iloc[0]["end_date"] == "2024-01-31"
    assert frame.iloc[1]["start_date"] == "2024-03-01"
    assert frame.iloc[2]["start_date"] == "2024-01-01"
    assert frame.iloc[2]["end_date"] == "2024-02-29"
    assert list(frame.columns) == ["stock_code", "start_date", "end_date", "source", "snapshot_version"]
