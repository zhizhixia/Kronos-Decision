"""Qlib 点时数据读取器测试（mock D API）。"""
from __future__ import annotations

import pandas as pd

from evaluation.qlib_data import QlibPointData


class _FakeD:
    def __init__(self):
        self.last_codes = None

    def calendar(self, start_time=None, end_time=None):
        return pd.bdate_range("2024-01-01", "2024-01-31").to_pydatetime().tolist()

    def instruments(self, market=None):
        return object()

    def list_instruments(self, instruments=None, start_time=None, end_time=None, as_list=True):
        return ["SH600519", "SZ000001"]

    def features(self, codes, fields, start_time=None, end_time=None, freq="day"):
        self.last_codes = list(codes)
        index = pd.MultiIndex.from_product([["600519"], pd.bdate_range("2024-01-02", "2024-01-10")], names=["instrument", "datetime"])
        return pd.DataFrame(
            {"$open": 10.0, "$high": 11.0, "$low": 9.0, "$close": 10.0, "$volume": 1000.0, "$factor": 1.0, "$vwap": 10.2},
            index=index,
        )


def test_qlib_point_data_universe_and_anchors() -> None:
    point = QlibPointData("fixture")
    point._D = _FakeD()
    anchors = point.weekly_anchors("2024-01-01", "2024-01-31")
    assert len(anchors) >= 4
    weeks = {(anchor.isocalendar().year, anchor.isocalendar().week) for anchor in anchors}
    assert len(weeks) == len(anchors)
    assert all(anchor <= pd.Timestamp("2024-01-31") for anchor in anchors)
    universe = point.universe_at("2024-01-05")
    assert universe == ["600519", "000001"]


def test_monthly_anchors_returns_one_anchor_per_month() -> None:
    point = QlibPointData("fixture")
    fake = _FakeD()
    fake.calendar = lambda start_time=None, end_time=None: pd.bdate_range("2024-01-01", "2024-03-29").to_pydatetime().tolist()
    point._D = fake
    anchors = point.monthly_anchors("2024-01-01", "2024-03-31")
    months = {(anchor.year, anchor.month) for anchor in anchors}
    assert months == {(2024, 1), (2024, 2), (2024, 3)}
    assert anchors == sorted(anchors)
    assert all(anchor.day >= 20 for anchor in anchors)


def test_universe_excludes_benchmark_index() -> None:
    """csi300 成员表若混入 SH000300 基准，必须从股票池排除。"""
    point = QlibPointData("fixture")
    fake = _FakeD()
    fake.list_instruments = lambda instruments=None, start_time=None, end_time=None, as_list=True: ["SH600519", "SZ000001", "SH000300"]
    point._D = fake
    universe = point.universe_at("2024-01-05")
    assert universe == ["600519", "000001"]


def test_benchmark_bars_uses_sh000300_directly() -> None:
    """基准读取必须使用 SH000300，不能经 000300 -> SZ000300 的股票映射。"""
    point = QlibPointData("fixture")
    fake = _FakeD()
    point._D = fake
    bars = point.benchmark_bars("2024-01-10")
    assert not bars.empty
    assert fake.last_codes == ["SH000300"]


def test_latest_session_returns_calendar_end() -> None:
    point = QlibPointData("fixture")
    point._D = _FakeD()
    assert point.latest_session() == pd.Timestamp("2024-01-31")


def test_qlib_bars_only_contain_visible_days() -> None:
    point = QlibPointData("fixture")
    point._D = _FakeD()
    bars = point.bars_at("600519", "2024-01-10")
    assert list(bars.columns) == ["date", "open", "high", "low", "close", "volume", "amount"]
    assert bars["date"].max() <= pd.Timestamp("2024-01-10")
    assert (bars["amount"] > 0).all()


def test_qlib_bars_drop_rows_with_partial_nan() -> None:
    """停牌/缺口日仅 close 有效而 OHLCV 其余字段缺失时，整行必须丢弃。"""
    point = QlibPointData("fixture")
    fake = _FakeD()

    def features_with_gap(codes, fields, start_time=None, end_time=None, freq="day"):
        index = pd.MultiIndex.from_product([["600519"], pd.bdate_range("2024-01-02", "2024-01-05")], names=["instrument", "datetime"])
        frame = pd.DataFrame(
            {"$open": 10.0, "$high": 11.0, "$low": 9.0, "$close": 10.0, "$volume": 1000.0, "$factor": 1.0, "$vwap": 10.2},
            index=index,
        )
        frame.loc[pd.IndexSlice["600519", "2024-01-04"], ["$open", "$high", "$low", "$volume", "$vwap"]] = float("nan")
        return frame

    fake.features = features_with_gap
    point._D = fake
    bars = point.bars_at("600519", "2024-01-05")
    assert "2024-01-04" not in set(bars["date"].astype(str))
    assert len(bars) == 3
