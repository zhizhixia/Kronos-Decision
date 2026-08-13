"""从 Qlib instruments 生成历史沪深300成员表（带有效区间）。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from data.universe import HistoricalUniverse


def import_from_qlib(provider_uri: str | Path | None = None, output: str | Path = "data/cache/csi300_membership.csv") -> int:
    """从 Qlib csi300 重建每只成分股的有效区间并写入标准化成员表（月度点时精度）。"""
    import qlib
    from qlib.data import D

    uri = str(provider_uri or (Path.home() / ".qlib" / "qlib_data" / "cn_data"))
    if not Path(uri).exists():
        raise RuntimeError(f"Qlib 中国市场数据目录不存在：{uri}")
    qlib.init(provider_uri=uri, region="cn")
    return _import_memberships(D, output)


def _import_memberships(D: Any, output: str | Path) -> int:
    """按月检查点枚举 csi300 成员并写入成员表；D 可注入便于离线测试。"""
    instruments = D.instruments(market="csi300")
    calendar = pd.DatetimeIndex(D.calendar(start_time="2005-01-01", end_time="2999-12-31")).sort_values()
    memberships: dict[str, list[tuple[str, str]]] = {}
    current_month: tuple[int, int] | None = None
    month_last_day: pd.Timestamp | None = None
    for day in calendar:
        month = (day.year, day.month)
        if current_month is not None and month != current_month:
            _collect_month_members(D, instruments, month_last_day, memberships)
        current_month = month
        month_last_day = day
    if current_month is not None:
        _collect_month_members(D, instruments, month_last_day, memberships)
    rows = []
    for code, intervals in memberships.items():
        for start, end in intervals:
            rows.append({"stock_code": code, "start_date": start, "end_date": end, "source": "qlib-csi300", "snapshot_version": datetime.now().strftime("%Y%m%d")})
    frame = pd.DataFrame(rows)
    frame = frame[frame["stock_code"].str.fullmatch(r"\d{6}") & (frame["stock_code"] != "000300")]
    frame = frame.drop_duplicates(keep="last").sort_values(["stock_code", "start_date"]).reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("Qlib csi300 instruments 中没有有效 A 股成员。")
    universe = HistoricalUniverse(output)
    universe._validate(frame)
    universe._atomic_write(frame)
    return len(frame)


def _collect_month_members(D: Any, instruments: Any, day: pd.Timestamp, memberships: dict[str, list[tuple[str, str]]]) -> None:
    """把某个月最后一个交易日的成分并入该股票的连续区间列表。"""
    codes = D.list_instruments(instruments=instruments, start_time=str(day), end_time=str(day), as_list=True)
    month_start = day.replace(day=1)
    month_end = day.strftime("%Y-%m-%d")
    previous_month = (day.year, day.month - 1) if day.month > 1 else (day.year - 1, 12)
    for code in codes:
        local = str(code).replace("SH", "").replace("SZ", "").zfill(6)
        intervals = memberships.setdefault(local, [])
        last_end = pd.Timestamp(intervals[-1][1]) if intervals else None
        if last_end is not None and (last_end.year, last_end.month) == previous_month:
            intervals[-1] = (intervals[-1][0], month_end)
        else:
            intervals.append((month_start.strftime("%Y-%m-%d"), month_end))


def _instrument_bounds(instrument: Any) -> tuple[str, str, str]:
    """兼容 Qlib Instrument 对象的多种字段形式。"""
    if isinstance(instrument, (tuple, list)):
        code, start, end = instrument[0], instrument[1], instrument[2]
    else:
        code = getattr(instrument, "code", None) or str(instrument).split(",")[0].strip("(' ")
        start = getattr(instrument, "start_time", None) or "1990-01-01"
        end = getattr(instrument, "end_time", None) or "2999-12-31"
    return str(code).replace("SH", "").replace("SZ", "").zfill(6), str(start)[:10], str(end)[:10]
