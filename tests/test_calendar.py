"""Qlib 优先的 A 股交易日历测试。"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from data.calendar import AshareCalendar

SHANGHAI = ZoneInfo("Asia/Shanghai")


def _calendar(tmp_path, sessions: list[str]) -> AshareCalendar:
    path = tmp_path / "calendars"
    path.mkdir(parents=True)
    (path / "day.txt").write_text("\n".join(sessions), encoding="utf-8")
    return AshareCalendar(tmp_path)


def test_qlib_calendar_skips_holiday_and_uses_future_sessions(tmp_path) -> None:
    calendar = _calendar(tmp_path, ["2024-02-08", "2024-02-19", "2024-02-20"])
    now = datetime(2024, 2, 12, 16, 0, tzinfo=SHANGHAI)

    complete, flags = calendar.latest_complete_session(now)
    future = calendar.future_sessions(pd.Timestamp("2024-02-08"), 2)

    assert complete == pd.Timestamp("2024-02-08")
    assert flags == ()
    assert future.tolist() == [pd.Timestamp("2024-02-19"), pd.Timestamp("2024-02-20")]
    assert calendar.version.startswith("qlib-day-")


def test_qlib_calendar_returns_auditable_sessions_between_dates(tmp_path) -> None:
    calendar = _calendar(tmp_path, ["2024-02-08", "2024-02-19", "2024-02-20"])

    sessions, flags = calendar.sessions_between(pd.Timestamp("2024-02-08"), pd.Timestamp("2024-02-20"))

    assert sessions.tolist() == [pd.Timestamp("2024-02-08"), pd.Timestamp("2024-02-19"), pd.Timestamp("2024-02-20")]
    assert flags == ()


def test_qlib_calendar_excludes_intraday_session(tmp_path) -> None:
    calendar = _calendar(tmp_path, ["2024-02-08", "2024-02-09"])
    now = datetime(2024, 2, 9, 15, 29, tzinfo=SHANGHAI)

    complete, flags = calendar.latest_complete_session(now)

    assert complete == pd.Timestamp("2024-02-08")
    assert flags == ()


def test_requested_session_respects_intraday_and_historical_calendar(tmp_path) -> None:
    calendar = _calendar(tmp_path, ["2024-02-08", "2024-02-09"])
    intraday = datetime(2024, 2, 9, 15, 29, tzinfo=SHANGHAI)
    far_future = datetime(2026, 1, 5, 16, 0, tzinfo=SHANGHAI)

    current, current_flags = calendar.complete_session_for(pd.Timestamp("2024-02-09"), intraday)
    historical, historical_flags = calendar.complete_session_for(pd.Timestamp("2024-02-09"), far_future)

    assert current == pd.Timestamp("2024-02-08")
    assert current_flags == ()
    assert historical == pd.Timestamp("2024-02-09")
    assert historical_flags == ()


def test_weekend_after_calendar_end_uses_known_friday_without_downgrade(tmp_path) -> None:
    calendar = _calendar(tmp_path, ["2024-02-08", "2024-02-09"])
    now = datetime(2024, 2, 11, 16, 0, tzinfo=SHANGHAI)

    complete, flags = calendar.latest_complete_session(now)

    assert complete == pd.Timestamp("2024-02-09")
    assert flags == ()


def test_outdated_qlib_calendar_falls_back_with_quality_flag(tmp_path) -> None:
    calendar = _calendar(tmp_path, ["2024-01-02"])
    now = datetime(2024, 1, 5, 16, 0, tzinfo=SHANGHAI)

    complete, flags = calendar.latest_complete_session(now)

    assert complete == pd.Timestamp("2024-01-05")
    assert flags == ("CALENDAR_QLIB_OUTDATED",)


def test_missing_calendar_uses_explicit_weekday_fallback(tmp_path) -> None:
    calendar = AshareCalendar(tmp_path)
    now = datetime(2024, 1, 8, 12, 0, tzinfo=SHANGHAI)

    complete, flags = calendar.latest_complete_session(now)

    assert complete == pd.Timestamp("2024-01-05")
    assert flags == ("CALENDAR_WEEKDAY_FALLBACK",)
