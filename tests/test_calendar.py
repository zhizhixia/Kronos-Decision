"""Qlib 优先的 A 股交易日历测试。"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

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
    assert flags == ("NON_TRADING_DAY_HOLIDAY",)
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
    assert flags == ("INTRADAY_SESSION",)


def test_requested_session_respects_intraday_and_historical_calendar(tmp_path) -> None:
    calendar = _calendar(tmp_path, ["2024-02-08", "2024-02-09"])
    intraday = datetime(2024, 2, 9, 15, 29, tzinfo=SHANGHAI)
    far_future = datetime(2026, 1, 5, 16, 0, tzinfo=SHANGHAI)

    current, current_flags = calendar.complete_session_for(pd.Timestamp("2024-02-09"), intraday)
    historical, historical_flags = calendar.complete_session_for(pd.Timestamp("2024-02-09"), far_future)

    assert current == pd.Timestamp("2024-02-08")
    assert current_flags == ("INTRADAY_SESSION", "ERROR_INCOMPLETE_LATEST_SESSION")
    assert historical == pd.Timestamp("2024-02-09")
    assert historical_flags == ()


def test_weekend_after_calendar_end_uses_known_friday_without_downgrade(tmp_path) -> None:
    calendar = _calendar(tmp_path, ["2024-02-08", "2024-02-09"])
    now = datetime(2024, 2, 11, 16, 0, tzinfo=SHANGHAI)

    complete, flags = calendar.latest_complete_session(now)

    assert complete == pd.Timestamp("2024-02-09")
    assert flags == ("NON_TRADING_DAY_WEEKEND",)


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
    assert flags == ("INTRADAY_SESSION", "CALENDAR_WEEKDAY_FALLBACK")


def test_session_diagnostics_are_available_without_changing_return_flags(tmp_path) -> None:
    calendar = _calendar(tmp_path, ["2024-02-08", "2024-02-19"])
    now = datetime(2024, 2, 12, 16, 0, tzinfo=SHANGHAI)

    complete, flags = calendar.latest_complete_session(now)

    assert complete == pd.Timestamp("2024-02-08")
    assert flags == ("NON_TRADING_DAY_HOLIDAY",)
    assert calendar.diagnostics == ("NON_TRADING_DAY_HOLIDAY",)
    assert calendar.session_diagnostics(now) == ("NON_TRADING_DAY_HOLIDAY",)


def test_historical_as_of_never_uses_future_calendar_session(tmp_path) -> None:
    calendar = _calendar(tmp_path, ["2024-02-08", "2024-02-19"])
    now = datetime(2024, 2, 9, 16, 0, tzinfo=SHANGHAI)

    historical, flags = calendar.complete_session_for(pd.Timestamp("2024-02-19"), now)

    assert pd.isna(historical)
    assert "ERROR_AS_OF_IN_FUTURE" in flags


def test_historical_as_of_outside_calendar_does_not_use_weekday_fallback(tmp_path) -> None:
    calendar = _calendar(tmp_path, ["2024-01-02", "2024-01-03"])
    now = datetime(2024, 1, 8, 16, 0, tzinfo=SHANGHAI)

    historical, flags = calendar.complete_session_for(pd.Timestamp("2024-01-05"), now)

    assert pd.isna(historical)
    assert "ERROR_AS_OF_SESSION_UNAVAILABLE" in flags
    assert "AS_OF_AFTER_CALENDAR" in flags


def test_historical_weekend_resolves_to_previous_session_with_status(tmp_path) -> None:
    calendar = _calendar(tmp_path, ["2024-02-08", "2024-02-09"])
    now = datetime(2024, 2, 12, 16, 0, tzinfo=SHANGHAI)

    historical, flags = calendar.complete_session_for(pd.Timestamp("2024-02-10"), now)

    assert historical == pd.Timestamp("2024-02-09")
    assert flags == ("NON_TRADING_DAY_WEEKEND",)


def test_future_sessions_refuse_uncovered_calendar_tail(tmp_path) -> None:
    calendar = _calendar(tmp_path, ["2024-01-02", "2024-01-03"])

    future = calendar.future_sessions(pd.Timestamp("2024-01-03"), 1)

    assert future.empty


def test_invalid_calendar_arguments_are_rejected(tmp_path) -> None:
    calendar = _calendar(tmp_path, ["2024-01-02", "2024-01-03"])

    with pytest.raises(ValueError):
        calendar.sessions_between(pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-02"))
    with pytest.raises(ValueError):
        calendar.future_sessions(pd.Timestamp("2024-01-03"), 0)
