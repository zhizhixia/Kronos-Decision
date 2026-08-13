"""A 股完整交易日判断：优先本地 Qlib 日历，缺失时失败闭合降级。"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

SHANGHAI = ZoneInfo("Asia/Shanghai")
_DEFAULT_PROVIDER = Path.home() / ".qlib" / "qlib_data" / "cn_data"
_PROJECT_PROVIDER = Path(__file__).resolve().parents[1] / "qlib_data" / "cn_data"


class AshareCalendar:
    """使用 Qlib 交易日历；仅在缺失或过期时回退到工作日候选。"""

    def __init__(self, provider_uri: str | Path | None = None) -> None:
        provider = self._resolve_provider(provider_uri)
        self._sessions, self._version = self._load_sessions(provider / "calendars" / "day.txt")

    @staticmethod
    def _resolve_provider(provider_uri: str | Path | None) -> Path:
        if provider_uri is not None:
            return Path(provider_uri)
        configured = os.environ.get("KRONOS_QLIB_DATA")
        if configured:
            return Path(configured)
        return _PROJECT_PROVIDER if _PROJECT_PROVIDER.exists() else _DEFAULT_PROVIDER

    @property
    def version(self) -> str:
        """返回日历内容版本，供数据包与评估工件审计。"""
        return self._version

    def latest_complete_session(self, now: datetime | None = None) -> tuple[pd.Timestamp, tuple[str, ...]]:
        """15:30 前排除当日；日历覆盖候选日时严格使用真实交易日。"""
        current = now or datetime.now(SHANGHAI)
        candidate = pd.Timestamp(current.date())
        if current.time() < time(15, 30):
            candidate -= pd.Timedelta(days=1)
        if self._covers(candidate):
            visible = self._sessions[self._sessions <= candidate]
            if not visible.empty:
                return visible[-1], ()
        if not self._sessions.empty and candidate > self._sessions[-1] and candidate.dayofweek >= 5 and candidate - self._sessions[-1] <= pd.Timedelta(days=2):
            return self._sessions[-1], ()
        return self._weekday_complete_session(candidate), self._fallback_flags()

    def complete_session_for(self, as_of: pd.Timestamp | None, now: datetime | None = None) -> tuple[pd.Timestamp, tuple[str, ...]]:
        """解析请求可用的完整会话，历史请求不继承当前日历状态。"""
        current, current_flags = self.latest_complete_session() if now is None else self.latest_complete_session(now)
        if as_of is None:
            return current, current_flags
        requested = pd.Timestamp(as_of).normalize()
        if requested >= current:
            return current, current_flags
        return self._session_at_or_before(requested)

    def future_sessions(self, last_date: pd.Timestamp, count: int) -> pd.Series:
        """返回后续交易日；本地日历不足时仅生成工作日展示候选。"""
        if count < 1:
            return pd.Series(dtype="datetime64[ns]", name="date")
        last = pd.Timestamp(last_date).normalize()
        future = self._sessions[self._sessions > last]
        if len(future) >= count:
            return pd.Series(future[:count], name="date")
        return self._weekday_future_sessions(last, count)

    def sessions_between(self, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DatetimeIndex, tuple[str, ...]]:
        """返回闭区间内可审计交易日；日历未覆盖时明确拒绝推断。"""
        first = pd.Timestamp(start).normalize()
        last = pd.Timestamp(end).normalize()
        if first > last:
            return pd.DatetimeIndex([]), ()
        if not self._covers(first) or not self._covers(last):
            return pd.DatetimeIndex([]), self._fallback_flags()
        return self._sessions[(self._sessions >= first) & (self._sessions <= last)], ()

    def _session_at_or_before(self, value: pd.Timestamp) -> tuple[pd.Timestamp, tuple[str, ...]]:
        """返回历史日期之前的真实会话；日历未覆盖时失败闭合降级。"""
        candidate = pd.Timestamp(value).normalize()
        if self._covers(candidate):
            visible = self._sessions[self._sessions <= candidate]
            if not visible.empty:
                return visible[-1], ()
        return self._weekday_complete_session(candidate), self._fallback_flags()

    @staticmethod
    def _load_sessions(path: Path) -> tuple[pd.DatetimeIndex, str]:
        try:
            payload = path.read_bytes()
            sessions = pd.DatetimeIndex(pd.to_datetime(payload.decode("utf-8").splitlines())).normalize().sort_values().unique()
            if sessions.empty:
                raise ValueError("empty")
            digest = hashlib.sha256(payload).hexdigest()[:12]
            return sessions, f"qlib-day-{digest}"
        except (OSError, UnicodeError, ValueError):
            return pd.DatetimeIndex([]), "weekday-fallback-v1"

    def _covers(self, candidate: pd.Timestamp) -> bool:
        return not self._sessions.empty and self._sessions[0] <= candidate <= self._sessions[-1]

    def _fallback_flags(self) -> tuple[str, ...]:
        return ("CALENDAR_QLIB_OUTDATED",) if not self._sessions.empty else ("CALENDAR_WEEKDAY_FALLBACK",)

    @staticmethod
    def _weekday_complete_session(candidate: pd.Timestamp) -> pd.Timestamp:
        day = candidate
        while day.dayofweek >= 5:
            day -= pd.Timedelta(days=1)
        return day

    @staticmethod
    def _weekday_future_sessions(last: pd.Timestamp, count: int) -> pd.Series:
        dates: list[pd.Timestamp] = []
        current = last + pd.Timedelta(days=1)
        while len(dates) < count:
            if current.dayofweek < 5:
                dates.append(current)
            current += pd.Timedelta(days=1)
        return pd.Series(dates, name="date")
