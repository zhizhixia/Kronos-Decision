"""A 股完整交易日判断：未知日期和历史时点均失败闭合。"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

SHANGHAI = ZoneInfo("Asia/Shanghai")
_MARKET_CLOSE = time(15, 30)
_DEFAULT_PROVIDER = Path.home() / ".qlib" / "qlib_data" / "cn_data"
_PROJECT_PROVIDER = Path(__file__).resolve().parents[1] / "qlib_data" / "cn_data"


class AshareCalendar:
    """使用已校验的 Qlib 交易日历，无法证明时返回可审计的失败状态。"""

    def __init__(self, provider_uri: str | Path | None = None) -> None:
        provider = self._resolve_provider(provider_uri)
        self._sessions, self._version = self._load_sessions(provider / "calendars" / "day.txt")
        self._session_set = frozenset(self._sessions.tolist())
        self._last_diagnostics: tuple[str, ...] = ()

    @staticmethod
    def _resolve_provider(provider_uri: str | Path | None) -> Path:
        """解析数据目录；不为缺失目录创建或下载数据。"""
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

    @property
    def diagnostics(self) -> tuple[str, ...]:
        """返回最近一次解析的盘中、非交易日和时点诊断。"""
        return self._last_diagnostics

    def session_diagnostics(self, now: datetime | None = None) -> tuple[str, ...]:
        """返回当前会话诊断，不改变既有会话解析返回契约。"""
        self.latest_complete_session(now)
        return self._last_diagnostics

    def latest_complete_session(self, now: datetime | None = None) -> tuple[pd.Timestamp, tuple[str, ...]]:
        """返回当前已完成会话，并标记盘中、非交易日和日历降级状态。"""
        current = self._coerce_now(now)
        current_date = pd.Timestamp(current.date())
        intraday = current.time() < _MARKET_CLOSE
        cutoff = current_date - pd.Timedelta(days=1) if intraday else current_date
        flags: list[str] = []
        diagnostics: list[str] = []
        if intraday:
            flags.append("INTRADAY_SESSION")
            diagnostics.append("INTRADAY_SESSION")
        non_trading = self._non_trading_flag(current_date)
        if non_trading is not None:
            flags.append(non_trading)
            diagnostics.append(non_trading)

        if self._sessions.empty:
            flags.append("CALENDAR_WEEKDAY_FALLBACK")
            diagnostics.append("CALENDAR_WEEKDAY_FALLBACK")
            return self._finish_session(
                self._weekday_complete_session(cutoff),
                flags,
                diagnostics,
            )

        visible = self._sessions[self._sessions <= cutoff]
        if visible.empty:
            flags.append("CALENDAR_QLIB_OUTDATED")
            diagnostics.append("CALENDAR_QLIB_OUTDATED")
            return self._finish_session(
                self._weekday_complete_session(cutoff),
                flags,
                diagnostics,
            )

        selected = visible[-1]
        if current_date > self._sessions[-1]:
            if current_date.dayofweek >= 5 and current_date - selected <= pd.Timedelta(days=2):
                return self._finish_session(selected, flags, diagnostics)
            flags.append("CALENDAR_QLIB_OUTDATED")
            diagnostics.append("CALENDAR_QLIB_OUTDATED")
            return self._finish_session(
                self._weekday_complete_session(cutoff),
                flags,
                diagnostics,
            )
        if current_date < self._sessions[0]:
            flags.append("CALENDAR_QLIB_OUTDATED")
            diagnostics.append("CALENDAR_QLIB_OUTDATED")
            return self._finish_session(selected, flags, diagnostics)
        return self._finish_session(selected, flags, diagnostics)

    def complete_session_for(
        self,
        as_of: pd.Timestamp | None,
        now: datetime | None = None,
    ) -> tuple[pd.Timestamp, tuple[str, ...]]:
        """解析指定时点的完整会话，不借用未来日历或当前最新会话。"""
        if as_of is None:
            if now is None:
                return self.latest_complete_session()
            return self.latest_complete_session(now)
        requested = self._coerce_date(as_of, "as_of")
        if self._sessions.empty:
            return pd.NaT, self._unavailable_flags("CALENDAR_WEEKDAY_FALLBACK")

        historical, historical_flags = self._historical_session(requested)
        if pd.isna(historical):
            return pd.NaT, historical_flags

        current_clock = self._coerce_now(now)
        current_date = pd.Timestamp(current_clock.date())
        if requested > current_date:
            flags = ["ERROR_AS_OF_IN_FUTURE"]
            if current_clock.time() < _MARKET_CLOSE:
                flags.append("INTRADAY_SESSION")
            self._last_diagnostics = self._unique_flags(flags)
            return pd.NaT, self._last_diagnostics

        current = self._latest_verified_session(current_clock)
        if current is None:
            return pd.NaT, self._unavailable_flags("CALENDAR_QLIB_OUTDATED")
        if historical > current:
            if current_clock.time() < _MARKET_CLOSE and requested == current_date:
                flags = ("INTRADAY_SESSION", "ERROR_INCOMPLETE_LATEST_SESSION")
                self._last_diagnostics = flags
                return current, flags
            flags = ["ERROR_AS_OF_IN_FUTURE"]
            if current_clock.time() < _MARKET_CLOSE:
                flags.append("INTRADAY_SESSION")
            self._last_diagnostics = self._unique_flags(flags)
            return pd.NaT, self._last_diagnostics
        self._last_diagnostics = self._unique_flags(historical_flags)
        return historical, historical_flags

    def future_sessions(self, last_date: pd.Timestamp, count: int) -> pd.Series:
        """只返回日历明确列出的未来会话；不足时返回空序列而不猜工作日。"""
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("count 必须是正整数。")
        last = self._coerce_date(last_date, "last_date")
        future = self._sessions[self._sessions > last]
        if len(future) < count:
            missing_reason = "CALENDAR_WEEKDAY_FALLBACK" if self._sessions.empty else "ERROR_FUTURE_SESSION_UNAVAILABLE"
            self._last_diagnostics = (missing_reason, "ERROR_FUTURE_SESSION_UNAVAILABLE")
            return pd.Series(dtype="datetime64[ns]", name="date")
        self._last_diagnostics = ()
        return pd.Series(future[:count], name="date")

    def sessions_between(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> tuple[pd.DatetimeIndex, tuple[str, ...]]:
        """返回闭区间内可审计交易日；覆盖不足时拒绝推断。"""
        first = self._coerce_date(start, "start")
        last = self._coerce_date(end, "end")
        if first > last:
            raise ValueError("start 不能晚于 end。")
        if self._sessions.empty:
            self._last_diagnostics = ("CALENDAR_WEEKDAY_FALLBACK",)
            return pd.DatetimeIndex([]), ("CALENDAR_WEEKDAY_FALLBACK",)
        if first < self._sessions[0] or last > self._sessions[-1]:
            return pd.DatetimeIndex([]), self._unavailable_flags("CALENDAR_QLIB_OUTDATED")

        selected = self._sessions[(self._sessions >= first) & (self._sessions <= last)]
        flags: list[str] = []
        start_flag = self._non_trading_flag(first)
        end_flag = self._non_trading_flag(last)
        if start_flag is not None:
            flags.append(start_flag)
        if end_flag is not None:
            flags.append(end_flag)
        self._last_diagnostics = self._unique_flags(flags)
        return selected, self._unique_flags(flags)

    def _historical_session(self, requested: pd.Timestamp) -> tuple[pd.Timestamp, tuple[str, ...]]:
        """只按请求日期解析历史会话，不对日历范围外日期作工作日猜测。"""
        if requested < self._sessions[0]:
            return pd.NaT, self._unavailable_flags("AS_OF_BEFORE_CALENDAR")

        visible = self._sessions[self._sessions <= requested]
        if visible.empty:
            return pd.NaT, self._unavailable_flags("AS_OF_BEFORE_CALENDAR")
        selected = visible[-1]

        if requested > self._sessions[-1]:
            weekend_gap = requested.dayofweek >= 5 and requested - selected <= pd.Timedelta(days=2)
            if not weekend_gap:
                return pd.NaT, self._unavailable_flags("AS_OF_AFTER_CALENDAR")

        if requested == selected:
            return selected, ()
        flag = "NON_TRADING_DAY_WEEKEND" if requested.dayofweek >= 5 else "NON_TRADING_DAY_HOLIDAY"
        return selected, (flag,)

    def _latest_verified_session(self, now: datetime | None) -> pd.Timestamp | None:
        """返回当前时钟之前日历能证明的最后会话，不使用降级候选。"""
        current = self._coerce_now(now)
        current_date = pd.Timestamp(current.date())
        cutoff = current_date - pd.Timedelta(days=1) if current.time() < _MARKET_CLOSE else current_date
        visible = self._sessions[self._sessions <= cutoff]
        return visible[-1] if not visible.empty else None

    @staticmethod
    def _load_sessions(path: Path) -> tuple[pd.DatetimeIndex, str]:
        """读取并严格校验日历；损坏、重复或乱序文件进入显式降级状态。"""
        try:
            payload = path.read_bytes()
            lines = payload.decode("utf-8-sig").splitlines()
            if not lines or any(not line.strip() for line in lines):
                raise ValueError("交易日日历为空或含空行")

            parsed: list[pd.Timestamp] = []
            for line in lines:
                value = pd.Timestamp(line.strip())
                if pd.isna(value) or value.tzinfo is not None or value != value.normalize():
                    raise ValueError("交易日日历含非法日级日期")
                parsed.append(value.normalize())
            sessions = pd.DatetimeIndex(parsed)
            if sessions.has_duplicates or not sessions.is_monotonic_increasing:
                raise ValueError("交易日日历必须严格递增且无重复")
            digest = hashlib.sha256(payload).hexdigest()[:12]
            return sessions, f"qlib-day-{digest}"
        except (OSError, UnicodeError, ValueError, TypeError, OverflowError):
            return pd.DatetimeIndex([]), "weekday-fallback-v1"

    def _covers(self, candidate: pd.Timestamp) -> bool:
        """判断日期是否处于已加载日历的覆盖范围内。"""
        return not self._sessions.empty and self._sessions[0] <= candidate <= self._sessions[-1]

    def _non_trading_flag(self, candidate: pd.Timestamp) -> str | None:
        """为周末或已覆盖范围内的非交易日返回明确原因。"""
        if candidate.dayofweek >= 5:
            return "NON_TRADING_DAY_WEEKEND"
        if self._covers(candidate) and candidate not in self._session_set:
            return "NON_TRADING_DAY_HOLIDAY"
        return None

    def _unavailable_flags(self, reason: str) -> tuple[str, ...]:
        """构造不可审计会话的硬失败原因。"""
        self._last_diagnostics = self._unique_flags((reason, "ERROR_AS_OF_SESSION_UNAVAILABLE"))
        return self._last_diagnostics

    @staticmethod
    def _coerce_now(value: datetime | None) -> datetime:
        """将时钟统一到上海时区；错误类型不静默转换。"""
        if value is None:
            current = datetime.now(SHANGHAI)
        elif isinstance(value, pd.Timestamp):
            current = value.to_pydatetime()
        elif isinstance(value, datetime):
            current = value
        else:
            raise TypeError("now 必须是 datetime 或 None。")
        if current.tzinfo is None:
            return current.replace(tzinfo=SHANGHAI)
        return current.astimezone(SHANGHAI)

    @staticmethod
    def _coerce_date(value: pd.Timestamp, field_name: str) -> pd.Timestamp:
        """将输入严格解析为无时分秒的日期。"""
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{field_name} 必须是有效日期。") from exc
        if pd.isna(timestamp):
            raise ValueError(f"{field_name} 必须是有效日期。")
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert(SHANGHAI).tz_localize(None)
        if timestamp != timestamp.normalize():
            raise ValueError(f"{field_name} 必须是日级日期。")
        return timestamp.normalize()

    def _finish_session(
        self,
        session: pd.Timestamp,
        flags: tuple[str, ...] | list[str],
        diagnostics: tuple[str, ...] | list[str],
    ) -> tuple[pd.Timestamp, tuple[str, ...]]:
        """记录诊断并返回兼容的会话与质量旗标。"""
        self._last_diagnostics = self._unique_flags(diagnostics)
        return session, self._unique_flags(flags)

    @staticmethod
    def _weekday_complete_session(candidate: pd.Timestamp) -> pd.Timestamp:
        """仅生成降级诊断候选；调用方必须检查返回的质量旗标。"""
        day = candidate
        while day.dayofweek >= 5:
            day -= pd.Timedelta(days=1)
        return day

    @staticmethod
    def _unique_flags(flags: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        """稳定去重状态旗标，保留首次出现顺序。"""
        return tuple(dict.fromkeys(str(flag) for flag in flags))
