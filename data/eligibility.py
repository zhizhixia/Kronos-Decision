"""股票交易资格规则：状态、日线证据和流动性均缺证据即拒绝。"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd

# 计划锁定的固定门槛：上市不足 252 个交易日、日均成交额低于 2000 万元、
# 最新成交日距锚点超过 14 个自然日（约 10 个交易日）视为停牌。
MIN_LISTED_SESSIONS = 252
MIN_AVG_DAILY_AMOUNT = 20_000_000.0
MAX_SUSPENSION_DAYS = 14
LIQUIDITY_LOOKBACK_SESSIONS = 20
_REQUIRED_BAR_COLUMNS = ("date", "open", "high", "low", "close", "volume", "amount")
_KNOWN_QUALITY_FLAGS = {
    "ABNORMAL_PRICE_JUMP",
    "CALENDAR_QLIB_OUTDATED",
    "CALENDAR_WEEKDAY_FALLBACK",
    "ERROR_INCOMPLETE_LATEST_SESSION",
    "ERROR_UNRESOLVED_SESSION_GAP",
    "INTRADAY_SESSION",
    "NON_TRADING_DAY_HOLIDAY",
    "NON_TRADING_DAY_WEEKEND",
    "POSSIBLE_SUSPENSION_ZERO_VOLUME",
    "STALE_CACHE",
}
_NORMAL_SECURITY_STATES = {"NORMAL", "ACTIVE", "TRADING", "LISTED", "正常", "正常交易"}


@dataclass(frozen=True)
class EligibilityResult:
    """交易资格结论；不合格股票不能获得增持。"""

    eligible: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """返回供报告和审计记录使用的字典。"""
        return {"eligible": self.eligible, "reasons": list(self.reasons)}


def evaluate_stock_eligibility(
    bars: pd.DataFrame | None,
    stock_name: str = "",
    as_of: pd.Timestamp | None = None,
    *,
    security_status: str | None = None,
    security_state: str | None = None,
    status: str | None = None,
    data_complete: bool | None = None,
    quality_flags: Iterable[str] | None = None,
    data_quality_flags: Iterable[str] | None = None,
    latest_complete_session: pd.Timestamp | None = None,
    calendar: Any | None = None,
) -> EligibilityResult:
    """按固定规则评估资格；未知状态、坏数据和缺证据一律不可增持。"""
    reasons: list[str] = []
    add = lambda reason: _add_reason(reasons, reason)

    _check_security_identity(stock_name, add)
    if _security_name_is_restricted(stock_name):
        add("ST_OR_DELISTING")

    frame = _prepare_frame(bars, add)
    if frame is None:
        add("LISTED_LESS_THAN_252_SESSIONS")
        add("LIQUIDITY_EVIDENCE_MISSING")
        _check_security_status(bars, security_status, security_state, status, add=add)
        return _result(reasons)

    if frame.columns.duplicated().any():
        add("DATA_SCHEMA_INVALID")
        add("LISTED_LESS_THAN_252_SESSIONS")
        add("LIQUIDITY_EVIDENCE_MISSING")
        _check_security_status(frame, security_status, security_state, status, add=add)
        return _result(reasons)

    _check_security_status(frame, security_status, security_state, status, add=add)
    _check_quality_flags(frame, quality_flags, data_quality_flags, add=add)

    anchor, anchor_error = _parse_as_of(as_of)
    if as_of is None:
        add("AS_OF_MISSING")
    elif anchor is None:
        add(anchor_error or "INVALID_AS_OF")
    elif anchor_error is not None:
        add(anchor_error)

    missing_columns = [column for column in _REQUIRED_BAR_COLUMNS if column not in frame.columns]
    if missing_columns:
        add("DATA_SCHEMA_INVALID")
        if "amount" in missing_columns:
            add("LIQUIDITY_EVIDENCE_MISSING")
        add("LISTED_LESS_THAN_252_SESSIONS")
        return _result(reasons)

    dates = _validated_dates(frame["date"], add)
    numeric = _validated_numeric(frame, add)
    if dates is None or numeric is None:
        if numeric is None:
            add("LIQUIDITY_EVIDENCE_MISSING")
        if len(frame) < MIN_LISTED_SESSIONS:
            add("LISTED_LESS_THAN_252_SESSIONS")
        return _result(reasons)

    if len(frame) < MIN_LISTED_SESSIONS:
        add("LISTED_LESS_THAN_252_SESSIONS")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        add("DATA_DATE_ORDER_INVALID")

    _check_ohlc_invariants(numeric, add)
    _check_completeness_evidence(
        frame,
        dates,
        anchor,
        data_complete,
        latest_complete_session,
        calendar,
        add,
    )
    _check_liquidity(numeric, add)
    return _result(reasons)


def _result(reasons: list[str]) -> EligibilityResult:
    """将去重后的原因列表封装为失败闭合结果。"""
    return EligibilityResult(not reasons, tuple(reasons))


def _add_reason(reasons: list[str], reason: str | None) -> None:
    """按首次出现顺序加入非空原因码。"""
    if reason and reason not in reasons:
        reasons.append(reason)


def _prepare_frame(bars: pd.DataFrame | None, add) -> pd.DataFrame | None:
    """验证输入是否为非空表格；错误类型不被当成空数据放宽。"""
    if bars is None:
        add("DATA_UNAVAILABLE")
        return None
    if not isinstance(bars, pd.DataFrame):
        add("INVALID_DATA_INPUT")
        return None
    if bars.empty:
        add("DATA_EMPTY")
        return None
    return bars.copy(deep=False)


def _check_security_identity(stock_name: str, add) -> None:
    """拒绝缺失或非字符串的证券身份。"""
    if not isinstance(stock_name, str) or not stock_name.strip():
        add("SECURITY_IDENTITY_UNKNOWN")


def _security_name_is_restricted(stock_name: str) -> bool:
    """识别名称中的 ST、退市和终止上市标记。"""
    if not isinstance(stock_name, str):
        return False
    name = stock_name.strip().upper()
    compact = re.sub(r"\s+", "", name)
    st_match = re.search(r"(?:^|[^A-Z0-9])\*?ST(?:$|[^A-Z0-9])", name)
    return bool(st_match or compact.startswith("S*ST") or "退市" in compact or compact.endswith("退") or "DELIST" in compact)


def _check_security_status(frame: pd.DataFrame | None, *values, add) -> None:
    """只接受明确的正常证券状态，冲突或未知状态均拒绝。"""
    candidates: list[str] = []
    invalid = False
    for value in values:
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            invalid = True
            continue
        candidates.append(value.strip())

    if frame is not None:
        metadata = getattr(frame, "attrs", {})
        for key in ("security_status", "security_state", "status"):
            value = metadata.get(key)
            if value is not None:
                if isinstance(value, str) and value.strip():
                    candidates.append(value.strip())
                else:
                    invalid = True
            if key in frame.columns:
                series = frame[key]
                if series.isna().any():
                    invalid = True
                unique = {str(item).strip() for item in series.dropna().tolist() if str(item).strip()}
                if len(unique) != 1:
                    invalid = True
                else:
                    candidates.extend(unique)

    normalized = {_normalize_status(value) for value in candidates if value}
    normalized.discard("")
    if invalid and not normalized:
        add("UNKNOWN_SECURITY_STATUS")
        return
    if invalid or len(normalized) > 1:
        add("SECURITY_STATUS_CONFLICT")
        return
    if not normalized:
        add("UNKNOWN_SECURITY_STATUS")
        return

    state = next(iter(normalized))
    if state in _NORMAL_SECURITY_STATES:
        return
    if state in {"ST", "*ST", "S*ST", "DELISTED", "DELISTING", "退市", "终止上市"}:
        add("ST_OR_DELISTING")
    elif state in {"SUSPENDED", "停牌"}:
        add("SECURITY_SUSPENDED")
    elif state in {"LONG_SUSPENSION", "LONGSUSPENSION", "长期停牌"}:
        add("SECURITY_SUSPENDED")
        add("LONG_SUSPENSION")
    else:
        add("UNKNOWN_SECURITY_STATUS")


def _normalize_status(value: str) -> str:
    """标准化证券状态文本，不把未知枚举映射为正常。"""
    return value.strip().upper().replace(" ", "").replace("_", "").replace("-", "")


def _check_quality_flags(frame: pd.DataFrame, *values, add) -> None:
    """审计数据质量旗标；错误、过期和未知旗标都阻断资格。"""
    sources = [value for value in values if value is not None]
    metadata = getattr(frame, "attrs", {})
    if not sources and "quality_flags" in metadata:
        sources.append(metadata["quality_flags"])

    for source in sources:
        flags = _coerce_flags(source, add)
        if flags is None:
            continue
        for flag in flags:
            if flag not in _KNOWN_QUALITY_FLAGS and not flag.startswith("ERROR_"):
                add(flag)
                add("UNKNOWN_DATA_QUALITY")
                continue
            if flag.startswith("ERROR_") or flag in {
                "ABNORMAL_PRICE_JUMP",
                "CALENDAR_QLIB_OUTDATED",
                "CALENDAR_WEEKDAY_FALLBACK",
                "POSSIBLE_SUSPENSION_ZERO_VOLUME",
                "STALE_CACHE",
            }:
                add(flag)
                add("DATA_QUALITY_UNVERIFIED")
            if "INCOMPLETE" in flag:
                add("INCOMPLETE_LATEST_SESSION")
            if "SUSPENSION" in flag:
                add("SUSPENSION_EVIDENCE_MISSING")


def _coerce_flags(source: Iterable[str], add) -> list[str] | None:
    """把质量旗标解析为字符串列表；非法容器不静默忽略。"""
    if isinstance(source, str):
        raw = [source]
    else:
        try:
            raw = list(source)
        except TypeError:
            add("INVALID_DATA_QUALITY_INPUT")
            return None
    flags: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            add("INVALID_DATA_QUALITY_INPUT")
            continue
        flag = item.strip()
        if flag not in flags:
            flags.append(flag)
    return flags


def _parse_as_of(value: pd.Timestamp | None) -> tuple[pd.Timestamp | None, str | None]:
    """严格解析日级锚点，并显式拒绝盘中锚点。"""
    if not isinstance(value, (str, pd.Timestamp, datetime, date)):
        return None, "INVALID_AS_OF"
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None, "INVALID_AS_OF"
    if pd.isna(timestamp):
        return None, "INVALID_AS_OF"
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("Asia/Shanghai").tz_localize(None)
    intraday = timestamp != timestamp.normalize()
    return timestamp.normalize(), "INTRADAY_AS_OF" if intraday else None


def _validated_dates(series: pd.Series, add) -> pd.Series | None:
    """验证日期列可解析、为日级且不含时区混合值。"""
    try:
        dates = pd.to_datetime(series, errors="coerce")
        if dates.isna().any():
            add("DATA_DATE_INVALID")
            return None
        if getattr(dates.dt, "tz", None) is not None:
            add("DATA_DATE_TIMEZONE_INVALID")
            return None
        normalized = dates.dt.normalize()
        if (dates != normalized).any():
            add("DATA_DATE_NOT_DAILY")
            return None
        return normalized
    except (AttributeError, TypeError, ValueError):
        add("DATA_DATE_INVALID")
        return None


def _validated_numeric(frame: pd.DataFrame, add) -> pd.DataFrame | None:
    """验证价格、成交量和成交额是有限数值。"""
    numeric_columns = [column for column in _REQUIRED_BAR_COLUMNS if column != "date"]
    try:
        numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    except (KeyError, TypeError, ValueError):
        add("DATA_NUMERIC_INVALID")
        return None
    invalid = numeric.isna() | numeric.eq(float("inf")) | numeric.eq(float("-inf"))
    if invalid.any().any():
        add("DATA_NUMERIC_INVALID")
        return None
    return numeric


def _check_ohlc_invariants(numeric: pd.DataFrame, add) -> None:
    """验证 OHLC 正值关系和成交量/成交额非负约束。"""
    high_ok = numeric["high"] >= numeric[["open", "close", "low"]].max(axis=1)
    low_ok = numeric["low"] <= numeric[["open", "close", "high"]].min(axis=1)
    if not high_ok.all() or not low_ok.all() or (numeric[["open", "high", "low", "close"]] <= 0).any().any():
        add("DATA_OHLC_INVALID")
    if (numeric[["volume", "amount"]] < 0).any().any():
        add("DATA_TURNOVER_INVALID")
    if (numeric["volume"] == 0).any():
        add("POSSIBLE_SUSPENSION_ZERO_VOLUME")
        add("SUSPENSION_EVIDENCE_MISSING")


def _check_completeness_evidence(
    frame: pd.DataFrame,
    dates: pd.Series,
    anchor: pd.Timestamp | None,
    data_complete: bool | None,
    latest_complete_session: pd.Timestamp | None,
    calendar: Any | None,
    add,
) -> None:
    """检查锚点可见性、最新会话完整性和日历证据，不借用未来数据。"""
    metadata = getattr(frame, "attrs", {})
    if data_complete is None and "data_complete" in metadata:
        data_complete = metadata["data_complete"]
    if data_complete is not None and not isinstance(data_complete, bool):
        add("INVALID_DATA_COMPLETENESS_INPUT")
        data_complete = False
    if data_complete is False:
        add("INCOMPLETE_LATEST_SESSION")

    if anchor is None:
        return
    latest = dates.iloc[-1]
    if latest > anchor:
        add("DATA_AFTER_AS_OF")
    elif anchor - latest > pd.Timedelta(days=MAX_SUSPENSION_DAYS):
        add("LONG_SUSPENSION")

    expected, expected_flags = _resolve_expected_session(calendar, latest_complete_session, anchor, add)
    if calendar is not None or latest_complete_session is not None:
        if expected is None:
            add("DATA_COMPLETENESS_UNKNOWN")
        else:
            if latest < expected:
                add("INCOMPLETE_LATEST_SESSION")
            elif latest > expected:
                add("DATA_AFTER_AS_OF")
        for flag in expected_flags:
            if flag.startswith("ERROR_") or flag.startswith("CALENDAR_"):
                add(flag)
                add("DATA_COMPLETENESS_UNKNOWN")


def _resolve_expected_session(calendar: Any | None, explicit: pd.Timestamp | None, anchor: pd.Timestamp, add) -> tuple[pd.Timestamp | None, tuple[str, ...]]:
    """从显式会话或日历解析预期截止日，失败时返回未知证据。"""
    if explicit is not None:
        expected = _normalize_expected_session(explicit)
        if expected is None:
            add("INVALID_LATEST_COMPLETE_SESSION")
        return expected, ()
    if calendar is None:
        return None, ()
    resolver = getattr(calendar, "complete_session_for", None)
    if not callable(resolver):
        add("INVALID_CALENDAR_EVIDENCE")
        return None, ()
    try:
        result = resolver(anchor)
        if not isinstance(result, tuple) or len(result) != 2:
            add("INVALID_CALENDAR_EVIDENCE")
            return None, ()
        expected = _normalize_expected_session(result[0])
        flags = tuple(str(flag) for flag in result[1])
    except (TypeError, ValueError, AttributeError, KeyError):
        add("INVALID_CALENDAR_EVIDENCE")
        return None, ()
    if expected is None:
        return None, flags
    return expected, flags


def _normalize_expected_session(value: Any) -> pd.Timestamp | None:
    """严格把预期会话转换为无时分秒时间戳。"""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("Asia/Shanghai").tz_localize(None)
    if timestamp != timestamp.normalize():
        return None
    return timestamp.normalize()


def _check_liquidity(numeric: pd.DataFrame, add) -> None:
    """要求最近 20 个会话具有可证明的正成交额证据。"""
    amounts = numeric["amount"].tail(LIQUIDITY_LOOKBACK_SESSIONS)
    if len(amounts) < LIQUIDITY_LOOKBACK_SESSIONS or amounts.isna().any() or (amounts <= 0).any():
        add("LIQUIDITY_EVIDENCE_MISSING")
        return
    mean_amount = float(amounts.mean())
    if mean_amount < MIN_AVG_DAILY_AMOUNT:
        add("INSUFFICIENT_LIQUIDITY")


# 保留一个无副作用的中文别名，便于调用方明确使用“证券状态”术语。
evaluate_security_eligibility = evaluate_stock_eligibility
