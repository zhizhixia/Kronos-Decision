"""沪深 300 股票池与证券状态管理。

旧版 ``get_hs300_pool`` 继续提供当前缓存字典；新增的时点快照接口不把
今天的成分股冒充历史成分股。缺少有效状态、来源或可见时间时，快照保持
失败闭合，调用方只能看到明确的拒绝原因。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from data.contracts import InstrumentState, UniverseSnapshot
from decision.config import get_config

SHANGHAI = ZoneInfo("Asia/Shanghai")
_CACHE_PATH = Path(get_config().data.cache_dir) / "pool_hs300.csv"
_HISTORY_PATH = Path(get_config().data.cache_dir) / "pool_hs300_history.csv"
_META_PATH = _CACHE_PATH.with_suffix(".meta.json")
_REFRESH_INTERVAL_DAYS = 7
_SOURCE_TIMEOUT_SECONDS = 60
_CODE_PATTERN = re.compile(r"^\d{6}$")
_REQUIRED_HISTORY_COLUMNS = {
    "code",
    "name",
    "effective_date",
    "available_at",
    "security_status",
    "listed_date",
    "delisted_date",
    "source",
    "source_revision",
    "corporate_action_version",
}
_ACTIVE_STATES = {"NORMAL", "ACTIVE", "TRADING", "LISTED", "正常", "正常交易"}
_REMOVED_STATES = {"REMOVED", "EXITED", "DELISTED", "DELISTING", "退市", "终止上市"}


def get_hs300_pool() -> dict[str, str]:
    """获取当前沪深 300 成分股字典，保持旧接口兼容。"""
    from decision.errors import DataSourceError

    cached = _load_cache()
    if cached is not None and _is_fresh():
        return cached

    try:
        from data.timeout import call_with_timeout

        pool = call_with_timeout(_fetch_from_akshare, _SOURCE_TIMEOUT_SECONDS, name="fetch-hs300-pool")
        _save_cache(pool)
        _save_pool_metadata("akshare")
        return pool
    except Exception as exc:
        if cached is not None:
            return cached
        raise DataSourceError("无法获取沪深 300 成分股列表") from exc


def get_current_universe_snapshot(
    as_of: pd.Timestamp | str | None = None,
) -> UniverseSnapshot:
    """返回当前股票池快照；旧缓存没有状态时明确拒绝使用。"""
    anchor = _normalize_day(as_of) if as_of is not None else _today()
    pool = get_hs300_pool()
    available_at = _cache_available_at()
    metadata = _load_pool_metadata()
    source = metadata.get("source", "current-cache")
    source_revision = metadata.get("source_revision", "")
    records = tuple(
        _legacy_record(code, name, anchor, available_at, source, source_revision)
        for code, name in sorted(pool.items())
    )
    reasons = _snapshot_reasons(records, scope="current")
    return _make_snapshot(
        scope="current",
        as_of=anchor,
        available_at=available_at,
        source=source,
        records=records,
        history_available=False,
        reasons=reasons,
    )


def get_universe_snapshot(
    as_of: pd.Timestamp | str,
    *,
    history_path: str | Path | None = None,
) -> UniverseSnapshot:
    """按 ``as_of`` 读取点时股票池，不使用未来可见记录或当前池回填。"""
    anchor = _normalize_day(as_of)
    path = Path(history_path) if history_path is not None else _HISTORY_PATH
    if not path.exists():
        return _unavailable_snapshot(anchor, "HISTORY_SOURCE_UNAVAILABLE")
    try:
        records = _load_historical_records(path, anchor)
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        del exc
        return _unavailable_snapshot(anchor, "HISTORY_SCHEMA_INVALID")
    if not records:
        return _unavailable_snapshot(anchor, "HISTORY_NO_MEMBERSHIP_AT_AS_OF")
    reasons = _snapshot_reasons(records, scope="history")
    return _make_snapshot(
        scope="history",
        as_of=anchor,
        available_at=max(item.available_at for item in records),
        source="historical-csv",
        records=records,
        history_available=True,
        reasons=reasons,
    )


def _unavailable_snapshot(as_of: pd.Timestamp, reason: str) -> UniverseSnapshot:
    """构造不可用于研究的历史快照。"""
    return _make_snapshot(
        scope="history",
        as_of=as_of,
        available_at=as_of,
        source="unavailable",
        records=(),
        history_available=False,
        reasons=(reason,),
    )


def _make_snapshot(
    *,
    scope: str,
    as_of: pd.Timestamp,
    available_at: pd.Timestamp,
    source: str,
    records: tuple[InstrumentState, ...],
    history_available: bool,
    reasons: tuple[str, ...],
) -> UniverseSnapshot:
    """根据规范化记录生成稳定版本的股票池快照。"""
    canonical = [
        {
            "code": item.code,
            "name": item.name,
            "security_status": item.security_status,
            "effective_date": item.effective_date.isoformat(),
            "available_at": item.available_at.isoformat(),
            "listed_date": item.listed_date.isoformat() if item.listed_date is not None else None,
            "delisted_date": item.delisted_date.isoformat() if item.delisted_date is not None else None,
            "source": item.source,
            "source_revision": item.source_revision,
            "corporate_action_version": item.corporate_action_version,
        }
        for item in records
    ]
    payload = json.dumps(
        {
            "universe": "hs300",
            "scope": scope,
            "as_of": as_of.isoformat(),
            "available_at": available_at.isoformat(),
            "source": source,
            "records": canonical,
            "reasons": list(reasons),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return UniverseSnapshot(
        universe="hs300",
        scope=scope,
        as_of=as_of,
        available_at=available_at,
        universe_version=hashlib.sha256(payload).hexdigest(),
        source=source,
        instruments=records,
        history_available=history_available,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _snapshot_reasons(
    records: tuple[InstrumentState, ...],
    *,
    scope: str,
) -> tuple[str, ...]:
    """汇总状态、来源和时点缺口；任何关键缺口都不放行。"""
    reasons: list[str] = []
    if not records:
        reasons.append("UNIVERSE_EMPTY")
    for item in records:
        if not _CODE_PATTERN.fullmatch(item.code):
            _add_reason(reasons, "INVALID_SECURITY_CODE")
        if not item.name.strip():
            _add_reason(reasons, "SECURITY_NAME_MISSING")
        if not item.is_status_known:
            _add_reason(reasons, "UNKNOWN_SECURITY_STATUS")
        elif not item.is_active:
            _add_reason(reasons, "SECURITY_NOT_TRADABLE")
        if not item.source.strip() or not item.source_revision.strip():
            _add_reason(reasons, "SOURCE_PROVENANCE_MISSING")
        if item.delisted_date is not None and item.delisted_date <= item.effective_date:
            _add_reason(reasons, "SECURITY_DATE_RANGE_INVALID")
    return tuple(reasons)


def _load_historical_records(path: Path, as_of: pd.Timestamp) -> tuple[InstrumentState, ...]:
    """严格读取历史股票池记录，并按可见时点选择每个代码的最后状态。"""
    frame = pd.read_csv(path, dtype=str)
    if not _REQUIRED_HISTORY_COLUMNS.issubset(frame.columns):
        missing = sorted(_REQUIRED_HISTORY_COLUMNS - set(frame.columns))
        raise ValueError(f"历史股票池缺少字段：{missing}")
    frame = frame[sorted(_REQUIRED_HISTORY_COLUMNS)].copy()
    for column in ("effective_date", "available_at", "listed_date", "delisted_date"):
        frame[column] = _parse_history_dates(frame[column], column, allow_empty=column in {"listed_date", "delisted_date"})
    for column in ("code", "name"):
        if frame[column].isna().any():
            raise ValueError(f"历史股票池字段缺失：{column}")
        frame[column] = frame[column].astype(str)
    for column in ("effective_date", "available_at"):
        if frame[column].isna().any():
            raise ValueError(f"历史股票池字段缺失：{column}")
    for column in ("security_status", "source", "source_revision"):
        frame[column] = frame[column].fillna("").astype(str)
    frame["corporate_action_version"] = frame["corporate_action_version"].where(
        frame["corporate_action_version"].notna(), None
    )
    visible = frame.loc[
        (frame["effective_date"] <= as_of)
        & (frame["available_at"] <= as_of)
    ].copy()
    if visible.empty:
        return ()
    visible = visible.sort_values(
        ["code", "effective_date", "available_at", "source_revision"],
        kind="mergesort",
    ).drop_duplicates("code", keep="last")
    records: list[InstrumentState] = []
    for row in visible.to_dict("records"):
        status = str(row["security_status"]).strip()
        if status.upper() in _REMOVED_STATES:
            continue
        delisted = _optional_timestamp(row["delisted_date"])
        if delisted is not None and delisted <= as_of:
            continue
        records.append(
            InstrumentState(
                code=_validate_code(row["code"]),
                name=_validate_text(row["name"], "name"),
                security_status=status,
                effective_date=_required_timestamp(row["effective_date"], "effective_date"),
                available_at=_required_timestamp(row["available_at"], "available_at"),
                listed_date=_optional_timestamp(row["listed_date"]),
                delisted_date=delisted,
                source=str(row["source"]).strip(),
                source_revision=str(row["source_revision"]).strip(),
                corporate_action_version=_optional_text(row["corporate_action_version"]),
            )
        )
    return tuple(sorted(records, key=lambda item: item.code))


def _legacy_record(
    code: str,
    name: str,
    as_of: pd.Timestamp,
    available_at: pd.Timestamp,
    source: str,
    source_revision: str,
) -> InstrumentState:
    """把旧版当前字典转换为明确但默认未知状态的记录。"""
    return InstrumentState(
        code=str(code),
        name=str(name),
        security_status="",
        effective_date=as_of,
        available_at=available_at,
        source=source,
        source_revision=source_revision,
    )


def _parse_history_dates(series: pd.Series, name: str, *, allow_empty: bool) -> pd.Series:
    """解析历史日期并拒绝无法审计的时区或时间值。"""
    values = series.astype("object").copy()
    values = values.mask(values.isna())
    values = values.mask(values.astype(str).isin({"", "None", "nan"}))
    parsed = pd.to_datetime(values, errors="coerce", utc=False)
    if parsed.isna().any() and not allow_empty:
        raise ValueError(f"历史股票池日期无效：{name}")
    if getattr(parsed.dt, "tz", None) is not None:
        raise ValueError(f"历史股票池日期不允许带时区：{name}")
    normalized = parsed.dt.normalize()
    if not allow_empty and (parsed != normalized).any():
        raise ValueError(f"历史股票池日期必须为日级：{name}")
    return normalized


def _required_timestamp(value: object, name: str) -> pd.Timestamp:
    """返回已经通过日期校验的必填时间。"""
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"历史股票池时间缺失：{name}")
    return timestamp.normalize()


def _optional_timestamp(value: object) -> pd.Timestamp | None:
    """把可选日期转换为日级时间。"""
    if value is None or pd.isna(value):
        return None
    return _required_timestamp(value, "optional_date")


def _optional_text(value: object) -> str | None:
    """把可选来源文本转换为非空值或空值。"""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _validate_code(value: object) -> str:
    """严格验证六位证券代码，不替换或补齐错误输入。"""
    code = str(value).strip()
    if _CODE_PATTERN.fullmatch(code) is None:
        raise ValueError("证券代码必须是六位数字。")
    return code


def _validate_text(value: object, name: str) -> str:
    """严格验证必填文本字段。"""
    text = str(value).strip()
    if not text:
        raise ValueError(f"历史股票池字段为空：{name}")
    return text


def _add_reason(reasons: list[str], reason: str) -> None:
    """按首次出现顺序记录拒绝原因。"""
    if reason not in reasons:
        reasons.append(reason)


def _today() -> pd.Timestamp:
    """返回上海时区的当前日期，不暴露时分秒到时点契约。"""
    return pd.Timestamp.now(tz=SHANGHAI).tz_localize(None).normalize()


def _normalize_day(value: pd.Timestamp | str) -> pd.Timestamp:
    """解析无时区日级锚点，拒绝无法确定的值。"""
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("股票池 as_of 不能为空。")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(SHANGHAI).tz_localize(None)
    if timestamp != timestamp.normalize():
        raise ValueError("股票池 as_of 必须是日级时间。")
    return timestamp.normalize()


def _fetch_from_akshare() -> dict[str, str]:
    """从 AkShare 获取当前成分股名称。"""
    import akshare as ak

    df = ak.index_stock_cons(symbol="000300")
    code_col = next(c for c in ["品种代码", "constituent_code"] if c in df.columns)
    name_col = next(c for c in ["品种名称", "constituent_name"] if c in df.columns)
    pool: dict[str, str] = {}
    for _, row in df.iterrows():
        code = str(row[code_col])
        name = str(row[name_col])
        pool[code] = name
    return pool


def _load_cache() -> dict[str, str] | None:
    """读取兼容旧版的当前股票池 CSV。"""
    if not _CACHE_PATH.exists():
        return None
    try:
        df = pd.read_csv(_CACHE_PATH, dtype=str)
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None
    if not {"code", "name"}.issubset(df.columns):
        return None
    return dict(zip(df["code"], df["name"]))


def _save_cache(pool: dict[str, str]) -> None:
    """原子写入当前股票池兼容缓存。"""
    df = pd.DataFrame(pool.items(), columns=["code", "name"])
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", delete=False, dir=_CACHE_PATH.parent, suffix=".tmp"
        ) as handle:
            temporary_path = Path(handle.name)
            df.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, _CACHE_PATH)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                temporary_path = None


def _save_pool_metadata(source: str) -> None:
    """记录当前池来源与可见修订，不写入证券状态猜测。"""
    payload = {
        "source": source,
        "source_revision": datetime.now(SHANGHAI).date().isoformat(),
        "available_at": datetime.now(SHANGHAI).isoformat(),
    }
    _META_PATH.parent.mkdir(parents=True, exist_ok=True)
    _META_PATH.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _load_pool_metadata() -> dict[str, str]:
    """读取当前池来源元数据；损坏时返回空对象并由状态门禁拒绝。"""
    try:
        payload = json.loads(_META_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): str(value) for key, value in payload.items() if value is not None}


def _cache_available_at() -> pd.Timestamp:
    """返回当前池缓存的文件可见时间。"""
    try:
        return pd.Timestamp.fromtimestamp(_CACHE_PATH.stat().st_mtime, tz=SHANGHAI).tz_localize(None)
    except (OSError, ValueError):
        return _today()


def _is_fresh() -> bool:
    """判断当前池缓存是否仍在刷新周期内。"""
    if not _CACHE_PATH.exists():
        return False
    mtime = datetime.fromtimestamp(_CACHE_PATH.stat().st_mtime)
    return datetime.now() - mtime < timedelta(days=_REFRESH_INTERVAL_DAYS)


def refresh_pool() -> dict[str, str]:
    """强制刷新当前股票池，供 Web 设置页面调用。"""
    pool = _fetch_from_akshare()
    _save_cache(pool)
    _save_pool_metadata("akshare")
    return pool


__all__ = [
    "get_current_universe_snapshot",
    "get_hs300_pool",
    "get_universe_snapshot",
    "refresh_pool",
]
