"""A 股日线获取：AkShare、BaoStock 与 CSV 缓存三级降级。"""
from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from data.calendar import AshareCalendar, SHANGHAI
from data.contracts import MarketDataBundle, bars_hash
from decision.config import get_config
from decision.errors import DataSourceError, DataValidationError

REQUIRED_COLUMNS = ["open", "high", "low", "close"]
OPTIONAL_COLUMNS = ["volume", "amount"]
COLUMN_ALIASES = {
    "日期": "date", "date": "date", "trade_date": "date",
    "开盘": "open", "open": "open", "最高": "high", "high": "high",
    "最低": "low", "low": "low", "收盘": "close", "close": "close",
    "成交量": "volume", "volume": "volume", "vol": "volume",
    "成交额": "amount", "amount": "amount",
}


class DataFetcher:
    """返回可追溯 ``MarketDataBundle`` 的 A 股日线获取器。"""

    def __init__(self) -> None:
        cfg = get_config().data
        self._cache_dir = Path(cfg.cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._ttl_hours = cfg.cache_ttl_hours
        self._retry_max = cfg.retry_max
        self._retry_backoff = cfg.retry_backoff_seconds
        self._source_timeout_seconds = 60
        self._calendar = AshareCalendar()
        optional = getattr(cfg, "optional_sources", [])
        optional = [optional] if isinstance(optional, str) else optional
        self._optional_sources = tuple(dict.fromkeys(str(source).lower() for source in optional))
        self._tushare_token_env = str(getattr(cfg, "tushare_token_env", "TUSHARE_TOKEN"))

    def fetch_daily(self, stock_code: str) -> pd.DataFrame:
        """兼容 v1：仅返回日线 DataFrame。"""
        return self.fetch_daily_bundle(stock_code).bars

    def fetch_daily_bundle(self, stock_code: str, as_of: pd.Timestamp | None = None) -> MarketDataBundle:
        """获取截至完整交易日的日线及其来源、新鲜度和质量信息。"""
        cached = self._load_cache(stock_code)
        cache_meta = self._load_cache_metadata(stock_code)
        cached_flags: tuple[str, ...] = ()
        if cached is not None:
            try:
                cached_flags = self._validate(cached, stock_code, as_of)
            except DataValidationError:
                cached = None
                cache_meta = {}
        if cached is not None and self._is_cache_fresh(stock_code) and not self._has_unresolved_gap(cached_flags):
            return self._build_bundle(cached, "csv_cache", False, self._cache_fallback_chain(cache_meta), as_of, cached_flags)
        chain: list[str] = []
        degraded: list[tuple[pd.DataFrame, str, tuple[str, ...]]] = []
        for source, func in self._source_candidates():
            chain.append(source)
            try:
                bars = self._fetch_with_retry(func, stock_code)
                flags = self._validate(bars, stock_code, as_of)
                if self._has_unresolved_gap(flags):
                    degraded.append((bars, source, flags))
                    continue
                bundle = self._build_bundle(bars, source, False, tuple(chain), as_of, flags)
                self._save_cache(stock_code, bundle.bars, bundle)
                return bundle
            except (DataSourceError, DataValidationError):
                continue
        if degraded:
            bars, source, flags = degraded[0]
            bundle = self._build_bundle(bars, source, False, tuple(chain), as_of, flags)
            self._save_cache(stock_code, bundle.bars, bundle)
            return bundle
        if cached is not None:
            return self._build_bundle(cached, "csv_cache", True, tuple(chain + ["csv_cache"]), as_of, (*cached_flags, "STALE_CACHE"))
        raise DataSourceError(f"无法获取 {stock_code} 的可验证日线数据。")

    def _source_candidates(self) -> tuple[tuple[str, Callable[[str], pd.DataFrame]], ...]:
        """返回实际可用的数据源顺序；付费源仅在显式配置且凭据存在时启用。"""
        sources: list[tuple[str, Callable[[str], pd.DataFrame]]] = [("akshare", self._fetch_akshare), ("baostock", self._fetch_baostock)]
        if "tushare" in self._optional_sources and os.getenv(self._tushare_token_env, "").strip():
            sources.append(("tushare", self._fetch_tushare))
        return tuple(sources)

    def _build_bundle(self, bars, source, is_stale, chain, as_of, flags=()):
        cutoff, complete, calendar_flags = self._resolve_cutoff(as_of)
        usable = self._visible_bars(bars, cutoff)
        if usable.empty:
            raise DataValidationError("截至指定日期没有完整日线数据。")
        actual_as_of = pd.Timestamp(usable["date"].max()).normalize()
        incomplete_session = actual_as_of < cutoff
        effective_stale = is_stale or incomplete_session
        final_flags = (*flags, *calendar_flags, *(("ERROR_INCOMPLETE_LATEST_SESSION",) if incomplete_session else ()), *(("STALE_CACHE",) if is_stale else ()))
        return MarketDataBundle(usable, source, datetime.now(SHANGHAI), actual_as_of, complete, "qfq", self._calendar.version, None, effective_stale, tuple(dict.fromkeys(final_flags)), chain, bars_hash(usable))

    def _resolve_cutoff(self, as_of: pd.Timestamp | None) -> tuple[pd.Timestamp, pd.Timestamp, tuple[str, ...]]:
        """统一解析数据包构建和质量校验使用的完整会话上限。"""
        requested = pd.Timestamp(as_of).normalize() if as_of is not None else None
        resolver = getattr(self._calendar, "complete_session_for", None)
        complete, flags = resolver(requested) if callable(resolver) else self._calendar.latest_complete_session()
        cutoff = complete if requested is None else min(requested, complete)
        return cutoff, complete, flags

    @staticmethod
    def _visible_bars(bars: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
        """只保留截止完整会话前的数据，避免历史报告校验未来行。"""
        if "date" not in bars:
            return bars.copy()
        visible = bars.copy()
        visible["date"] = pd.to_datetime(visible["date"]).dt.normalize()
        return visible.loc[visible["date"] <= cutoff].copy()

    def _fetch_with_retry(self, func: Callable[[str], pd.DataFrame], stock_code: str) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(self._retry_max):
            try:
                return self._run_with_timeout(func, stock_code)
            except Exception as exc:
                last_error = exc
                if attempt < self._retry_max - 1:
                    time.sleep(self._retry_backoff[attempt])
        raise DataSourceError(f"获取 {stock_code} 失败：{last_error}")

    def _run_with_timeout(self, func: Callable[[str], pd.DataFrame], stock_code: str) -> pd.DataFrame:
        """用独立线程执行数据源调用，防止无超时的网络请求挂起主线程。"""
        from data.timeout import call_with_timeout

        try:
            return call_with_timeout(lambda: func(stock_code), self._source_timeout_seconds, name=f"fetch-{stock_code}")
        except TimeoutError:
            raise DataSourceError(f"{stock_code} 数据源请求超过 {self._source_timeout_seconds} 秒。")

    def _fetch_akshare(self, stock_code: str) -> pd.DataFrame:
        import akshare as ak
        data = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date="20000101", end_date=datetime.now().strftime("%Y%m%d"), adjust="qfq")
        if data is None or data.empty:
            raise DataSourceError(f"AkShare 未返回 {stock_code} 数据。")
        return self._normalize_dataframe(data)

    def _fetch_baostock(self, stock_code: str) -> pd.DataFrame:
        import baostock as bs
        login = bs.login()
        if login.error_code != "0":
            raise DataSourceError(f"BaoStock 登录失败：{login.error_msg}")
        try:
            market = "sh" if stock_code.startswith(("5", "6", "9")) else "sz"
            result = bs.query_history_k_data_plus(f"{market}.{stock_code}", "date,open,high,low,close,volume,amount", start_date="2000-01-01", end_date=datetime.now().strftime("%Y-%m-%d"), frequency="d", adjustflag="2")
            if result.error_code != "0":
                raise DataSourceError(f"BaoStock 查询失败：{result.error_msg}")
            rows = []
            while result.next():
                rows.append(result.get_row_data())
            return self._normalize_dataframe(pd.DataFrame(rows, columns=["date", *REQUIRED_COLUMNS, *OPTIONAL_COLUMNS]))
        finally:
            bs.logout()

    def _fetch_tushare(self, stock_code: str) -> pd.DataFrame:
        """使用可选 Tushare Pro SDK 获取前复权日线；凭据只从环境变量读取。"""
        token = os.getenv(self._tushare_token_env, "").strip()
        if not token:
            raise DataSourceError(f"Tushare Pro 未配置环境变量 {self._tushare_token_env}。")
        try:
            import tushare as ts
        except ImportError as exc:
            raise DataSourceError("未安装可选依赖 tushare；请安装 requirements-optional.txt。") from exc
        market = "SH" if stock_code.startswith(("5", "6", "9")) else "SZ"
        try:
            api = ts.pro_api(token)
            data = ts.pro_bar(ts_code=f"{stock_code}.{market}", pro_api=api, adj="qfq", asset="E", freq="D", start_date="20000101", end_date=datetime.now().strftime("%Y%m%d"))
        except Exception as exc:
            raise DataSourceError(f"Tushare Pro 获取 {stock_code} 失败。") from exc
        if data is None or data.empty:
            raise DataSourceError(f"Tushare Pro 未返回 {stock_code} 数据。")
        bars = self._normalize_dataframe(data)
        bars["volume"] *= 100.0
        bars["amount"] *= 1000.0
        return bars

    @staticmethod
    def _normalize_dataframe(data: pd.DataFrame) -> pd.DataFrame:
        aliases = {column: COLUMN_ALIASES.get(str(column).strip(), COLUMN_ALIASES.get(str(column).strip().lower(), str(column).strip().lower())) for column in data.columns}
        bars = data.rename(columns=aliases).copy()
        if "date" not in bars:
            raise DataValidationError("日线数据缺少可识别的日期列。")
        bars["date"] = pd.to_datetime(bars["date"], errors="coerce")
        for col in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
            bars[col] = pd.to_numeric(bars[col], errors="coerce") if col in bars else 0.0
        return bars[["date", *REQUIRED_COLUMNS, *OPTIONAL_COLUMNS]].dropna(subset=["date", *REQUIRED_COLUMNS]).sort_values("date").reset_index(drop=True)

    def _validate(self, bars: pd.DataFrame, stock_code: str, as_of: pd.Timestamp | None = None) -> tuple[str, ...]:
        expected = ["date", *REQUIRED_COLUMNS, *OPTIONAL_COLUMNS]
        if any(col not in bars for col in expected):
            raise DataValidationError(f"{stock_code} 缺少必需日线列。")
        visible = self._visible_bars(bars, self._resolve_cutoff(as_of)[0])
        if len(visible) < 50 or visible["date"].duplicated().any() or not visible["date"].is_monotonic_increasing:
            raise DataValidationError(f"{stock_code} 日线数量、日期重复或排序异常。")
        high_ok = visible["high"] >= visible[["open", "close", "low"]].max(axis=1)
        low_ok = visible["low"] <= visible[["open", "close", "high"]].min(axis=1)
        if not high_ok.all() or not low_ok.all() or (visible[REQUIRED_COLUMNS] <= 0).any().any() or (visible[OPTIONAL_COLUMNS] < 0).any().any():
            raise DataValidationError(f"{stock_code} 存在 OHLC 或成交量不变量错误。")
        flags = self._session_quality_flags(visible)
        if (visible["close"].pct_change().abs() > 0.5).any():
            flags.append("ABNORMAL_PRICE_JUMP")
        return tuple(flags)

    def _session_quality_flags(self, bars: pd.DataFrame) -> list[str]:
        """区分零成交记录与本地日历可证实的缺失交易日。"""
        flags = ["POSSIBLE_SUSPENSION_ZERO_VOLUME"] if (bars["volume"] == 0).any() else []
        sessions_between = getattr(self._calendar, "sessions_between", None)
        if not callable(sessions_between):
            return flags
        observed = pd.DatetimeIndex(pd.to_datetime(bars["date"])).normalize().unique()
        expected, calendar_flags = sessions_between(observed.min(), observed.max())
        if calendar_flags or expected.empty:
            return flags
        if not expected.isin(observed).all():
            flags.append("ERROR_UNRESOLVED_SESSION_GAP")
        return flags

    @staticmethod
    def _has_unresolved_gap(flags: tuple[str, ...]) -> bool:
        """缺失交易日不能被缓存新鲜度或来源降级掩盖。"""
        return "ERROR_UNRESOLVED_SESSION_GAP" in flags

    def _cache_path(self, stock_code: str) -> Path:
        return self._cache_dir / f"{stock_code}.csv"

    def _load_cache(self, stock_code: str) -> pd.DataFrame | None:
        """读取缓存；截断或无法解析的 CSV 视为未命中并继续降级链。"""
        path = self._cache_path(stock_code)
        if not path.exists():
            return None
        try:
            return pd.read_csv(path, parse_dates=["date"])
        except (OSError, UnicodeError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
            return None

    def _load_cache_metadata(self, stock_code: str) -> dict:
        """读取缓存来源审计信息；损坏的元数据不影响 CSV 质量校验。"""
        path = self._cache_path(stock_code).with_suffix(".meta.json")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _cache_fallback_chain(metadata: dict) -> tuple[str, ...]:
        """保留缓存最初的数据源链，并注明本次实际从 CSV 读取。"""
        stored = metadata.get("fallback_chain") or [metadata.get("source")]
        clean = [str(source) for source in stored if source]
        return tuple(dict.fromkeys([*clean, "csv_cache"]))

    def _save_cache(self, stock_code: str, bars: pd.DataFrame, bundle: MarketDataBundle) -> None:
        path = self._cache_path(stock_code)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=path.parent, suffix=".tmp") as handle:
            bars.to_csv(handle, index=False)
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
        self._atomic_json(path.with_suffix(".meta.json"), {
            "source": bundle.source,
            "retrieved_at": bundle.retrieved_at.isoformat(),
            "as_of": bundle.as_of.isoformat(),
            "latest_complete_session": bundle.latest_complete_session.isoformat(),
            "adjustment": bundle.adjustment,
            "calendar_version": bundle.calendar_version,
            "universe_version": bundle.universe_version,
            "is_stale": bundle.is_stale,
            "quality_flags": list(bundle.quality_flags),
            "fallback_chain": list(bundle.fallback_chain),
            "content_hash": bundle.content_hash,
        })

    @staticmethod
    def _atomic_json(path: Path, data: dict) -> None:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as handle:
            json.dump(data, handle, ensure_ascii=False, sort_keys=True)
            temp_path = Path(handle.name)
        os.replace(temp_path, path)

    def _is_cache_fresh(self, stock_code: str) -> bool:
        path = self._cache_path(stock_code)
        return path.exists() and (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).total_seconds() <= self._ttl_hours * 3600
