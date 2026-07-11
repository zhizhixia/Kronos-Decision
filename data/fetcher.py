"""A 股数据获取模块。

支持 akshare（主）→ baostock（备）→ 缓存（降级）三级策略。
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from decision.config import get_config
from decision.errors import DataSourceError, DataValidationError

# 必需的 OHLC 列
REQUIRED_COLUMNS = ["open", "high", "low", "close"]
# 可选列（缺失时填 0）
OPTIONAL_COLUMNS = ["volume", "amount"]


class DataFetcher:
    """A 股日 K 数据获取器。

    用法:
        fetcher = DataFetcher()
        df = fetcher.fetch_daily("600519")
    """

    def __init__(self) -> None:
        cfg = get_config().data
        self._cache_dir = Path(cfg.cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._ttl_hours = cfg.cache_ttl_hours
        self._retry_max = cfg.retry_max
        self._retry_backoff = cfg.retry_backoff_seconds

    def fetch_daily(self, stock_code: str) -> pd.DataFrame:
        """获取指定股票日 K 数据。

        优先级: 缓存（有效期内）→ akshare → baostock → 缓存（过期）。

        Args:
            stock_code: 6 位股票代码，如 "600519"

        Returns:
            包含 OHLCV 列的 DataFrame，按日期升序排列

        Raises:
            DataSourceError: 所有数据源均不可用且无缓存
        """
        # 检查有效缓存
        cached = self._load_cache(stock_code)
        if cached is not None and self._is_cache_fresh(stock_code):
            return cached

        # 尝试主数据源
        try:
            df = self._fetch_with_retry(self._fetch_akshare, stock_code)
            self._validate(df, stock_code)
            self._save_cache(stock_code, df)
            return df
        except (DataSourceError, DataValidationError):
            pass

        # 尝试备用数据源
        try:
            df = self._fetch_with_retry(self._fetch_baostock, stock_code)
            self._validate(df, stock_code)
            self._save_cache(stock_code, df)
            return df
        except (DataSourceError, DataValidationError):
            pass

        # 降级到过期缓存
        if cached is not None:
            import warnings
            warnings.warn(
                f"⚠️ 数据源不可用，使用 {stock_code} 缓存数据（可能不是最新）"
            )
            return cached

        raise DataSourceError(
            f"无法获取 {stock_code} 数据：所有数据源均不可用且无缓存"
        )

    # ── 私有方法 ──────────────────────────────────────

    def _fetch_with_retry(self, func, stock_code: str) -> pd.DataFrame:
        last_error = None
        for attempt in range(self._retry_max):
            try:
                return func(stock_code)
            except Exception as e:
                last_error = e
                if attempt < self._retry_max - 1:
                    delay = self._retry_backoff[attempt]
                    time.sleep(delay)
        raise DataSourceError(
            f"获取 {stock_code} 数据失败（已重试 {self._retry_max} 次）: {last_error}"
        )

    def _fetch_akshare(self, stock_code: str) -> pd.DataFrame:
        import akshare as ak
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date="20000101",
            end_date=datetime.now().strftime("%Y%m%d"),
            adjust="qfq",
        )
        if df is None or df.empty:
            raise DataSourceError(f"akshare 返回 {stock_code} 空数据")
        return self._normalize_dataframe(df)

    def _fetch_baostock(self, stock_code: str) -> pd.DataFrame:
        import baostock as bs
        bs.login()
        try:
            code = f"sh.{stock_code}" if stock_code.startswith(("5", "6", "9")) else f"sz.{stock_code}"
            rs = bs.query_history_k_data_plus(
                code,
                "date,open,high,low,close,volume,amount",
                start_date="2000-01-01",
                end_date=datetime.now().strftime("%Y-%m-%d"),
                frequency="d",
                adjustflag="2",
            )
            if rs.error_code != "0":
                raise DataSourceError(f"baostock 查询错误: {rs.error_msg}")
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            df = pd.DataFrame(rows, columns=["日期", "open", "high", "low", "close", "volume", "amount"])
            if df.empty:
                raise DataSourceError(f"baostock 返回 {stock_code} 空数据")
            return self._normalize_dataframe(df)
        finally:
            bs.logout()

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """统一 DataFrame 格式：日期索引、数值列、升序排列。"""
        # 找到日期列（akshare 用 "日期"，baostock 用 "date"）
        date_col = next((c for c in ["日期", "date"] if c in df.columns), df.columns[0])
        df = df.rename(columns={date_col: "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        # 统一列名
        col_map = {c: c.lower() for c in df.columns}
        df = df.rename(columns=col_map)
        # 转换数值类型
        for col in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # 补充缺失的可选列
        for col in OPTIONAL_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0
        return df

    def _validate(self, df: pd.DataFrame, stock_code: str) -> None:
        """校验数据完整性。"""
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise DataValidationError(
                f"{stock_code} 缺少必需列: {missing}"
            )
        null_count = df[REQUIRED_COLUMNS].isnull().any(axis=1).sum()
        if null_count > 0:
            df = df.dropna(subset=REQUIRED_COLUMNS)
        if len(df) < 50:
            raise DataValidationError(
                f"{stock_code} 有效数据行数不足（{len(df)} < 50）"
            )
        # 高低价合理性检查
        invalid_hl = (df["high"] < df["low"]).sum()
        if invalid_hl > len(df) * 0.05:
            raise DataValidationError(
                f"{stock_code} 超过 5% 行的高价低于低价"
            )

    def _cache_path(self, stock_code: str) -> Path:
        return self._cache_dir / f"{stock_code}.csv"

    def _load_cache(self, stock_code: str) -> pd.DataFrame | None:
        path = self._cache_path(stock_code)
        if path.exists():
            return pd.read_csv(path, parse_dates=["date"])
        return None

    def _save_cache(self, stock_code: str, df: pd.DataFrame) -> None:
        df.to_csv(self._cache_path(stock_code), index=False)

    def _is_cache_fresh(self, stock_code: str) -> bool:
        """判断缓存是否在有效期内。今日收盘后（16:00）缓存视为过期。"""
        path = self._cache_path(stock_code)
        if not path.exists():
            return False
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        age = datetime.now() - mtime
        if age > timedelta(hours=self._ttl_hours):
            return False
        # 今日 16:00 后缓存立即过期
        now = datetime.now()
        if now.hour >= 16 and mtime.date() < now.date():
            return False
        return True