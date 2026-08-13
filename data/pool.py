"""沪深 300 股票池管理。

通过 akshare 动态获取最新成分股，本地 CSV 缓存每周自动刷新。
"""
from __future__ import annotations

import os
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

from decision.config import get_config

_CACHE_PATH = Path(get_config().data.cache_dir) / "pool_hs300.csv"
_REFRESH_INTERVAL_DAYS = 7
_SOURCE_TIMEOUT_SECONDS = 60


def get_hs300_pool() -> dict[str, str]:
    """获取沪深 300 成分股字典。

    Returns:
        {股票代码: 股票名称}，如 {"600519": "贵州茅台", ...}

    Raises:
        DataSourceError: 首次获取失败且无缓存
    """
    from decision.errors import DataSourceError

    # 缓存有效则直接返回
    cached = _load_cache()
    if cached is not None and _is_fresh():
        return cached

    # 尝试 akshare 获取
    try:
        from data.timeout import call_with_timeout

        pool = call_with_timeout(_fetch_from_akshare, _SOURCE_TIMEOUT_SECONDS, name="fetch-hs300-pool")
        _save_cache(pool)
        return pool
    except Exception:
        if cached is not None:
            return cached  # 降级
        raise DataSourceError("无法获取沪深 300 成分股列表")


def _fetch_from_akshare() -> dict[str, str]:
    import akshare as ak
    df = ak.index_stock_cons(symbol="000300")
    code_col = next(c for c in ["品种代码", "constituent_code"] if c in df.columns)
    name_col = next(c for c in ["品种名称", "constituent_name"] if c in df.columns)
    pool = {}
    for _, row in df.iterrows():
        code = str(row[code_col])
        name = str(row[name_col])
        pool[code] = name
    return pool


def _load_cache() -> dict[str, str] | None:
    if not _CACHE_PATH.exists():
        return None
    import pandas as pd

    try:
        df = pd.read_csv(_CACHE_PATH, dtype=str)
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None
    if not {"code", "name"}.issubset(df.columns):
        return None
    return dict(zip(df["code"], df["name"]))


def _save_cache(pool: dict[str, str]) -> None:
    import pandas as pd

    df = pd.DataFrame(pool.items(), columns=["code", "name"])
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=_CACHE_PATH.parent, suffix=".tmp") as handle:
        df.to_csv(handle, index=False)
        temporary_path = Path(handle.name)
    os.replace(temporary_path, _CACHE_PATH)


def _is_fresh() -> bool:
    if not _CACHE_PATH.exists():
        return False
    mtime = datetime.fromtimestamp(_CACHE_PATH.stat().st_mtime)
    return datetime.now() - mtime < timedelta(days=_REFRESH_INTERVAL_DAYS)


def refresh_pool() -> dict[str, str]:
    """强制刷新股票池（供 Web 设置页面调用）。"""
    pool = _fetch_from_akshare()
    _save_cache(pool)
    return pool
