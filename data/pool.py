"""沪深 300 股票池管理。

通过 akshare 动态获取最新成分股，本地缓存每周自动刷新。
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

from decision.config import get_config

_CACHE_PATH = Path(get_config().data.cache_dir) / "pool_hs300.parquet"
_REFRESH_INTERVAL_DAYS = 7


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
        pool = _fetch_from_akshare()
        _save_cache(pool)
        return pool
    except Exception:
        if cached is not None:
            return cached  # 降级
        raise DataSourceError("无法获取沪深 300 成分股列表")


def _fetch_from_akshare() -> dict[str, str]:
    import akshare as ak
    df = ak.index_stock_cons(symbol="000300")
    # akshare 返回列: "品种代码", "品种名称"
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
    df = pd.read_parquet(_CACHE_PATH)
    return dict(zip(df["code"], df["name"]))


def _save_cache(pool: dict[str, str]) -> None:
    import pandas as pd
    df = pd.DataFrame(pool.items(), columns=["code", "name"])
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_CACHE_PATH, index=False)


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