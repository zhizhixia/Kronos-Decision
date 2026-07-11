"""测试股票池模块。"""
import pytest
from data.pool import get_hs300_pool, refresh_pool


def test_get_pool_returns_dict():
    """获取的股票池应为非空字典。"""
    pool = get_hs300_pool()
    assert isinstance(pool, dict)
    assert len(pool) > 0
    assert "600519" in pool or "000001" in pool


def test_pool_has_code_and_name():
    """每项包含代码和名称。"""
    pool = get_hs300_pool()
    for code, name in pool.items():
        assert len(code) == 6
        assert len(name) > 0


def test_refresh_pool():
    """强制刷新不报错。"""
    pool = refresh_pool()
    assert len(pool) > 0