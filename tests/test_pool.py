"""测试股票池模块。"""
import pytest


@pytest.fixture
def cached_pool(tmp_path, monkeypatch):
    """以临时 CSV 缓存隔离股票池合同测试，默认不访问网络。"""
    import data.pool as pool_module

    monkeypatch.setattr(pool_module, "_CACHE_PATH", tmp_path / "pool_hs300.csv")
    pool = {"600519": "贵州茅台", "000001": "平安银行"}
    pool_module._save_cache(pool)
    return pool_module, pool


def test_get_pool_returns_dict(cached_pool):
    """获取的股票池应为非空字典。"""
    pool_module, _ = cached_pool
    pool = pool_module.get_hs300_pool()
    assert isinstance(pool, dict)
    assert len(pool) > 0
    assert "600519" in pool or "000001" in pool


def test_pool_has_code_and_name(cached_pool):
    """每项包含代码和名称。"""
    pool_module, _ = cached_pool
    pool = pool_module.get_hs300_pool()
    for code, name in pool.items():
        assert len(code) == 6
        assert len(name) > 0


def test_refresh_pool_writes_mocked_source_to_cache(cached_pool, monkeypatch):
    """强制刷新使用数据源结果并原子写入缓存，不依赖真实 AkShare。"""
    pool_module, _ = cached_pool
    refreshed = {"600519": "贵州茅台", "601318": "中国平安"}
    monkeypatch.setattr(pool_module, "_fetch_from_akshare", lambda: refreshed)
    assert pool_module.refresh_pool() == refreshed
    assert pool_module._load_cache() == refreshed


def test_corrupt_cache_falls_back_to_source(cached_pool, monkeypatch):
    """截断或缺列的股票池缓存不能阻止主源恢复。"""
    pool_module, _ = cached_pool
    pool_module._CACHE_PATH.write_text("broken\n", encoding="utf-8")
    recovered = {"600519": "贵州茅台"}
    monkeypatch.setattr(pool_module, "_fetch_from_akshare", lambda: recovered)

    assert pool_module.get_hs300_pool() == recovered
