"""测试数据获取模块。"""
import pytest
from data.fetcher import DataFetcher, DataSourceError


@pytest.fixture
def fetcher():
    return DataFetcher()


def test_fetch_real_stock(fetcher):
    """真实获取一只股票数据，验证格式。"""
    df = fetcher.fetch_daily("600519")
    assert len(df) >= 50
    assert "open" in df.columns
    assert "high" in df.columns
    assert "low" in df.columns
    assert "close" in df.columns
    assert df["high"].iloc[-1] >= df["low"].iloc[-1]


def test_fetch_invalid_code_raises(fetcher):
    """无效股票代码应该抛出异常或返回空。"""
    with pytest.raises(DataSourceError):
        fetcher.fetch_daily("999999")


def test_cache_hit(fetcher):
    """两次连续请求应从缓存命中（速度更快）。"""
    import time
    fetcher.fetch_daily("000001")
    start = time.time()
    fetcher.fetch_daily("000001")
    elapsed = time.time() - start
    assert elapsed < 1.0  # 缓存命中应极快