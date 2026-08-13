"""股票交易资格规则测试。"""
from __future__ import annotations

import pandas as pd

from data.eligibility import (
    MIN_AVG_DAILY_AMOUNT,
    MIN_LISTED_SESSIONS,
    evaluate_stock_eligibility,
)


def _bars(days: int = 400, amount: float = 100_000_000.0, last_date: str = "2024-05-31") -> pd.DataFrame:
    dates = pd.bdate_range(end=last_date, periods=days)
    return pd.DataFrame({
        "date": dates,
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.0,
        "volume": 1_000_000.0,
        "amount": amount,
    })


def test_normal_stock_is_eligible() -> None:
    result = evaluate_stock_eligibility(_bars(), "贵州茅台", pd.Timestamp("2024-06-01"))
    assert result.eligible
    assert result.reasons == ()


def test_st_name_blocks_eligibility() -> None:
    result = evaluate_stock_eligibility(_bars(), "*ST 某某", pd.Timestamp("2024-06-01"))
    assert not result.eligible
    assert "ST_OR_DELISTING" in result.reasons


def test_delisting_name_blocks_eligibility() -> None:
    result = evaluate_stock_eligibility(_bars(), "某某退", pd.Timestamp("2024-06-01"))
    assert not result.eligible
    assert "ST_OR_DELISTING" in result.reasons


def test_listed_less_than_252_sessions_blocks() -> None:
    result = evaluate_stock_eligibility(_bars(days=MIN_LISTED_SESSIONS - 1), "某股", pd.Timestamp("2024-06-01"))
    assert not result.eligible
    assert "LISTED_LESS_THAN_252_SESSIONS" in result.reasons


def test_long_suspension_blocks() -> None:
    result = evaluate_stock_eligibility(_bars(last_date="2024-05-01"), "某股", pd.Timestamp("2024-06-01"))
    assert not result.eligible
    assert "LONG_SUSPENSION" in result.reasons


def test_insufficient_liquidity_blocks() -> None:
    result = evaluate_stock_eligibility(_bars(amount=MIN_AVG_DAILY_AMOUNT - 1), "某股", pd.Timestamp("2024-06-01"))
    assert not result.eligible
    assert "INSUFFICIENT_LIQUIDITY" in result.reasons


def test_missing_amount_data_does_not_punish() -> None:
    """成交额字段缺失（全零）时不把数据缺口误判为流动性不足。"""
    result = evaluate_stock_eligibility(_bars(amount=0.0), "某股", pd.Timestamp("2024-06-01"))
    assert result.eligible
