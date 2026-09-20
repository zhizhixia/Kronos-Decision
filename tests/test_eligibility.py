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
    result = evaluate_stock_eligibility(
        _bars(), "贵州茅台", pd.Timestamp("2024-06-01"), security_status="NORMAL"
    )
    assert result.eligible
    assert result.reasons == ()


def test_missing_security_status_blocks() -> None:
    result = evaluate_stock_eligibility(_bars(), "某股", pd.Timestamp("2024-06-01"))
    assert not result.eligible
    assert "UNKNOWN_SECURITY_STATUS" in result.reasons


def test_explicit_unknown_security_status_blocks() -> None:
    result = evaluate_stock_eligibility(
        _bars(), "某股", pd.Timestamp("2024-06-01"), security_status="UNKNOWN"
    )
    assert not result.eligible
    assert "UNKNOWN_SECURITY_STATUS" in result.reasons


def test_explicit_suspended_security_status_blocks() -> None:
    result = evaluate_stock_eligibility(
        _bars(), "某股", pd.Timestamp("2024-06-01"), security_status="SUSPENDED"
    )
    assert not result.eligible
    assert "SECURITY_SUSPENDED" in result.reasons


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


def test_missing_amount_column_blocks_liquidity_claim() -> None:
    result = evaluate_stock_eligibility(
        _bars().drop(columns=["amount"]), "某股", pd.Timestamp("2024-06-01")
    )
    assert not result.eligible
    assert "LIQUIDITY_EVIDENCE_MISSING" in result.reasons


def test_missing_amount_data_blocks_eligibility() -> None:
    """全零成交额缺少流动性证据时必须拒绝资格。"""
    result = evaluate_stock_eligibility(
        _bars(amount=0.0), "某股", pd.Timestamp("2024-06-01"), security_status="NORMAL"
    )
    assert not result.eligible
    assert "LIQUIDITY_EVIDENCE_MISSING" in result.reasons


def test_calendar_display_flags_are_known_non_blocking_statuses() -> None:
    result = evaluate_stock_eligibility(
        _bars(),
        "某股",
        pd.Timestamp("2024-06-01"),
        security_status="NORMAL",
        quality_flags=("INTRADAY_SESSION", "NON_TRADING_DAY_WEEKEND", "NON_TRADING_DAY_HOLIDAY"),
    )
    assert result.eligible
    assert result.reasons == ()


def test_incomplete_latest_session_blocks() -> None:
    result = evaluate_stock_eligibility(
        _bars(), "某股", pd.Timestamp("2024-06-01"), data_complete=False
    )
    assert not result.eligible
    assert "INCOMPLETE_LATEST_SESSION" in result.reasons


def test_incomplete_quality_flag_blocks() -> None:
    result = evaluate_stock_eligibility(
        _bars(),
        "某股",
        pd.Timestamp("2024-06-01"),
        quality_flags=("ERROR_INCOMPLETE_LATEST_SESSION",),
    )
    assert not result.eligible
    assert "ERROR_INCOMPLETE_LATEST_SESSION" in result.reasons


def test_future_bars_are_not_silently_trimmed_for_historical_as_of() -> None:
    result = evaluate_stock_eligibility(_bars(), "某股", pd.Timestamp("2024-05-01"))
    assert not result.eligible
    assert "DATA_AFTER_AS_OF" in result.reasons
