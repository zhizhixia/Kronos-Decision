"""v1 接口兼容快照：记录并固定旧 API 的响应契约。"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from decision.engine import DecisionReport


def test_v1_decision_report_contract_snapshot() -> None:
    """记录 v1 /api/decision-report 响应结构，防止字段丢失。"""
    report = DecisionReport(status="ok", signal=None, prediction={"pred_df": [], "pred_len": 60, "current_price": 100.0}, stock_code="600519", stock_name="贵州茅台", elapsed_seconds=1.5)
    payload = dataclasses.asdict(report)
    assert set(payload) == {"status", "error_message", "signal", "prediction", "stock_code", "stock_name", "generated_at", "elapsed_seconds"}
    assert payload["status"] == "ok"
    assert payload["prediction"]["pred_len"] == 60
    snapshot = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    assert "stock_code" in snapshot and "elapsed_seconds" in snapshot


def test_v1_error_report_contract() -> None:
    """v1 错误报告保持 status=error 与 error_message 字段。"""
    report = DecisionReport(status="error", error_message="模型未就绪", stock_code="600519", elapsed_seconds=0.2)
    payload = dataclasses.asdict(report)
    assert payload["status"] == "error"
    assert payload["error_message"] == "模型未就绪"


def test_v1_fetch_daily_still_returns_dataframe_contract(tmp_path, monkeypatch) -> None:
    """v1 fetch_daily 的 DataFrame 接口契约保持不变。"""
    import pandas as pd
    from data.fetcher import DataFetcher
    from data.contracts import MarketDataBundle

    fetcher = DataFetcher()
    fetcher._cache_dir = tmp_path
    bars = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=60), "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1000.0, "amount": 10000.0})
    bundle = MarketDataBundle(bars, "fixture", pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-01"), "qfq", "v1", None, False, (), ("fixture",), "hash")
    monkeypatch.setattr(fetcher, "fetch_daily_bundle", lambda code: bundle)
    frame = fetcher.fetch_daily("600519")
    assert list(frame.columns) == ["date", "open", "high", "low", "close", "volume", "amount"]
