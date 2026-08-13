"""Qlib 真实回测集成测试（使用小型点数据目录，自动跳过未安装环境）。"""
from __future__ import annotations

import shutil
import subprocess
import sys
import os
from pathlib import Path

import pandas as pd
import pytest

from evaluation.contracts import prediction_key


def _qlib_available() -> bool:
    try:
        import qlib  # noqa: F401
        return True
    except Exception:
        return False


def _build_mini_data(tmp_path: Path) -> Path:
    cache = Path("data/cache")
    if not cache.joinpath("600519.csv").exists():
        pytest.skip("缺少缓存数据，无法构建小型 Qlib 数据。")
    target = tmp_path / "cn_data"
    subprocess.run(
        [sys.executable, "scripts/build_mini_qlib.py", "--cache-dir", str(cache), "--output", str(target), "--codes", "600519,000001"],
        check=True,
        capture_output=True,
    )
    return target


def _predictions() -> pd.DataFrame:
    rows = []
    anchors = ["2024-02-01", "2024-02-08", "2024-02-23"]
    for anchor in anchors:
        execution = (pd.Timestamp(anchor) + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
        for index, code in enumerate(["600519", "000001"]):
            rows.append({
                "prediction_key": prediction_key(code, anchor, "m", "c"),
                "anchor_date": anchor,
                "execution_date": execution,
                "stock_code": code,
                "horizon": 20,
                "score": float(2 - index),
                "predicted_return": 0.01,
                "actual_return": 0.01,
                "data_as_of": anchor,
                "model_hash": "m",
                "config_hash": "c",
                "data_hash": "d",
            })
    return pd.DataFrame(rows)


@pytest.mark.skipif(not _qlib_available(), reason="pyqlib 未安装")
def test_qlib_backtest_end_to_end(tmp_path) -> None:
    from evaluation.qlib_adapter import QlibBacktestAdapter, QlibBacktestConfig
    from evaluation.metrics import portfolio_metrics

    target = _build_mini_data(tmp_path)
    adapter = QlibBacktestAdapter(QlibBacktestConfig(provider_uri=str(target), topk=1, n_drop=0))
    report, extra = adapter.run(_predictions())
    assert "return" in report and "cost" in report and "bench" in report
    assert len(report) > 20
    metrics = portfolio_metrics(report)
    assert "annualized_excess_return" in metrics
    assert (report["cost"].abs().sum() >= 0)


@pytest.mark.skipif(not _qlib_available(), reason="pyqlib 未安装")
def test_qlib_backtest_uses_registered_total_return_benchmark(tmp_path) -> None:
    from evaluation.qlib_adapter import QlibBacktestAdapter, QlibBacktestConfig
    from scripts.import_total_return_benchmark import INSTRUMENT, import_total_return

    target = _build_mini_data(tmp_path)
    dates = pd.to_datetime((target / "calendars" / "day.txt").read_text(encoding="utf-8").splitlines())
    source = tmp_path / "h00300.csv"
    pd.DataFrame({"date": dates, "close": range(1000, 1000 + len(dates))}).to_csv(source, index=False)
    import_total_return(source, target, source="test_fixture", source_url="https://www.csindex.com.cn/indices/H00300")

    adapter = QlibBacktestAdapter(QlibBacktestConfig(provider_uri=str(target), benchmark=INSTRUMENT, benchmark_is_total_return=True, topk=1, n_drop=0))
    report, _ = adapter.run(_predictions())

    assert "bench" in report and report["bench"].notna().all()
