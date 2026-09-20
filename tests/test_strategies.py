"""验证无模型基线的时点截断和事实语义。"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from data.contracts import MarketDataBundle
from decision.strategies import BaselineDataError, run_momentum_baseline


def _bundle(include_future: bool = True) -> MarketDataBundle:
    """构造包含可截断未来行的不可变输入。"""
    dates = pd.bdate_range("2024-01-02", periods=30)
    bars = pd.DataFrame(
        {
            "date": dates,
            "close": [100.0 + index for index in range(len(dates))],
        }
    )
    if include_future:
        future_dates = pd.bdate_range(dates[-1] + pd.Timedelta(1, unit="D"), periods=3)
        bars = pd.concat(
            [
                bars,
                pd.DataFrame({"date": future_dates, "close": [1.0, 1.0, 1.0]}),
            ],
            ignore_index=True,
        )
    return MarketDataBundle(
        bars=bars,
        source="fixture",
        retrieved_at=datetime.now(timezone.utc),
        as_of=pd.Timestamp(dates[-1]),
        latest_complete_session=pd.Timestamp(dates[-1]),
        adjustment="none",
        calendar_version="fixture-v1",
        universe_version=None,
        is_stale=False,
        content_hash="fixture-hash",
        snapshot_id="snapshot-fixture",
    )


def test_baseline_uses_only_visible_rows_and_exposes_5_10_20_facts() -> None:
    """as_of 之后的行不能影响基线事实，且三档 horizon 都可追溯。"""
    report = run_momentum_baseline(_bundle(), "600519", "测试股票").report

    assert set(report["horizons"]) == {"5", "10", "20"}
    assert report["data_provenance"]["future_rows_excluded"] == 3
    assert report["data_provenance"]["max_input_date"] == "2024-02-12T00:00:00"
    assert report["recommendation"]["action"] is None
    assert report["action_permission"] == "NONE"
    assert report["model_provenance"]["model_revision"] == "baseline-momentum-v1"
    assert all(item["direction"] == "UP" for item in report["baseline_facts"]["supporting"])


def test_baseline_is_replayable_when_future_rows_change() -> None:
    """只改变 as_of 之后的行时，基线输出的历史事实保持不变。"""
    first = run_momentum_baseline(_bundle(), "600519").report
    changed = _bundle()
    changed.bars.loc[changed.bars["date"] > changed.as_of, "close"] = 9999.0
    second = run_momentum_baseline(changed, "600519").report

    assert first["horizons"] == second["horizons"]
    assert first["baseline_facts"] == second["baseline_facts"]


def test_baseline_rejects_missing_snapshot_or_invalid_prices() -> None:
    """没有不可变引用或价格质量错误时必须失败闭合。"""
    missing_snapshot = _bundle()
    object.__setattr__(missing_snapshot, "snapshot_id", "")
    with pytest.raises(BaselineDataError, match="snapshot_id"):
        run_momentum_baseline(missing_snapshot, "600519")

    invalid = _bundle(include_future=False)
    invalid.bars.loc[0, "close"] = 0.0
    with pytest.raises(BaselineDataError, match="非有限值或非正"):
        run_momentum_baseline(invalid, "600519")
