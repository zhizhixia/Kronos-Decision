"""可信研究第一阶段的纯离线回归测试。"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from data.fetcher import DataFetcher
from data.universe import HistoricalUniverse
from evaluation.artifacts import EvaluationArtifacts
from evaluation.gates import evaluate_evidence_gate
from evaluation.run import run
from model.kronos import KronosPredictor
from portfolio.store import PortfolioStore


def _bars(rows: int = 60) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=rows)
    return pd.DataFrame({"date": dates, "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1000.0, "amount": 10000.0})


def test_path_repair_preserves_ohlc_and_nonnegative_volume() -> None:
    paths = np.array([[[10.0, 8.0, 12.0, 11.0, -1.0, -2.0]]])
    repaired, count = KronosPredictor._repair_paths(paths)
    assert count == 4
    assert repaired[0, 0, 1] >= max(repaired[0, 0, 0], repaired[0, 0, 3], repaired[0, 0, 2])
    assert repaired[0, 0, 2] <= min(repaired[0, 0, 0], repaired[0, 0, 3], repaired[0, 0, 1])
    assert (repaired[:, :, 4:] >= 0).all()


def test_bundle_excludes_incomplete_session_and_marks_calendar_fallback(tmp_path) -> None:
    fetcher = DataFetcher()
    fetcher._cache_dir = tmp_path
    fetcher._calendar.latest_complete_session = lambda: (pd.Timestamp("2024-03-01"), ("CALENDAR_WEEKDAY_FALLBACK",))
    bars = _bars().assign(date=pd.bdate_range("2023-12-11", periods=60))
    bundle = fetcher._build_bundle(bars, "fixture", False, ("fixture",), None)
    assert bundle.bars["date"].max() <= pd.Timestamp("2024-03-01")
    assert "CALENDAR_WEEKDAY_FALLBACK" in bundle.quality_flags
    assert len(bundle.content_hash) == 64


def test_gate_requires_every_metric_and_passes_complete_metrics() -> None:
    failed = evaluate_evidence_gate({})
    assert not failed.passed and "FRESH_ARTIFACT" in failed.failed_codes
    metrics = {"data_complete": True, "quality_error": False, "coverage_80": 0.8, "up_probability_ece": 0.08, "rank_ic_mean": 0.01, "rank_ic_positive_probability": 0.8, "annualized_excess_return": 0.01, "information_ratio": 0.5, "positive_12m_window_ratio": 0.6, "drawdown_worsening": 0.05, "versions_match": True, "formal_protocol": True, "evaluated_at": datetime.now().isoformat()}
    assert evaluate_evidence_gate(metrics).passed


def test_universe_requires_effective_dates_and_queries_anchor(tmp_path) -> None:
    source = tmp_path / "source.csv"
    pd.DataFrame({"stock_code": ["600519", "000001"], "start_date": ["2020-01-01", "2024-01-01"], "end_date": ["2023-12-31", "2024-12-31"], "source": ["fixture", "fixture"], "snapshot_version": ["v1", "v1"]}).to_csv(source, index=False)
    universe = HistoricalUniverse(tmp_path / "members.csv")
    assert universe.import_csv(source) == 2
    assert universe.members_at("2023-06-01") == {"600519"}
    assert universe.members_at("2024-06-01") == {"000001"}


def test_universe_membership_changes_across_anchors(tmp_path) -> None:
    """动态成分：股票加入与退出在不同锚点必须反映在成员集中。"""
    source = tmp_path / "source.csv"
    pd.DataFrame({"stock_code": ["600519", "000001"], "start_date": ["2020-01-01", "2022-01-01"], "end_date": ["2021-12-31", "2024-12-31"], "source": ["fixture", "fixture"], "snapshot_version": ["v1", "v1"]}).to_csv(source, index=False)
    universe = HistoricalUniverse(tmp_path / "members.csv")
    universe.import_csv(source)
    assert universe.members_at("2020-06-01") == {"600519"}
    assert universe.members_at("2022-06-01") == {"000001"}
    assert universe.members_at("2026-06-01") == set()


def test_artifact_and_portfolio_are_local_and_replayable(tmp_path) -> None:
    artifacts = EvaluationArtifacts.open("smoke", tmp_path / "artifacts")
    artifacts.write_manifest({"status": "completed", "as_of": "2024-01-01", "seed": 1, "horizons": [20], "model": "fixture", "config_hash": "a", "data_hash": "b"})
    artifacts.write_frame("predictions", pd.DataFrame({"prediction_key": ["k1"], "sampling_seed": [1], "sampling_params_hash": ["fixture"]}))
    assert artifacts.has_prediction("k1")
    store = PortfolioStore(tmp_path / "kronos.db")
    result = store.replace_portfolio({"risk_profile": "balanced", "cash": 1000, "holdings": [{"stock_code": "600519", "shares": 100, "cost_basis": 10}]})
    assert result["holdings"][0]["shares"] == 100


def test_evaluation_preflight_records_blockers_without_claiming_success(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    # 测试环境已安装 pyqlib（vendor 目录），这里显式模拟 qlib 缺失以验证预检失败闭合
    import sys

    monkeypatch.setitem(sys.modules, "qlib", None)
    artifacts, manifest = run("full", "2024-01-01", "preflight")
    assert artifacts.root.joinpath("manifest.json").exists()
    assert manifest["status"] == "blocked"
    assert "POINT_IN_TIME_CSI300_MISSING" in manifest["failure_reason"]
    assert "PYQLIB_NOT_INSTALLED" in manifest["failure_reason"]
