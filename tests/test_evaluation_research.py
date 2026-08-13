"""Qlib 正式研究适配层的离线合同测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.contracts import prediction_key, validate_prediction_frame
from evaluation.metrics import (
    block_bootstrap_positive_probability,
    expected_calibration_error,
    interval_coverage,
    portfolio_metrics,
    rank_ic_by_anchor,
)
from evaluation.qlib_adapter import QlibBacktestAdapter, build_qlib_signal


def _predictions() -> pd.DataFrame:
    rows = []
    for anchor_index, anchor in enumerate(("2024-01-05", "2024-01-12")):
        for stock_index, code in enumerate(("600519", "000001", "300750")):
            score = float(stock_index + anchor_index)
            rows.append({"prediction_key": prediction_key(code, anchor, "m", "c"), "anchor_date": anchor, "execution_date": pd.Timestamp(anchor) + pd.offsets.BDay(1), "stock_code": code, "horizon": 20, "score": score, "predicted_return": score / 100, "actual_return": score / 100, "data_as_of": anchor, "model_hash": "m", "config_hash": "c", "data_hash": "d"})
    return pd.DataFrame(rows)


def test_prediction_contract_rejects_future_data() -> None:
    frame = _predictions()
    frame.loc[0, "data_as_of"] = "2024-01-06"
    with pytest.raises(ValueError, match="未来数据"):
        validate_prediction_frame(frame, horizon=20)


def test_qlib_signal_uses_next_session_and_market_prefix() -> None:
    signal = build_qlib_signal(_predictions())
    assert signal.index.names == ["datetime", "instrument"]
    assert "SH600519" in signal.index.get_level_values("instrument")
    assert "SZ000001" in signal.index.get_level_values("instrument")
    assert signal.index.get_level_values("datetime").min() > pd.Timestamp("2024-01-05")


def test_qlib_signal_filters_multi_horizon_frame() -> None:
    """predictions.csv 同时含 5/20/60 期限时，回测信号只取 20 日。"""
    frame = _predictions()
    multi = pd.concat([frame, frame.assign(horizon=5), frame.assign(horizon=60)], ignore_index=True)
    signal = build_qlib_signal(multi)
    assert len(signal) == len(frame)


def test_prediction_contract_allows_shared_path_key_across_horizons() -> None:
    """同一路径派生的5/20/60日行共享 prediction_key，但同期限不可重复。"""
    frame = _predictions()
    multi = pd.concat([frame, frame.assign(horizon=5), frame.assign(horizon=60)], ignore_index=True)

    checked = validate_prediction_frame(multi)

    assert len(checked) == len(multi)


def test_exchange_contract_uses_open_trade_unit_cost_and_limits() -> None:
    config = QlibBacktestAdapter()._exchange_kwargs()
    assert config["deal_price"] == "$open"
    assert config["trade_unit"] == 100
    assert config["min_cost"] == 5.0
    assert config["limit_threshold"] == 0.095
    assert config["volume_threshold"][0] == "current"


def test_rank_ic_calibration_and_bootstrap_metrics() -> None:
    predictions = _predictions()
    rank_ic = rank_ic_by_anchor(predictions)
    assert np.allclose(rank_ic, 1.0)
    repeated = pd.Series([0.01] * 24)
    assert block_bootstrap_positive_probability(repeated, samples=100, seed=1) == 1.0
    assert expected_calibration_error(pd.Series([0.1, 0.9]), pd.Series([0, 1]), bins=2) == pytest.approx(0.1)
    assert interval_coverage(pd.Series([-0.1, -0.1]), pd.Series([0.1, 0.1]), pd.Series([0.0, 0.2])) == 0.5


def test_rank_ic_uses_twenty_day_horizon_by_default() -> None:
    """5/60日标签不能混入20日正式 RankIC 门禁。"""
    main = _predictions()
    short = main.assign(horizon=5, actual_return=-main["actual_return"])
    multi = pd.concat([main, short], ignore_index=True)

    assert np.allclose(rank_ic_by_anchor(multi), 1.0)
    assert np.allclose(rank_ic_by_anchor(multi, horizon=5), -1.0)


def test_portfolio_metrics_include_cost_and_drawdown() -> None:
    report = pd.DataFrame({"return": [0.01] * 260, "bench": [0.005] * 260, "cost": [0.001] * 260})
    metrics = portfolio_metrics(report)
    assert metrics["annualized_excess_return"] > 0
    assert metrics["information_ratio"] > 0
    assert metrics["positive_12m_window_ratio"] == 1.0
    assert metrics["drawdown_worsening"] == 0.0


def test_cost_before_and_after_metrics_differ() -> None:
    """门禁要求：Qlib 成本前后结果必须有明确差异。"""
    from evaluation.metrics import _annualized_return

    gross = pd.Series([0.001] * 252)
    net = gross - 0.0005
    assert _annualized_return(gross) > _annualized_return(net)
    gross_metrics = portfolio_metrics(pd.DataFrame({"return": gross, "bench": [0.0005] * 252, "cost": [0.0] * 252}))
    net_metrics = portfolio_metrics(pd.DataFrame({"return": gross, "bench": [0.0005] * 252, "cost": [0.0005] * 252}))
    assert net_metrics["annualized_excess_return"] < gross_metrics["annualized_excess_return"]


class _FakePoint:
    """universe_at 返回点时成分；bars_at 对 short_codes 只返回 40 根（模拟数据不足）。"""

    def __init__(self, universe: list[str], short_codes: set[str] | None = None) -> None:
        self._universe = universe
        self._short = set(short_codes or ())

    def universe_at(self, anchor) -> list[str]:
        return list(self._universe)

    def bars_at(self, code: str, anchor) -> pd.DataFrame:
        count = 40 if code in self._short else 100
        return pd.DataFrame({"date": pd.bdate_range("2023-01-02", periods=count), "close": 10.0})

    def future_sessions(self, last_date, count: int) -> pd.Series:
        start = pd.Timestamp(last_date) + pd.offsets.BDay(1)
        return pd.Series(pd.bdate_range(start, periods=count), name="date")


class _FakeResearch:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str]] = []

    def run(self, universe: list[str], anchors: list, skip) -> pd.DataFrame:
        self.calls.append((list(universe), pd.Timestamp(anchors[0]).date().isoformat()))
        return _chunk(universe, anchors[0])


def _chunk(codes: list[str], anchor) -> pd.DataFrame:
    rows = []
    for code in codes:
        key = prediction_key(code, pd.Timestamp(anchor).date().isoformat(), "m", "c")
        rows.append({"prediction_key": key, "anchor_date": pd.Timestamp(anchor).date().isoformat(), "execution_date": (pd.Timestamp(anchor) + pd.offsets.BDay(1)).date().isoformat(), "stock_code": code, "horizon": 20, "score": 0.5, "predicted_return": 0.01, "actual_return": float("nan"), "data_as_of": pd.Timestamp(anchor).date().isoformat(), "model_hash": "m", "config_hash": "c", "data_hash": "d", "sampling_seed": 1, "sampling_params_hash": "h"})
    return pd.DataFrame(rows)


def test_full_predictions_filters_stocks_and_short_history(tmp_path) -> None:
    """完整阶段必须尊重 --stocks 过滤，并剔除历史不足 60 根的成分。"""
    from evaluation.artifacts import EvaluationArtifacts
    from evaluation.run import _generate_full_predictions

    artifacts = EvaluationArtifacts.open("filter-test", tmp_path)
    point = _FakePoint(["600519", "000001", "600036"], short_codes={"600036"})
    research = _FakeResearch()
    frames = _generate_full_predictions(research, point, [pd.Timestamp("2024-02-02")], artifacts, {}, "600519,000001")
    assert research.calls[0][0] == ["600519", "000001"]
    assert len(frames) == 2
    research2 = _FakeResearch()
    frames2 = _generate_full_predictions(research2, point, [pd.Timestamp("2024-02-02")], artifacts, {})
    assert research2.calls[0][0] == ["600519", "000001"]
    assert len(frames2) == 2


def test_full_predictions_resumes_from_existing_artifacts(tmp_path) -> None:
    """断点续跑：全部命中已有预测时复用 predictions.csv，不重复推理。"""
    from evaluation.artifacts import EvaluationArtifacts
    from evaluation.run import _generate_full_predictions

    artifacts = EvaluationArtifacts.open("resume-test", tmp_path)
    point = _FakePoint(["600519", "000001"])
    research = _FakeResearch()
    frames = _generate_full_predictions(research, point, [pd.Timestamp("2024-02-02")], artifacts, {})
    artifacts.write_frame("predictions", frames)

    class _SkippingResearch(_FakeResearch):
        def run(self, universe, anchors, skip):
            return pd.DataFrame()

    resumed = _generate_full_predictions(_SkippingResearch(), point, [pd.Timestamp("2024-02-02")], artifacts, {})
    assert len(resumed) == len(frames)
    assert resumed["prediction_key"].tolist() == frames["prediction_key"].tolist()


def test_legacy_predictions_without_sampling_provenance_are_not_resumed(tmp_path) -> None:
    """旧工件缺少路径种子或采样哈希时必须重新生成。"""
    from evaluation.artifacts import EvaluationArtifacts

    artifacts = EvaluationArtifacts.open("legacy-prediction-test", tmp_path)
    frame = _chunk(["600519"], pd.Timestamp("2024-02-02")).drop(columns=["sampling_seed", "sampling_params_hash"])
    artifacts.write_frame("predictions", frame)

    assert not artifacts.has_prediction(str(frame["prediction_key"].iloc[0]))


def test_incompatible_prediction_checkpoint_is_filtered(tmp_path) -> None:
    """模型、配置或数据哈希变化后，旧预测不得并入当前正式信号。"""
    from evaluation.artifacts import EvaluationArtifacts
    from evaluation.run import _load_existing_predictions

    artifacts = EvaluationArtifacts.open("incompatible-prediction-test", tmp_path)
    frame = _chunk(["600519"], pd.Timestamp("2024-02-02"))
    artifacts.write_frame("predictions", frame)

    loaded = _load_existing_predictions(artifacts, {"model_hash": "other-model", "config_hash": "c", "data_hash": "d"})

    assert loaded.empty


def test_version_match_checks_prediction_rows_and_current_hashes(monkeypatch) -> None:
    """正式门禁的版本匹配不能再由固定 True 代替。"""
    import importlib

    run_module = importlib.import_module("evaluation.run")
    frame = _predictions()
    frame.loc[:, "model_hash"] = "model"
    frame.loc[:, "config_hash"] = "config"
    frame.loc[:, "data_hash"] = "data"
    frame.loc[:, "sampling_params_hash"] = "sampling"
    manifest = {"model_hash": "model", "config_hash": "config", "data_hash": "data", "sampling_params_hash": "sampling", "rules_hash": "rules", "model_provenance": {"revisions_pinned": True}}
    monkeypatch.setattr(run_module, "current_config_hash", lambda: "config")
    monkeypatch.setattr(run_module, "rules_hash", lambda: "rules")
    monkeypatch.setattr(run_module, "model_provenance", lambda: {"pipeline_hash": "model", "revisions_pinned": True})

    assert run_module._versions_match(frame, manifest)
    frame.loc[0, "data_hash"] = "other"
    assert not run_module._versions_match(frame, manifest)
    frame.loc[:, "data_hash"] = "data"
    frame.loc[0, "sampling_params_hash"] = "other"
    assert not run_module._versions_match(frame, manifest)


def test_full_predictions_merges_new_rows_with_checkpoint(tmp_path) -> None:
    """断点续跑产生新锚点时，检查点与新增预测必须共同保留。"""
    from evaluation.artifacts import EvaluationArtifacts
    from evaluation.run import _generate_full_predictions

    class _CacheAwareResearch(_FakeResearch):
        def run(self, universe, anchors, skip) -> pd.DataFrame:
            self.calls.append((list(universe), pd.Timestamp(anchors[0]).date().isoformat()))
            frame = _chunk(universe, anchors[0])
            return frame.loc[~frame["prediction_key"].map(skip)].reset_index(drop=True)

    artifacts = EvaluationArtifacts.open("resume-merge-test", tmp_path)
    point = _FakePoint(["600519", "000001"])
    first_anchor = pd.Timestamp("2024-02-02")
    checkpoint = _chunk(["600519", "000001"], first_anchor)
    checkpoint.loc[:, "actual_return"] = 0.123
    artifacts.write_frame("predictions", checkpoint)

    anchors = [first_anchor, *pd.bdate_range("2024-02-09", periods=5, freq="W-FRI")]
    resumed = _generate_full_predictions(_CacheAwareResearch(), point, anchors, artifacts, {})
    persisted = pd.read_csv(artifacts.root / "predictions.csv", dtype={"stock_code": str})

    assert len(resumed) == 12
    assert len(persisted) == 12
    assert resumed["prediction_key"].is_unique
    assert resumed.loc[resumed["anchor_date"] == "2024-02-02", "actual_return"].tolist() == [0.123, 0.123]


def test_manifest_records_anchor_frequency_and_replay_command() -> None:
    """轻量月度研究运行也必须完整记录可重放参数。"""
    from evaluation.run import _manifest

    manifest = _manifest("smoke", "2024-02-01", "monthly-unit", 30, "600519,000001", "2024-01-01", "monthly", [])

    assert manifest["anchor_frequency"] == "monthly"
    assert not manifest["formal_protocol"]
    assert manifest["stocks"] == ["600519", "000001"]
    assert "--anchors-start 2024-01-01" in manifest["command"]
    assert "--anchor-frequency monthly" in manifest["command"]
    assert "--sample-count 30" in manifest["command"]
    assert '--stocks "600519,000001"' in manifest["command"]
    assert "--run-id monthly-unit" in manifest["command"]


def test_parser_defaults_to_weekly_anchor_frequency() -> None:
    """未指定频率时保持正式每周锚点协议。"""
    from evaluation.run import build_parser

    args = build_parser().parse_args(["--stage", "full", "--as-of", "2024-02-01"])
    assert args.anchor_frequency == "weekly"


def test_parser_accepts_bounded_prediction_collection() -> None:
    """长周期评估可限制本次新锚点数量，避免每次都触发完整回测。"""
    from evaluation.run import build_parser

    args = build_parser().parse_args(["--stage", "full", "--as-of", "2024-02-01", "--collect-only", "--max-new-anchors", "3"])
    assert args.collect_only
    assert args.max_new_anchors == 3


def test_bounded_anchor_run_forces_collection_mode(monkeypatch) -> None:
    """带锚点上限的运行必须走仅收集路径，不能意外触发正式结论。"""
    import importlib

    run_module = importlib.import_module("evaluation.run")
    seen = {}

    class _Artifacts:
        def __init__(self, run_id: str) -> None:
            self.run_id = run_id

        @classmethod
        def open(cls, run_id: str):
            return cls(run_id)

        def write_manifest(self, manifest: dict) -> None:
            seen["manifest"] = dict(manifest)

        def write_gate_result(self, result: dict) -> None:
            seen["gate"] = dict(result)

    def fake_full(as_of, anchors_start, artifacts, manifest, stocks, sample_count, frequency, collect_only, maximum) -> None:
        seen.update({"as_of": as_of, "collect_only": collect_only, "maximum": maximum, "formal_protocol": manifest["formal_protocol"]})

    monkeypatch.setattr(run_module, "EvaluationArtifacts", _Artifacts)
    monkeypatch.setattr(run_module, "_preflight", lambda *args: [])
    monkeypatch.setattr(run_module, "_run_full", fake_full)

    _, manifest = run_module.run("full", "2020-09-25", run_id="bounded-test", max_new_anchors=2)

    assert seen["collect_only"]
    assert seen["maximum"] == 2
    assert not seen["formal_protocol"]
    assert manifest["collection_only"]
    assert "--collect-only" in manifest["command"]
    assert "--max-new-anchors 2" in manifest["command"]


def test_preflight_allows_explicit_diagnostic_stocks_without_universe(monkeypatch) -> None:
    """显式诊断股票可运行，但缺少历史股票池时仍不可作为正式全量评估。"""
    import importlib

    run_module = importlib.import_module("evaluation.run")

    class _NoHistoricalUniverse:
        def members_at(self, as_of: str) -> list[str]:
            return []

    monkeypatch.setattr(run_module, "HistoricalUniverse", _NoHistoricalUniverse)
    without_stocks = run_module._preflight("full", "2020-09-25")
    with_stocks = run_module._preflight("full", "2020-09-25", stocks_arg="600519,000001")

    assert "POINT_IN_TIME_CSI300_MISSING" in without_stocks
    assert "POINT_IN_TIME_CSI300_MISSING" not in with_stocks


def test_formal_protocol_rejects_partial_universe_history_and_price_benchmark() -> None:
    """抽样、缺失历史、收集模式或价格指数均不得伪装成正式协议。"""
    from evaluation.run import _is_formal_protocol

    common = ("2020-09-25", "2017-01-01", "weekly", 100, False, None, False, pd.Timestamp("2020-09-25"), True)
    assert _is_formal_protocol(*common)
    assert not _is_formal_protocol(*common[:5], "600519", *common[6:])
    assert not _is_formal_protocol("2020-06-19", *common[1:])
    assert not _is_formal_protocol(*common[:6], True, *common[7:])
    assert not _is_formal_protocol(*common[:-1], False)


def test_manifest_json_replaces_nonfinite_numbers_with_null(tmp_path) -> None:
    """基线指标中的 NaN 不得写成非标准 JSON 常量。"""
    from evaluation.artifacts import EvaluationArtifacts

    artifacts = EvaluationArtifacts.open("strict-json", tmp_path)
    manifest = {"status": "completed", "as_of": "2024-02-01", "seed": 1, "horizons": [20], "model": "Kronos", "config_hash": "cfg", "data_hash": "data", "metrics": {"missing": float("nan"), "overflow": float("inf")}}
    artifacts.write_manifest(manifest)

    text = (artifacts.root / "manifest.json").read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert '"missing": null' in text and '"overflow": null' in text


class _FakePosition:
    def __init__(self, amounts: dict[str, float]) -> None:
        self._amounts = amounts

    def get_stock_amount_dict(self) -> dict[str, float]:
        return dict(self._amounts)


def test_positions_and_trades_derived_from_snapshots() -> None:
    """持仓快照落盘 positions.csv，相邻快照差派生模拟成交 trades.csv。"""
    from evaluation.run import _positions_and_trades

    snapshots = {
        pd.Timestamp("2024-02-06"): _FakePosition({"SH600519": 95_000_000.0}),
        pd.Timestamp("2024-02-07"): _FakePosition({"SH600519": 50_000_000.0, "SZ000001": 45_000_000.0}),
    }
    point = _FakePoint(["600519", "000001"])
    positions, trades = _positions_and_trades(snapshots, point)
    assert len(positions) == 3
    assert positions.iloc[0]["stock_code"] == "600519"
    assert positions.iloc[0]["amount"] == 95_000_000.0
    assert len(trades) == 3
    first = trades[trades["date"] == "2024-02-06"].iloc[0]
    assert first["direction"] == "BUY"
    assert first["notional"] == 95_000_000.0
    assert first["shares"] == 9_500_000.0
    second = trades[trades["date"] == "2024-02-07"].set_index("stock_code")
    assert second.loc["600519", "direction"] == "SELL"
    assert second.loc["000001", "direction"] == "BUY"
