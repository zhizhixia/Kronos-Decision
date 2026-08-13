"""v2 五态动作和安全 API 契约。"""
from __future__ import annotations

from decision.v2 import RecommendationAction, decide_action, legacy_signal
from decision.engine import DecisionEngine, DecisionReport

import pandas as pd
import numpy as np


def test_evidence_failure_never_maps_to_sell() -> None:
    action, reasons = decide_action(False, {5: 0.0, 20: 0.0, 60: 0.0}, {5: 0.0, 20: 0.0, 60: 0.0}, True, True)
    assert action is RecommendationAction.INSUFFICIENT_EVIDENCE
    assert legacy_signal(action) == "HOLD"
    assert reasons == ("EVIDENCE_GATE_FAILED",)


def test_stale_cache_or_path_flags_force_degraded_hold() -> None:
    """过期缓存或路径质量标记必须把旧三态动作强制为 HOLD 并标记 degraded。"""
    from decision.analyzer import RiskResult, SignalResult, TrendResult
    from decision.engine import DecisionEngine

    signal = SignalResult("BUY", "测试信号", TrendResult("↑", 0.9, "高", 0.8), RiskResult(-0.02, "低", "低"))
    assert DecisionEngine._apply_legacy_evidence_floor(signal, is_stale=True, path_flags=()) == "degraded"
    assert signal.signal == "HOLD"
    assert "证据不足" in signal.signal_reason
    fresh = SignalResult("BUY", "测试信号", TrendResult("↑", 0.9, "高", 0.8), RiskResult(-0.02, "低", "低"))
    assert DecisionEngine._apply_legacy_evidence_floor(fresh, is_stale=False, path_flags=("PATH_REPAIR_EXCESSIVE",)) == "degraded"
    assert fresh.signal == "HOLD"
    normal = SignalResult("BUY", "测试信号", TrendResult("↑", 0.9, "高", 0.8), RiskResult(-0.02, "低", "低"))
    assert DecisionEngine._apply_legacy_evidence_floor(normal, is_stale=False, path_flags=()) == "ok"
    assert normal.signal == "BUY"


def test_add_needs_all_three_horizons_to_avoid_veto() -> None:
    action, _ = decide_action(True, {5: 0.8, 20: 0.8, 60: 0.8}, {5: 0.7, 20: 0.7, 60: 0.7}, False, False)
    assert action is RecommendationAction.ADD
    action, _ = decide_action(True, {5: 0.1, 20: 0.8, 60: 0.8}, {5: 0.7, 20: 0.7, 60: 0.7}, False, False)
    assert action is RecommendationAction.HOLD


def test_horizon_statistics_come_from_real_path_distribution() -> None:
    paths = np.zeros((100, 60, 6), dtype=float)
    paths[:, :, 3] = np.linspace(100, 120, 60)
    paths[:50, 19, 3] = 90
    paths[50:, 19, 3] = 110
    horizons = DecisionEngine._path_horizons(paths, 100)
    assert np.isclose(horizons["20"]["q05"], -0.1)
    assert np.isclose(horizons["20"]["q95"], 0.1)
    assert horizons["20"]["up_probability"] == 0.5
    assert np.isclose(horizons["60"]["q50"], 0.2)


def test_cpu_sampling_reduction_is_marked(monkeypatch) -> None:
    """CPU 上超过 30 条的采样请求必须降级并明确标记。"""
    engine = DecisionEngine()
    calls = {}

    class _FakePredictor:
        device = "cpu"

        def predict_paths(self, *args, **kwargs):
            calls["sample_count"] = kwargs.get("sample_count") if "sample_count" in kwargs else args[5]
            calls["sample_batch_size"] = kwargs.get("sample_batch_size") if "sample_batch_size" in kwargs else args[6]
            return _FakePaths()

    class _FakePaths:
        sample_count = 30
        seed = 1
        model_id = "m"
        tokenizer_id = "t"
        sampling_params_hash = "h"
        repair_ratio = 0.0
        quality_flags = ()
        paths = np.zeros((30, 60, 6))
        timestamps = pd.date_range("2024-01-02", periods=60, freq="B")
        mean_df = pd.DataFrame(np.zeros((60, 6)), columns=["open", "high", "low", "close", "volume", "amount"])

    monkeypatch.setattr(engine._model_manager, "get_predictor", lambda: (None, _FakePredictor()))
    bars = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=100), "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1000.0, "amount": 10000.0})
    bundle = type("Bundle", (), {"source": "fixture", "as_of": pd.Timestamp("2024-05-01"), "is_stale": False, "quality_flags": (), "content_hash": "h"})()
    payload = engine._prediction_payload(_FakePaths(), bundle, bars, False, {"device": "cpu", "reduced_for_cpu": True, "requested_sample_count": 100})
    assert payload["sampling"]["reduced_for_cpu"] is True
    assert payload["sampling"]["requested_sample_count"] == 100


def test_v2_report_propagates_actual_historical_cutoff(monkeypatch) -> None:
    engine = DecisionEngine()
    observed = {}
    chart_data = {"history": [], "forecast": {"timestamps": [], "mean_close": []}}
    report = DecisionReport(
        "ok",
        prediction={
            "data_provenance": {"as_of": "2024-01-05", "quality_flags": []},
            "sampling": {},
            "horizons": {},
            "chart_data": chart_data,
        },
        stock_code="600519",
    )

    def fake_predict(code, params):
        observed["params"] = params
        return report

    def fake_eligibility(code, as_of):
        observed["eligibility_as_of"] = as_of
        return {"eligible": False}

    def fake_market(as_of=None):
        observed["market_as_of"] = as_of
        return {"available": False}

    monkeypatch.setattr(engine, "predict_and_analyze", fake_predict)
    monkeypatch.setattr("decision.engine.EvaluationBinding.load_latest", lambda: None)
    monkeypatch.setattr(engine, "_eligibility_for", fake_eligibility)
    monkeypatch.setattr(engine, "_market_context_payload", fake_market)

    payload = engine.decision_report_v2("600519", as_of="2024-01-06")

    assert observed["params"]["as_of"] == pd.Timestamp("2024-01-06")
    assert observed["eligibility_as_of"] == pd.Timestamp("2024-01-05")
    assert observed["market_as_of"] == pd.Timestamp("2024-01-05")
    assert payload["data_provenance"]["as_of"] == "2024-01-05"
    assert payload["chart_data"] is chart_data


def test_held_position_uses_reduce_instead_of_avoid() -> None:
    from evaluation.gates import EvidenceGateResult

    evidence = type(
        "Evidence",
        (),
        {
            "short_probability": 0.7,
            "calibrated_up_probability": 0.3,
            "long_probability": 0.7,
            "short_rank": 0.7,
            "cross_section_rank": 0.1,
            "long_rank": 0.7,
        },
    )()
    gate = EvidenceGateResult(True, (), {})

    held, _ = DecisionEngine._decide_with_evidence(gate, evidence, False, True)
    unheld, _ = DecisionEngine._decide_with_evidence(gate, evidence, False, False)

    assert held is RecommendationAction.REDUCE
    assert unheld is RecommendationAction.AVOID


def test_quality_warnings_are_not_hard_errors_but_error_flags_are() -> None:
    assert not DecisionEngine._has_hard_quality_error(
        {"quality_flags": ["POSSIBLE_SUSPENSION_ZERO_VOLUME"], "is_stale": False},
        {"quality_flags": []},
    )
    assert DecisionEngine._has_hard_quality_error(
        {"quality_flags": ["ERROR_UNRESOLVED_SESSION_GAP"], "is_stale": False},
        {"quality_flags": []},
    )


def test_prediction_payload_chart_data_caps_history_and_orders_quantiles() -> None:
    """chart_data 历史最多 120 条且截止于数据日；预测数组等长；分位数逐日单调；display_paths 仍兼容。"""
    sample_count, pred_len = 40, 18
    rng = np.random.RandomState(0)
    close_paths = np.linspace(100, 120, pred_len)[None, :] + rng.randn(sample_count, pred_len)
    paths_arr = np.zeros((sample_count, pred_len, 6), dtype=float)
    paths_arr[:, :, 3] = close_paths
    timestamps = pd.bdate_range("2024-06-03", periods=pred_len)
    mean_df = pd.DataFrame(
        {
            "open": np.linspace(100, 120, pred_len),
            "high": np.linspace(101, 121, pred_len),
            "low": np.linspace(99, 119, pred_len),
            "close": close_paths.mean(axis=0),
            "volume": np.zeros(pred_len),
            "amount": np.zeros(pred_len),
        },
        index=timestamps,
    )
    fake_paths = type("FakePaths", (), {
        "sample_count": sample_count, "seed": 7, "model_id": "m", "tokenizer_id": "t",
        "sampling_params_hash": "h", "repair_ratio": 0.0, "quality_flags": (),
        "paths": paths_arr, "mean_df": mean_df, "timestamps": timestamps,
    })()

    bar_count = 150
    bars = pd.DataFrame({
        "date": pd.bdate_range("2024-01-02", periods=bar_count),
        "open": np.arange(bar_count, dtype=float),
        "high": np.arange(1, bar_count + 1, dtype=float),
        "low": np.arange(bar_count, dtype=float) - 1.0,
        "close": np.arange(bar_count, dtype=float),
        "volume": np.full(bar_count, 1000.0),
        "amount": np.full(bar_count, 10000.0),
    })
    last_date = pd.Timestamp(bars["date"].iloc[-1])
    bundle = type("Bundle", (), {
        "source": "fixture", "as_of": last_date, "is_stale": False,
        "quality_flags": (), "content_hash": "h",
    })()

    payload = DecisionEngine()._prediction_payload(fake_paths, bundle, bars, True, {"device": "cpu"})

    # 旧契约保持
    assert "pred_df" in payload and "horizons" in payload and "sampling" in payload
    assert payload["display_paths"] == paths_arr[:20].tolist()

    chart = payload["chart_data"]
    history = chart["history"]
    assert len(history) == 120  # 超过 120 条截断
    assert history[-1]["timestamp"] == last_date.isoformat()  # 截止日为最后一根 K 线
    assert all(set(b.keys()) == {"timestamp", "open", "high", "low", "close", "volume"} for b in history)
    assert all(pd.Timestamp(b["timestamp"]) <= last_date for b in history)

    forecast = chart["forecast"]
    assert len(forecast["timestamps"]) == pred_len
    for key in ("mean_close", "q05_close", "q50_close", "q95_close"):
        assert len(forecast[key]) == pred_len, f"{key} 长度应等于预测期"
    for i in range(pred_len):
        assert forecast["q05_close"][i] <= forecast["q50_close"][i] <= forecast["q95_close"][i]
        assert forecast["timestamps"][i] == pd.Timestamp(timestamps[i]).isoformat()


def test_v2_report_payload_always_includes_chart_data(monkeypatch) -> None:
    """最终成功 v2 payload 必须始终包含 report.prediction.chart_data，不受 include_display_paths 影响。"""
    engine = DecisionEngine()
    chart_data = {"history": [], "forecast": {"timestamps": [], "mean_close": []}}

    def fake_predict(code, params):
        prediction = {
            "data_provenance": {"as_of": "2024-01-05", "quality_flags": []},
            "sampling": {"model_id": "m", "seed": 1, "sample_count": 10},
            "chart_data": chart_data,
            "horizons": {"5": {}, "20": {}, "60": {}},
        }
        if params.get("include_display_paths"):
            prediction["display_paths"] = []
        return DecisionReport(
            "ok",
            prediction=prediction,
            stock_code=code,
            stock_name="测试",
            elapsed_seconds=0.1,
        )

    monkeypatch.setattr(engine, "predict_and_analyze", fake_predict)
    monkeypatch.setattr("decision.engine.EvaluationBinding.load_latest", lambda: None)
    monkeypatch.setattr(engine, "_market_context_payload", lambda as_of=None: {"available": False})
    monkeypatch.setattr(engine, "_eligibility_for", lambda code, as_of=None: {"eligible": False})

    payload_with_paths = engine.decision_report_v2("600519", include_display_paths=True)
    payload_without_paths = engine.decision_report_v2("600519", include_display_paths=False)

    for payload in (payload_with_paths, payload_without_paths):
        assert payload["schema_version"] == "2.0"
        assert "chart_data" in payload
        assert payload["chart_data"] is chart_data

    assert "display_paths" not in payload_without_paths
    assert "display_paths" in payload_with_paths
