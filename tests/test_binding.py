"""评估工件与在线报告绑定测试。"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from decision.engine import DecisionEngine
from decision.v2 import RecommendationAction
from evaluation.artifacts import EvaluationArtifacts
from evaluation.binding import EvaluationBinding
from evaluation.gates import evaluate_evidence_gate


def _gate_metrics(passed: bool = True) -> dict:
    if not passed:
        return {"data_complete": False}
    return {"data_complete": True, "quality_error": False, "coverage_80": 0.8, "up_probability_ece": 0.08, "rank_ic_mean": 0.01, "rank_ic_positive_probability": 0.8, "annualized_excess_return": 0.01, "information_ratio": 0.5, "positive_12m_window_ratio": 0.6, "drawdown_worsening": 0.05, "versions_match": True, "formal_protocol": True, "evaluated_at": datetime.now().isoformat()}


def _write_artifacts(root: Path, gate_metrics: dict, calibration: dict | None = None, scores: dict[str, float] | None = None, with_short_long: bool = False, prediction_data_hash: str = "data") -> None:
    artifacts = EvaluationArtifacts.open("binding-test", root)
    artifacts.write_manifest({"status": "completed", "as_of": "2026-08-01", "seed": 1, "horizons": [20], "model": "Kronos", "model_hash": "m", "rules_hash": "rules", "config_hash": "cfg", "data_hash": "data", "sampling_params_hash": "params"})
    artifacts.write_gate_result(evaluate_evidence_gate(gate_metrics).as_dict())
    if calibration:
        rows = []
        scores = scores or {}
        for code, score in (("600519", scores.get("600519", 0.9)), ("000001", scores.get("000001", 0.3))):
            rows.append({"prediction_key": f"k-{code}-20", "anchor_date": "2026-08-01", "execution_date": "2026-08-04", "stock_code": code, "horizon": 20, "score": score, "predicted_return": 0.01, "actual_return": 0.01, "data_as_of": "2026-08-01", "model_hash": "m", "config_hash": "cfg", "data_hash": prediction_data_hash, "raw_up_probability": 0.7 if code == "600519" else 0.3, "q05": -0.05, "q50": 0.05, "q95": 0.15, "sampling_seed": 101, "sampling_params_hash": "params"})
        if with_short_long:
            for code, score in (("600519", scores.get("600519", 0.9)), ("000001", scores.get("000001", 0.3))):
                rows.append({"prediction_key": f"k-{code}-5", "anchor_date": "2026-08-01", "execution_date": "2026-08-04", "stock_code": code, "horizon": 5, "score": score, "predicted_return": 0.0, "actual_return": 0.0, "data_as_of": "2026-08-01", "model_hash": "m", "config_hash": "cfg", "data_hash": prediction_data_hash, "raw_up_probability": 0.7 if code == "600519" else 0.3, "q05": -0.01, "q50": 0.01, "q95": 0.03, "sampling_seed": 101, "sampling_params_hash": "params"})
                rows.append({"prediction_key": f"k-{code}-60", "anchor_date": "2026-08-01", "execution_date": "2026-08-04", "stock_code": code, "horizon": 60, "score": score, "predicted_return": 0.02, "actual_return": 0.02, "data_as_of": "2026-08-01", "model_hash": "m", "config_hash": "cfg", "data_hash": prediction_data_hash, "raw_up_probability": 0.7 if code == "600519" else 0.3, "q05": -0.1, "q50": 0.1, "q95": 0.3, "sampling_seed": 101, "sampling_params_hash": "params"})
        artifacts.write_frame("predictions", pd.DataFrame(rows))
        profile_path = root / "binding-test" / "calibration.json"
        profile_path.write_text(json.dumps(calibration, ensure_ascii=False), encoding="utf-8")


def _calibration_profile(horizon: int, now: datetime) -> dict:
    return {"horizon": horizon, "fitted_at": (now - timedelta(days=1)).isoformat(), "expires_at": (now + timedelta(days=29)).isoformat(), "anchor_count": 25, "observation_count": 2000, "conformal_adjustment": 0.02, "isotonic_x": [0.0, 0.5, 1.0], "isotonic_y": [0.0, 0.5, 1.0], "model_hash": "m", "data_hash": "data", "config_hash": "cfg"}


def _calibration_payload(now: datetime | None = None, horizons: tuple[int, ...] = (5, 20, 60)) -> dict:
    now = now or datetime.now()
    return {"schema_version": "2.0", "profiles": {str(horizon): _calibration_profile(horizon, now) for horizon in horizons}}


def test_gate_and_evidence_allow_add_with_calibrated_probability(tmp_path) -> None:
    _write_artifacts(tmp_path, _gate_metrics(True), _calibration_payload(), {"600519": 0.9, "000001": 0.3}, with_short_long=True)
    binding = EvaluationBinding.load_latest(tmp_path)
    evidence = binding.evidence_for("600519", 20)
    assert evidence is not None and evidence.calibrated_up_probability is not None
    gate = evaluate_evidence_gate(_gate_metrics(True))
    action, _ = DecisionEngine._decide_with_evidence(gate, evidence)
    assert action is RecommendationAction.ADD


def test_missing_five_and_sixty_day_evidence_vetoes_add(tmp_path) -> None:
    _write_artifacts(tmp_path, _gate_metrics(True), _calibration_payload(), {"600519": 0.9, "000001": 0.3})
    binding = EvaluationBinding.load_latest(tmp_path)
    evidence = binding.evidence_for("600519", 20)
    gate = evaluate_evidence_gate(_gate_metrics(True))
    action, reasons = DecisionEngine._decide_with_evidence(gate, evidence)
    assert action is RecommendationAction.INSUFFICIENT_EVIDENCE
    assert reasons == ("MULTI_HORIZON_CALIBRATION_UNAVAILABLE",)


def test_failed_gate_returns_insufficient_evidence(tmp_path) -> None:
    _write_artifacts(tmp_path, _gate_metrics(False), _calibration_payload())
    binding = EvaluationBinding.load_latest(tmp_path)
    gate = evaluate_evidence_gate(_gate_metrics(False))
    action, reasons = DecisionEngine._decide_with_evidence(gate, binding.evidence_for("600519", 20))
    assert action is RecommendationAction.INSUFFICIENT_EVIDENCE
    assert reasons == ("EVIDENCE_GATE_FAILED",)


def test_nonformal_protocol_cannot_pass_evidence_gate() -> None:
    """月度或低采样 canary 不得成为正式交易动作的证据。"""
    metrics = _gate_metrics(True)
    metrics["formal_protocol"] = False

    gate = evaluate_evidence_gate(metrics)

    assert not gate.passed
    assert "FORMAL_PROTOCOL" in gate.failed_codes


def test_expired_calibration_blocks_action(tmp_path) -> None:
    payload = _calibration_payload()
    for profile in payload["profiles"].values():
        profile["expires_at"] = (datetime.now() - timedelta(days=1)).isoformat()
    _write_artifacts(tmp_path, _gate_metrics(True), payload, {"600519": 0.9, "000001": 0.3})
    binding = EvaluationBinding.load_latest(tmp_path)
    evidence = binding.evidence_for("600519", 20)
    assert evidence is not None and evidence.calibrated_up_probability is None
    gate = evaluate_evidence_gate(_gate_metrics(True))
    action, _ = DecisionEngine._decide_with_evidence(gate, evidence)
    assert action is RecommendationAction.INSUFFICIENT_EVIDENCE


def test_merge_horizons_marks_uncalibrated_without_evidence() -> None:
    path_horizons = {"20": {"q05": -0.1, "q50": 0.0, "q95": 0.1, "up_probability": 0.5, "predicted_drawdown": -0.2, "calibration_status": "unavailable"}}
    merged = DecisionEngine._merge_horizons(path_horizons, None)
    assert merged["20"]["calibration_status"] == "unavailable"
    assert merged["20"]["up_probability"] == 0.5


def test_merge_horizons_uses_calibrated_evidence(tmp_path) -> None:
    _write_artifacts(tmp_path, _gate_metrics(True), _calibration_payload(), {"600519": 0.9, "000001": 0.3})
    binding = EvaluationBinding.load_latest(tmp_path)
    evidence = binding.evidence_for("600519", 20)
    path_horizons = {"20": {"q05": -0.1, "q50": 0.0, "q95": 0.1, "up_probability": 0.5, "predicted_drawdown": -0.2, "calibration_status": "unavailable"}}
    merged = DecisionEngine._merge_horizons(path_horizons, evidence)
    assert merged["20"]["calibration_status"] == "calibrated"
    assert merged["20"]["up_probability"] == evidence.calibrated_up_probability
    assert merged["20"]["q50"] == evidence.q50


def test_short_and_long_profiles_must_be_calibrated(tmp_path) -> None:
    """即使存在5/60日预测，缺失对应档案也必须阻止动作。"""
    _write_artifacts(tmp_path, _gate_metrics(True), _calibration_payload(horizons=(20,)), {"600519": 0.9, "000001": 0.3}, with_short_long=True)
    binding = EvaluationBinding.load_latest(tmp_path)
    evidence = binding.evidence_for("600519", 20)
    gate = evaluate_evidence_gate(_gate_metrics(True))
    action, reasons = DecisionEngine._decide_with_evidence(gate, evidence)

    assert action is RecommendationAction.INSUFFICIENT_EVIDENCE
    assert reasons == ("MULTI_HORIZON_CALIBRATION_UNAVAILABLE",)


def test_merge_horizons_calibrates_all_three_horizons(tmp_path) -> None:
    _write_artifacts(tmp_path, _gate_metrics(True), _calibration_payload(), {"600519": 0.9, "000001": 0.3}, with_short_long=True)
    evidence = EvaluationBinding.load_latest(tmp_path).evidence_for("600519", 20)
    path_horizons = {str(horizon): {"q05": -0.1, "q50": 0.0, "q95": 0.1, "up_probability": 0.5, "predicted_drawdown": -0.2, "calibration_status": "unavailable"} for horizon in (5, 20, 60)}
    merged = DecisionEngine._merge_horizons(path_horizons, evidence)

    assert all(merged[str(horizon)]["calibration_status"] == "calibrated" for horizon in (5, 20, 60))


def test_calibration_hash_mismatch_blocks_action(tmp_path) -> None:
    """预测的数据哈希与校准档案不一致时，校准概率不可用，动作必须证据不足。"""
    _write_artifacts(tmp_path, _gate_metrics(True), _calibration_payload(), {"600519": 0.9, "000001": 0.3}, with_short_long=True, prediction_data_hash="other-data")
    binding = EvaluationBinding.load_latest(tmp_path)
    evidence = binding.evidence_for("600519", 20)
    assert evidence is not None and evidence.calibrated_up_probability is None
    gate = evaluate_evidence_gate(_gate_metrics(True))
    action, reasons = DecisionEngine._decide_with_evidence(gate, evidence)
    assert action is RecommendationAction.INSUFFICIENT_EVIDENCE
    assert reasons == ("MULTI_HORIZON_CALIBRATION_UNAVAILABLE",)


def test_gate_fails_on_version_mismatch() -> None:
    """评估/模型/配置/数据版本任一不匹配时 VERSION_MATCH 门禁失败。"""
    metrics = _gate_metrics(True)
    metrics["versions_match"] = False
    gate = evaluate_evidence_gate(metrics)
    assert not gate.passed
    assert "VERSION_MATCH" in gate.failed_codes


def test_risk_blocked_returns_avoid_when_gate_passes(tmp_path) -> None:
    """资格阻断（ST/停牌/流动性不足等）在门禁通过时输出 AVOID，而不是增持。"""
    _write_artifacts(tmp_path, _gate_metrics(True), _calibration_payload(), {"600519": 0.9, "000001": 0.3}, with_short_long=True)
    binding = EvaluationBinding.load_latest(tmp_path)
    evidence = binding.evidence_for("600519", 20)
    gate = evaluate_evidence_gate(_gate_metrics(True))
    action, reasons = DecisionEngine._decide_with_evidence(gate, evidence, risk_blocked=True)
    assert action is RecommendationAction.AVOID
    assert "TRADE_ELIGIBILITY_OR_RISK_BLOCK" in reasons


def test_online_report_must_match_evaluation_versions(monkeypatch, tmp_path) -> None:
    """在线模型、规则、配置、采样参数和数据截止日任一变化都阻断动作。"""
    import importlib

    _write_artifacts(tmp_path, _gate_metrics(True), _calibration_payload(), {"600519": 0.9, "000001": 0.3}, with_short_long=True)
    binding = EvaluationBinding.load_latest(tmp_path)
    report = SimpleNamespace(
        stock_code="600519",
        prediction={"data_provenance": {"as_of": "2026-08-01", "quality_flags": []}, "sampling": {"model_hash": "m", "config_hash": "cfg", "rules_hash": "rules", "sampling_seed": 101, "sampling_params_hash": "params"}},
    )
    engine_module = importlib.import_module("decision.engine")
    monkeypatch.setattr(engine_module, "config_hash", lambda: "cfg")
    monkeypatch.setattr(engine_module, "rules_hash", lambda: "rules")

    gate, _ = DecisionEngine._bind_evidence(binding, report)
    assert gate.passed

    report.prediction["sampling"]["sampling_params_hash"] = "other"
    mismatched, _ = DecisionEngine._bind_evidence(binding, report)
    assert not mismatched.passed
    assert not mismatched.checks["VERSION_MATCH"]


def test_online_hard_quality_error_overrides_passing_research_gate(
    monkeypatch, tmp_path
) -> None:
    """在线数据硬错误不能被历史正式门禁的通过状态覆盖。"""
    import importlib

    _write_artifacts(
        tmp_path, _gate_metrics(True), _calibration_payload(),
        {"600519": 0.9, "000001": 0.3}, with_short_long=True,
    )
    binding = EvaluationBinding.load_latest(tmp_path)
    report = SimpleNamespace(
        stock_code="600519",
        prediction={
            "data_provenance": {
                "as_of": "2026-08-01",
                "quality_flags": [],
                "has_hard_quality_error": True,
            },
            "sampling": {
                "model_hash": "m", "config_hash": "cfg",
                "rules_hash": "rules", "sampling_seed": 101,
                "sampling_params_hash": "params",
            },
        },
    )
    engine_module = importlib.import_module("decision.engine")
    monkeypatch.setattr(engine_module, "config_hash", lambda: "cfg")
    monkeypatch.setattr(engine_module, "rules_hash", lambda: "rules")

    gate, evidence = DecisionEngine._bind_evidence(binding, report)
    action, _ = DecisionEngine._decide_with_evidence(gate, evidence)

    assert not gate.passed
    assert not gate.checks["NO_QUALITY_ERROR"]
    assert action is RecommendationAction.INSUFFICIENT_EVIDENCE


def test_latest_binding_prefers_formal_run_over_newer_diagnostic(tmp_path) -> None:
    """新的 smoke/canary 工件不能遮蔽仍可追溯的正式运行。"""
    formal = EvaluationArtifacts.open("formal", tmp_path)
    formal.write_manifest({"status": "completed", "as_of": "2026-08-01", "seed": 1, "horizons": [5, 20, 60], "model": "Kronos", "config_hash": "cfg", "data_hash": "data", "formal_protocol": True})
    formal.write_gate_result(evaluate_evidence_gate(_gate_metrics(True)).as_dict())
    diagnostic = EvaluationArtifacts.open("diagnostic", tmp_path)
    diagnostic.write_manifest({"status": "completed-smoke", "as_of": "2026-08-02", "seed": 2, "horizons": [5, 20, 60], "model": "Kronos", "config_hash": "cfg", "data_hash": "data", "formal_protocol": False})
    diagnostic.write_gate_result(evaluate_evidence_gate({"formal_protocol": False}).as_dict())
    os.utime(formal.root / "manifest.json", (1, 1))
    os.utime(diagnostic.root / "manifest.json", (2, 2))

    binding = EvaluationBinding.load_latest(tmp_path)

    assert binding is not None and binding.run_id == "formal"
