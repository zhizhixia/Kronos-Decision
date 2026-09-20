"""报告与预测 Web API 的统一安全契约测试。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import importlib
from types import SimpleNamespace

app_module = importlib.import_module("webui.app")
app = app_module.app
LOCAL_ORIGIN_HEADERS = {"Origin": "http://localhost:7070"}


class _FakeStore:
    """隔离建议历史写入，测试只验证 API 契约。"""

    def save_recommendation(self, *args, **kwargs):
        return 1


def _chart_data(with_path: bool = True) -> dict:
    """构造只用于契约测试的预测图表数据。"""
    timestamps = ["2024-01-02"] if with_path else []
    return {
        "history": [],
        "forecast": {
            "timestamps": timestamps,
            "mean_close": [100.0] if with_path else [],
            "q05_close": [99.0] if with_path else [],
            "q50_close": [100.0] if with_path else [],
            "q95_close": [101.0] if with_path else [],
        },
    }


def _v2_report(**overrides) -> dict:
    """返回带完整字段的可行动 v2 夹具，调用方可显式删除字段。"""
    report = {
        "schema_version": "2.0",
        "status": "ok",
        "run_status": "SUCCEEDED",
        "evidence_status": "QUALIFIED",
        "action_permission": "CONDITIONAL_REFERENCE",
        "stock": {"code": "600519", "name": "测试"},
        "data_provenance": {"as_of": "2024-01-02", "is_stale": False, "quality_flags": []},
        "model_provenance": {"quality_flags": []},
        "chart_data": _chart_data(),
        "horizons": {"5": {}, "20": {}, "60": {}},
        "recommendation": {"action": "ADD", "legacy_signal": "BUY", "reason_codes": []},
        "evidence_gate": {"passed": True, "checks": {"DATA_COMPLETE": True}},
        "generated_at": "2024-01-02T16:00:00",
        "elapsed_seconds": 0.1,
    }
    report.update(overrides)
    return report


def _patch_v2(monkeypatch, report: dict) -> None:
    """将 v2 引擎替换为确定性夹具，并阻断测试数据库写入。"""
    class FakePipeline:
        """返回确定性报告夹具，避免契约测试访问真实行情源。"""

        def run(self, *args, **kwargs):
            return SimpleNamespace(
                report=report,
                report_id="decision:test:2024-01-02",
                version=1,
                snapshot_id="snapshot-test",
            )

    monkeypatch.setattr(app_module, "_get_report_pipeline_class", lambda: FakePipeline)
    monkeypatch.setattr("portfolio.store.PortfolioStore", _FakeStore)


def _patch_legacy(monkeypatch, report: dict) -> None:
    """将旧版分析引擎替换为确定性夹具。"""
    class FakeEngine:
        def predict_and_analyze(self, *args, **kwargs):
            return report

    monkeypatch.setattr(app_module, "_get_decision_engine_class", lambda: FakeEngine)


def test_v2_missing_contract_fields_default_to_no_action(monkeypatch) -> None:
    """缺少证据状态或动作权限时不能凭 gate 和路径推断出 ADD/BUY。"""
    for missing_field in ("evidence_status", "action_permission"):
        report = _v2_report()
        report.pop(missing_field)
        _patch_v2(monkeypatch, report)

        response = app.test_client().post("/api/v2/decision-report", json={"stock_code": "600519"}, headers=LOCAL_ORIGIN_HEADERS)

        assert response.status_code == 200
        body = response.get_json()
        assert body["run_status"] == "SUCCEEDED"
        assert body["action_permission"] == "NONE"
        assert body["recommendation"]["action"] == "INSUFFICIENT_EVIDENCE"
        assert body["recommendation"]["legacy_signal"] is None
        assert body["evidence_status"] == "INSUFFICIENT"


def test_v2_missing_gate_defaults_to_no_action(monkeypatch) -> None:
    """缺少证据门禁时，即使三类状态字段声称合格也必须拒绝动作。"""
    report = _v2_report()
    report.pop("evidence_gate")
    _patch_v2(monkeypatch, report)

    body = app.test_client().post("/api/v2/decision-report", json={"stock_code": "600519"}, headers=LOCAL_ORIGIN_HEADERS).get_json()

    assert body["action_permission"] == "NONE"
    assert body["recommendation"]["action"] == "INSUFFICIENT_EVIDENCE"


def test_v2_empty_path_stale_data_and_error_flags_never_restore_legacy_signal(monkeypatch) -> None:
    """空路径、过期数据和 ERROR 质量标记都不能恢复 BUY/HOLD/SELL。"""
    cases = [
        {"chart_data": _chart_data(with_path=False)},
        {"data_provenance": {"as_of": "2024-01-02", "is_stale": True, "quality_flags": []}},
        {"data_provenance": {"as_of": "2024-01-02", "is_stale": False, "quality_flags": ["ERROR_BAD_DATA"]}},
    ]
    for change in cases:
        report = _v2_report(**change)
        _patch_v2(monkeypatch, report)

        response = app.test_client().post("/api/v2/decision-report", json={"stock_code": "600519"}, headers=LOCAL_ORIGIN_HEADERS)
        body = response.get_json()

        assert response.status_code == 200
        assert body["action_permission"] == "NONE"
        assert body["recommendation"]["action"] == "INSUFFICIENT_EVIDENCE"
        assert body["recommendation"]["legacy_signal"] is None
        assert body["evidence_status"] in {"INVALID", "STALE"}


def test_v2_unknown_status_is_non_2xx_and_has_safe_chinese_error(monkeypatch) -> None:
    """未知运行状态必须拒绝请求，错误正文不能回显绝对路径或凭据。"""
    report = _v2_report(run_status="MYSTERY", detail=r"C:\private\token=secret")
    _patch_v2(monkeypatch, report)

    response = app.test_client().post("/api/v2/decision-report", json={"stock_code": "600519"}, headers=LOCAL_ORIGIN_HEADERS)
    body = response.get_json()

    assert response.status_code >= 400
    assert body["error"]["message"]
    assert any("\u4e00" <= char <= "\u9fff" for char in body["error"]["message"])
    assert "C:\\private" not in response.get_data(as_text=True)
    assert "secret" not in response.get_data(as_text=True)
    assert body["action_permission"] == "NONE"


def test_v2_model_failure_does_not_leak_exception_text(monkeypatch) -> None:
    """模型异常只返回稳定中文错误，不暴露本地路径或凭据文本。"""
    def fail(*args, **kwargs):
        raise RuntimeError(r"C:\models\private\credential=secret")

    class FakePipeline:
        def run(self, *args, **kwargs):
            return fail(*args, **kwargs)

    monkeypatch.setattr(app_module, "_get_report_pipeline_class", lambda: FakePipeline)
    response = app.test_client().post("/api/v2/decision-report", json={"stock_code": "600519"}, headers=LOCAL_ORIGIN_HEADERS)
    body = response.get_json()

    assert response.status_code >= 500
    assert body["error"]["message"] == "无法取得可验证的完整日线数据。"
    assert "C:\\models" not in response.get_data(as_text=True)
    assert "credential" not in response.get_data(as_text=True)
    assert body["action_permission"] == "NONE"


def test_legacy_error_is_non_2xx_and_cannot_restore_signal(monkeypatch) -> None:
    """旧报告入口的 error 状态必须是非 2xx，且不能由页面回退为 HOLD。"""
    _patch_legacy(
        monkeypatch,
        {
            "status": "error",
            "error_message": r"C:\private\credential=secret",
            "stock_code": "600519",
        },
    )

    response = app.test_client().post("/api/decision-report", json={"stock_code": "600519"}, headers=LOCAL_ORIGIN_HEADERS)
    body = response.get_json()

    assert response.status_code >= 400
    assert body["action_permission"] == "NONE"
    assert body["recommendation"]["legacy_signal"] is None
    assert "C:\\private" not in response.get_data(as_text=True)
    assert "secret" not in response.get_data(as_text=True)


def test_legacy_stale_signal_is_removed_from_response(monkeypatch) -> None:
    """旧接口即使引擎返回 BUY，stale 证据也只能展示无动作状态。"""
    _patch_legacy(
        monkeypatch,
        {
            "status": "ok",
            "stock_code": "600519",
            "stock_name": "测试",
            "signal": {"signal": "BUY", "signal_reason": "不可信"},
            "prediction": {
                "pred_df": [{"close": 100.0}],
                "data_provenance": {"is_stale": True, "quality_flags": []},
                "sampling": {"quality_flags": []},
            },
        },
    )

    response = app.test_client().post("/api/decision-report", json={"stock_code": "600519"}, headers=LOCAL_ORIGIN_HEADERS)
    body = response.get_json()

    assert response.status_code == 200
    assert body["action_permission"] == "NONE"
    assert body["evidence_status"] == "STALE"
    assert body["signal"] is None
    assert body["recommendation"]["legacy_signal"] is None


def test_prediction_empty_path_returns_safe_contract(tmp_path: Path, monkeypatch) -> None:
    """旧预测入口收到空路径时返回非 2xx 的统一安全契约。"""
    data_directory = tmp_path / "data"
    data_directory.mkdir()
    source = data_directory / "bars.csv"
    pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=4),
            "open": [1, 2, 3, 4],
            "high": [2, 3, 4, 5],
            "low": [0, 1, 2, 3],
            "close": [1, 2, 3, 4],
        }
    ).to_csv(source, index=False)

    class EmptyPredictor:
        def predict(self, **kwargs):
            return pd.DataFrame()

    monkeypatch.setattr(app_module, "DATA_DIRECTORY", data_directory)
    monkeypatch.setattr(app_module, "MODEL_AVAILABLE", True)
    monkeypatch.setattr(app_module, "predictor", EmptyPredictor())

    response = app.test_client().post(
        "/api/predict",
        json={"file_name": source.name, "lookback": 2, "pred_len": 1},
        headers=LOCAL_ORIGIN_HEADERS,
    )
    body = response.get_json()

    assert response.status_code == 422
    assert body["run_status"] == "FAILED"
    assert body["evidence_status"] == "INVALID"
    assert body["action_permission"] == "NONE"
    assert body["error"]["message"] == "模型未返回有效预测路径。"
    assert str(source) not in response.get_data(as_text=True)


def test_report_pages_start_without_a_default_action() -> None:
    """旧版兼容页和新版报告页的初始 DOM 都必须是无动作状态。"""
    client = app.test_client()
    legacy = client.get("/report/legacy").get_data(as_text=True)
    modern = client.get("/report").get_data(as_text=True)

    assert 'id="signal-en">NONE<' in legacy
    assert 'id="action-permission">NONE<' in legacy
    assert "signalStr || 'HOLD'" not in legacy
    assert 'id="run-status"' in modern
    assert 'id="evidence-status"' in modern
    assert 'id="action-permission"' in modern
    assert "缺失字段猜测成功状态" in modern
