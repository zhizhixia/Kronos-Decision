"""外部审查缺陷的轻量回归测试。"""
from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

from decision.analyzer import RiskResult, SignalResult, TrendResult
from decision.engine import DecisionEngine
from decision.model_manager import ModelManager
from data.contracts import has_hard_quality_flags
from evaluation.contracts import prediction_key
from model.prediction import sampling_params_hash
from portfolio.optimizer import build_simulated_orders


def test_sampling_identity_changes_with_batch_size() -> None:
    first = sampling_params_hash(60, 100, 20, 0.6, 0.9, 0)
    second = sampling_params_hash(60, 100, 10, 0.6, 0.9, 0)
    assert first != second


def test_prediction_checkpoint_key_binds_sampling_identity() -> None:
    first = prediction_key("600519", "2026-08-01", "model", "config", "params-a")
    second = prediction_key("600519", "2026-08-01", "model", "config", "params-b")
    assert first != second


def test_expired_prediction_deadline_fails_synchronously() -> None:
    from model.prediction import ensure_before_deadline

    with pytest.raises(TimeoutError, match="PREDICTION_TIMEOUT"):
        ensure_before_deadline(time.monotonic() - 0.01)


@pytest.mark.parametrize(
    "flag",
    ["CALENDAR_QLIB_OUTDATED", "CALENDAR_WEEKDAY_FALLBACK"],
)
def test_unverified_calendar_is_a_hard_quality_error(flag: str) -> None:
    assert has_hard_quality_flags((flag,))


def test_display_only_warning_is_not_a_hard_quality_error() -> None:
    assert not has_hard_quality_flags(("POSSIBLE_SUSPENSION_ZERO_VOLUME",))


def test_sell_from_round_lot_remains_a_round_lot() -> None:
    orders, _, _ = build_simulated_orders(
        {"600519": 0.0555}, {"600519": 1000}, {"600519": 10.0}, 100_000
    )
    sell = next(order for order in orders if order["side"] == "SELL")
    assert sell["shares"] == 400
    assert sell["shares"] % 100 == 0


def test_sell_can_include_complete_odd_lot_once() -> None:
    orders, _, _ = build_simulated_orders(
        {"600519": 0.06}, {"600519": 1050}, {"600519": 10.0}, 100_000
    )
    sell = next(order for order in orders if order["side"] == "SELL")
    assert sell["shares"] == 450
    assert (1050 - sell["shares"]) % 100 == 0


def test_model_cleanup_never_releases_an_active_lease(monkeypatch) -> None:
    manager = ModelManager()

    def fake_load() -> None:
        manager._tokenizer = object()
        manager._predictor = object()

    monkeypatch.setattr(manager, "_load_model", fake_load)
    manager._idle_timeout = timedelta(seconds=1)

    with manager.lease():
        manager._last_access = datetime.now() - timedelta(seconds=10)
        manager._cleanup()
        assert manager._predictor is not None

    manager._cancel_cleanup_timer()
    manager._last_access = datetime.now() - timedelta(seconds=10)
    manager._cleanup()
    assert manager._predictor is None


def test_v1_without_formal_evidence_is_degraded_hold_and_uses_cpu_budget(monkeypatch) -> None:
    engine = DecisionEngine()
    observed: dict[str, object] = {}
    bars = pd.DataFrame({
        "date": pd.bdate_range("2024-01-02", periods=100),
        "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0,
        "volume": 1000.0, "amount": 10000.0,
    })
    bundle = type("Bundle", (), {
        "bars": bars, "source": "fixture", "as_of": pd.Timestamp("2024-05-20"),
        "is_stale": False, "quality_flags": (), "content_hash": "bars",
        "has_hard_quality_error": False,
    })()

    class FakePaths:
        sample_count = 10
        seed = 1
        model_id = "model"
        tokenizer_id = "tokenizer"
        sampling_params_hash = "params"
        repair_ratio = 0.0
        quality_flags = ()
        paths = np.full((10, 60, 6), 10.0, dtype=float)
        timestamps = pd.bdate_range("2024-05-21", periods=60)
        mean_df = pd.DataFrame(paths.mean(axis=0), columns=["open", "high", "low", "close", "volume", "amount"], index=timestamps)

    class FakePredictor:
        device = "cpu"
        model = None

        def predict_paths(self, *args):
            observed["sample_count"] = args[4]
            observed["batch_size"] = args[5]
            observed["deadline"] = args[10]
            return FakePaths()

    class FakeManager:
        @contextmanager
        def lease(self):
            yield None, FakePredictor()

    monkeypatch.setattr(engine._fetcher, "fetch_daily_bundle", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(engine._calendar, "future_sessions", lambda *args: FakePaths.timestamps)
    monkeypatch.setattr(engine._analyzer, "analyze", lambda *args: SignalResult("SELL", "raw", TrendResult("down", 0.9, "high", 0.1), RiskResult(-0.1, "high", "high")))
    monkeypatch.setattr(engine, "_model_manager", FakeManager())
    monkeypatch.setattr("decision.engine.EvaluationBinding.load_latest", lambda: None)

    started = time.monotonic()
    report = engine.predict_and_analyze("600519")

    assert report.status == "degraded"
    assert report.signal is not None and report.signal.signal == "HOLD"
    assert observed["sample_count"] == 10
    assert observed["batch_size"] == 10
    assert started < float(observed["deadline"]) <= started + 61


def test_windows_launcher_uses_runtime_errorlevel() -> None:
    content = Path("start.bat").read_text(encoding="utf-8")
    assert "%errorlevel%" not in content.lower()
    assert "if errorlevel 1" in content.lower()
    assert "KRONOS_STARTUP_CHECK_ONLY" in content
    assert "py -3.11" in content
    assert "py -3.10" in content


def test_start_bat_locked_design() -> None:
    """锁定设计的轻量回归：uvexternally-managed 修复后的关键不变量。"""
    content = Path("start.bat").read_text(encoding="utf-8")
    lower = content.lower()

    # 无 Hermes 硬编码
    assert "hermes" not in lower

    # 绝不执行 --break-system-packages（仅在 rem 注释中允许出现）
    exec_break = [
        ln for ln in content.splitlines()
        if "--break-system-packages" in ln.lower()
        and ln.strip() and not ln.strip().lower().startswith("rem")
    ]
    assert not exec_break

    # uv venv 仅以 --no-python-downloads 离线使用，且创建项目 .venv
    uv_lines = [ln for ln in content.splitlines() if "uv venv" in ln.lower()]
    assert uv_lines, "expected uv venv invocation"
    for ln in uv_lines:
        assert "--no-python-downloads" in ln
        assert "--offline" in ln
        assert r"%PROJECT%\.venv" in ln

    # uv python find 也必须离线
    find_lines = [ln for ln in content.splitlines() if "uv python find" in ln]
    assert find_lines, "expected uv python find invocation"
    for ln in find_lines:
        assert "--no-python-downloads" in ln

    # 绝不直接选择 APPDATA 下 uv 裸 Python
    assert r"%APPDATA%\uv\python" not in content
    assert "dir /b /s" not in lower

    # check-only 模式不联网安装：失败分支给出的 pip install 仅出现在 echo 提示行，不实际执行
    check_block = content[content.find('KRONOS_STARTUP_CHECK_ONLY!"=="1'):]
    fail_idx = check_block.find("Dependencies missing")
    end_idx = check_block.find("exit /b 1", fail_idx)
    fail_block = check_block[fail_idx:end_idx]
    assert "pip install" in fail_block  # 作为提示命令文本出现
    exec_pip_in_check = [
        ln for ln in fail_block.splitlines()
        if "pip install" in ln.lower()
        and ln.strip() and not ln.strip().lower().startswith("echo")
    ]
    assert not exec_pip_in_check  # 无实际执行的 pip install

    # 每个路径候选都经过 :try_python 版本校验
    assert ":try_python" in content
    assert "version_info[:2]in((3,10),(3,11))" in content.replace(" ", "")

    # 正常启动仍可在缺依赖时 pip install requirements
    assert "-r " in content and "requirements.txt" in content


def test_temporary_sitecustomize_is_removed() -> None:
    assert not Path("sitecustomize.py").exists()


def test_python_sources_have_no_utf8_bom() -> None:
    roots = ("data", "decision", "evaluation", "examples", "finetune", "finetune_csv", "model", "portfolio", "scripts", "tests", "webui")
    offenders = [
        path
        for root in roots
        for path in Path(root).rglob("*.py")
        if ".tmp" not in path.parts and path.read_bytes().startswith(b"\xef\xbb\xbf")
    ]
    assert offenders == []
