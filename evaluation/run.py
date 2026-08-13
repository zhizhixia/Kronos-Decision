"""Qlib 正式研究的命令行入口：预测、回测、指标与门禁工件。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from data.universe import HistoricalUniverse
from decision.config import get_config
from decision.versioning import config_hash as current_config_hash, model_provenance, rules_hash
from evaluation.artifacts import EvaluationArtifacts, PREDICTION_PROVENANCE_COLUMNS
from evaluation.gates import evaluate_evidence_gate
from evaluation.metrics import block_bootstrap_positive_probability, portfolio_metrics, rank_ic_by_anchor

SAMPLE_STOCKS = ["600519", "000001", "601318", "600036", "000858", "601166", "600030", "000333", "601398", "600900", "601288", "600276", "601012", "000651", "600887", "601888", "000725", "600309", "601601", "002594", "600028", "601988", "000002", "601668", "600104", "300059", "002415", "603288", "601899", "000063"]
RESEARCH_DATA_START = "2011-01-01"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kronos-Decision Qlib 评估")
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--as-of", required=True, help="锚点日期 YYYY-MM-DD")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--stocks", default=None, help="逗号分隔的股票代码，默认30只代表股")
    parser.add_argument("--anchors-start", default="2017-01-01", help="周锚点起点（默认 2017-01-01）")
    parser.add_argument("--anchor-frequency", choices=("weekly", "monthly"), default="weekly", help="锚点频率：weekly（每周最后交易日，默认）或 monthly（每月最后交易日）")
    parser.add_argument("--collect-only", action="store_true", help="仅收集预测工件，不执行回测或生成正式结论")
    parser.add_argument("--max-new-anchors", type=int, default=None, help="本次最多生成的新锚点批次；自动启用仅收集模式")
    return parser


def run(
    stage: str,
    as_of: str,
    run_id: str | None = None,
    sample_count: int = 100,
    stocks: str | None = None,
    anchors_start: str = "2017-01-01",
    anchor_frequency: str = "weekly",
    collect_only: bool = False,
    max_new_anchors: int | None = None,
) -> tuple[EvaluationArtifacts, dict[str, Any]]:
    """执行安全预检并持久化工件；正式回测仅在预检全通过后启动。"""
    resolved_id = run_id or f"{stage}-{as_of.replace('-', '')}"
    artifacts = EvaluationArtifacts.open(resolved_id)
    effective_collect = collect_only or max_new_anchors is not None
    failures = _preflight(stage, as_of, max_new_anchors, effective_collect, stocks)
    manifest = _manifest(stage, as_of, run_id, sample_count, stocks, anchors_start, anchor_frequency, failures, effective_collect, max_new_anchors)
    artifacts.write_manifest(manifest)
    if failures:
        gate = evaluate_evidence_gate({"data_complete": False, "quality_error": True, "versions_match": False})
        artifacts.write_gate_result(gate.as_dict())
        manifest["completed_at"] = datetime.now().isoformat()
        artifacts.write_manifest(manifest)
        return artifacts, manifest
    if stage == "smoke":
        _run_smoke(as_of, sample_count, stocks, artifacts, manifest)
    else:
        _run_full(as_of, anchors_start, artifacts, manifest, stocks, sample_count, anchor_frequency, effective_collect, max_new_anchors)
    return artifacts, manifest


def _preflight(
    stage: str,
    as_of: str,
    max_new_anchors: int | None = None,
    collect_only: bool = False,
    stocks_arg: str | None = None,
) -> list[str]:
    failures: list[str] = []
    try:
        datetime.fromisoformat(as_of)
    except ValueError:
        failures.append("INVALID_AS_OF")
    if max_new_anchors is not None and max_new_anchors < 1:
        failures.append("INVALID_MAX_NEW_ANCHORS")
    if collect_only and stage != "full":
        failures.append("COLLECTION_REQUIRES_FULL_STAGE")
    if stage == "full":
        try:
            import qlib  # noqa: F401
        except ImportError:
            failures.append("PYQLIB_NOT_INSTALLED")
        if not Path(_qlib_data_dir()).exists():
            failures.append("QLIB_CN_DATA_MISSING")
        if not HistoricalUniverse().members_at(as_of) and not (stocks_arg or "").strip():
            failures.append("POINT_IN_TIME_CSI300_MISSING")
    return failures


def _run_smoke(as_of: str, sample_count: int, stocks_arg: str | None, artifacts: EvaluationArtifacts, manifest: dict) -> None:
    """真实数据冒烟：固定股票、真实采样与工件落盘。"""
    from decision.engine import DecisionEngine
    from decision.model_manager import get_model_manager
    from evaluation.predictor import ResearchPredictor

    engine = DecisionEngine()
    fetcher = engine._fetcher
    calendar = engine._calendar.future_sessions
    stocks = stocks_arg.split(",") if stocks_arg else SAMPLE_STOCKS
    requested_count = sample_count

    def fetch_bars(code: str, anchor: pd.Timestamp):
        bundle = fetcher.fetch_daily_bundle(code)
        return bundle.bars

    with get_model_manager().lease() as (tokenizer, predictor):
        model_hash = _attach_model_provenance(manifest, tokenizer, predictor)
        sample_count, reduced = _resolve_sample_count(predictor, sample_count)
        batch_size = min(20, sample_count)
        research = ResearchPredictor(
            predictor.predict_paths, fetch_bars, calendar, model_hash,
            manifest["config_hash"], manifest["data_hash"], sample_count, batch_size,
        )
        manifest["sampling_params_hash"] = research.sampling_params_hash
        existing = _load_existing_predictions(artifacts, manifest)
        existing_keys = set(existing.get("prediction_key", pd.Series(dtype=str)).astype(str))
        predictions = research.run(stocks, [as_of], existing_keys.__contains__)
    predictions["actual_return"] = float("nan")
    merged = _merge_predictions(existing, predictions)
    artifacts.write_frame("predictions", merged)
    manifest["predictions_count"] = int(len(merged))
    manifest["sample_count"] = sample_count
    manifest["requested_sample_count"] = requested_count
    manifest["reduced_for_cpu"] = reduced
    manifest["status"] = "completed-smoke"
    manifest["completed_at"] = datetime.now().isoformat()
    artifacts.write_manifest(manifest)
    gate = evaluate_evidence_gate({"data_complete": True, "quality_error": False, "versions_match": False})
    artifacts.write_gate_result(gate.as_dict())


def _run_full(
    as_of: str,
    anchors_start: str,
    artifacts: EvaluationArtifacts,
    manifest: dict,
    stocks_arg: str | None = None,
    sample_count: int = 100,
    anchor_frequency: str = "weekly",
    collect_only: bool = False,
    max_new_anchors: int | None = None,
) -> None:
    """完整阶段：Qlib 点数据预测、正式回测、指标与校准。"""
    from evaluation.benchmark import resolve_benchmark
    from evaluation.predictor import ResearchPredictor
    from evaluation.qlib_data import QlibPointData
    from evaluation.qlib_adapter import QlibBacktestConfig
    from decision.model_manager import get_model_manager

    point = QlibPointData(_qlib_data_dir())
    point.init()
    print(f"[full] Qlib 数据目录: {_qlib_data_dir()}", flush=True)
    anchor_fn = point.monthly_anchors if anchor_frequency == "monthly" else point.weekly_anchors
    anchors = [anchor for anchor in anchor_fn(anchors_start, as_of)]
    print(f"[full] 锚点数: {len(anchors)}", flush=True)
    base_config = QlibBacktestConfig(provider_uri=_qlib_data_dir())
    benchmark = resolve_benchmark(base_config.provider_uri, RESEARCH_DATA_START, as_of)
    backtest_config = replace(
        base_config,
        benchmark=benchmark.instrument,
        benchmark_is_total_return=benchmark.is_total_return,
    )
    requested_count = sample_count
    with get_model_manager().lease() as (tokenizer, predictor):
        model_hash = _attach_model_provenance(manifest, tokenizer, predictor)
        sample_count, reduced = _resolve_sample_count(predictor, sample_count)
        protocol_complete = _is_formal_protocol(
            as_of, anchors_start, anchor_frequency, sample_count, reduced, stocks_arg,
            collect_only, point.latest_session(), backtest_config.benchmark_is_total_return,
        )
        formal_protocol = protocol_complete and bool(manifest["model_provenance"]["revisions_pinned"])
        research = ResearchPredictor(
            predictor.predict_paths, point.bars_at, point.future_sessions, model_hash,
            manifest["config_hash"], manifest["data_hash"], sample_count, min(20, sample_count),
        )
        manifest.update({
            "sample_count": sample_count, "requested_sample_count": requested_count,
            "reduced_for_cpu": reduced, "formal_protocol": formal_protocol,
            "sampling_params_hash": research.sampling_params_hash,
            "benchmark": backtest_config.benchmark,
            "benchmark_is_total_return": backtest_config.benchmark_is_total_return,
            "benchmark_provenance": benchmark.as_dict(),
        })
        predictions = _generate_full_predictions(
            research, point, anchors, artifacts, manifest, stocks_arg, max_new_anchors
        )
    print(f"[full] 预测行数: {len(predictions)}", flush=True)
    if collect_only:
        _record_prediction_collection(artifacts, manifest, predictions, len(anchors))
        return
    _finalize_full_run(predictions, point, artifacts, manifest, backtest_config, formal_protocol, len(anchors))


def _record_prediction_collection(artifacts: EvaluationArtifacts, manifest: dict, predictions: pd.DataFrame, anchors_count: int) -> None:
    """写入分片预测检查点，并显式禁止其产生正式建议。"""
    artifacts.write_frame("predictions", predictions)
    manifest.update({"status": "collecting", "predictions_count": int(len(predictions)), "anchors_count": anchors_count, "formal_protocol": False, "completed_at": datetime.now().isoformat()})
    artifacts.write_manifest(manifest)
    gate = evaluate_evidence_gate({"data_complete": False, "quality_error": False, "versions_match": False, "formal_protocol": False})
    artifacts.write_gate_result(gate.as_dict())


def _finalize_full_run(
    predictions: pd.DataFrame,
    point: Any,
    artifacts: EvaluationArtifacts,
    manifest: dict,
    backtest_config: Any,
    formal_protocol: bool,
    anchors_count: int,
) -> None:
    """对完整预测工件回填、回测、校准并写入最终门禁。"""
    from evaluation.backfill import backfill_actuals
    from evaluation.qlib_adapter import QlibBacktestAdapter

    predictions = backfill_actuals(predictions, point.bars_at)
    print(f"[full] 回填实际收益: {int(predictions['actual_return'].notna().sum())}/{len(predictions)}", flush=True)
    artifacts.write_frame("predictions", predictions)
    manifest["backfilled_actuals"] = int(predictions["actual_return"].notna().sum())
    manifest["predictions_count"] = int(len(predictions))
    manifest["anchors_count"] = anchors_count
    artifacts.write_manifest(manifest)
    universe_size = int(predictions["stock_code"].nunique())
    topk = min(10, universe_size)
    n_drop = min(2, max(0, universe_size - topk))
    adapter_config = replace(backtest_config, topk=topk, n_drop=n_drop)
    adapter = QlibBacktestAdapter(adapter_config)
    report, extra = adapter.run(predictions)
    print(f"[full] 回测完成: {len(report)} 交易日", flush=True)
    artifacts.write_frame("backtest_daily", report.reset_index())
    positions, trades = _positions_and_trades(extra.get("positions", {}), point)
    artifacts.write_frame("positions", positions)
    artifacts.write_frame("trades", trades)
    artifacts.write_frame("horizon_metrics", _horizon_metrics(predictions))
    manifest["trades_count"] = int(len(trades))
    manifest["backtest_days"] = int(len(report))
    metrics = portfolio_metrics(report)
    rank_ic = rank_ic_by_anchor(predictions)
    calibration = _calibration_metrics(predictions, manifest)
    calibration_20 = calibration.metrics[20]
    artifacts.write_calibration(calibration.as_dict())
    manifest["calibration"] = calibration.as_dict()
    bootstrap_probability = block_bootstrap_positive_probability(rank_ic, seed=manifest["seed"])
    metrics.update({
        "rank_ic_mean": float(rank_ic.mean()) if not rank_ic.empty else 0.0,
        "rank_ic_positive_probability": bootstrap_probability,
        "coverage_80": calibration_20["coverage_80"],
        "up_probability_ece": calibration_20["up_probability_ece"],
        "calibration_by_horizon": calibration.metrics,
        "data_complete": True,
        "quality_error": False,
        "versions_match": _versions_match(predictions, manifest),
        "formal_protocol": formal_protocol,
        "benchmark_is_total_return": adapter_config.benchmark_is_total_return,
        "evaluated_at": datetime.now().isoformat(),
    })
    gate = evaluate_evidence_gate(metrics)
    artifacts.write_gate_result(gate.as_dict())
    baselines = _baseline_comparison(predictions, point, adapter_config)
    print(f"[full] 基线比较完成", flush=True)
    forward = _forward_validation(predictions, point)
    print(f"[full] 前瞻验证完成", flush=True)
    manifest.update({"status": "completed", "metrics": metrics, "baselines": baselines, "forward_validation": forward, "completed_at": datetime.now().isoformat()})
    artifacts.write_manifest(manifest)


def _baseline_comparison(predictions: pd.DataFrame, point, backtest_config: Any) -> dict:
    """收集期末可见价格序列并与三个基线比较；结果仅作研究展示，不计入门禁。"""
    from evaluation.baselines import compare_baselines

    codes = sorted(predictions["stock_code"].astype(str).str.zfill(6).unique())
    frames = {}
    for code in codes:
        bars = point.bars_at(code, predictions["anchor_date"].max(), lookback_days=1400)
        if bars is not None and not bars.empty:
            frames[code] = bars.set_index("date")["close"].astype(float)
    if not frames:
        return {"error": "PRICE_HISTORY_UNAVAILABLE"}
    prices = pd.DataFrame(frames).sort_index()
    hs300 = point.benchmark_bars(predictions["anchor_date"].max(), lookback_days=1400, benchmark=backtest_config.benchmark)
    hs300_close = hs300.set_index("date")["close"].astype(float) if hs300 is not None and not hs300.empty else pd.Series(dtype=float)
    summary = compare_baselines(predictions, prices, hs300_close, backtest_config.benchmark_is_total_return)
    summary["benchmark"] = backtest_config.benchmark
    summary["point_in_time"] = bool(summary["point_in_time_equal_weight"])
    summary["disclaimer"] = "基线使用评估期末可见序列，仅作研究展示，不计入正式证据门禁。"
    return summary


def _forward_validation(predictions: pd.DataFrame, point) -> dict:
    """20 日周期前瞻模拟；结果与历史回测分开报告。"""
    from evaluation.forward_sim import forward_report, simulate_forward

    codes = sorted(predictions["stock_code"].astype(str).str.zfill(6).unique())
    end = point.latest_session()
    frames = {}
    for code in codes:
        bars = point.bars_at(code, end, lookback_days=1400)
        if bars is not None and not bars.empty:
            frames[code] = bars.set_index("date")["close"].astype(float)
    if not frames:
        return {"status": "no_observations"}
    prices = pd.DataFrame(frames).sort_index()
    simulated = simulate_forward(predictions, prices, topk=10, horizon=20)
    return forward_report(simulated)


def _generate_full_predictions(
    research,
    point: Any,
    anchors: list,
    artifacts: EvaluationArtifacts,
    manifest: dict,
    stocks_arg: str | None = None,
    max_new_anchors: int | None = None,
) -> pd.DataFrame:
    """按每周锚点和点时成分生成预测；`--stocks` 限定范围；同一键命中已有工件时跳过推理。"""
    requested = {str(code).zfill(6) for code in stocks_arg.split(",")} if stocks_arg else None
    persisted = _load_existing_predictions(artifacts, manifest)
    existing_keys = set(persisted.get("prediction_key", pd.Series(dtype=str)).astype(str))
    rows: list[pd.DataFrame] = []
    new_batches = 0
    for anchor in anchors:
        if max_new_anchors is not None and new_batches >= max_new_anchors:
            print(f"[full] 已达到本次 {max_new_anchors} 个新锚点批次上限", flush=True)
            break
        universe = point.universe_at(anchor)
        if not universe:
            continue
        if requested is not None:
            skipped = sorted(requested - set(universe))
            if skipped:
                print(f"[full] {anchor.date()} 请求股票不在点时成分中，跳过: {','.join(skipped)}", flush=True)
            universe = [code for code in universe if code in requested]
            if not universe:
                continue
        # 预过滤数据不足 60 根的股票，避免单只缺失导致整个运行中断
        usable = []
        for code in universe:
            bars = point.bars_at(code, anchor)
            if bars is None or len(bars) < 60:
                continue
            future = point.future_sessions(pd.Timestamp(bars["date"].iloc[-1]), 1)
            if future.empty:
                continue
            usable.append(code)
        universe = usable
        if not universe:
            print(f"[full] {anchor.date()} 无可用历史数据的成分，跳过该锚点", flush=True)
            continue
        chunk = research.run(universe, [anchor], existing_keys.__contains__)
        if not chunk.empty:
            rows.append(chunk)
            new_batches += 1
            existing_keys.update(chunk["prediction_key"].astype(str))
            if len(rows) == 5:
                persisted = _merge_predictions(persisted, *rows)
                artifacts.write_frame("predictions", persisted)
                rows.clear()
                print(f"[full] 检查点已落盘: {new_batches} 个新增锚点批次", flush=True)
    predictions = _merge_predictions(persisted, *rows)
    if predictions.empty:
        raise RuntimeError("完整阶段没有生成任何预测；请检查 Qlib 数据、点时成分与 --stocks 范围。")
    if new_batches == 0:
        print("[full] 无新增预测，复用已有工件实现断点续跑", flush=True)
    return predictions


def _load_existing_predictions(artifacts: EvaluationArtifacts, manifest: dict | None = None) -> pd.DataFrame:
    """读取已原子落盘的预测检查点，避免每个股票重复扫描 CSV。"""
    path = artifacts.root / "predictions.csv"
    if not path.exists() or path.stat().st_size < 20:
        return pd.DataFrame()
    frame = pd.read_csv(path, dtype={"stock_code": str})
    if not PREDICTION_PROVENANCE_COLUMNS.issubset(frame.columns):
        print("[full] 旧预测工件缺少采样审计字段，将重新生成", flush=True)
        return pd.DataFrame()
    base_provenance = ("model_hash", "config_hash", "data_hash")
    if manifest is not None and all(manifest.get(column) for column in base_provenance):
        expected = {column: str(manifest.get(column, "")) for column in base_provenance}
        if manifest.get("sampling_params_hash"):
            expected["sampling_params_hash"] = str(manifest["sampling_params_hash"])
        compatible = pd.Series(True, index=frame.index)
        for column, value in expected.items():
            compatible &= frame[column].astype(str).eq(value) if column in frame else False
        if not compatible.all():
            print("[full] 检查点版本不兼容，将仅保留当前模型、配置和数据版本的预测", flush=True)
            frame = frame.loc[compatible].copy()
    return frame


def _merge_predictions(persisted: pd.DataFrame, *chunks: pd.DataFrame) -> pd.DataFrame:
    """合并检查点和新增预测，保留完整历史而不重复同一预测期限。"""
    frames = [frame for frame in (persisted, *chunks) if not frame.empty]
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["prediction_key", "horizon"], keep="last")
    return merged.sort_values(["anchor_date", "stock_code", "horizon"]).reset_index(drop=True)


def _resolve_sample_count(predictor: Any, sample_count: int) -> tuple[int, bool]:
    """CPU 环境使用配置的降级采样数，与在线报告行为一致。"""
    device = str(getattr(predictor, "device", "cpu"))
    fallback = get_config().prediction.fallback_sample_count
    reduced = device.startswith("cpu") and sample_count > fallback
    return (fallback if reduced else sample_count), reduced


def _calibration_metrics(predictions: pd.DataFrame, manifest: dict) -> Any:
    """拟合 5/20/60 日独立校准档案；20 日指标仍作为正式主门禁。"""
    from evaluation.calibration import apply_multi_horizon_calibration_metrics

    return apply_multi_horizon_calibration_metrics(predictions, manifest["model_hash"], manifest["data_hash"], manifest["config_hash"])


def _attach_model_provenance(manifest: dict, tokenizer: Any, predictor: Any) -> str:
    """把实际加载的固定模型/Tokenizer 标识写入运行清单。"""
    provenance = model_provenance(getattr(predictor, "model", None), tokenizer)
    manifest.update({"model": provenance["model_id"], "tokenizer": provenance["tokenizer_id"], "model_hash": provenance["pipeline_hash"], "model_provenance": provenance})
    return str(provenance["pipeline_hash"])


def _versions_match(predictions: pd.DataFrame, manifest: dict) -> bool:
    """验证评估行、当前配置、规则和固定模型修订完全一致。"""
    required = {"model_hash", "config_hash", "data_hash", "sampling_params_hash"}
    if predictions.empty or not required.issubset(predictions) or manifest.get("config_hash") != current_config_hash() or manifest.get("rules_hash") != rules_hash():
        return False
    provenance = manifest.get("model_provenance", {})
    if not provenance.get("revisions_pinned") or manifest.get("model_hash") != model_provenance()["pipeline_hash"]:
        return False
    for column in required:
        if set(predictions[column].dropna().astype(str)) != {str(manifest.get(column))}:
            return False
    return True


def _horizon_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """按期限汇总预测与回填统计，供研究页展示。"""
    grouped = predictions.groupby("horizon").agg(count=("prediction_key", "size"), backfilled=("actual_return", lambda series: int(series.notna().sum())), mean_score=("score", "mean"), mean_actual=("actual_return", "mean"), q05_mean=("q05", "mean"), q50_mean=("q50", "mean"), q95_mean=("q95", "mean"))
    return grouped.reset_index()


def _positions_and_trades(positions: dict, point: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把 Qlib 每日持仓快照转为 positions.csv，并用相邻快照差派生 trades.csv。"""
    from evaluation.qlib_data import _to_local_code

    position_rows = []
    trade_rows = []
    previous: dict[str, float] = {}
    for day in sorted(positions):
        amounts = positions[day].get_stock_amount_dict()
        normalized = {_to_local_code(code): float(amount) for code, amount in amounts.items()}
        for code, amount in normalized.items():
            position_rows.append({"date": day.date().isoformat(), "stock_code": code, "amount": amount})
        for code, amount in normalized.items():
            delta = amount - previous.get(code, 0.0)
            if abs(delta) < 1.0:
                continue
            bars = point.bars_at(code, day)
            price = float(bars["close"].iloc[-1]) if bars is not None and not bars.empty else float("nan")
            trade_rows.append({"date": day.date().isoformat(), "stock_code": code, "direction": "BUY" if delta > 0 else "SELL", "notional": abs(delta), "price": price, "shares": abs(delta) / price if price == price else float("nan")})
        previous = normalized
    return pd.DataFrame(position_rows), pd.DataFrame(trade_rows)


def _qlib_data_hash(uri: str) -> str:
    """对 Qlib 日历与成员文件做内容哈希，作为数据版本标识。"""
    digest = hashlib.sha256()
    for relative in ("calendars/day.txt", "instruments/all.txt", "instruments/csi300.txt"):
        path = Path(uri) / relative
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _cache_data_hash() -> str:
    """对 data/cache 下全部 CSV 内容做哈希，作为冒烟数据版本标识。"""
    digest = hashlib.sha256()
    for path in sorted(Path("data/cache").glob("*.csv")):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _manifest(
    stage: str,
    as_of: str,
    run_id: str | None,
    sample_count: int,
    stocks: str | None,
    anchors_start: str,
    anchor_frequency: str,
    failures: list[str],
    collect_only: bool = False,
    max_new_anchors: int | None = None,
) -> dict:
    config_digest = current_config_hash()
    provenance = model_provenance()
    data_hash = _qlib_data_hash(_qlib_data_dir()) if stage == "full" else _cache_data_hash()
    command_parts = ["python -m evaluation.run", f"--stage {stage}", f"--as-of {as_of}", f"--anchors-start {anchors_start}", f"--anchor-frequency {anchor_frequency}", f"--sample-count {sample_count}"]
    stock_codes = [str(code).strip().zfill(6) for code in stocks.split(",") if str(code).strip()] if stocks else []
    if stock_codes:
        command_parts.append(f'--stocks "{",".join(stock_codes)}"')
    if collect_only:
        command_parts.append("--collect-only")
    if max_new_anchors is not None:
        command_parts.append(f"--max-new-anchors {max_new_anchors}")
    if run_id:
        command_parts.append(f"--run-id {run_id}")
    return {"status": "blocked" if failures else "ready", "failure_reason": failures, "command": " ".join(command_parts), "as_of": as_of, "anchors_start": anchors_start, "anchor_frequency": anchor_frequency, "stocks": stock_codes, "collection_only": collect_only, "max_new_anchors": max_new_anchors, "formal_protocol": False, "seed": _seed(stage, as_of), "sampling_seed_strategy": "sha256(model_hash|stock_code|last_visible_date|config_hash)", "horizons": [5, 20, 60], "model": provenance["model_id"], "tokenizer": provenance["tokenizer_id"], "model_hash": provenance["pipeline_hash"], "model_provenance": provenance, "config_hash": config_digest, "rules_hash": rules_hash(), "data_hash": data_hash, "git_commit": _git("rev-parse", "HEAD"), "worktree_dirty": bool(_git("status", "--porcelain")), "started_at": datetime.now().isoformat(), "completed_at": None}


def _is_formal_protocol(
    as_of: str,
    anchors_start: str,
    anchor_frequency: str,
    sample_count: int,
    reduced: bool,
    stocks_arg: str | None,
    collect_only: bool,
    latest_session: pd.Timestamp,
    benchmark_is_total_return: bool,
) -> bool:
    """仅在全收益基准等完整研究协议满足时允许正式门禁继续评估。"""
    try:
        return (
            anchor_frequency == "weekly"
            and sample_count == 100
            and not reduced
            and not (stocks_arg or "").strip()
            and not collect_only
            and benchmark_is_total_return
            and pd.Timestamp(anchors_start).normalize() == pd.Timestamp("2017-01-01")
            and pd.Timestamp(as_of).normalize() >= pd.Timestamp(latest_session).normalize()
        )
    except (TypeError, ValueError):
        return False


def _seed(stage: str, as_of: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{stage}|{as_of}".encode()).digest()[:8], "big") % (2**63 - 1)


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _qlib_data_dir() -> str:
    """Qlib 数据目录：优先环境变量 KRONOS_QLIB_DATA，回退到用户默认路径。"""
    return os.environ.get("KRONOS_QLIB_DATA") or str(Path.home() / ".qlib" / "qlib_data" / "cn_data")


def main() -> int:
    args = build_parser().parse_args()
    artifacts, manifest = run(args.stage, args.as_of, args.run_id, args.sample_count, args.stocks, args.anchors_start, args.anchor_frequency, args.collect_only, args.max_new_anchors)
    print(json.dumps({"run_id": artifacts.run_id, "status": manifest["status"], "failure_reason": manifest.get("failure_reason", [])}, ensure_ascii=False))
    return 0 if manifest["status"] not in ("blocked",) else 2


if __name__ == "__main__":
    raise SystemExit(main())
