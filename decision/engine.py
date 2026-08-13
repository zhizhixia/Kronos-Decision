"""决策编排：真实采样路径、数据来源和旧接口兼容。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from model.prediction import PredictionPaths

import pandas as pd
import numpy as np

from data.calendar import AshareCalendar
from data.contracts import has_hard_quality_flags
from data.eligibility import evaluate_stock_eligibility
from data.fetcher import DataFetcher
from decision.analyzer import SignalAnalyzer, SignalResult
from decision.config import get_config
from decision.errors import KronosError, PredictionTimeoutError
from decision.model_manager import get_model_manager
from decision.versioning import config_hash, derive_sampling_seed, model_provenance, rules_hash
from decision.v2 import RecommendationAction, decide_action, legacy_signal
from evaluation.gates import evaluate_evidence_gate
from evaluation.gates import EvidenceGateResult
from evaluation.binding import EvaluationBinding
from data.market_context import MarketContextProvider
from data.fundamental import FundamentalAdapter
from data.events import EventRiskProvider
from portfolio.store import PortfolioStore


@dataclass
class DecisionReport:
    """v1 报告；新增字段不改变原有调用方。"""

    status: Literal["ok", "error", "degraded"]
    error_message: str = ""
    signal: SignalResult | None = None
    prediction: dict[str, Any] | None = None
    stock_code: str = ""
    stock_name: str = ""
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    elapsed_seconds: float = 0.0


class DecisionEngine:
    """只生成本地研究建议，绝不连接券商或自动下单。"""

    def __init__(self) -> None:
        cfg = get_config()
        self._pred_cfg = cfg.prediction
        self._analyzer = SignalAnalyzer()
        self._fetcher = DataFetcher()
        self._model_manager = get_model_manager()
        self._calendar = AshareCalendar()

    def predict_and_analyze(self, stock_code: str, params: dict | None = None) -> DecisionReport:
        """兼容 v1 的完整预测入口；不再构造人工高斯路径。"""
        started = time.time()
        try:
            return self._run_legacy_report(stock_code, params or {}, started)
        except TimeoutError as exc:
            return self._error_report(stock_code, f"PREDICTION_TIMEOUT：{exc}", started)
        except KronosError as exc:
            return self._error_report(stock_code, str(exc), started)
        except Exception as exc:
            return self._error_report(stock_code, f"未知错误：{exc}", started)

    def decision_report_v2(self, stock_code: str, portfolio_id: str = "default", include_display_paths: bool = False, as_of: str | pd.Timestamp | None = None) -> dict[str, Any]:
        """返回不允许网页覆写采样参数的 v2 可审计报告。"""
        requested_as_of = self._normalize_as_of(as_of)
        params: dict[str, Any] = {"include_display_paths": include_display_paths}
        if requested_as_of is not None:
            params["as_of"] = requested_as_of
        report = self.predict_and_analyze(stock_code, params)
        if report.status == "error" or report.prediction is None:
            message = report.error_message or "无法生成决策报告。"
            if "PREDICTION_TIMEOUT" in message:
                raise PredictionTimeoutError(message)
            raise KronosError(message)
        binding = EvaluationBinding.load_latest()
        gate, evidence = self._bind_evidence(binding, report)
        report_as_of = pd.Timestamp(report.prediction["data_provenance"]["as_of"]).normalize()
        eligibility = self._eligibility_for(stock_code, report_as_of)
        risk_blocked = not bool(eligibility.get("eligible"))
        is_held = self._is_held(portfolio_id, stock_code)
        action, reasons = self._decide_with_evidence(
            gate, evidence, risk_blocked=risk_blocked, is_held=is_held
        )
        horizons = self._merge_horizons(report.prediction["horizons"], evidence)
        payload = {"schema_version": "2.0", "status": report.status, "stock": {"code": stock_code, "name": report.stock_name}, "data_provenance": report.prediction["data_provenance"], "model_provenance": report.prediction["sampling"], "chart_data": report.prediction["chart_data"], "horizons": horizons, "recommendation": {"action": action.value, "reason_codes": list(reasons), "legacy_signal": legacy_signal(action)}, "evidence_gate": gate.as_dict(), "eligibility": eligibility, "market_context": self._market_context_payload(report_as_of), "fundamentals": FundamentalAdapter().snapshot(stock_code, as_of=report.prediction["data_provenance"].get("as_of")), "events": EventRiskProvider().events(stock_code, as_of=report_as_of.isoformat()), "portfolio_impact": {"portfolio_id": portfolio_id, "is_held": is_held, "eligible_for_rebalance": bool(gate.passed and evidence is not None)}, "generated_at": report.generated_at, "elapsed_seconds": report.elapsed_seconds}
        if include_display_paths and "display_paths" in report.prediction:
            payload["display_paths"] = report.prediction["display_paths"]
        return payload

    @staticmethod
    def _normalize_as_of(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
        """规范化可选报告截止日，拒绝无效时间点。"""
        if value is None:
            return None
        try:
            result = pd.Timestamp(value).normalize()
        except (TypeError, ValueError) as exc:
            raise KronosError("as_of 必须是可识别的日期。") from exc
        if pd.isna(result):
            raise KronosError("as_of 必须是可识别的日期。")
        return result

    def _market_context_payload(self, as_of: pd.Timestamp | None = None, timeout: float = 20.0) -> dict:
        """市场状态与行业强弱展示信息；总预算超时即返回 unavailable，不阻塞报告。"""
        import threading

        def build() -> dict:
            def fetch_index():
                if as_of is None:
                    return self._fetcher.fetch_daily_bundle("000300").bars
                return self._fetcher.fetch_daily_bundle("000300", as_of=as_of).bars

            return MarketContextProvider(fetch_index_bars=fetch_index).context()

        result: dict = {}

        def run() -> None:
            try:
                result["value"] = build()
            except Exception:
                result["value"] = {"available": False, "reason": "MARKET_DATA_UNAVAILABLE"}

        thread = threading.Thread(target=run, daemon=True, name="market-ctx")
        thread.start()
        thread.join(timeout=timeout)
        if thread.is_alive():
            return {"available": False, "reason": "MARKET_CONTEXT_TIMEOUT_OR_UNAVAILABLE"}
        return result.get("value", {"available": False, "reason": "MARKET_DATA_UNAVAILABLE"})

    @staticmethod
    def _bind_evidence(report_binding, report) -> tuple:
        """组合数据质量门禁与最新评估运行门禁，并提取该股票证据。"""
        provenance = report.prediction["data_provenance"]
        sampling = report.prediction.get("sampling", {})
        quality_error = DecisionEngine._has_hard_quality_error(provenance, sampling)
        quality_gate = evaluate_evidence_gate({
            "data_complete": bool(provenance.get("as_of")) and not bool(provenance.get("is_stale")),
            "quality_error": quality_error,
            "versions_match": False,
        })
        if report_binding is None or not report_binding.gate_result.get("checks"):
            return quality_gate, None
        research_checks = report_binding.gate_result["checks"]
        combined_checks = {**research_checks}
        for code in ("DATA_COMPLETE", "NO_QUALITY_ERROR"):
            if code in quality_gate.checks:
                combined_checks[code] = quality_gate.checks[code]
        stock = str(report.stock_code).zfill(6)
        evidence = report_binding.evidence_for(stock, 20)
        combined_checks["VERSION_MATCH"] = bool(research_checks.get("VERSION_MATCH")) and DecisionEngine._online_versions_match(report_binding, report, evidence)
        combined_checks["ONLINE_AS_OF_MATCH"] = DecisionEngine._online_as_of_matches(report_binding, report)
        failed = tuple(code for code, passed in combined_checks.items() if not passed)
        combined_gate = EvidenceGateResult(not failed, failed, combined_checks)
        return combined_gate, evidence

    @staticmethod
    def _online_versions_match(report_binding, report, evidence) -> bool:
        """在线模型、配置、规则和采样参数必须与评估证据逐项相同。"""
        if evidence is None:
            return False
        sampling = report.prediction.get("sampling", {})
        manifest = report_binding.manifest
        return (
            sampling.get("model_hash") == evidence.model_hash == manifest.get("model_hash")
            and sampling.get("config_hash") == evidence.config_hash == manifest.get("config_hash") == config_hash()
            and sampling.get("rules_hash") == manifest.get("rules_hash") == rules_hash()
            and sampling.get("sampling_seed") == evidence.sampling_seed
            and sampling.get("sampling_params_hash") == evidence.sampling_params_hash == manifest.get("sampling_params_hash")
        )

    @staticmethod
    def _online_as_of_matches(report_binding, report) -> bool:
        """在线日线截止日必须与正式评估的截止日相同。"""
        try:
            online = pd.Timestamp(report.prediction["data_provenance"]["as_of"]).normalize()
            evaluated = pd.Timestamp(report_binding.manifest["as_of"]).normalize()
            return online == evaluated
        except (KeyError, TypeError, ValueError):
            return False

    @staticmethod
    def _decide_with_evidence(
        gate,
        evidence,
        risk_blocked: bool = False,
        is_held: bool = False,
    ) -> tuple[RecommendationAction, tuple[str, ...]]:
        """门禁和点时证据都满足时输出五态动作；资格/风险阻断时输出 AVOID。"""
        if not gate.passed or evidence is None:
            return RecommendationAction.INSUFFICIENT_EVIDENCE, ("EVIDENCE_GATE_FAILED",)
        probabilities = {5: evidence.short_probability, 20: evidence.calibrated_up_probability, 60: evidence.long_probability}
        if any(value is None for value in probabilities.values()):
            return RecommendationAction.INSUFFICIENT_EVIDENCE, ("MULTI_HORIZON_CALIBRATION_UNAVAILABLE",)
        ranks = {5: evidence.short_rank if evidence.short_rank is not None else 0.0, 20: evidence.cross_section_rank, 60: evidence.long_rank if evidence.long_rank is not None else 0.0}
        return decide_action(
            True,
            ranks,
            {horizon: float(value) for horizon, value in probabilities.items()},
            is_held,
            risk_blocked,
        )

    @staticmethod
    def _merge_horizons(path_horizons: dict[str, dict[str, Any]], evidence) -> dict[str, dict[str, Any]]:
        """用独立评估档案校准5/20/60日区间与上涨概率。"""
        merged = {horizon: dict(stats) for horizon, stats in path_horizons.items()}
        if evidence is None:
            for stats in merged.values():
                stats["calibration_status"] = "unavailable"
            return merged
        for horizon, item in evidence.horizons.items():
            stats = merged.get(str(horizon))
            if stats is None:
                continue
            stats["cross_section_rank"] = item.cross_section_rank
            stats["anchor_date"] = evidence.anchor_date.isoformat()
            if item.calibrated_up_probability is None:
                stats["calibration_status"] = "unavailable"
                continue
            stats.update({"q05": item.q05, "q50": item.q50, "q95": item.q95, "up_probability": item.calibrated_up_probability, "calibration_status": "calibrated"})
        return merged

    def _run_legacy_report(self, stock_code: str, params: dict, started: float) -> DecisionReport:
        bundle = self._fetcher.fetch_daily_bundle(stock_code, as_of=params.get("as_of"))
        bars = self._clean_bars(bundle.bars)
        pred_len = int(params.get("pred_len", self._pred_cfg.default_pred_len))
        model_params = self._resolve_params(params)
        recent = bars.iloc[-model_params["max_context"]:].copy()
        timestamps = self._calendar.future_sessions(pd.Timestamp(bars["date"].iloc[-1]), pred_len)
        with self._model_manager.lease() as (tokenizer, predictor):
            provenance = model_provenance(getattr(predictor, "model", None), tokenizer)
            config_digest = config_hash()
            requested_count = int(params.get("sample_count", self._pred_cfg.default_sample_count))
            device = str(getattr(predictor, "device", "cpu"))
            reduced_for_cpu = device.startswith("cpu") and requested_count > self._pred_cfg.fallback_sample_count
            sample_count = self._pred_cfg.fallback_sample_count if reduced_for_cpu else requested_count
            batch_size = min(20, sample_count)
            seed = derive_sampling_seed(str(provenance["pipeline_hash"]), stock_code, bundle.as_of, config_digest)
            deadline = time.monotonic() + self._pred_cfg.timeout_seconds
            paths = predictor.predict_paths(
                recent, pd.Series(recent["date"].values), timestamps, pred_len,
                sample_count, batch_size, seed, model_params["temperature"],
                model_params["top_p"], 0, deadline,
            )
        sampling_meta = {"device": device, "reduced_for_cpu": reduced_for_cpu, "requested_sample_count": requested_count, "sample_batch_size": batch_size, "sampling_seed": seed, "model_hash": provenance["pipeline_hash"], "model_provenance": provenance, "config_hash": config_digest, "rules_hash": rules_hash()}
        signal = self._analyzer.analyze(paths.paths, float(bars["close"].iloc[-1]), bars["close"].to_numpy())
        prediction = self._prediction_payload(paths, bundle, bars, bool(params.get("include_display_paths", False)), sampling_meta)
        report = DecisionReport("ok", signal=signal, prediction=prediction, stock_code=stock_code, stock_name=self._get_stock_name(stock_code), elapsed_seconds=round(time.time() - started, 2))
        gate, evidence = self._bind_evidence(EvaluationBinding.load_latest(), report)
        eligibility = evaluate_stock_eligibility(
            bars, self._get_stock_name(stock_code), bundle.as_of
        )
        action, reasons = self._decide_with_evidence(
            gate,
            evidence,
            risk_blocked=not eligibility.eligible,
            is_held=self._is_held("default", stock_code),
        )
        signal.signal = legacy_signal(action)
        signal.signal_reason = self._legacy_reason(action, reasons)
        report.status = "ok" if gate.passed and action is not RecommendationAction.INSUFFICIENT_EVIDENCE else "degraded"
        return report

    def _eligibility_for(self, stock_code: str, as_of: pd.Timestamp | None = None) -> dict[str, Any]:
        """v2 专用：按固定资格规则评估股票是否可增持；失败返回不可用不阻断报告。"""
        try:
            bundle = self._fetcher.fetch_daily_bundle(stock_code, as_of=as_of)
            bars = self._clean_bars(bundle.bars)
            return evaluate_stock_eligibility(bars, self._get_stock_name(stock_code), bundle.as_of).as_dict()
        except Exception:
            return {"eligible": False, "reasons": ["ELIGIBILITY_DATA_UNAVAILABLE"]}

    @staticmethod
    def _clean_bars(bars: pd.DataFrame) -> pd.DataFrame:
        required = ["open", "high", "low", "close"]
        optional = ["volume", "amount"]
        cleaned = bars.copy()
        for column in optional:
            if column not in cleaned:
                cleaned[column] = 0.0
        return cleaned.dropna(subset=required).fillna({column: 0.0 for column in optional})

    def _resolve_params(self, params: dict) -> dict[str, float | int]:
        return {"max_context": int(params.get("max_context", get_config().model.max_context)), "temperature": float(params.get("temperature", self._pred_cfg.default_temperature)), "top_p": float(params.get("top_p", self._pred_cfg.default_top_p))}

    @staticmethod
    def _apply_legacy_evidence_floor(signal: SignalResult, is_stale: bool, path_flags: tuple[str, ...]) -> Literal["ok", "degraded"]:
        if not is_stale and not path_flags:
            return "ok"
        signal.signal = "HOLD"
        signal.signal_reason = "证据不足：数据新鲜度或预测路径质量门禁未通过。"
        return "degraded"

    @staticmethod
    def _has_hard_quality_error(data_provenance: dict, sampling: dict) -> bool:
        """只把陈旧数据、ERROR_* 与路径硬错误视为建议阻断项。"""
        data_flags = tuple(str(flag) for flag in data_provenance.get("quality_flags", ()))
        path_flags = tuple(str(flag) for flag in sampling.get("quality_flags", ()))
        return (
            bool(data_provenance.get("is_stale"))
            or bool(data_provenance.get("has_hard_quality_error"))
            or has_hard_quality_flags(data_flags)
            or "PATH_REPAIR_EXCESSIVE" in path_flags
        )

    @staticmethod
    def _legacy_reason(action: RecommendationAction, reason_codes: tuple[str, ...]) -> str:
        """用固定模板生成兼容 v1 的中文说明。"""
        if action is RecommendationAction.INSUFFICIENT_EVIDENCE:
            return "证据不足：正式评估、版本或数据质量门禁未通过。"
        labels = {
            RecommendationAction.ADD: "正式证据支持增持。",
            RecommendationAction.HOLD: "正式证据通过，但未达到调仓阈值。",
            RecommendationAction.REDUCE: "正式证据支持降低现有持仓。",
            RecommendationAction.AVOID: "正式证据提示回避该股票。",
        }
        return f"{labels[action]} 原因代码：{','.join(reason_codes)}"

    @staticmethod
    def _is_held(portfolio_id: str, stock_code: str) -> bool:
        """从本地组合读取持仓状态；读取失败时按未持有处理。"""
        try:
            portfolio = PortfolioStore().get_portfolio(portfolio_id)
            normalized = str(stock_code).zfill(6)
            return any(
                str(item.get("stock_code", "")).zfill(6) == normalized
                and int(item.get("shares", 0)) > 0
                for item in portfolio.get("holdings", [])
            )
        except Exception:
            return False

    def _prediction_payload(self, paths, bundle, bars: pd.DataFrame, include_display_paths: bool, sampling_meta: dict[str, Any] | None = None) -> dict[str, Any]:
        current = float(bars["close"].iloc[-1])
        sampling = {"sample_count": paths.sample_count, "seed": paths.seed, "model_id": paths.model_id, "tokenizer_id": paths.tokenizer_id, "sampling_params_hash": paths.sampling_params_hash, "repair_ratio": paths.repair_ratio, "quality_flags": list(paths.quality_flags)}
        sampling.update(sampling_meta or {})
        payload = {"pred_df": paths.mean_df.reset_index(names="date").to_dict(orient="records"), "pred_len": len(paths.timestamps), "current_price": current, "horizons": DecisionEngine._path_horizons(paths.paths, current), "data_provenance": {"source": bundle.source, "as_of": bundle.as_of.isoformat(), "is_stale": bundle.is_stale, "quality_flags": list(bundle.quality_flags), "has_hard_quality_error": bool(getattr(bundle, "has_hard_quality_error", False)), "content_hash": bundle.content_hash}, "sampling": sampling}
        payload["chart_data"] = DecisionEngine._chart_payload(bars, paths)
        if include_display_paths:
            payload["display_paths"] = paths.paths[:20].tolist()
        return payload

    @staticmethod
    def _chart_payload(bars: pd.DataFrame, paths: "PredictionPaths") -> dict[str, Any]:
        """构造前端 K 线副图所需的历史与预测序列；分位数逐日从真实 close 路径计算。"""
        history_bars = bars.iloc[-120:]
        history = [
            {
                "timestamp": pd.Timestamp(row["date"]).isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
            for _, row in history_bars.iterrows()
        ]
        close_paths = np.asarray(paths.paths)[:, :, 3]
        q05, q50, q95 = (np.quantile(close_paths, q, axis=0) for q in (0.05, 0.50, 0.95))
        if "close" in paths.mean_df.columns:
            mean_close = paths.mean_df["close"].to_numpy()
        else:
            mean_close = close_paths.mean(axis=0)
        timestamps = [pd.Timestamp(ts).isoformat() for ts in paths.timestamps]
        forecast = {
            "timestamps": timestamps,
            "mean_close": [float(v) for v in mean_close],
            "q05_close": [float(v) for v in q05],
            "q50_close": [float(v) for v in q50],
            "q95_close": [float(v) for v in q95],
        }
        return {"history": history, "forecast": forecast}

    @staticmethod
    def _path_horizons(paths: np.ndarray, current: float) -> dict[str, dict[str, float | None]]:
        result: dict[str, dict[str, float | None]] = {}
        close_paths = paths[:, :, 3]
        price_paths = np.column_stack([np.full(len(paths), current), close_paths])
        running_peak = np.maximum.accumulate(price_paths, axis=1)
        drawdowns = price_paths / running_peak - 1
        for horizon in (5, 20, 60):
            if paths.shape[1] < horizon:
                result[str(horizon)] = {"q05": None, "q50": None, "q95": None, "up_probability": None, "predicted_drawdown": None}
                continue
            returns = close_paths[:, horizon - 1] / current - 1
            result[str(horizon)] = {"q05": float(np.quantile(returns, 0.05)), "q50": float(np.quantile(returns, 0.50)), "q95": float(np.quantile(returns, 0.95)), "up_probability": float(np.mean(returns > 0)), "predicted_drawdown": float(np.median(drawdowns[:, : horizon + 1].min(axis=1))), "calibration_status": "unavailable"}
        return result

    @staticmethod
    def _error_report(stock_code: str, message: str, started: float) -> DecisionReport:
        return DecisionReport("error", message, stock_code=stock_code, elapsed_seconds=round(time.time() - started, 2))

    @staticmethod
    def _get_stock_name(stock_code: str) -> str:
        try:
            from data.pool import get_hs300_pool
            return get_hs300_pool().get(stock_code, "")
        except Exception:
            return ""
