"""决策编排：真实采样路径、数据来源和旧接口兼容。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Mapping

if TYPE_CHECKING:
    from model.prediction import PredictionPaths

import pandas as pd
import numpy as np

from data.calendar import AshareCalendar
from data.contracts import MarketDataBundle, has_hard_quality_flags
from data.eligibility import evaluate_stock_eligibility
from data.fetcher import DataFetcher
from decision.analyzer import SignalAnalyzer, SignalResult
from decision.config import get_config
from decision.errors import KronosError
from decision.model_manager import get_model_manager
from decision.versioning import config_hash, derive_sampling_seed, model_provenance, rules_hash
from decision.v2 import RecommendationAction, decide_action
from evaluation.gates import evaluate_evidence_gate
from evaluation.gates import EvidenceGateResult
from evaluation.binding import EvaluationBinding
from data.market_context import MarketContextProvider
from data.fundamental import FundamentalAdapter
from data.events import EventRiskProvider
from portfolio.store import PortfolioStore
from decision.contracts import (
    ActionPermission,
    DecisionReportContract,
    EvidenceStatus,
    RunStatus,
    safe_legacy_signal,
)
from decision.gates import (
    evaluate_data_gate as evaluate_decision_data_gate,
    evaluate_eligibility_gate,
    evaluate_evidence_gate as evaluate_decision_evidence_gate,
)


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

    def predict_and_analyze(
        self,
        stock_code: str,
        params: dict | None = None,
        *,
        prepared_bundle: MarketDataBundle | None = None,
    ) -> DecisionReport:
        """兼容 v1 的完整预测入口；不再构造人工高斯路径。"""
        started = time.time()
        try:
            return self._run_legacy_report(
                stock_code,
                params or {},
                started,
                prepared_bundle=prepared_bundle,
            )
        except TimeoutError as exc:
            return self._error_report(stock_code, f"PREDICTION_TIMEOUT：{exc}", started)
        except KronosError as exc:
            return self._error_report(stock_code, str(exc), started)
        except Exception as exc:
            return self._error_report(stock_code, f"未知错误：{exc}", started)

    def decision_report_v2(
        self,
        stock_code: str,
        portfolio_id: str = "default",
        include_display_paths: bool = False,
        as_of: str | pd.Timestamp | None = None,
        *,
        prepared_bundle: MarketDataBundle | None = None,
    ) -> dict[str, Any]:
        """返回不允许网页覆写采样参数的 v2 可审计报告。"""
        requested_as_of = self._normalize_as_of(as_of)
        params: dict[str, Any] = {"include_display_paths": include_display_paths}
        if requested_as_of is not None:
            params["as_of"] = requested_as_of
        if prepared_bundle is None:
            report = self.predict_and_analyze(stock_code, params)
        else:
            report = self.predict_and_analyze(
                stock_code,
                params,
                prepared_bundle=prepared_bundle,
            )
        if report.status not in {"ok", "degraded", "error"}:
            return self._rejected_v2_report(stock_code, report, "RUN_STATUS_UNKNOWN", prepared_bundle)
        if report.status == "error" or report.prediction is None:
            return self._rejected_v2_report(stock_code, report, "MODEL_OR_DATA_FAILURE", prepared_bundle)
        if not self._prediction_payload_is_valid(report.prediction):
            return self._rejected_v2_report(stock_code, report, "PREDICTION_PAYLOAD_INVALID", prepared_bundle)
        binding = EvaluationBinding.load_latest()
        gate, evidence = self._bind_evidence(binding, report)
        try:
            report_as_of = pd.Timestamp(report.prediction["data_provenance"]["as_of"]).normalize()
            if pd.isna(report_as_of):
                raise ValueError("NaT")
        except (KeyError, TypeError, ValueError):
            return self._rejected_v2_report(stock_code, report, "DATA_AS_OF_INVALID", prepared_bundle)
        if prepared_bundle is None:
            eligibility = self._eligibility_for(stock_code, report_as_of)
        else:
            eligibility = self._eligibility_for(
                stock_code,
                report_as_of,
                prepared_bundle=prepared_bundle,
            )
        risk_blocked = not bool(eligibility.get("eligible"))
        eligibility_gate = evaluate_eligibility_gate(eligibility)
        is_held = self._is_held(portfolio_id, stock_code)
        prediction = report.prediction
        data_gate = evaluate_decision_data_gate(
            prediction.get("data_provenance"),
            prediction.get("sampling"),
            None,
        )
        evidence_gate = evaluate_decision_evidence_gate(
            evidence,
            research_gate=gate,
            synthetic=bool(prediction.get("sampling", {}).get("synthetic", False)),
            requires_calibration=True,
        )
        signal_available = bool(
            report.signal is not None
            and getattr(report.signal, "signal", None) is not None
            and getattr(report.signal, "analysis_valid", True)
        )
        reasons: tuple[str, ...]
        if not data_gate.passed or not evidence_gate.passed or not eligibility_gate.passed or not signal_available:
            action = RecommendationAction.INSUFFICIENT_EVIDENCE
            reason_list = list(data_gate.failed_codes) + list(evidence_gate.failed_codes) + list(eligibility_gate.failed_codes)
            if not signal_available:
                reason_list.append("SIGNAL_MISSING_OR_INVALID")
            reasons = self._unique_reason_codes(reason_list)
        else:
            action, reasons = self._decide_with_evidence(
                gate, evidence, risk_blocked=risk_blocked, is_held=is_held
            )
        evidence_status = self._structured_evidence_status(
            prediction, data_gate, evidence_gate, gate, action
        )
        if (
            action in {
                RecommendationAction.ADD,
                RecommendationAction.HOLD,
                RecommendationAction.REDUCE,
            }
            and evidence_status is EvidenceStatus.QUALIFIED
            and data_gate.passed
            and evidence_gate.passed
            and eligibility_gate.passed
            and signal_available
            and not risk_blocked
        ):
            permission = ActionPermission.CONDITIONAL_REFERENCE
        else:
            permission = ActionPermission.NONE
        contract = DecisionReportContract(
            run_status=RunStatus.SUCCEEDED,
            evidence_status=evidence_status,
            action_permission=permission,
            action=action,
            reason_codes=reasons,
        )
        horizons = self._merge_horizons(
            prediction.get("horizons", {}),
            evidence,
            synthetic=bool(prediction.get("sampling", {}).get("synthetic", False)),
        )
        payload = {
            "schema_version": "2.0",
            "status": report.status,
            "run_status": contract.run_status.value,
            "evidence_status": contract.evidence_status.value,
            "action_permission": contract.action_permission.value,
            "stock": {"code": stock_code, "name": report.stock_name},
            "data_provenance": prediction["data_provenance"],
            "model_provenance": prediction["sampling"],
            "chart_data": prediction.get("chart_data", {"history": [], "forecast": {}}),
            "horizons": horizons,
            "recommendation": {
                "action": contract.action.value if contract.action is not None else None,
                "reason_codes": list(contract.reason_codes),
                "legacy_signal": safe_legacy_signal(
                    contract.action,
                    contract.evidence_status,
                    contract.action_permission,
                ),
            },
            "evidence_gate": gate.as_dict(),
            "evaluation_id": binding.run_id if binding is not None else "none",
            "eligibility": eligibility,
            "market_context": self._market_context_payload(report_as_of),
            "fundamentals": FundamentalAdapter().snapshot(stock_code, as_of=prediction["data_provenance"].get("as_of")),
            "events": EventRiskProvider().events(stock_code, as_of=report_as_of.isoformat()),
            "portfolio_impact": {
                "portfolio_id": portfolio_id,
                "is_held": is_held,
                "eligible_for_rebalance": bool(contract.action_permission is ActionPermission.CONDITIONAL_REFERENCE),
            },
            "generated_at": report.generated_at,
            "elapsed_seconds": report.elapsed_seconds,
        }
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
        if not bool(getattr(gate, "passed", False)) or evidence is None:
            return RecommendationAction.INSUFFICIENT_EVIDENCE, ("EVIDENCE_GATE_FAILED",)
        try:
            probabilities = {
                5: DecisionEngine._evidence_field(evidence, "short_probability"),
                20: DecisionEngine._evidence_field(evidence, "calibrated_up_probability"),
                60: DecisionEngine._evidence_field(evidence, "long_probability"),
            }
            ranks = {
                5: DecisionEngine._evidence_field(evidence, "short_rank"),
                20: DecisionEngine._evidence_field(evidence, "cross_section_rank"),
                60: DecisionEngine._evidence_field(evidence, "long_rank"),
            }
        except (AttributeError, TypeError):
            return RecommendationAction.INSUFFICIENT_EVIDENCE, ("EVIDENCE_FIELDS_INVALID",)
        if any(not DecisionEngine._finite_unit_interval(value) for value in probabilities.values()):
            return RecommendationAction.INSUFFICIENT_EVIDENCE, ("MULTI_HORIZON_CALIBRATION_UNAVAILABLE",)
        if any(not DecisionEngine._finite_unit_interval(value) for value in ranks.values()):
            return RecommendationAction.INSUFFICIENT_EVIDENCE, ("EVIDENCE_RANK_INVALID",)
        try:
            return decide_action(
                True,
                {horizon: float(value) for horizon, value in ranks.items()},
                {horizon: float(value) for horizon, value in probabilities.items()},
                is_held,
                risk_blocked,
            )
        except (KeyError, TypeError, ValueError):
            return RecommendationAction.INSUFFICIENT_EVIDENCE, ("ACTION_RULE_INPUT_INVALID",)

    @staticmethod
    def _evidence_field(evidence: Any, key: str, default: Any = None) -> Any:
        """从映射或评估对象读取证据字段。"""
        if isinstance(evidence, Mapping):
            return evidence.get(key, default)
        return getattr(evidence, key, default)

    @staticmethod
    def _unique_reason_codes(codes) -> tuple[str, ...]:
        """去重并稳定保留原因码顺序。"""
        result: list[str] = []
        for code in codes:
            value = str(code)
            if value and value not in result:
                result.append(value)
        return tuple(result)

    @staticmethod
    def _finite_unit_interval(value: Any) -> bool:
        """安全判断一个证据数值是否为有限的 [0, 1] 数。"""
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return False
        return bool(np.isfinite(number) and 0.0 <= number <= 1.0)

    @staticmethod
    def _finite_float(value: Any) -> bool:
        """安全判断一个报告数值是否为有限浮点数。"""
        try:
            return bool(np.isfinite(float(value)))
        except (TypeError, ValueError, OverflowError):
            return False

    @staticmethod
    def _contains_nonfinite(value: Any) -> bool:
        """递归检测报告载荷中的 NaN/Infinity。"""
        if value is None or isinstance(value, (str, bytes, bool, pd.Timestamp)):
            return False
        if isinstance(value, Mapping):
            return any(DecisionEngine._contains_nonfinite(item) for item in value.values())
        if isinstance(value, (list, tuple, set)):
            return any(DecisionEngine._contains_nonfinite(item) for item in value)
        if isinstance(value, np.ndarray):
            try:
                return not bool(np.isfinite(value.astype(float)).all())
            except (TypeError, ValueError):
                return True
        if isinstance(value, (int, float, np.number)):
            return not bool(np.isfinite(value))
        return False

    @staticmethod
    def _prediction_payload_is_valid(prediction: Any) -> bool:
        """判断报告预测载荷存在且没有 NaN；宽松兼容旧测试夹具的可选字段。"""
        if not isinstance(prediction, Mapping):
            return False
        if DecisionEngine._contains_nonfinite(prediction):
            return False
        provenance = prediction.get("data_provenance")
        sampling = prediction.get("sampling")
        if not isinstance(provenance, Mapping) or not isinstance(sampling, Mapping):
            return False
        return True

    @staticmethod
    def _structured_evidence_status(
        prediction: Mapping[str, Any],
        data_gate,
        evidence_gate,
        research_gate,
        action: RecommendationAction,
    ) -> EvidenceStatus:
        """把旧评估门禁和新输入门禁合成为稳定证据状态。"""
        provenance = prediction.get("data_provenance") or {}
        if bool(provenance.get("is_stale")):
            return EvidenceStatus.STALE
        if not data_gate.passed or evidence_gate.evidence_status is EvidenceStatus.INVALID:
            return EvidenceStatus.INVALID if not data_gate.passed else evidence_gate.evidence_status
        if action is RecommendationAction.INSUFFICIENT_EVIDENCE or not bool(getattr(research_gate, "passed", False)):
            return EvidenceStatus.INSUFFICIENT
        return EvidenceStatus.QUALIFIED

    @staticmethod
    def _merge_horizons(
        path_horizons: dict[str, dict[str, Any]], evidence, *, synthetic: bool = False
    ) -> dict[str, dict[str, Any]]:
        """用独立评估档案校准5/20/60日区间与上涨概率。"""
        source = path_horizons if isinstance(path_horizons, Mapping) else {}
        merged = {horizon: dict(stats) for horizon, stats in source.items() if isinstance(stats, Mapping)}
        if evidence is None or synthetic:
            for stats in merged.values():
                stats["calibration_status"] = "unavailable"
                stats["calibrated_up_probability"] = None
            return merged
        horizon_items = (
            evidence.get("horizons", {})
            if isinstance(evidence, Mapping)
            else getattr(evidence, "horizons", {})
        ) or {}
        if not isinstance(horizon_items, Mapping):
            for stats in merged.values():
                stats["calibration_status"] = "unavailable"
                stats["calibrated_up_probability"] = None
            return merged
        for horizon, item in horizon_items.items():
            stats = merged.get(str(horizon))
            if stats is None:
                continue
            try:
                item_value = lambda key: item.get(key) if isinstance(item, Mapping) else getattr(item, key)
                anchor_value = evidence.get("anchor_date") if isinstance(evidence, Mapping) else evidence.anchor_date
                anchor_date = anchor_value.isoformat() if hasattr(anchor_value, "isoformat") else str(anchor_value)
                rank = item_value("cross_section_rank")
                calibrated = item_value("calibrated_up_probability")
                interval = (item_value("q05"), item_value("q50"), item_value("q95"))
            except (AttributeError, TypeError, ValueError):
                stats["calibration_status"] = "unavailable"
                stats["calibrated_up_probability"] = None
                continue
            if (
                not DecisionEngine._finite_unit_interval(rank)
                or not DecisionEngine._finite_unit_interval(calibrated)
                or any(not DecisionEngine._finite_float(value) for value in interval)
            ):
                stats["calibration_status"] = "unavailable"
                stats["calibrated_up_probability"] = None
                continue
            stats["cross_section_rank"] = rank
            stats["anchor_date"] = anchor_date
            stats.update({"q05": interval[0], "q50": interval[1], "q95": interval[2], "up_probability": calibrated, "calibrated_up_probability": calibrated, "calibration_status": "calibrated"})
        return merged

    def _run_legacy_report(
        self,
        stock_code: str,
        params: dict,
        started: float,
        *,
        prepared_bundle: MarketDataBundle | None = None,
    ) -> DecisionReport:
        bundle = prepared_bundle or self._fetcher.fetch_daily_bundle(
            stock_code,
            as_of=params.get("as_of"),
        )
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
        eligibility_gate = evaluate_eligibility_gate(eligibility)
        if not eligibility_gate.passed:
            action = RecommendationAction.INSUFFICIENT_EVIDENCE
            reasons = eligibility_gate.failed_codes
        else:
            action, reasons = self._decide_with_evidence(
                gate,
                evidence,
                risk_blocked=False,
                is_held=self._is_held("default", stock_code),
            )
        if not getattr(signal, "analysis_valid", True):
            action = RecommendationAction.INSUFFICIENT_EVIDENCE
            reasons = self._unique_reason_codes((*reasons, *getattr(signal, "reason_codes", ())))
        referenceable = (
            action in {
                RecommendationAction.ADD,
                RecommendationAction.HOLD,
                RecommendationAction.REDUCE,
            }
            and bool(getattr(gate, "passed", False))
            and evidence is not None
            and bool(getattr(eligibility_gate, "passed", False))
            and bool(getattr(signal, "analysis_valid", True))
        )
        signal.signal = safe_legacy_signal(
            action,
            EvidenceStatus.QUALIFIED if referenceable else EvidenceStatus.INSUFFICIENT,
            ActionPermission.CONDITIONAL_REFERENCE if referenceable else ActionPermission.NONE,
        )
        signal.signal_reason = self._legacy_reason(action, reasons)
        report.status = "ok" if gate.passed and action is not RecommendationAction.INSUFFICIENT_EVIDENCE else "degraded"
        return report

    def _eligibility_for(
        self,
        stock_code: str,
        as_of: pd.Timestamp | None = None,
        *,
        prepared_bundle: MarketDataBundle | None = None,
    ) -> dict[str, Any]:
        """v2 专用：按固定资格规则评估股票是否可增持；失败返回不可用不阻断报告。"""
        try:
            bundle = prepared_bundle or self._fetcher.fetch_daily_bundle(stock_code, as_of=as_of)
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
        signal.signal = None
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
        if not self._prediction_paths_are_valid(paths):
            raise KronosError("预测路径为空或包含非有限值。")
        current = float(bars["close"].iloc[-1])
        sampling = {"sample_count": paths.sample_count, "seed": paths.seed, "model_id": paths.model_id, "tokenizer_id": paths.tokenizer_id, "sampling_params_hash": paths.sampling_params_hash, "repair_ratio": paths.repair_ratio, "quality_flags": list(paths.quality_flags), "paths_valid": True, "paths_finite": True, "synthetic": bool(getattr(paths, "synthetic", False)), "calibration_status": "unavailable"}
        sampling.update(sampling_meta or {})
        fallback_chain = tuple(getattr(bundle, "fallback_chain", ()))
        payload = {"pred_df": paths.mean_df.reset_index(names="date").to_dict(orient="records"), "pred_len": len(paths.timestamps), "current_price": current, "horizons": DecisionEngine._path_horizons(paths.paths, current), "data_provenance": {"source": self._stable_data_source(bundle), "retrieval_source": bundle.source, "fallback_chain": list(fallback_chain), "as_of": bundle.as_of.isoformat(), "is_stale": bundle.is_stale, "quality_flags": list(bundle.quality_flags), "has_hard_quality_error": bool(getattr(bundle, "has_hard_quality_error", False)), "content_hash": bundle.content_hash, "snapshot_id": getattr(bundle, "snapshot_id", ""), "universe_version": getattr(bundle, "universe_version", None)}, "sampling": sampling}
        payload["chart_data"] = DecisionEngine._chart_payload(bars, paths)
        if include_display_paths:
            payload["display_paths"] = paths.paths[:20].tolist()
        return payload

    @staticmethod
    def _stable_data_source(bundle: MarketDataBundle) -> str:
        """返回不因读取缓存而变化的原始数据源标识。"""
        source = str(getattr(bundle, "source", "unknown"))
        if source != "csv_cache":
            return source
        fallback_chain = tuple(getattr(bundle, "fallback_chain", ()))
        candidates = [item for item in fallback_chain if item != "csv_cache"]
        return candidates[-1] if candidates else "csv_cache"

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
        empty = {
            str(horizon): {
                "q05": None,
                "q50": None,
                "q95": None,
                "up_probability": None,
                "raw_up_probability": None,
                "calibrated_up_probability": None,
                "predicted_drawdown": None,
                "calibration_status": "unavailable",
            }
            for horizon in (5, 20, 60)
        }
        try:
            array = np.asarray(paths, dtype=float)
        except (TypeError, ValueError):
            return empty
        if (
            array.ndim < 3
            or array.shape[0] == 0
            or array.shape[1] == 0
            or array.shape[-1] <= 3
            or not np.isfinite(array).all()
            or not np.isfinite(current)
            or current <= 0
        ):
            return empty
        close_paths = array[:, :, 3]
        if np.any(close_paths <= 0):
            return empty
        price_paths = np.column_stack([np.full(array.shape[0], current), close_paths])
        running_peak = np.maximum.accumulate(price_paths, axis=1)
        drawdowns = price_paths / running_peak - 1
        for horizon in (5, 20, 60):
            if array.shape[1] < horizon:
                result[str(horizon)] = dict(empty[str(horizon)])
                continue
            returns = close_paths[:, horizon - 1] / current - 1
            raw_probability = float(np.mean(returns > 0))
            result[str(horizon)] = {"q05": float(np.quantile(returns, 0.05)), "q50": float(np.quantile(returns, 0.50)), "q95": float(np.quantile(returns, 0.95)), "up_probability": raw_probability, "raw_up_probability": raw_probability, "calibrated_up_probability": None, "predicted_drawdown": float(np.median(drawdowns[:, : horizon + 1].min(axis=1))), "calibration_status": "unavailable"}
        return result

    @staticmethod
    def _error_report(stock_code: str, message: str, started: float) -> DecisionReport:
        return DecisionReport("error", message, stock_code=stock_code, elapsed_seconds=round(time.time() - started, 2))

    @staticmethod
    def _prediction_paths_are_valid(paths: Any) -> bool:
        """判断模型路径具备最小非空、有限数值结构。"""
        try:
            array = np.asarray(paths.paths, dtype=float)
        except (AttributeError, TypeError, ValueError):
            return False
        return bool(
            array.ndim >= 3
            and array.shape[0] > 0
            and array.shape[1] > 0
            and array.shape[-1] > 3
            and np.isfinite(array).all()
        )

    @staticmethod
    def _rejected_v2_report(
        stock_code: str,
        report: DecisionReport,
        reason_code: str,
        prepared_bundle: MarketDataBundle | None = None,
    ) -> dict[str, Any]:
        """把模型/数据失败封装为无动作的稳定 v2 报告，不把异常升级为旧信号。"""
        prediction = report.prediction if isinstance(report.prediction, Mapping) else {}
        provenance = prediction.get("data_provenance") if isinstance(prediction.get("data_provenance"), Mapping) else {}
        sampling = prediction.get("sampling") if isinstance(prediction.get("sampling"), Mapping) else {}
        if DecisionEngine._contains_nonfinite(provenance):
            provenance = {}
        if DecisionEngine._contains_nonfinite(sampling):
            sampling = {}
        if prepared_bundle is not None and not provenance:
            provenance = {
                "snapshot_id": prepared_bundle.snapshot_id,
                "as_of": prepared_bundle.as_of.isoformat(),
                "source": DecisionEngine._stable_data_source(prepared_bundle),
                "retrieval_source": prepared_bundle.source,
                "is_stale": prepared_bundle.is_stale,
                "quality_flags": list(prepared_bundle.quality_flags),
                "content_hash": prepared_bundle.content_hash,
                "universe_version": prepared_bundle.universe_version,
            }
        run_status = RunStatus.TIMED_OUT if "TIMEOUT" in str(report.error_message or "") else RunStatus.FAILED
        evidence_status = EvidenceStatus.STALE if provenance.get("is_stale") is True else EvidenceStatus.INVALID
        contract = DecisionReportContract(
            run_status=run_status,
            evidence_status=evidence_status,
            action_permission=ActionPermission.NONE,
            action=None,
            reason_codes=(reason_code,),
        )
        return {
            "schema_version": "2.0",
            "status": report.status,
            "run_status": contract.run_status.value,
            "evidence_status": contract.evidence_status.value,
            "action_permission": contract.action_permission.value,
            "stock": {"code": stock_code, "name": report.stock_name},
            "data_provenance": dict(provenance),
            "model_provenance": dict(sampling),
            "chart_data": {"history": [], "forecast": {"timestamps": [], "mean_close": [], "q05_close": [], "q50_close": [], "q95_close": []}},
            "horizons": {},
            "recommendation": {
                "action": None,
                "reason_codes": list(contract.reason_codes),
                "legacy_signal": safe_legacy_signal(
                    None,
                    contract.evidence_status,
                    contract.action_permission,
                ),
            },
            "evidence_gate": {"passed": False, "failed_codes": [reason_code], "checks": {reason_code: False}},
            "eligibility": {"eligible": False, "reasons": [reason_code]},
            "market_context": {"available": False, "reason": reason_code},
            "fundamentals": {},
            "events": [],
            "portfolio_impact": {"portfolio_id": "default", "is_held": False, "eligible_for_rebalance": False},
            "generated_at": report.generated_at,
            "elapsed_seconds": report.elapsed_seconds,
        }

    @staticmethod
    def _get_stock_name(stock_code: str) -> str:
        try:
            from data.pool import get_hs300_pool
            return get_hs300_pool().get(stock_code, "")
        except Exception:
            return ""
