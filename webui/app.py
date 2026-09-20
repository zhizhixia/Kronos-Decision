import os
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, render_template, request, jsonify, send_file
import sys
import warnings
import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
warnings.filterwarnings('ignore')

# 本地 WebUI 只接受默认启动端口上的 HTTP 回环 Origin。
_LOCAL_ORIGIN_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_LOCAL_ORIGIN_SCHEME = "http"
_LOCAL_ORIGIN_PORT = 7070

# Add project root directory to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: Kronos model cannot be imported, will use simulated data for demonstration")

app = Flask(__name__)


def _is_allowed_local_origin(origin: str | None) -> bool:
    """严格校验本机 Origin 的 scheme、主机名、端口和 URL 结构。"""
    if not isinstance(origin, str) or not origin:
        return False
    try:
        parsed = urlsplit(origin)
        hostname = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme == _LOCAL_ORIGIN_SCHEME
        and hostname in _LOCAL_ORIGIN_HOSTS
        and port == _LOCAL_ORIGIN_PORT
        and parsed.username is None
        and parsed.password is None
        and parsed.path == ""
        and parsed.query == ""
        and parsed.fragment == ""
    )


def _origin_error(code: str, message: str) -> Any:
    """返回保持 JSON 结构的 Origin 拒绝响应。"""
    error = {"code": code, "message": message, "retryable": False, "details": {}}
    return jsonify({"status": "error", "error": error}), 403


@app.before_request
def reject_foreign_write_origin() -> Any | None:
    """拒绝缺少或不符合本机约束的写请求，避免本地 API 被跨站调用。"""
    if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
        return None
    origin = request.headers.get("Origin")
    if not origin:
        return _origin_error("ORIGIN_REQUIRED", "写请求必须携带本机 Origin。")
    if _is_allowed_local_origin(origin):
        return None
    try:
        hostname = (urlsplit(origin).hostname or "").lower()
    except ValueError:
        hostname = ""
    if hostname in _LOCAL_ORIGIN_HOSTS:
        return _origin_error("INVALID_LOCAL_ORIGIN", "本机 Origin 的 scheme 或端口无效。")
    return _origin_error("FOREIGN_ORIGIN_BLOCKED", "仅允许本机页面调用写接口。")


@app.after_request
def add_security_headers(response: Any) -> Any:
    """为 HTML、JSON 和错误响应统一添加安全响应头。"""
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; base-uri 'self'; object-src 'none'; "
        "frame-ancestors 'none'; script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
        "connect-src 'self'; font-src 'self'",
    )
    return response

# Global variables to store models
tokenizer = None
model = None
predictor = None

# 报告接口的安全状态白名单。旧接口只能在完整成功路径上保留三态信号。
_REPORT_RUN_STATUSES = frozenset({
    "QUEUED", "RUNNING", "CANCELLING", "SUCCEEDED", "FAILED", "CANCELLED", "TIMED_OUT",
})
_REPORT_EVIDENCE_STATUSES = frozenset({
    "RESEARCH_ONLY", "QUALIFIED", "INSUFFICIENT", "STALE", "INVALID",
})
_REPORT_ACTION_PERMISSIONS = frozenset({"NONE", "CONDITIONAL_REFERENCE", "INSUFFICIENT_EVIDENCE"})
_V2_ACTIONS = frozenset({"ADD", "HOLD", "REDUCE", "AVOID", "INSUFFICIENT_EVIDENCE"})
_LEGACY_SIGNALS = frozenset({"BUY", "HOLD", "SELL"})
_V2_LEGACY_SIGNALS = {"ADD": "BUY", "HOLD": "HOLD", "REDUCE": "SELL", "AVOID": "SELL"}


def _get_decision_engine_class():
    """延迟加载决策引擎，便于 API 层在无模型依赖时保持可测试。"""
    from decision.engine import DecisionEngine

    return DecisionEngine


def _get_report_pipeline_class():
    """延迟加载报告流水线，确保 API 不绕过快照和持久化边界。"""
    from decision.report_pipeline import DecisionReportPipeline

    return DecisionReportPipeline


def _get_job_service():
    """按需创建有界任务服务，不在 WebUI 导入时启动执行器。"""
    service = app.extensions.get("kronos_decision_job_service")
    if service is None:
        from webui.job_service import DecisionJobService

        service = DecisionJobService(lambda: _get_report_pipeline_class()())
        app.extensions["kronos_decision_job_service"] = service
    return service


def _report_contract_fields(
    run_status: str,
    evidence_status: str,
    action_permission: str,
) -> dict[str, str]:
    """构造所有报告/预测接口都使用的稳定状态字段。"""
    return {
        "run_status": run_status,
        "evidence_status": evidence_status,
        "action_permission": action_permission,
    }


def _report_error_response(
    code: str,
    message: str,
    http_status: int,
    retryable: bool = False,
    *,
    run_status: str = "FAILED",
    evidence_status: str = "INVALID",
):
    """返回不含异常详情、绝对路径或凭据的中文错误报告。"""
    from decision.v2 import error_payload

    body = error_payload(code, message, retryable)
    body.update(_report_contract_fields(run_status, evidence_status, "NONE"))
    body["error_message"] = message
    body["action"] = None
    body["recommendation"] = {
        "action": "INSUFFICIENT_EVIDENCE",
        "legacy_signal": None,
        "reason_codes": [code],
    }
    return jsonify(body), http_status


def _safe_reason_codes(value: Any) -> list[str]:
    """只保留短的机器原因码，避免把异常文本带回页面。"""
    if not isinstance(value, (list, tuple, set)):
        return []
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not 0 < len(item) <= 80:
            continue
        if not all(char.isalnum() or char in {"_", "-", ":"} for char in item):
            continue
        if item not in result:
            result.append(item)
    return result


def _quality_flags(*sources: Any) -> list[str]:
    """提取质量标记；只用于门禁，不把原始错误文本返回给客户端。"""
    flags: list[str] = []
    for source in sources:
        if isinstance(source, dict):
            source = source.get("quality_flags", [])
        if isinstance(source, (list, tuple, set)):
            flags.extend(str(flag).upper() for flag in source)
    return flags


def _has_hard_report_error(data_provenance: Any, model_provenance: Any) -> bool:
    """识别会阻断动作的质量错误。"""
    data = data_provenance if isinstance(data_provenance, dict) else {}
    model = model_provenance if isinstance(model_provenance, dict) else {}
    flags = _quality_flags(data, model)
    return bool(
        data.get("has_hard_quality_error")
        or str(data.get("status", "")).upper() in {"ERROR", "INVALID", "FAILED"}
        or str(model.get("status", "")).upper() in {"ERROR", "INVALID", "FAILED"}
        or any(
            flag.startswith("ERROR")
            or flag in {"PATH_REPAIR_EXCESSIVE", "INVALID", "FAILED"}
            for flag in flags
        )
    )


def _is_report_stale(data_provenance: Any, explicit_evidence: Any = None) -> bool:
    """识别过期数据，过期数据永远不能开放动作权限。"""
    data = data_provenance if isinstance(data_provenance, dict) else {}
    freshness = str(data.get("freshness_status", data.get("freshness", ""))).upper()
    return bool(data.get("is_stale")) or freshness in {"STALE", "EXPIRED"} or explicit_evidence == "STALE"


def _value_has_items(value: Any) -> bool:
    """判断预测路径/序列是否真的包含数据，而不是只包含字段名。"""
    if value is None:
        return False
    if isinstance(value, dict):
        for nested_key in ("data", "records", "timestamps", "paths", "values"):
            if nested_key in value:
                return _value_has_items(value[nested_key])
        return bool(value)
    if isinstance(value, (list, tuple, set)):
        return bool(value)
    return True


def _prediction_path_state(payload: Any, keys: tuple[str, ...]) -> tuple[bool, bool]:
    """返回（路径可用，是否明确出现了空路径）而不猜测缺失数据。"""
    if not isinstance(payload, dict):
        return False, False
    for key in keys:
        if key in payload:
            value = payload.get(key)
            available = _value_has_items(value)
            return available, not available
    chart_data = payload.get("chart_data")
    if isinstance(chart_data, dict) and "forecast" in chart_data:
        forecast = chart_data.get("forecast")
        available = _value_has_items(forecast)
        return available, not available
    return False, False


def _normalise_v2_report(report: Any) -> tuple[dict[str, Any] | None, tuple[str, str, int] | None]:
    """把 v2 引擎结果收敛为安全契约，失败时返回稳定错误码。"""
    if not isinstance(report, dict):
        return None, ("INVALID_REPORT", "报告响应格式无效。", 502)

    raw_run_status = report.get("run_status")
    raw_status = report.get("status")
    raw_status_value = str(raw_status).upper() if raw_status is not None else ""
    status_value = str(raw_run_status if raw_run_status is not None else raw_status or "").upper()
    if raw_run_status is not None:
        if raw_status_value in {"ERROR", "FAILED", "CANCELLED", "TIMED_OUT"}:
            return None, ("REPORT_FAILED", "报告运行未成功，无法提供动作参考。", 503)
        if status_value != "SUCCEEDED":
            if status_value in {"FAILED", "CANCELLED", "TIMED_OUT"}:
                return None, ("REPORT_FAILED", "报告运行未成功，无法提供动作参考。", 503)
            return None, ("UNKNOWN_RUN_STATUS", "报告运行状态未知，已拒绝动作参考。", 502)
    elif status_value in {"OK", "SUCCESS", "SUCCEEDED", "DEGRADED"}:
        status_value = "SUCCEEDED"
    elif status_value in {"FAILED", "ERROR", "CANCELLED", "TIMED_OUT"}:
        return None, ("REPORT_FAILED", "报告运行未成功，无法提供动作参考。", 503)
    elif status_value in {"STALE", "EXPIRED"}:
        status_value = "SUCCEEDED"
    else:
        return None, ("UNKNOWN_RUN_STATUS", "报告运行状态未知，已拒绝动作参考。", 502)

    evidence_raw = report.get("evidence_status")
    evidence_value = str(evidence_raw).upper() if evidence_raw is not None else ""
    if evidence_value and evidence_value not in _REPORT_EVIDENCE_STATUSES:
        return None, ("UNKNOWN_EVIDENCE_STATUS", "报告证据状态未知，已拒绝动作参考。", 502)

    permission_raw = report.get("action_permission")
    permission_value = str(permission_raw).upper() if permission_raw is not None else ""
    if permission_value and permission_value not in _REPORT_ACTION_PERMISSIONS:
        return None, ("UNKNOWN_ACTION_PERMISSION", "报告动作权限未知，已拒绝动作参考。", 502)

    recommendation = report.get("recommendation")
    recommendation = dict(recommendation) if isinstance(recommendation, dict) else {}
    raw_action = recommendation.get("action")
    action = str(raw_action).upper() if raw_action is not None else "INSUFFICIENT_EVIDENCE"
    if action not in _V2_ACTIONS:
        return None, ("UNKNOWN_ACTION", "报告动作状态未知，已拒绝动作参考。", 422)

    data_provenance = report.get("data_provenance")
    model_provenance = report.get("model_provenance")
    stale = _is_report_stale(data_provenance, evidence_value) or raw_status_value in {"STALE", "EXPIRED"}
    hard_error = _has_hard_report_error(data_provenance, model_provenance)
    degraded = raw_status_value == "DEGRADED"
    gate = report.get("evidence_gate") if isinstance(report.get("evidence_gate"), dict) else {}
    gate_passed = gate.get("passed") is True
    path_available, path_empty = _prediction_path_state(
        report,
        ("prediction_paths", "paths", "samples", "forecast_paths"),
    )
    if isinstance(report.get("prediction"), dict):
        nested_available, nested_empty = _prediction_path_state(
            report["prediction"],
            ("prediction_paths", "paths", "samples", "forecast_paths", "pred_df"),
        )
        path_available = path_available or nested_available
        path_empty = path_empty or nested_empty

    reason_codes = _safe_reason_codes(recommendation.get("reason_codes"))
    blocked_reason = ""
    derived_evidence = "QUALIFIED"
    if stale:
        blocked_reason = "STALE_DATA"
        derived_evidence = "STALE"
    elif hard_error:
        blocked_reason = "EVIDENCE_QUALITY_ERROR"
        derived_evidence = "INVALID"
    elif degraded:
        blocked_reason = "INSUFFICIENT_EVIDENCE"
        derived_evidence = "INSUFFICIENT"
    elif (
        action == "INSUFFICIENT_EVIDENCE"
        or not gate_passed
        or evidence_value != "QUALIFIED"
        or permission_value != "CONDITIONAL_REFERENCE"
    ):
        blocked_reason = "INSUFFICIENT_EVIDENCE"
        derived_evidence = "INSUFFICIENT"
    elif path_empty or not path_available:
        blocked_reason = "EMPTY_PREDICTION_PATH"
        derived_evidence = "INVALID"

    if blocked_reason:
        if blocked_reason not in reason_codes:
            reason_codes.append(blocked_reason)
        recommendation["action"] = "INSUFFICIENT_EVIDENCE"
        recommendation["legacy_signal"] = None
        recommendation["reason_codes"] = reason_codes
        report["action"] = None
        report.update(_report_contract_fields(status_value, derived_evidence, "NONE"))
        report["recommendation"] = recommendation
        report["error_message"] = "证据不足：当前报告不会生成买入、持有或卖出动作参考。"
        return report, None

    recommendation["action"] = action
    recommendation["legacy_signal"] = _V2_LEGACY_SIGNALS[action]
    recommendation["reason_codes"] = reason_codes
    report["action"] = action
    report.update(_report_contract_fields(status_value, "QUALIFIED", "CONDITIONAL_REFERENCE"))
    report["recommendation"] = recommendation
    report["error_message"] = ""
    return report, None


def _normalise_legacy_report(report: Any) -> tuple[dict[str, Any] | None, tuple[str, str, int] | None]:
    """把旧 DecisionReport 接到统一契约，阻断旧页面的默认三态回退。"""
    import dataclasses

    if dataclasses.is_dataclass(report):
        payload = dataclasses.asdict(report)
    elif isinstance(report, dict):
        payload = dict(report)
    else:
        return None, ("INVALID_REPORT", "报告响应格式无效。", 502)

    raw_status = str(payload.get("status", "")).upper()
    if raw_status in {"ERROR", "FAILED", "CANCELLED", "TIMED_OUT"}:
        return None, ("REPORT_FAILED", "报告运行未成功，无法提供动作参考。", 503)
    if raw_status not in {"OK", "DEGRADED"}:
        return None, ("UNKNOWN_RUN_STATUS", "报告运行状态未知，已拒绝动作参考。", 502)

    signal = payload.get("signal") if isinstance(payload.get("signal"), dict) else {}
    raw_signal = str(signal.get("signal", "")).upper()
    prediction = payload.get("prediction")
    data_provenance = prediction.get("data_provenance") if isinstance(prediction, dict) else None
    model_provenance = prediction.get("sampling") if isinstance(prediction, dict) else None
    stale = _is_report_stale(data_provenance)
    hard_error = _has_hard_report_error(data_provenance, model_provenance)
    path_available, path_empty = _prediction_path_state(
        prediction,
        ("pred_df", "prediction_results", "prediction_paths", "paths", "samples"),
    )

    blocked_reason = ""
    evidence_status = "QUALIFIED"
    if raw_status == "DEGRADED":
        blocked_reason = "INSUFFICIENT_EVIDENCE"
        evidence_status = "INSUFFICIENT"
    elif stale:
        blocked_reason = "STALE_DATA"
        evidence_status = "STALE"
    elif hard_error:
        blocked_reason = "EVIDENCE_QUALITY_ERROR"
        evidence_status = "INVALID"
    elif raw_signal not in _LEGACY_SIGNALS:
        blocked_reason = "UNKNOWN_SIGNAL"
        evidence_status = "INVALID"
    elif path_empty or not path_available:
        blocked_reason = "EMPTY_PREDICTION_PATH"
        evidence_status = "INVALID"

    if blocked_reason:
        payload["status"] = "degraded"
        payload["signal"] = None
        payload["action"] = None
        payload["recommendation"] = {
            "action": "INSUFFICIENT_EVIDENCE",
            "legacy_signal": None,
            "reason_codes": [blocked_reason],
        }
        payload.update(_report_contract_fields("SUCCEEDED", evidence_status, "NONE"))
        payload["error_message"] = "证据不足：当前报告不会生成买入、持有或卖出动作参考。"
        return payload, None

    payload["action"] = raw_signal
    payload["recommendation"] = {
        "action": raw_signal,
        "legacy_signal": raw_signal,
        "reason_codes": [],
    }
    payload.update(_report_contract_fields("SUCCEEDED", "QUALIFIED", "CONDITIONAL_REFERENCE"))
    payload["error_message"] = ""
    return payload, None

# Available model configurations
AVAILABLE_MODELS = {
    'kronos-mini': {
        'name': 'Kronos-mini',
        'model_id': 'NeoQuasar/Kronos-mini',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
        'context_length': 2048,
        'params': '4.1M',
        'description': 'Lightweight model, suitable for fast prediction'
    },
    'kronos-small': {
        'name': 'Kronos-small',
        'model_id': 'NeoQuasar/Kronos-small',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '24.7M',
        'description': 'Small model, balanced performance and speed'
    },
    'kronos-base': {
        'name': 'Kronos-base',
        'model_id': 'NeoQuasar/Kronos-base',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '102.3M',
        'description': 'Base model, provides better prediction quality'
    }
}

def load_data_files() -> list[dict[str, str]]:
    """扫描数据目录，只返回安全的相对文件名和 ID，不暴露绝对路径。"""
    data_files: list[dict[str, str]] = []
    if DATA_DIRECTORY.exists():
        for file_path in sorted(DATA_DIRECTORY.glob("*.csv")):
            if file_path.is_file():
                file_size = file_path.stat().st_size
                data_files.append({
                    "id": file_path.name,
                    "file_name": file_path.name,
                    "name": file_path.name,
                    "path": file_path.name,
                    "size": f"{file_size / 1024:.1f} KB" if file_size < 1024 * 1024 else f"{file_size / (1024 * 1024):.1f} MB",
                })
    return data_files


def _resolve_request_file(data: dict[str, Any]) -> tuple[str | None, str | None]:
    """将请求中的文件名/ID映射到服务端路径，拒绝伪造的文件名。"""
    for key in ("file_id", "file_name", "filename"):
        if key not in data:
            continue
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            return None, "INVALID_DATA_FILE_REFERENCE"
        reference = value.strip()
        if Path(reference).name != reference or reference in {".", ".."}:
            return None, "INVALID_DATA_FILE_REFERENCE"
        return str(DATA_DIRECTORY / reference), None

    value = data.get("file_path")
    if not isinstance(value, str) or not value.strip():
        return None, "EMPTY_DATA_PATH"
    reference = value.strip()
    if not Path(reference).is_absolute() and Path(reference).name == reference:
        return str(DATA_DIRECTORY / reference), None
    return reference, None


def load_data_file(file_path: str) -> tuple[pd.DataFrame | None, str | None]:
    """读取数据文件并要求文件提供真实的时间列。"""
    try:
        allowed = DATA_DIRECTORY.resolve(strict=True)
        candidate = Path(file_path).resolve(strict=True)
        if not candidate.is_relative_to(allowed):
            return None, "DATA_FILE_OUTSIDE_ALLOWED_DIRECTORY"
        if candidate.suffix.lower() != ".csv" or not candidate.is_file():
            return None, "UNSUPPORTED_DATA_FILE"
        df = pd.read_csv(candidate)

        required_cols = ["open", "high", "low", "close"]
        if not all(col in df.columns for col in required_cols):
            return None, f"Missing required columns: {required_cols}"

        timestamp_column = next(
            (column for column in ("timestamps", "timestamp", "date") if column in df.columns),
            None,
        )
        if timestamp_column is None:
            return None, "MISSING_TIMESTAMP"
        try:
            timestamps = pd.to_datetime(df[timestamp_column], errors="raise")
        except (TypeError, ValueError, OverflowError, pd.errors.ParserError):
            return None, "INVALID_TIMESTAMP"
        if timestamps.isna().any():
            return None, "INVALID_TIMESTAMP"
        df["timestamps"] = timestamps

        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        if "amount" in df.columns:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

        df = df.dropna()
        if df.empty:
            return None, "EMPTY_DATA"
        return df, None
    except (OSError, TypeError, ValueError, UnicodeError, pd.errors.ParserError) as exc:
        return None, f"Failed to load file: {exc}"


def _future_timestamps(frame: pd.DataFrame, periods: int) -> pd.DatetimeIndex:
    """按文件自身频率生成最后可见时间点之后的时间戳。"""
    values = pd.DatetimeIndex(pd.to_datetime(frame["timestamps"])).sort_values()
    if len(values) < 2:
        return pd.date_range(values[-1] + pd.Timedelta(days=1), periods=periods, freq="D")
    inferred = pd.infer_freq(values[-min(20, len(values)):])
    if inferred:
        return pd.date_range(values[-1], periods=periods + 1, freq=inferred)[1:]
    step = pd.Series(values).diff().dropna().median()
    return pd.date_range(values[-1] + step, periods=periods, freq=step)

def save_prediction_results(file_path, prediction_type, prediction_results, actual_data, input_data, prediction_params):
    """Save prediction results to file"""
    try:
        # Create prediction results directory
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'prediction_{timestamp}.json'
        filepath = os.path.join(results_dir, filename)
        
        # Prepare data for saving
        save_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'file_path': file_path,
            'prediction_type': prediction_type,
            'prediction_params': prediction_params,
            'input_data_summary': {
                'rows': len(input_data),
                'columns': list(input_data.columns),
                'price_range': {
                    'open': {'min': float(input_data['open'].min()), 'max': float(input_data['open'].max())},
                    'high': {'min': float(input_data['high'].min()), 'max': float(input_data['high'].max())},
                    'low': {'min': float(input_data['low'].min()), 'max': float(input_data['low'].max())},
                    'close': {'min': float(input_data['close'].min()), 'max': float(input_data['close'].max())}
                },
                'last_values': {
                    'open': float(input_data['open'].iloc[-1]),
                    'high': float(input_data['high'].iloc[-1]),
                    'low': float(input_data['low'].iloc[-1]),
                    'close': float(input_data['close'].iloc[-1])
                }
            },
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'analysis': {}
        }
        
        # If actual data exists, perform comparison analysis
        if actual_data and len(actual_data) > 0:
            # Calculate continuity analysis
            if len(prediction_results) > 0 and len(actual_data) > 0:
                last_pred = prediction_results[0]  # First prediction point
            first_actual = actual_data[0]      # First actual point
                
            save_data['analysis']['continuity'] = {
                    'last_prediction': {
                        'open': last_pred['open'],
                        'high': last_pred['high'],
                        'low': last_pred['low'],
                        'close': last_pred['close']
                    },
                    'first_actual': {
                        'open': first_actual['open'],
                        'high': first_actual['high'],
                        'low': first_actual['low'],
                        'close': first_actual['close']
                    },
                    'gaps': {
                        'open_gap': abs(last_pred['open'] - first_actual['open']),
                        'high_gap': abs(last_pred['high'] - first_actual['high']),
                        'low_gap': abs(last_pred['low'] - first_actual['low']),
                        'close_gap': abs(last_pred['close'] - first_actual['close'])
                    },
                    'gap_percentages': {
                        'open_gap_pct': (abs(last_pred['open'] - first_actual['open']) / first_actual['open']) * 100,
                        'high_gap_pct': (abs(last_pred['high'] - first_actual['high']) / first_actual['high']) * 100,
                        'low_gap_pct': (abs(last_pred['low'] - first_actual['low']) / first_actual['low']) * 100,
                        'close_gap_pct': (abs(last_pred['close'] - first_actual['close']) / first_actual['close']) * 100
                    }
                }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Prediction results saved to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Failed to save prediction results: {e}")
        return None

def create_prediction_chart(df, pred_df, lookback, pred_len, actual_df=None, historical_start_idx=0):
    """Create prediction chart"""
    # Use specified historical data start position, not always from the beginning of df
    if historical_start_idx + lookback + pred_len <= len(df):
        # Display lookback historical points + pred_len prediction points starting from specified position
        historical_df = df.iloc[historical_start_idx:historical_start_idx+lookback]
        prediction_range = range(historical_start_idx+lookback, historical_start_idx+lookback+pred_len)
    else:
        # If data is insufficient, adjust to maximum available range
        available_lookback = min(lookback, len(df) - historical_start_idx)
        available_pred_len = min(pred_len, max(0, len(df) - historical_start_idx - available_lookback))
        historical_df = df.iloc[historical_start_idx:historical_start_idx+available_lookback]
        prediction_range = range(historical_start_idx+available_lookback, historical_start_idx+available_lookback+available_pred_len)
    
    # Create chart
    fig = go.Figure()
    
    # Add historical data (candlestick chart)
    fig.add_trace(go.Candlestick(
        x=historical_df['timestamps'] if 'timestamps' in historical_df.columns else historical_df.index,
        open=historical_df['open'],
        high=historical_df['high'],
        low=historical_df['low'],
        close=historical_df['close'],
        name='Historical Data (400 data points)',
        increasing_line_color='#26A69A',
        decreasing_line_color='#EF5350'
    ))
    
    # Add prediction data (candlestick chart)
    if pred_df is not None and len(pred_df) > 0:
        # Calculate prediction data timestamps - ensure continuity with historical data
        if 'timestamps' in df.columns and len(historical_df) > 0:
            # Start from the last timestamp of historical data, create prediction timestamps with the same time interval
            last_timestamp = historical_df['timestamps'].iloc[-1]
            time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
            
            pred_timestamps = pd.date_range(
                start=last_timestamp + time_diff,
                periods=len(pred_df),
                freq=time_diff
            )
        else:
            # If no timestamps, use index
            pred_timestamps = range(len(historical_df), len(historical_df) + len(pred_df))
        
        fig.add_trace(go.Candlestick(
            x=pred_timestamps,
            open=pred_df['open'],
            high=pred_df['high'],
            low=pred_df['low'],
            close=pred_df['close'],
            name='Prediction Data (120 data points)',
            increasing_line_color='#66BB6A',
            decreasing_line_color='#FF7043'
        ))
    
    # Add actual data for comparison (if exists)
    if actual_df is not None and len(actual_df) > 0:
        # Actual data should be in the same time period as prediction data
        if 'timestamps' in df.columns:
            # Actual data should use the same timestamps as prediction data to ensure time alignment
            if 'pred_timestamps' in locals():
                actual_timestamps = pred_timestamps
            else:
                # If no prediction timestamps, calculate from the last timestamp of historical data
                if len(historical_df) > 0:
                    last_timestamp = historical_df['timestamps'].iloc[-1]
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
                    actual_timestamps = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=len(actual_df),
                        freq=time_diff
                    )
                else:
                    actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        else:
            actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        
        fig.add_trace(go.Candlestick(
            x=actual_timestamps,
            open=actual_df['open'],
            high=actual_df['high'],
            low=actual_df['low'],
            close=actual_df['close'],
            name='Actual Data (120 data points)',
            increasing_line_color='#FF9800',
            decreasing_line_color='#F44336'
        ))
    
    # Update layout
    fig.update_layout(
        title='Kronos Financial Prediction Results - 400 Historical Points + 120 Prediction Points vs 120 Actual Points',
        xaxis_title='Time',
        yaxis_title='Price',
        template='plotly_white',
        height=600,
        showlegend=True
    )
    
    # Ensure x-axis time continuity
    if 'timestamps' in historical_df.columns:
        # Get all timestamps and sort them
        all_timestamps = []
        if len(historical_df) > 0:
            all_timestamps.extend(historical_df['timestamps'])
        if 'pred_timestamps' in locals():
            all_timestamps.extend(pred_timestamps)
        if 'actual_timestamps' in locals():
            all_timestamps.extend(actual_timestamps)
        
        if all_timestamps:
            all_timestamps = sorted(all_timestamps)
            fig.update_xaxes(
                range=[all_timestamps[0], all_timestamps[-1]],
                rangeslider_visible=False,
                type='date'
            )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
def index():
    """重定向到决策报告页面。"""
    from flask import redirect
    return redirect('/report')

@app.route('/assets/plotly.min.js')
def plotly_asset() -> Any:
    """从本地安装的 plotly 包返回压缩版 JS，避免使用 CDN。"""
    import plotly
    asset_path = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
    if not asset_path.exists():
        error = {"code": "ASSET_NOT_FOUND", "message": "本地 plotly.min.js 不可用。", "retryable": False, "details": {}}
        return jsonify({"status": "error", "error": error}), 404
    response = send_file(str(asset_path), mimetype="application/javascript")
    response.headers["Cache-Control"] = "public, max-age=86400"
    return response

@app.route('/api/data-files')
def get_data_files():
    """Get available data file list"""
    data_files = load_data_files()
    return jsonify(data_files)

@app.route('/api/load-data', methods=['POST'])
def load_data():
    """加载数据文件，并返回统一研究状态契约。"""
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _report_error_response("INVALID_REQUEST", "请求必须是 JSON 对象。", 400)
        file_path, reference_error = _resolve_request_file(data)
        if reference_error:
            return _report_error_response(reference_error, "数据文件名或 ID 无效。", 400)

        df, error = load_data_file(file_path)
        if error:
            return _report_error_response("INVALID_DATA", "数据文件不可用。", 400)
        
        # Detect data time frequency
        def detect_timeframe(df):
            if len(df) < 2:
                return "Unknown"
            
            time_diffs = []
            for i in range(1, min(10, len(df))):  # Check first 10 time differences
                diff = df['timestamps'].iloc[i] - df['timestamps'].iloc[i-1]
                time_diffs.append(diff)
            
            if not time_diffs:
                return "Unknown"
            
            # Calculate average time difference
            avg_diff = sum(time_diffs, pd.Timedelta(0)) / len(time_diffs)
            
            # Convert to readable format
            if avg_diff < pd.Timedelta(minutes=1):
                return f"{avg_diff.total_seconds():.0f} seconds"
            elif avg_diff < pd.Timedelta(hours=1):
                return f"{avg_diff.total_seconds() / 60:.0f} minutes"
            elif avg_diff < pd.Timedelta(days=1):
                return f"{avg_diff.total_seconds() / 3600:.0f} hours"
            else:
                return f"{avg_diff.days} days"
        
        # Return data information
        data_info = {
            'rows': len(df),
            'columns': list(df.columns),
            'start_date': df['timestamps'].min().isoformat() if 'timestamps' in df.columns else 'N/A',
            'end_date': df['timestamps'].max().isoformat() if 'timestamps' in df.columns else 'N/A',
            'price_range': {
                'min': float(df[['open', 'high', 'low', 'close']].min().min()),
                'max': float(df[['open', 'high', 'low', 'close']].max().max())
            },
            'prediction_columns': ['open', 'high', 'low', 'close'] + (['volume'] if 'volume' in df.columns else []),
            'timeframe': detect_timeframe(df)
        }
        
        return jsonify({
            'status': 'ok',
            'success': True,
            'data_info': data_info,
            'message': f'数据已加载，共 {len(df)} 行。',
            **_report_contract_fields("SUCCEEDED", "RESEARCH_ONLY", "NONE"),
        })
        
    except Exception:
        return _report_error_response("DATA_LOAD_FAILED", "数据加载失败，请检查数据格式。", 503, True)

@app.route('/api/predict', methods=['POST'])
def predict():
    """执行预测，并返回统一安全报告契约。"""
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _report_error_response("INVALID_REQUEST", "请求必须是 JSON 对象。", 400)
        lookback = int(data.get('lookback', 400))
        pred_len = int(data.get('pred_len', 120))

        # Get prediction quality parameters
        temperature = float(data.get('temperature', 1.0))
        top_p = float(data.get('top_p', 0.9))
        sample_count = int(data.get('sample_count', 1))

        file_path, reference_error = _resolve_request_file(data)
        if reference_error:
            return _report_error_response(reference_error, "数据文件名或 ID 无效。", 400)

        # Load data
        df, error = load_data_file(file_path)
        if error:
            return _report_error_response("INVALID_DATA", "数据文件不可用。", 400)
        
        if len(df) < lookback:
            return _report_error_response("INSUFFICIENT_DATA", f"数据长度不足，至少需要 {lookback} 行。", 400)
        
        # Perform prediction
        if MODEL_AVAILABLE and predictor is not None:
            try:
                # Use real Kronos model
                # Only use necessary columns: OHLCV, excluding amount
                required_cols = ['open', 'high', 'low', 'close']
                if 'volume' in df.columns:
                    required_cols.append('volume')
                
                # Process time period selection
                start_date = data.get('start_date')
                
                if start_date:
                    # Custom time period - fix logic: use data within selected window
                    start_dt = pd.to_datetime(start_date)
                    
                    # Find data after start time
                    mask = df['timestamps'] >= start_dt
                    time_range_df = df[mask]
                    
                    # Ensure sufficient data: lookback + pred_len
                    if len(time_range_df) < lookback + pred_len:
                        return _report_error_response("INSUFFICIENT_DATA", "所选时间范围的数据长度不足。", 400)
                    
                    # Use first lookback data points within selected window for prediction
                    x_df = time_range_df.iloc[:lookback][required_cols]
                    x_timestamp = time_range_df.iloc[:lookback]['timestamps']
                    
                    # Use last pred_len data points within selected window as actual values
                    y_timestamp = time_range_df.iloc[lookback:lookback+pred_len]['timestamps']
                    
                    # Calculate actual time period length
                    start_timestamp = time_range_df['timestamps'].iloc[0]
                    end_timestamp = time_range_df['timestamps'].iloc[lookback+pred_len-1]
                    time_span = end_timestamp - start_timestamp
                    
                    prediction_type = f"Kronos model prediction (within selected window: first {lookback} data points for prediction, last {pred_len} data points for comparison, time span: {time_span})"
                else:
                    # Use latest data
                    x_df = df.iloc[-lookback:][required_cols]
                    x_timestamp = df.iloc[-lookback:]['timestamps']
                    y_timestamp = pd.Series(_future_timestamps(df, pred_len), name='timestamps')
                    prediction_type = "Kronos model prediction (latest data)"
                
                # Ensure timestamps are Series format, not DatetimeIndex, to avoid .dt attribute error in Kronos model
                if isinstance(x_timestamp, pd.DatetimeIndex):
                    x_timestamp = pd.Series(x_timestamp, name='timestamps')
                if isinstance(y_timestamp, pd.DatetimeIndex):
                    y_timestamp = pd.Series(y_timestamp, name='timestamps')
                
                pred_df = predictor.predict(
                    df=x_df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=pred_len,
                    T=temperature,
                    top_p=top_p,
                    sample_count=sample_count
                )
                
            except Exception:
                return _report_error_response("MODEL_FAILURE", "模型推理失败，未生成可用预测。", 503, True)
        else:
            return _report_error_response("MODEL_UNAVAILABLE", "模型尚未就绪，暂时不能生成预测。", 503, True)

        if pred_df is None or len(pred_df) == 0:
            return _report_error_response("EMPTY_PREDICTION_PATH", "模型未返回有效预测路径。", 422)
        
        # Prepare actual data for comparison (if exists)
        actual_data = []
        actual_df = None
        
        if start_date:  # Custom time period
            # Fix logic: use data within selected window
            # Prediction uses first 400 data points within selected window
            # Actual data should be last 120 data points within selected window
            start_dt = pd.to_datetime(start_date)
            
            # Find data starting from start_date
            mask = df['timestamps'] >= start_dt
            time_range_df = df[mask]
            
            if len(time_range_df) >= lookback + pred_len:
                # Get last 120 data points within selected window as actual values
                actual_df = time_range_df.iloc[lookback:lookback+pred_len]
                
                for i, (_, row) in enumerate(actual_df.iterrows()):
                    actual_data.append({
                        'timestamp': row['timestamps'].isoformat(),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']) if 'volume' in row else 0,
                        'amount': float(row['amount']) if 'amount' in row else 0
                    })
        else:  # Latest data
            # 最新窗口之后没有已知真实值，不伪造历史对照。
            actual_df = None
        
        # Create chart - pass historical data start position
        if start_date:
            # Custom time period: find starting position of historical data in original df
            start_dt = pd.to_datetime(start_date)
            mask = df['timestamps'] >= start_dt
            historical_start_idx = df[mask].index[0] if len(df[mask]) > 0 else 0
        else:
            # Latest data: display the same trailing window used by the model
            historical_start_idx = max(0, len(df) - lookback)
        
        chart_json = create_prediction_chart(df, pred_df, lookback, pred_len, actual_df, historical_start_idx)
        
        # Prepare prediction result data - fix timestamp calculation logic
        if 'timestamps' in df.columns:
            if start_date:
                # Custom time period: use selected window data to calculate timestamps
                start_dt = pd.to_datetime(start_date)
                mask = df['timestamps'] >= start_dt
                time_range_df = df[mask]
                
                if len(time_range_df) >= lookback:
                    # Calculate prediction timestamps starting from last time point of selected window
                    last_timestamp = time_range_df['timestamps'].iloc[lookback-1]
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]
                    future_timestamps = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=pred_len,
                        freq=time_diff
                    )
                else:
                    future_timestamps = []
            else:
                future_timestamps = _future_timestamps(df, pred_len)
        else:
            future_timestamps = range(len(df), len(df) + pred_len)
        
        prediction_results = []
        for i, (_, row) in enumerate(pred_df.iterrows()):
            prediction_results.append({
                'timestamp': future_timestamps[i].isoformat() if i < len(future_timestamps) else f"T{i}",
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0,
                'amount': float(row['amount']) if 'amount' in row else 0
            })
        
        # Save prediction results to file
        try:
            save_prediction_results(
                file_path=file_path,
                prediction_type=prediction_type,
                prediction_results=prediction_results,
                actual_data=actual_data,
                input_data=x_df,
                prediction_params={
                    'lookback': lookback,
                    'pred_len': pred_len,
                    'temperature': temperature,
                    'top_p': top_p,
                    'sample_count': sample_count,
                    'start_date': start_date if start_date else 'latest'
                }
            )
        except Exception as e:
            print(f"Failed to save prediction results: {e}")
        
        return jsonify({
            'status': 'ok',
            'success': True,
            'prediction_type': prediction_type,
            'chart': chart_json,
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'has_comparison': len(actual_data) > 0,
            'message': f'预测完成，共生成 {pred_len} 个预测点。' + (f'，包含 {len(actual_data)} 个实际数据点用于对照。' if len(actual_data) > 0 else ''),
            **_report_contract_fields("SUCCEEDED", "RESEARCH_ONLY", "NONE"),
        })
        
    except Exception:
        return _report_error_response("PREDICTION_FAILED", "预测失败，未生成可用报告。", 503, True)

@app.route('/api/load-model', methods=['POST'])
def load_model():
    """加载 Kronos 模型配置。"""
    global tokenizer, model, predictor
    
    try:
        if not MODEL_AVAILABLE:
            return _report_error_response("MODEL_UNAVAILABLE", "模型库不可用，暂时不能加载模型。", 503, True)
        
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return _report_error_response("INVALID_REQUEST", "请求必须是 JSON 对象。", 400)
        model_key = data.get('model_key', 'kronos-small')
        device = data.get('device', 'cpu')
        
        if model_key not in AVAILABLE_MODELS:
            return _report_error_response("UNKNOWN_MODEL", "请求的模型配置未知。", 400)
        
        model_config = AVAILABLE_MODELS[model_key]
        
        # Load tokenizer and model
        tokenizer = KronosTokenizer.from_pretrained(model_config['tokenizer_id'])
        model = Kronos.from_pretrained(model_config['model_id'])
        
        # Create predictor
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=model_config['context_length'])
        
        return jsonify({
            'status': 'ok',
            'success': True,
            'message': f'模型已加载：{model_config["name"]}（{model_config["params"]}，设备 {device}）。',
            'model_info': {
                'name': model_config['name'],
                'params': model_config['params'],
                'context_length': model_config['context_length'],
                'description': model_config['description']
            },
            **_report_contract_fields("SUCCEEDED", "RESEARCH_ONLY", "NONE"),
        })
        
    except Exception:
        return _report_error_response("MODEL_LOAD_FAILED", "模型加载失败，未发布可用模型。", 503, True)

@app.route('/api/available-models')
def get_available_models():
    """Get available model list"""
    return jsonify({
        'models': AVAILABLE_MODELS,
        'model_available': MODEL_AVAILABLE
    })

@app.route('/api/model-status')
def get_model_status():
    """Get model status"""
    if MODEL_AVAILABLE:
        if predictor is not None:
            return jsonify({
                'available': True,
                'loaded': True,
                'message': 'Kronos model loaded and available',
                'current_model': {
                    'name': predictor.model.__class__.__name__,
                    'device': str(next(predictor.model.parameters()).device)
                }
            })
        else:
            return jsonify({
                'available': True,
                'loaded': False,
                'message': 'Kronos model available but not loaded'
            })
    else:
        return jsonify({
            'available': False,
            'loaded': False,
            'message': 'Kronos model library not available, please install related dependencies'
        })

# === 决策报告 API ===

@app.route("/report")
def report_page():
    """v2 五态决策报告页面（主入口）。"""
    return render_template("report_v2.html")


@app.route("/report/legacy")
def report_page_legacy():
    """v1 兼容页面：允许修改采样参数，仅供旧行为对照。"""
    return render_template("report.html", show_migration_banner=True)


@app.route("/portfolio")
def portfolio_page():
    """本地组合页面（不提供下单控件）。"""
    return render_template("portfolio.html")


@app.route("/research")
def research_page():
    """正式研究评估页面。"""
    return render_template("research.html")


@app.route("/settings")
def settings_page():
    """高级设置页面：采样参数只读展示，修改会使证据档案失效。"""
    return render_template("settings.html")


@app.route("/api/decision-report", methods=["POST"])
def api_decision_report():
    """生成决策报告。

    请求: {"stock_code": "600519", "pred_len": 120, ...}
    响应: DecisionReport 的 JSON 序列化
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return _report_error_response("INVALID_REQUEST", "请求必须是 JSON 对象。", 400)
    stock_code = str(data.get("stock_code", "")).strip()
    if not stock_code:
        return _report_error_response("INVALID_REQUEST", "请提供股票代码。", 400)

    try:
        DecisionEngine = _get_decision_engine_class()
        report = DecisionEngine().predict_and_analyze(stock_code, data)
        safe_report, problem = _normalise_legacy_report(report)
        if problem:
            code, message, http_status = problem
            return _report_error_response(code, message, http_status, http_status >= 500)
        return jsonify(safe_report)
    except Exception:
        return _report_error_response("REPORT_FAILED", "报告生成失败，未提供动作参考。", 503, True)


@app.route("/api/stock-list")
def api_stock_list():
    """获取股票池列表。"""
    try:
        from data.pool import get_current_universe_snapshot

        snapshot = get_current_universe_snapshot()
        stocks = [
            {"code": item.code, "name": item.name}
            for item in snapshot.instruments
            if item.is_active
        ]
        return jsonify(
            {
                "status": "ok" if snapshot.usable else "degraded",
                "stocks": stocks if snapshot.usable else [],
                "pool_status": "USABLE" if snapshot.usable else "UNUSABLE",
                "universe_version": snapshot.universe_version,
                "reasons": list(snapshot.reasons),
            }
        )
    except Exception:
        return jsonify(
            {
                "status": "error",
                "stocks": [],
                "pool_status": "UNAVAILABLE",
                "error_message": "当前股票池无法校验，未提供可选证券。",
            }
        ), 503


# === v2 可审计研究 API ===

@app.route("/api/v2/decision-report/<path:report_id>", methods=["GET"])
def api_read_persisted_decision_report(report_id: str):
    """只从 SQLite 和快照读取历史报告，不重新取数或运行模型。"""
    if (
        not isinstance(report_id, str)
        or not report_id
        or len(report_id) > 240
        or "/" in report_id
        or "\\" in report_id
    ):
        return _report_error_response("INVALID_REPORT_ID", "报告编号无效。", 400)
    try:
        Pipeline = _get_report_pipeline_class()
        record = Pipeline().read_published(report_id)
        if record is None:
            return _report_error_response("REPORT_NOT_FOUND", "历史报告不存在。", 404)
        report, problem = _normalise_v2_report(record.payload)
        if problem:
            code, message, http_status = problem
            return _report_error_response(code, message, http_status, http_status >= 500)
        response_report = dict(report)
        response_report.update(
            {
                "report_id": record.report_id,
                "report_version": record.version,
                "snapshot_id": record.snapshot_id,
                "publication_status": "PUBLISHED",
            }
        )
        return jsonify(response_report)
    except Exception:
        return _report_error_response(
            "REPORT_REPLAY_UNAVAILABLE",
            "历史报告或其数据快照无法校验，未恢复动作建议。",
            503,
            True,
        )


@app.route("/api/v2/decision-report", methods=["POST"])
def api_v2_decision_report():
    """生成固定采样配置的五态决策报告。"""
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return _report_error_response("INVALID_REQUEST", "请求必须是 JSON 对象。", 400)
    stock_code = str(payload.get("stock_code", "")).strip()
    if len(stock_code) != 6 or not stock_code.isdigit():
        return _report_error_response("INVALID_REQUEST", "stock_code 必须是六位股票代码。", 400)

    requested_as_of = payload.get("as_of")
    if requested_as_of is not None:
        if not isinstance(requested_as_of, str):
            return _report_error_response("INVALID_REQUEST", "as_of 必须是 YYYY-MM-DD 日期字符串或 null。", 400)
        try:
            requested_as_of = pd.Timestamp(requested_as_of).normalize()
            if pd.isna(requested_as_of):
                raise ValueError("NaT")
        except (TypeError, ValueError):
            return _report_error_response("INVALID_REQUEST", "as_of 必须是可识别的日期。", 400)

    try:
        from decision.errors import PredictionTimeoutError

        Pipeline = _get_report_pipeline_class()
        mode = payload.get("mode", "model")
        if mode == "baseline":
            result = Pipeline().run_baseline(
                stock_code,
                as_of=requested_as_of,
            )
        elif mode == "model":
            result = Pipeline().run(
                stock_code,
                portfolio_id=str(payload.get("portfolio_id", "default")),
                include_display_paths=bool(payload.get("include_display_paths", False)),
                as_of=requested_as_of,
            )
        else:
            return _report_error_response("INVALID_REQUEST", "mode 只能是 model 或 baseline。", 400)
        report = result.report
        report, problem = _normalise_v2_report(report)
        if problem:
            code, message, http_status = problem
            return _report_error_response(code, message, http_status, http_status >= 500)
        if report.get("action_permission") != "NONE" and report.get("recommendation", {}).get("action"):
            try:
                from portfolio.store import PortfolioStore

                version_match = bool((report["evidence_gate"].get("checks") or {}).get("VERSION_MATCH"))
                PortfolioStore().save_recommendation(
                    str(payload.get("portfolio_id", "default")),
                    stock_code,
                    report["recommendation"]["action"],
                    report["horizons"],
                    "gate-v1" if version_match else "no-evidence",
                    report["generated_at"],
                    report["data_provenance"].get("as_of"),
                    bool(report["evidence_gate"].get("passed")),
                )
            except Exception:
                app.logger.warning("建议历史写入失败，已继续返回安全报告。")
        response_report = dict(report)
        response_report.update(
            {
                "report_id": result.report_id,
                "report_version": result.version,
                "snapshot_id": result.snapshot_id,
                "publication_status": "PUBLISHED",
            }
        )
        return jsonify(response_report)
    except PredictionTimeoutError:
        return _report_error_response(
            "PREDICTION_TIMEOUT",
            "模型推理超过配置的时间预算。",
            504,
            True,
            run_status="TIMED_OUT",
        )
    except Exception:
        return _report_error_response("DATA_UNAVAILABLE", "无法取得可验证的完整日线数据。", 503, True)


@app.route("/api/v2/decision-jobs", methods=["POST"])
def api_submit_decision_job():
    """提交有界异步决策任务。"""
    payload = request.get_json(silent=True) or {}
    try:
        service = _get_job_service()
        snapshot = service.submit(
            str(payload.get("stock_code", "")),
            as_of=payload.get("as_of"),
            mode=str(payload.get("mode", "model")),
        )
        return jsonify(snapshot.as_dict()), 202
    except Exception:
        return _report_error_response("JOB_SUBMIT_FAILED", "任务提交失败。", 400)


@app.route("/api/v2/decision-jobs/<job_id>", methods=["GET"])
def api_get_decision_job(job_id: str):
    """读取异步任务状态和受控结果。"""
    try:
        return jsonify(_get_job_service().get(job_id))
    except Exception:
        return _report_error_response("JOB_NOT_FOUND", "任务不存在。", 404)


@app.route("/api/v2/decision-jobs/<job_id>/cancel", methods=["POST"])
def api_cancel_decision_job(job_id: str):
    """请求取消尚未完成的异步任务。"""
    try:
        return jsonify(_get_job_service().cancel(job_id))
    except Exception:
        return _report_error_response("JOB_CANCEL_FAILED", "任务取消失败。", 409)


@app.route("/api/v2/evaluation/latest")
def api_v2_evaluation_latest():
    """返回最近一次可追溯评估运行的清单与门禁结果。"""
    from decision.v2 import error_payload
    from evaluation.binding import EvaluationBinding

    binding = EvaluationBinding.load_latest()
    if binding is None:
        return jsonify(error_payload("EVALUATION_NOT_FOUND", "尚无正式评估运行", False)), 404
    result = {"status": "ok", "run_id": binding.run_id, "manifest": binding.manifest, "gate_result": binding.gate_result}
    return jsonify(result)


@app.route("/api/v2/forward-ledger/latest")
def api_v2_forward_ledger_latest():
    """返回与历史回测隔离的最新每日建议前瞻台账。"""
    from decision.v2 import error_payload
    from evaluation.forward_ledger import load_latest_forward_report

    latest = load_latest_forward_report()
    if latest is None:
        return jsonify(error_payload("FORWARD_LEDGER_NOT_FOUND", "尚无每日建议前瞻台账", False)), 404
    return jsonify({"status": "ok", **latest})


@app.route("/api/v2/portfolio/preview", methods=["POST"])
def api_v2_portfolio_preview():
    """解析持仓 CSV 预览；预览阶段绝不写正式账本。"""
    from decision.portfolio import PortfolioLedger, PortfolioError
    from decision.config import get_config
    from decision.v2 import error_payload

    payload = request.get_json(silent=True) or {}
    content = payload.get("csv_content")
    try:
        preview = PortfolioLedger(get_config().data.portfolio_ledger).preview_csv(content, preview_id=str(payload.get("preview_id", "")))
        return jsonify({
            "status": "ok" if preview.valid else "invalid",
            "preview_id": preview.preview_id,
            "content_hash": preview.content_hash,
            "errors": list(preview.errors),
            "positions": [position.to_dict() for position in preview.positions],
        }), 200 if preview.valid else 422
    except (PortfolioError, TypeError) as exc:
        return jsonify(error_payload("INVALID_PORTFOLIO_PREVIEW", str(exc), False)), 400


@app.route("/api/v2/portfolio/confirm", methods=["POST"])
def api_v2_portfolio_confirm():
    """在显式确认后建立持仓快照；重复确认保持幂等。"""
    from decision.config import get_config
    from decision.portfolio import PortfolioError, PortfolioLedger
    from decision.v2 import error_payload

    payload = request.get_json(silent=True) or {}
    try:
        ledger = PortfolioLedger(get_config().data.portfolio_ledger)
        preview = ledger.preview_csv(str(payload.get("csv_content", "")), preview_id=str(payload.get("preview_id", "")))
        if payload.get("content_hash") and payload["content_hash"] != preview.content_hash:
            raise PortfolioError("持仓预览内容已变化，请重新预览")
        snapshot = ledger.confirm_preview(
            preview,
            cash=float(payload.get("cash", -1)),
            as_of=str(payload.get("as_of", "")),
            available_at=str(payload.get("available_at", "")),
            source=str(payload.get("source", "manual")),
            confirmed=payload.get("confirmed") is True,
        )
        return jsonify({"status": "ok", "snapshot": snapshot.to_dict()}), 200
    except (PortfolioError, TypeError, ValueError) as exc:
        return jsonify(error_payload("PORTFOLIO_CONFIRM_REJECTED", str(exc), False)), 400


@app.route("/api/v2/portfolio/snapshots/<snapshot_id>", methods=["GET"])
def api_v2_portfolio_snapshot(snapshot_id: str):
    """读回指定持仓快照并校验内容哈希。"""
    from decision.config import get_config
    from decision.portfolio import PortfolioLedger, PortfolioError
    from decision.v2 import error_payload

    if not snapshot_id or len(snapshot_id) > 160 or "/" in snapshot_id or "\\" in snapshot_id:
        return jsonify(error_payload("INVALID_SNAPSHOT_ID", "持仓快照标识无效", False)), 400
    try:
        snapshot = PortfolioLedger(get_config().data.portfolio_ledger).get(snapshot_id)
        if snapshot is None:
            return jsonify(error_payload("PORTFOLIO_SNAPSHOT_NOT_FOUND", "持仓快照不存在", False)), 404
        return jsonify({"status": "ok", "snapshot": snapshot.to_dict()}), 200
    except PortfolioError as exc:
        return jsonify(error_payload("PORTFOLIO_SNAPSHOT_INVALID", str(exc), False)), 503


@app.route("/api/v2/portfolio", methods=["GET", "PUT"])
def api_v2_portfolio():
    """读取或更新本地 SQLite 持仓，不保存外部交易凭据。"""
    from portfolio.store import PortfolioStore
    from decision.v2 import error_payload

    profile_id = request.args.get("portfolio_id", "default")
    store = PortfolioStore()
    try:
        if request.method == "PUT":
            return jsonify({"status": "ok", "portfolio": store.replace_portfolio(request.get_json(silent=True) or {}, profile_id)})
        return jsonify({"status": "ok", "portfolio": store.get_portfolio(profile_id)})
    except ValueError as exc:
        return jsonify(error_payload("INVALID_PORTFOLIO", str(exc), False)), 400


@app.route("/api/v2/portfolio/rebalance", methods=["POST"])
def api_v2_portfolio_rebalance():
    """门禁通过时生成模拟目标权重与订单；永不执行真实交易。"""
    from decision.v2 import error_payload
    from evaluation.binding import EvaluationBinding
    from portfolio.service import RebalanceService
    from data.fetcher import DataFetcher

    profile_id = (request.get_json(silent=True) or {}).get("portfolio_id", "default")
    binding = EvaluationBinding.load_latest()
    fetcher = DataFetcher()

    def fetch_bars(code: str):
        try:
            return fetcher.fetch_daily(code)
        except Exception:
            return None

    result = RebalanceService().rebalance(binding, profile_id, fetch_bars)
    if result["status"] == "ok":
        return jsonify({"status": "ok", "rebalance": result})
    return jsonify(error_payload("INSUFFICIENT_EVIDENCE", "证据不足，无法生成模拟调仓", False, {"reason_codes": result["reason_codes"]})), 409


@app.route("/api/v2/recommendations")
def api_v2_recommendations():
    """返回本地保存的建议历史。"""
    from portfolio.store import PortfolioStore
    from decision.v2 import error_payload

    profile_id = request.args.get("portfolio_id", "default")
    try:
        limit = int(request.args.get("limit", 100))
        if not 1 <= limit <= 500:
            raise ValueError("limit out of range")
        return jsonify({"status": "ok", "recommendations": PortfolioStore().list_recommendations(profile_id, limit)})
    except ValueError:
        return jsonify(error_payload("INVALID_REQUEST", "limit 必须为数字", False)), 400


@app.route("/api/config", methods=["GET", "PUT"])
def api_config():
    """获取或更新配置。"""
    from copy import deepcopy

    from decision.config import get_config, save_config
    import dataclasses

    if request.method == "GET":
        cfg = get_config()
        return jsonify(dataclasses.asdict(cfg))

    if request.method == "PUT":
        try:
            data = request.get_json(silent=True)
            if not isinstance(data, dict):
                raise ValueError("配置请求必须是 JSON 对象。")
            cfg = deepcopy(get_config())
            for section, values in data.items():
                if not hasattr(cfg, section) or not isinstance(values, dict):
                    raise ValueError(f"未知或无效的配置分组：{section}")
                sub = getattr(cfg, section)
                for key, val in values.items():
                    if not hasattr(sub, key):
                        raise ValueError(f"未知配置项：{section}.{key}")
                    setattr(sub, key, val)
            save_config(cfg)
            return jsonify({"status": "ok", "message": "配置已更新"})
        except Exception as e:
            return jsonify({"status": "error", "error_message": str(e)}), 400


if __name__ == '__main__':
    print("Starting Kronos Web UI...")
    print(f"Model availability: {MODEL_AVAILABLE}")
    if MODEL_AVAILABLE:
        print("Tip: You can load Kronos model through /api/load-model endpoint")
    else:
        print("Tip: Will use simulated data for demonstration")
    
    from decision.config import get_config

    web_config = get_config().webui
    app.run(debug=False, host=web_config.host, port=web_config.port, use_reloader=False)
