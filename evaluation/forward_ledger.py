"""本地前瞻建议台账：仅观察成熟信号，绝不写回量化门禁。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def evaluate_forward_ledger(recommendations: list[dict[str, Any]], price_history: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """评估已满期建议；缺少日期、门禁或价格时保留明确状态。"""
    if horizon < 1:
        raise ValueError("horizon 必须大于零。")
    prices = _prepare_prices(price_history)
    rows = [_evaluate_row(item, prices, horizon) for item in recommendations]
    return pd.DataFrame(rows, columns=_LEDGER_COLUMNS)


def forward_ledger_report(observations: pd.DataFrame) -> dict[str, Any]:
    """汇总成熟观察，明确区分未成熟和不合格信号。"""
    if observations.empty:
        return {"status": "no_recommendations", "disclaimer": _DISCLAIMER}
    mature = observations.loc[observations["status"] == "MATURED"].copy()
    report = {"status": "completed" if not mature.empty else "awaiting_maturity", "total_recommendations": int(len(observations)), "matured_recommendations": int(len(mature)), "pending_recommendations": int((observations["status"] == "PENDING_HORIZON").sum()), "excluded_recommendations": int((observations["status"] != "MATURED").sum() - (observations["status"] == "PENDING_HORIZON").sum()), "by_action": {}, "disclaimer": _DISCLAIMER}
    for action, group in mature.groupby("action"):
        returns = group["actual_return"]
        report["by_action"][str(action)] = {"count": int(len(group)), "mean_actual_return": float(returns.mean()), "up_rate": float((returns > 0).mean())}
    return report


def load_latest_forward_report(root: str | Path = "artifacts/forward") -> dict[str, Any] | None:
    """读取最新有效前瞻台账摘要；损坏工件不阻断研究页。"""
    candidates = sorted(Path(root).glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(report, dict) and "status" in report:
            return {"artifact": path.name, "report": report}
    return None


def _prepare_prices(price_history: pd.DataFrame) -> pd.DataFrame:
    """将宽表价格历史规范为按交易日索引、六码代码列。"""
    prices = price_history.copy()
    if "date" in prices:
        prices = prices.set_index("date")
    prices.index = pd.DatetimeIndex(pd.to_datetime(prices.index)).normalize()
    prices = prices.loc[~prices.index.duplicated(keep="last")].sort_index()
    prices.columns = [str(code).zfill(6) for code in prices.columns]
    return prices.apply(pd.to_numeric, errors="coerce")


def _evaluate_row(item: dict[str, Any], prices: pd.DataFrame, horizon: int) -> dict[str, Any]:
    """从单条历史建议生成不可提前得知收益的观察记录。"""
    result = {"recommendation_id": item.get("id"), "stock_code": str(item.get("stock_code", "")).zfill(6), "action": item.get("action"), "data_as_of": item.get("data_as_of"), "status": "", "entry_price": None, "exit_price": None, "actual_return": None}
    if not bool(item.get("evidence_gate_passed")):
        result["status"] = "EXCLUDED_EVIDENCE_GATE_FAILED"
        return result
    if not result["data_as_of"]:
        result["status"] = "MISSING_DATA_AS_OF"
        return result
    if result["stock_code"] not in prices:
        result["status"] = "MISSING_STOCK_PRICE"
        return result
    series = prices[result["stock_code"]].dropna()
    anchor = pd.Timestamp(result["data_as_of"]).normalize()
    location = series.index.get_indexer([anchor])[0]
    if location < 0:
        result["status"] = "MISSING_ENTRY_PRICE"
        return result
    result["entry_price"] = float(series.iloc[location])
    if location + horizon >= len(series):
        result["status"] = "PENDING_HORIZON"
        return result
    result["exit_price"] = float(series.iloc[location + horizon])
    result["actual_return"] = result["exit_price"] / result["entry_price"] - 1
    result["status"] = "MATURED"
    return result


_LEDGER_COLUMNS = ["recommendation_id", "stock_code", "action", "data_as_of", "status", "entry_price", "exit_price", "actual_return"]
_DISCLAIMER = "前瞻台账只观察成熟后可见的实际收益，与历史回测分开报告，不进入正式证据门禁，也不改变量化动作。"
