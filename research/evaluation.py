"""研究评价指标，明确区分不可计算和零分。"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

import numpy as np
import pandas as pd

from research.backtest import BacktestResult


class EvaluationError(ValueError):
    """评价输入无法满足统计口径。"""


@dataclass(frozen=True)
class EvaluationResult:
    """评价指标、有效样本数和不可计算原因。"""

    metrics: Mapping[str, float | None]
    sample_counts: Mapping[str, int]
    reasons: tuple[str, ...]

    @property
    def valid(self) -> bool:
        """是否至少存在一项有定义的指标。"""
        return any(value is not None for value in self.metrics.values())


def rank_ic_by_date(predictions: pd.DataFrame) -> EvaluationResult:
    """按日期计算横截面 Spearman Rank IC，不把无方差日期当作零。"""
    frame = _prepare_predictions(predictions)
    values: list[float] = []
    valid_dates = 0
    skipped_dates = 0
    for _, group in frame.groupby("date", sort=True):
        group = group.dropna(subset=["score", "realized_return"])
        if len(group) < 2:
            skipped_dates += 1
            continue
        if group["score"].nunique() < 2 or group["realized_return"].nunique() < 2:
            skipped_dates += 1
            continue
        score_rank = group["score"].rank(method="average")
        return_rank = group["realized_return"].rank(method="average")
        correlation = score_rank.corr(return_rank)
        if correlation is None or not isfinite(float(correlation)):
            skipped_dates += 1
            continue
        values.append(float(correlation))
        valid_dates += 1
    reasons: list[str] = []
    if not values:
        reasons.append("RANK_IC_UNDEFINED")
    if skipped_dates:
        reasons.append("RANK_IC_DATES_SKIPPED")
    return EvaluationResult(
        metrics={"rank_ic_mean": _mean_or_none(values)},
        sample_counts={"valid_dates": valid_dates, "skipped_dates": skipped_dates},
        reasons=tuple(reasons),
    )


def evaluate_backtest(result: BacktestResult | pd.DataFrame) -> EvaluationResult:
    """计算净收益、最大回撤、换手和失败成交比例。"""
    equity = result.equity_curve if isinstance(result, BacktestResult) else result
    if not isinstance(equity, pd.DataFrame) or equity.empty:
        return EvaluationResult({}, {"equity_points": 0}, ("EQUITY_UNAVAILABLE",))
    if "equity" not in equity.columns:
        raise EvaluationError("equity_curve 缺少 equity 字段")
    values = pd.to_numeric(equity["equity"], errors="coerce")
    if values.isna().any() or (values <= 0).any():
        return EvaluationResult({}, {"equity_points": int(values.notna().sum())}, ("EQUITY_INVALID",))
    returns = values.iloc[-1] / values.iloc[0] - 1
    running_max = values.cummax()
    drawdown = values / running_max - 1
    metrics: dict[str, float | None] = {
        "net_return": float(returns),
        "max_drawdown": float(drawdown.min()),
    }
    counts = {"equity_points": len(values)}
    reasons: list[str] = []
    if isinstance(result, BacktestResult):
        gross = [fill.gross_value for fill in result.fills if fill.gross_value > 0]
        equity_base = values.shift(1).dropna()
        metrics["turnover"] = float(sum(gross) / equity_base.sum()) if not equity_base.empty else None
        total_attempts = len(result.fills) + len(result.failures)
        metrics["failed_fill_rate"] = (
            float(len(result.failures) / total_attempts) if total_attempts else None
        )
        counts["fills"] = len(result.fills)
        counts["failures"] = len(result.failures)
        if not result.fills and not result.failures:
            reasons.append("NO_TRADES")
    return EvaluationResult(metrics=metrics, sample_counts=counts, reasons=tuple(reasons))


def calibration_table(
    predictions: pd.DataFrame,
    *,
    score_column: str = "score",
    outcome_column: str = "realized_return",
    bins: int = 5,
) -> EvaluationResult:
    """按冻结分箱计算预测分数与成熟结果的校准差异。"""
    if isinstance(bins, bool) or bins < 2:
        raise EvaluationError("校准箱数必须至少为 2")
    required = {score_column, outcome_column}
    if not required.issubset(predictions.columns):
        return EvaluationResult({}, {"rows": 0}, ("MISSING_CALIBRATION_COLUMNS",))
    frame = predictions[[score_column, outcome_column]].copy()
    frame[score_column] = pd.to_numeric(frame[score_column], errors="coerce")
    frame[outcome_column] = pd.to_numeric(frame[outcome_column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return EvaluationResult({}, {"rows": 0}, ("NO_MATURED_CALIBRATION_ROWS",))
    frame["bin"] = pd.qcut(frame[score_column], q=min(bins, len(frame)), duplicates="drop")
    grouped = frame.groupby("bin", observed=True)
    table = []
    for interval, group in grouped:
        table.append(
            {
                "bin": str(interval),
                "count": int(len(group)),
                "mean_score": float(group[score_column].mean()),
                "mean_outcome": float(group[outcome_column].mean()),
                "gap": float(group[outcome_column].mean() - group[score_column].mean()),
            }
        )
    error = float(np.average([abs(row["gap"]) for row in table], weights=[row["count"] for row in table]))
    return EvaluationResult(
        metrics={"calibration_error": error, "calibration_bins": table},
        sample_counts={"rows": int(len(frame)), "bins": len(table)},
        reasons=(),
    )


def cost_stress_matrix(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    initial_cash: float,
    commission_rates: tuple[float, ...] = (0.0003, 0.0006),
    slippage_bps: tuple[float, ...] = (5.0, 15.0),
) -> EvaluationResult:
    """在多组明确成本下重复执行同一规则，缺失结果不记为零。"""
    from research.backtest import BacktestEngine, ExecutionRules

    rows: list[dict[str, object]] = []
    for commission in commission_rates:
        for slippage in slippage_bps:
            result = BacktestEngine(
                ExecutionRules(commission_rate=commission, slippage_bps=slippage)
            ).run(bars, signals, initial_cash=initial_cash)
            metric = evaluate_backtest(result)
            rows.append(
                {
                    "commission_rate": commission,
                    "slippage_bps": slippage,
                    "net_return": metric.metrics.get("net_return"),
                    "failures": len(result.failures),
                }
            )
    return EvaluationResult(
        metrics={"stress_matrix": rows},
        sample_counts={"scenarios": len(rows)},
        reasons=() if rows else ("NO_STRESS_SCENARIOS",),
    )


def evaluate_periods(predictions: pd.DataFrame, period_days: int = 5) -> EvaluationResult:
    """按非重叠日期块给出平均收益，避免重叠日线虚增独立样本。"""
    if isinstance(period_days, bool) or period_days <= 0:
        raise EvaluationError("period_days 必须是正整数")
    frame = _prepare_predictions(predictions)
    frame = frame.sort_values("date")
    dates = frame["date"].drop_duplicates().reset_index(drop=True)
    period_by_date = {
        date_value: index // period_days for index, date_value in dates.items()
    }
    frame["period"] = frame["date"].map(period_by_date)
    values = frame.groupby("period", sort=True)["realized_return"].mean().dropna().tolist()
    reasons = () if values else ("PERIOD_RETURN_UNDEFINED",)
    return EvaluationResult(
        {"period_return_mean": _mean_or_none([float(item) for item in values])},
        {"valid_periods": len(values)},
        reasons,
    )


def _prepare_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        raise EvaluationError("predictions 不能为空")
    required = {"date", "code", "score", "realized_return"}
    missing = required.difference(predictions.columns)
    if missing:
        raise EvaluationError(f"predictions 缺少字段：{sorted(missing)}")
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame["realized_return"] = pd.to_numeric(frame["realized_return"], errors="coerce")
    if frame["date"].isna().any():
        raise EvaluationError("predictions.date 含无效值")
    return frame


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


__all__ = ["EvaluationError", "EvaluationResult", "evaluate_backtest", "evaluate_periods", "rank_ic_by_date"]
