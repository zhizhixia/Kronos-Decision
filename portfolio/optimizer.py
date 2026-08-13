"""PyPortfolioOpt 组合优化、HRP 回退与 A 股模拟订单取整。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd
from pypfopt import HRPOpt
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class RiskProfile:
    name: str
    stock_cap: float
    sector_cap: float
    minimum_cash: float
    turnover_cap: float
    risk_aversion: float


PROFILES = {
    "conservative": RiskProfile("conservative", 0.05, 0.20, 0.20, 0.10, 8.0),
    "balanced": RiskProfile("balanced", 0.08, 0.25, 0.10, 0.20, 4.0),
    "aggressive": RiskProfile("aggressive", 0.12, 0.30, 0.05, 0.30, 2.0),
}

# 池外股票（自选但不在点时评估股票池）总组合权重上限，计划锁定为 10%
OUTSIDE_POOL_CAP = 0.10


@dataclass(frozen=True)
class SimulatedTradeCosts:
    """模拟订单费用假设；仅用于研究展示，不代表券商实际费率。"""

    commission_rate: float = 0.0003
    sell_stamp_duty_rate: float = 0.001
    minimum_commission: float = 5.0


DEFAULT_SIMULATED_COSTS = SimulatedTradeCosts()


@dataclass(frozen=True)
class OptimizationResult:
    status: str
    method: str | None
    target_weights: dict[str, float]
    cash_weight: float
    reason_codes: tuple[str, ...]


def annualize_expected_returns(median_excess_20d: pd.Series) -> pd.Series:
    """1%/99%截尾后把20日中位超额收益转换为年化值。"""
    values = median_excess_20d.astype(float).clip(lower=-0.99)
    if len(values) >= 2:
        values = values.clip(values.quantile(0.01), values.quantile(0.99))
    return (1 + values) ** (252 / 20) - 1


class PortfolioOptimizer:
    """先最大二次效用（含现金与换手硬约束），失败后尝试 HRP；不使用等权兜底。"""

    def optimize(self, expected_returns: pd.Series, price_history: pd.DataFrame, current_weights: dict[str, float], sectors: dict[str, str], eligible: set[str], profile_name: str) -> OptimizationResult:
        profile = PROFILES[profile_name]
        assets = sorted(eligible | {asset for asset, weight in current_weights.items() if weight > 0})
        try:
            problem = self._prepare_problem(assets, expected_returns, price_history, current_weights, sectors, eligible, profile)
            weights = self._quadratic_utility(problem, profile)
            weights = self._project_numerical_tolerance(weights, current_weights, sectors, eligible, profile)
            self._validate(weights, current_weights, sectors, eligible, profile)
            return OptimizationResult("ok", "max_quadratic_utility", weights, 1 - sum(weights.values()), ())
        except Exception as primary_error:
            if "problem" not in locals():
                return OptimizationResult("insufficient_evidence", None, {}, 1.0, (str(primary_error),))
            return self._hrp_or_failure(problem, current_weights, sectors, eligible, profile, primary_error)

    @staticmethod
    def _prepare_problem(assets, expected_returns, price_history, current_weights, sectors, eligible, profile):
        if not assets or any(asset not in sectors for asset in assets):
            raise ValueError("INDUSTRY_MAPPING_MISSING")
        missing_prices = [asset for asset in assets if asset not in price_history]
        if missing_prices:
            raise ValueError(f"PRICE_HISTORY_MISSING:{missing_prices}")
        investable = 1 - profile.minimum_cash
        returns = expected_returns.reindex(assets).fillna(0.0)
        prices = price_history[assets].dropna(how="all").ffill().dropna()
        daily_returns = prices.pct_change().dropna(how="all")
        covariance_values = LedoitWolf().fit(daily_returns.to_numpy(dtype=float)).covariance_ * 252
        covariance = pd.DataFrame(covariance_values, index=assets, columns=assets)
        upper = {asset: profile.stock_cap if asset in eligible else min(current_weights.get(asset, 0.0), profile.stock_cap) for asset in assets}
        outside = [asset for asset in assets if asset not in eligible and current_weights.get(asset, 0.0) > 0]
        return {"assets": assets, "returns": returns, "prices": prices, "covariance": covariance, "upper": upper, "investable": investable, "current": current_weights, "sectors": sectors, "outside": outside}

    @staticmethod
    def _quadratic_utility(problem, profile):
        """直接求解与 PyPortfolioOpt 相同的最大二次效用，但支持现金与换手硬约束。"""
        assets = problem["assets"]
        returns = problem["returns"].to_numpy(dtype=float)
        covariance = problem["covariance"].to_numpy(dtype=float)
        current_total = np.array([problem["current"].get(asset, 0.0) for asset in assets], dtype=float)
        weight = cp.Variable(len(assets), nonneg=True)
        sector_upper = {sector: profile.sector_cap for sector in set(problem["sectors"].values())}
        constraints = [weight <= np.array([problem["upper"][asset] for asset in assets], dtype=float), cp.sum(weight) <= problem["investable"], cp.norm1(weight - current_total) <= profile.turnover_cap]
        if problem["outside"]:
            outside_indices = [assets.index(asset) for asset in problem["outside"]]
            constraints.append(cp.sum(weight[outside_indices]) <= OUTSIDE_POOL_CAP)
        for sector, cap in sector_upper.items():
            members = [index for index, asset in enumerate(assets) if problem["sectors"][asset] == sector]
            if members:
                constraints.append(cp.sum(weight[members]) <= cap)
        objective = returns @ weight - profile.risk_aversion / 2 * cp.quad_form(weight, covariance) - 0.1 * cp.sum_squares(weight) - 0.001 * cp.norm1(weight - current_total)
        problem_obj = cp.Problem(cp.Maximize(objective), constraints)
        problem_obj.solve(solver=cp.CLARABEL)
        if problem_obj.status not in ("optimal", "optimal_inaccurate"):
            raise ValueError(f"OPTIMIZER_INFEASIBLE:{problem_obj.status}")
        values = np.maximum(np.asarray(weight.value, dtype=float), 0.0)
        return {asset: float(values[index]) for index, asset in enumerate(assets)}

    def _hrp_or_failure(self, problem, current_weights, sectors, eligible, profile, primary_error):
        try:
            returns = problem["prices"].pct_change().dropna(how="all")
            clean = HRPOpt(returns=returns).optimize()
            weights = {asset: float(clean.get(asset, 0.0) * problem["investable"]) for asset in problem["assets"]}
            weights = self._project_numerical_tolerance(weights, current_weights, sectors, eligible, profile)
            self._validate(weights, current_weights, sectors, eligible, profile)
            return OptimizationResult("ok", "hrp", weights, 1 - sum(weights.values()), ("PRIMARY_OPTIMIZER_FAILED",))
        except Exception as fallback_error:
            reasons = (f"PRIMARY_OPTIMIZER_FAILED:{type(primary_error).__name__}", f"HRP_FAILED:{type(fallback_error).__name__}")
            return OptimizationResult("insufficient_evidence", None, {}, 1.0, reasons)

    @staticmethod
    def _project_numerical_tolerance(weights, current, sectors, eligible, profile):
        projected = {asset: max(0.0, min(weight, profile.stock_cap)) for asset, weight in weights.items()}
        for asset in projected:
            if asset not in eligible:
                projected[asset] = min(projected[asset], current.get(asset, 0.0))
        outside_total = sum(projected[asset] for asset in projected if asset not in eligible and current.get(asset, 0.0) > 0)
        if outside_total > OUTSIDE_POOL_CAP:
            ratio = OUTSIDE_POOL_CAP / outside_total
            for asset in list(projected):
                if asset not in eligible and current.get(asset, 0.0) > 0:
                    projected[asset] *= ratio
        for sector in set(sectors.values()):
            members = [asset for asset in projected if sectors[asset] == sector]
            total = sum(projected[asset] for asset in members)
            if total > profile.sector_cap:
                ratio = profile.sector_cap / total
                for asset in members:
                    projected[asset] *= ratio
        turnover = sum(abs(projected.get(asset, 0.0) - current.get(asset, 0.0)) for asset in set(projected) | set(current))
        if turnover > profile.turnover_cap:
            ratio = profile.turnover_cap / turnover
            projected = {asset: current.get(asset, 0.0) + ratio * (weight - current.get(asset, 0.0)) for asset, weight in projected.items()}
        return projected

    @staticmethod
    def _validate(weights, current, sectors, eligible, profile):
        tolerance = 1e-6
        if any(weight < -tolerance or weight > profile.stock_cap + tolerance for weight in weights.values()):
            raise ValueError("STOCK_CAP_VIOLATED")
        if 1 - sum(weights.values()) < profile.minimum_cash - tolerance:
            raise ValueError("CASH_FLOOR_VIOLATED")
        sector_totals: dict[str, float] = {}
        for asset, weight in weights.items():
            sector_totals[sectors[asset]] = sector_totals.get(sectors[asset], 0.0) + weight
            if asset not in eligible and weight > current.get(asset, 0.0) + tolerance:
                raise ValueError("INELIGIBLE_POSITION_INCREASED")
        if any(weight > profile.sector_cap + tolerance for weight in sector_totals.values()):
            raise ValueError("SECTOR_CAP_VIOLATED")
        outside_total = sum(weights.get(asset, 0.0) for asset in current if asset not in eligible and current[asset] > 0)
        if outside_total > OUTSIDE_POOL_CAP + tolerance:
            raise ValueError("OUTSIDE_POOL_WEIGHT_VIOLATED")
        turnover = sum(abs(weights.get(asset, 0.0) - current.get(asset, 0.0)) for asset in set(weights) | set(current))
        if turnover > profile.turnover_cap + tolerance:
            raise ValueError("TURNOVER_CAP_VIOLATED")


def build_simulated_orders(
    target_weights: dict[str, float],
    current_shares: dict[str, int],
    prices: dict[str, float],
    total_value: float,
    costs: SimulatedTradeCosts = DEFAULT_SIMULATED_COSTS,
) -> tuple[list[dict[str, Any]], float, float]:
    """按A股单位和费用假设生成可负担的模拟订单及剩余现金。"""
    orders: list[dict[str, Any]] = []
    assets = sorted(set(target_weights) | set(current_shares))
    if any(asset not in prices or float(prices[asset]) <= 0 for asset in assets):
        raise ValueError("PRICE_MISSING_OR_INVALID")
    cash = total_value - sum(int(current_shares.get(asset, 0)) * float(prices[asset]) for asset in current_shares)
    deltas = {asset: _target_delta(target_weights.get(asset, 0.0), int(current_shares.get(asset, 0)), float(prices[asset]), total_value) for asset in assets}
    fees = 0.0
    for asset in assets:
        if deltas[asset] < 0:
            order = _priced_order(asset, deltas[asset], float(prices[asset]), costs)
            orders.append(order)
            cash += order["cash_impact"]
            fees += order["estimated_fee"]
    for asset in assets:
        if deltas[asset] > 0:
            order = _affordable_buy(asset, deltas[asset], float(prices[asset]), cash, costs)
            if order is not None:
                orders.append(order)
                cash += order["cash_impact"]
                fees += order["estimated_fee"]
    return orders, max(0.0, cash), fees


def _target_delta(target_weight: float, current: int, price: float, total_value: float) -> int:
    desired = int(float(target_weight) * total_value / price)
    if desired >= current:
        target = current + (desired - current) // 100 * 100
    else:
        target = _legal_sell_target(current, desired)
    return target - current


def _legal_sell_target(current: int, desired: int) -> int:
    """返回不低于期望值且可由一次合法卖单达到的最小余额。"""
    odd_lot = current % 100
    candidates = set(range(odd_lot, current + 1, 100))
    candidates.update(range(0, current - odd_lot + 1, 100))
    candidates.add(current)
    feasible = [target for target in candidates if target >= max(0, desired)]
    return min(feasible) if feasible else current


def _priced_order(asset: str, delta: int, price: float, costs: SimulatedTradeCosts) -> dict[str, Any]:
    shares = abs(int(delta))
    side = "BUY" if delta > 0 else "SELL"
    notional = shares * price
    commission = max(costs.minimum_commission, notional * costs.commission_rate)
    stamp_duty = notional * costs.sell_stamp_duty_rate if side == "SELL" else 0.0
    fee = commission + stamp_duty
    cash_impact = -(notional + fee) if side == "BUY" else notional - fee
    return {"stock_code": asset, "side": side, "shares": shares, "reference_price": price, "notional": notional, "commission": commission, "stamp_duty": stamp_duty, "estimated_fee": fee, "cash_impact": cash_impact}


def _affordable_buy(asset: str, shares: int, price: float, cash: float, costs: SimulatedTradeCosts) -> dict[str, Any] | None:
    """费用计入现金后向下调整到可买的100股单位。"""
    candidate = shares // 100 * 100
    while candidate >= 100:
        order = _priced_order(asset, candidate, price, costs)
        if cash + order["cash_impact"] >= -1e-8:
            return order
        candidate -= 100
    return None
