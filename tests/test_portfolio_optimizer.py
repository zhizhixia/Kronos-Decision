"""组合硬约束、回退和 A 股订单取整测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio.optimizer import PortfolioOptimizer, annualize_expected_returns, build_simulated_orders


def _problem():
    rng = np.random.default_rng(7)
    assets = [f"{index:06d}" for index in range(20)]
    returns = rng.normal(0.0004, 0.01, size=(300, len(assets)))
    prices = pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), columns=assets)
    expected = annualize_expected_returns(pd.Series(np.linspace(-0.01, 0.03, len(assets)), index=assets))
    current = {asset: 0.045 for asset in assets}
    sectors = {asset: f"sector-{index % 4}" for index, asset in enumerate(assets)}
    return assets, prices, expected, current, sectors


def test_optimizer_respects_balanced_hard_constraints() -> None:
    assets, prices, expected, current, sectors = _problem()
    result = PortfolioOptimizer().optimize(expected, prices, current, sectors, set(assets), "balanced")
    assert result.status == "ok"
    assert result.method == "max_quadratic_utility"
    assert result.cash_weight >= 0.10 - 1e-6
    assert max(result.target_weights.values()) <= 0.08 + 1e-6
    for sector in set(sectors.values()):
        assert sum(weight for asset, weight in result.target_weights.items() if sectors[asset] == sector) <= 0.25 + 1e-6
    turnover = sum(abs(result.target_weights[asset] - current[asset]) for asset in assets)
    assert turnover <= 0.20 + 1e-6


def test_missing_sector_mapping_returns_insufficient_evidence() -> None:
    assets, prices, expected, current, sectors = _problem()
    sectors.pop(assets[0])
    result = PortfolioOptimizer().optimize(expected, prices, current, sectors, set(assets), "balanced")
    assert result.status == "insufficient_evidence"
    assert "INDUSTRY_MAPPING_MISSING" in result.reason_codes


def test_ineligible_holding_cannot_increase() -> None:
    assets, prices, expected, current, sectors = _problem()
    ineligible = assets[0]
    result = PortfolioOptimizer().optimize(expected, prices, current, sectors, set(assets[1:]), "balanced")
    assert result.status == "ok"
    assert result.target_weights[ineligible] <= current[ineligible] + 1e-6


def test_outside_pool_total_weight_capped_at_ten_percent() -> None:
    """池外股票（不在评估股票池）总组合权重不得超过 10%。"""
    assets, prices, expected, current, sectors = _problem()
    outside = assets[:3]
    eligible = set(assets[3:])
    current = {asset: 0.05 for asset in assets}
    result = PortfolioOptimizer().optimize(expected, prices, current, sectors, eligible, "balanced")
    assert result.status == "ok"
    outside_total = sum(result.target_weights[asset] for asset in outside)
    assert outside_total <= 0.10 + 1e-6
    assert all(result.target_weights[asset] <= current[asset] + 1e-6 for asset in outside)


def test_simulated_orders_round_buys_and_liquidate_odd_lot() -> None:
    orders, cash, fees = build_simulated_orders({"600519": 0.2, "000001": 0.0}, {"600519": 0, "000001": 50}, {"600519": 100.0, "000001": 10.0}, 100_000)
    buy = next(order for order in orders if order["side"] == "BUY")
    sell = next(order for order in orders if order["side"] == "SELL")
    assert buy["shares"] % 100 == 0
    assert sell["shares"] == 50
    assert sell["stamp_duty"] > 0
    assert all(order["estimated_fee"] > 0 for order in orders)
    assert fees == sum(order["estimated_fee"] for order in orders)
    assert cash >= 0


def test_simulated_orders_reduce_buy_when_fees_exceed_cash() -> None:
    orders, cash, fees = build_simulated_orders({"600519": 1.0}, {}, {"600519": 100.0}, 10_000)
    assert orders == []
    assert cash == 10_000
    assert fees == 0.0


def test_buy_from_existing_odd_lot_only_adds_round_lots() -> None:
    orders, _, _ = build_simulated_orders(
        {"600519": 0.025}, {"600519": 50}, {"600519": 10.0}, 100_000
    )
    buy = next(order for order in orders if order["side"] == "BUY")
    assert buy["shares"] == 200
    assert buy["shares"] % 100 == 0
