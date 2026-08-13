"""证据门禁通过时的模拟调仓服务（永不执行真实交易）。"""
from __future__ import annotations

from typing import Any

import pandas as pd

from data.industry import IndustryMap
from evaluation.binding import EvaluationBinding
from portfolio.optimizer import PortfolioOptimizer, annualize_expected_returns, build_simulated_orders
from portfolio.store import PortfolioStore


class RebalanceService:
    """根据最新评估和本地组合生成目标权重与模拟订单。"""

    def __init__(self, store: PortfolioStore | None = None, industry_map: IndustryMap | None = None) -> None:
        self.store = store or PortfolioStore()
        self.industry_map = industry_map or IndustryMap()
        self.optimizer = PortfolioOptimizer()

    def rebalance(self, binding: EvaluationBinding, profile_id: str = "default", fetch_bars=None) -> dict[str, Any]:
        """执行一次受约束的模拟调仓；任何前提缺失都返回明确的证据不足。"""
        if binding is None or not binding.gate_result.get("passed"):
            return {"status": "insufficient_evidence", "reason_codes": ("EVIDENCE_GATE_FAILED",)}
        portfolio = self.store.get_portfolio(profile_id)
        evidence = self._latest_evidence_frame(binding)
        if evidence.empty:
            return {"status": "insufficient_evidence", "reason_codes": ("NO_EVALUATION_PREDICTIONS",)}
        sectors = self.industry_map.load()
        if not sectors:
            return {"status": "insufficient_evidence", "reason_codes": ("INDUSTRY_MAPPING_MISSING",)}
        current_shares = {holding["stock_code"]: holding["shares"] for holding in portfolio["holdings"] if holding["shares"] > 0}
        price_history, prices = self._prices(sorted(set(evidence.index) | set(current_shares)), fetch_bars)
        if price_history.empty:
            return {"status": "insufficient_evidence", "reason_codes": ("PRICE_HISTORY_MISSING",)}
        if any(code not in prices for code in current_shares):
            return {"status": "insufficient_evidence", "reason_codes": ("HELD_POSITION_PRICE_MISSING",)}
        total_value = portfolio["cash"] + sum(shares * prices[code] for code, shares in current_shares.items() if code in prices)
        current_weights = {code: shares * prices[code] / total_value for code, shares in current_shares.items() if code in prices}
        expected = annualize_expected_returns(evidence["median_excess_20d"])
        eligible = set(evidence.index)
        result = self.optimizer.optimize(expected, price_history, current_weights, sectors, eligible, portfolio["risk_profile"])
        if result.status != "ok":
            return {"status": "insufficient_evidence", "reason_codes": result.reason_codes}
        orders, remaining_cash, estimated_fees = build_simulated_orders(result.target_weights, current_shares, prices, total_value)
        run_id = self.store.save_rebalance(profile_id, {"weights": result.target_weights, "cash_weight": result.cash_weight}, orders, estimated_fees)
        return {"status": "ok", "run_id": run_id, "method": result.method, "target_weights": result.target_weights, "cash_weight": result.cash_weight, "orders": orders, "remaining_cash": remaining_cash, "estimated_fees": estimated_fees, "reason_codes": list(result.reason_codes)}

    @staticmethod
    def _latest_evidence_frame(binding: EvaluationBinding) -> pd.DataFrame:
        predictions = binding.predictions
        required = {"anchor_date", "stock_code", "horizon", "score", "raw_up_probability", "q05", "q50", "q95"}
        if predictions.empty or not required.issubset(predictions):
            return pd.DataFrame()
        frame = predictions[predictions["horizon"].astype(int) == 20].copy()
        if frame.empty:
            return pd.DataFrame()
        frame["anchor_date"] = pd.to_datetime(frame["anchor_date"])
        latest_anchor = frame["anchor_date"].max()
        latest = frame[frame["anchor_date"] == latest_anchor]
        latest["median_excess_20d"] = latest["q50"].astype(float)
        latest = latest.set_index(latest["stock_code"].astype(str).str.zfill(6))
        return latest[["median_excess_20d"]]

    def _prices(self, codes: list[str], fetch_bars) -> tuple[pd.DataFrame, dict[str, float]]:
        if fetch_bars is None:
            return pd.DataFrame(), {}
        frames = {}
        latest: dict[str, float] = {}
        for code in codes:
            bars = fetch_bars(code)
            if bars is None or bars.empty:
                continue
            series = bars.set_index("date")["close"].astype(float)
            frames[code] = series
            latest[code] = float(series.iloc[-1])
        if not frames:
            return pd.DataFrame(), {}
        return pd.DataFrame(frames).ffill().dropna(), latest
