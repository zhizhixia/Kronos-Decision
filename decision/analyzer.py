"""决策信号分析器。

接收 Kronos 预测结果，计算多维决策信号并通过矩阵判定最终 BUY/HOLD/SELL。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from decision.config import get_config


@dataclass
class TrendResult:
    """趋势分析结果。"""
    direction: str         # "↑" / "↓" / "→"
    strength: float        # 0.0 ~ 1.0
    confidence: str        # "高" / "中" / "低"
    up_probability: float  # 0.0 ~ 1.0


@dataclass
class RiskResult:
    """风险评估结果。"""
    var_95: float          # 95% VaR（负数表示潜在亏损比例）
    volatility: str        # "高" / "中" / "低"
    reversal_risk: str     # "高" / "中" / "低"


@dataclass
class SignalResult:
    """最终决策信号。"""
    signal: str | None     # "BUY" / "HOLD" / "SELL"；无效输入时为空
    signal_reason: str     # 人类可读的决策理由
    trend: TrendResult
    risk: RiskResult
    score_detail: dict = field(default_factory=dict)
    analysis_valid: bool = True
    reason_codes: tuple[str, ...] = field(default_factory=tuple)


class SignalAnalyzer:
    """决策信号分析器。

    用法:
        analyzer = SignalAnalyzer()
        result = analyzer.analyze(prediction_paths, current_price)
    """

    def __init__(self) -> None:
        cfg = get_config().signal
        self._buy_threshold = cfg.trend_strength_buy_threshold
        self._sell_threshold = cfg.trend_strength_sell_threshold
        self._var_high_threshold = cfg.var_risk_high_threshold
        self._consistency_threshold = cfg.consistency_std_threshold
        self._rsi_oversold = cfg.reversal_rsi_oversold
        self._rsi_overbought = cfg.reversal_rsi_overbought

    def analyze(
        self,
        prediction_paths: np.ndarray,
        current_price: float,
        historical_closes: np.ndarray | None = None,
    ) -> SignalResult:
        """分析预测路径，输出决策信号。

        Args:
            prediction_paths: 形状 (sample_count, pred_len, n_features)
                             每个 path 的最后一维是预测的 OHLCV
            current_price: 当前最新收盘价
            historical_closes: 历史收盘价序列（用于反转信号检测）

        Returns:
            结构化的 SignalResult
        """
        try:
            paths = np.asarray(prediction_paths, dtype=float)
        except (TypeError, ValueError):
            return self._invalid_result("PREDICTION_PATHS_INVALID")
        if (
            paths.ndim < 3
            or paths.shape[0] == 0
            or paths.shape[1] == 0
            or paths.shape[-1] <= 3
        ):
            return self._invalid_result("PREDICTION_PATHS_EMPTY")
        if not np.isfinite(paths).all() or not np.isfinite(current_price) or current_price <= 0:
            return self._invalid_result("PREDICTION_PATHS_NONFINITE")
        if historical_closes is not None:
            try:
                historical = np.asarray(historical_closes, dtype=float)
            except (TypeError, ValueError):
                return self._invalid_result("HISTORICAL_CLOSES_INVALID")
            if historical.size and not np.isfinite(historical).all():
                return self._invalid_result("HISTORICAL_CLOSES_NONFINITE")

        # 提取各路径的最终价格
        final_prices = paths[:, -1, 3]  # close 是第 4 列

        # ── 趋势分析 ──────────────────────────
        trend = self._analyze_trend(final_prices, current_price)

        # ── 风险评估 ──────────────────────────
        risk = self._analyze_risk(paths, final_prices, current_price)

        # ── 反转信号（如有历史数据） ──────────
        if historical_closes is not None and len(historical_closes) > 30:
            risk.reversal_risk = self._detect_reversal(np.asarray(historical_closes, dtype=float))

        # ── 决策矩阵判定 ──────────────────────
        signal, reason = self._decide(trend, risk)

        return SignalResult(
            signal=signal,
            signal_reason=reason,
            trend=trend,
            risk=risk,
            score_detail={
                "trend_strength": trend.strength,
                "up_probability": trend.up_probability,
                "calibrated_up_probability": None,
                "calibration_status": "unavailable",
                "var_95": risk.var_95,
                "volatility": risk.volatility,
                "reversal_risk": risk.reversal_risk,
                "consistency": trend.confidence,
            },
        )

    @staticmethod
    def _invalid_result(reason_code: str) -> SignalResult:
        """为无效输入返回不带动作的有限值结果，避免异常或 NaN 外泄。"""
        return SignalResult(
            signal=None,
            signal_reason=f"证据不足：{reason_code}。",
            trend=TrendResult(direction="→", strength=0.0, confidence="低", up_probability=0.0),
            risk=RiskResult(var_95=0.0, volatility="高", reversal_risk="高"),
            score_detail={
                "analysis_valid": False,
                "reason_codes": [reason_code],
                "calibrated_up_probability": None,
                "calibration_status": "unavailable",
            },
            analysis_valid=False,
            reason_codes=(reason_code,),
        )

    # ── 私有方法 ──────────────────────────────

    def _analyze_trend(
        self, final_prices: np.ndarray, current_price: float
    ) -> TrendResult:
        median_price = float(np.median(final_prices))
        change_ratio = (median_price - current_price) / current_price

        # 方向判定
        abs_change = abs(change_ratio)
        if change_ratio > 0.02:
            direction = "↑"
        elif change_ratio < -0.02:
            direction = "↓"
        else:
            direction = "→"

        # 强度：涨幅 + 方向占比归一化
        up_count = int(np.sum(final_prices > current_price))
        up_ratio = up_count / len(final_prices)
        strength = float(
            np.clip(abs_change * 10 * up_ratio, 0.0, 1.0)
        )

        # 一致性：路径间标准差
        path_std = float(np.std(final_prices) / current_price)
        if path_std < self._consistency_threshold:
            confidence = "高"
        elif path_std < self._consistency_threshold * 2:
            confidence = "中"
        else:
            confidence = "低"

        return TrendResult(
            direction=direction,
            strength=round(strength, 4),
            confidence=confidence,
            up_probability=round(up_ratio, 4),
        )

    def _analyze_risk(
        self,
        prediction_paths: np.ndarray,
        final_prices: np.ndarray,
        current_price: float,
    ) -> RiskResult:
        # VaR 95%：第 5 百分位的最大亏损
        var_95 = float(np.percentile(final_prices, 5))
        var_95_ratio = (var_95 - current_price) / current_price

        # 波动率：价格区间宽度
        price_range = float(np.std(final_prices) / current_price)
        if price_range > 0.10:
            volatility = "高"
        elif price_range > 0.05:
            volatility = "中"
        else:
            volatility = "低"

        return RiskResult(
            var_95=round(var_95_ratio, 4),
            volatility=volatility,
            reversal_risk="中",  # 默认值，有历史数据时覆盖
        )

    def _detect_reversal(self, closes: np.ndarray) -> str:
        """基于 RSI 检测反转风险。"""
        rsi = self._calc_rsi(closes[-14:])
        if rsi > self._rsi_overbought:
            return "高"
        elif rsi < self._rsi_oversold:
            return "高"
        elif rsi > self._rsi_overbought - 5:
            return "中"
        elif rsi < self._rsi_oversold + 5:
            return "中"
        return "低"

    @staticmethod
    def _calc_rsi(closes: np.ndarray, period: int = 14) -> float:
        diffs = np.diff(closes)
        gains = np.maximum(diffs, 0)
        losses = np.maximum(-diffs, 0)
        avg_gain = float(np.mean(gains[-period:])) if len(gains) >= period else float(np.mean(gains))
        avg_loss = float(np.mean(losses[-period:])) if len(losses) >= period else float(np.mean(losses))
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - 100.0 / (1.0 + rs))

    def _decide(self, trend: TrendResult, risk: RiskResult) -> tuple[str, str]:
        """决策矩阵判定。"""
        # 规则 6: 风险极高 → 回避（仅在潜在亏损超过阈值时触发）
        if risk.var_95 < -self._var_high_threshold:
            return "SELL", f"风险过大（VaR 95%: {risk.var_95:.1%}），建议回避"

        # 规则 1: 强趋势 + 高一致 + 低风险 → 买入
        if (
            trend.strength > self._buy_threshold
            and trend.confidence == "高"
            and risk.volatility != "高"
        ):
            return "BUY", "趋势明确上行，50条路径高度一致，风险可控"

        # 规则 2: 强趋势 + 高风险 → 偏多持有
        if trend.strength > self._buy_threshold and trend.confidence == "高":
            return "HOLD", "趋势上行但伴有较高风险，建议偏多持有"

        # 规则 3: 强趋势 + 低一致 → 持有
        if trend.strength > self._buy_threshold:
            return "HOLD", "趋势信号较强但路径间分歧大，模型不确定，建议持有观望"

        # 规则 4: 中等趋势 → 持有
        if trend.strength >= self._sell_threshold:
            return "HOLD", "趋势方向不明确，建议持有观望"

        # 中性或弱上行不能凭空变成卖出；只有明确下行才返回旧版 SELL。
        if trend.direction == "↓":
            return "SELL", "趋势明确下行，建议回避"
        return "HOLD", "趋势不足以支持买卖，建议持有观望"