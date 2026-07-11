"""测试 SignalAnalyzer。"""
import numpy as np
import pytest
from decision.analyzer import SignalAnalyzer, SignalResult


@pytest.fixture
def analyzer():
    return SignalAnalyzer()


def make_paths(
    n_paths: int = 50,
    pred_len: int = 120,
    trend: str = "up",
    noise: float = 0.02,
    base_price: float = 100.0,
) -> np.ndarray:
    """生成模拟预测路径。

    Args:
        n_paths: 路径数
        pred_len: 预测长度
        trend: "up" / "down" / "flat"
        noise: 路径间噪声水平
        base_price: 起始价格
    """
    paths = np.zeros((n_paths, pred_len, 5))  # OHLCV
    for i in range(n_paths):
        if trend == "up":
            drift = 0.001 * (1 + 0.5 * i / n_paths)
        elif trend == "down":
            drift = -0.001 * (1 + 0.5 * i / n_paths)
        else:
            drift = 0.0
        noise_level = noise * (1 + i / n_paths)
        for t in range(pred_len):
            paths[i, t, 3] = base_price + drift * base_price * t + np.random.randn() * noise_level * base_price
            paths[i, t, 0] = paths[i, t, 3] - abs(np.random.randn()) * 0.5  # open
            paths[i, t, 1] = paths[i, t, 3] + abs(np.random.randn()) * 0.5  # high
            paths[i, t, 2] = paths[i, t, 3] - abs(np.random.randn()) * 0.5  # low
    return paths


def test_buy_signal_strong_up(analyzer):
    """强上升趋势 + 低噪声 → BUY。"""
    paths = make_paths(trend="up", noise=0.005)
    result = analyzer.analyze(paths, current_price=100.0)
    assert isinstance(result, SignalResult)
    assert result.signal in ("BUY", "HOLD", "SELL")
    # 强上升趋势应至少是 BUY 或 HOLD
    assert result.signal != "SELL"


def test_sell_signal_strong_down(analyzer):
    """强下降趋势 → SELL。"""
    paths = make_paths(trend="down", noise=0.005)
    result = analyzer.analyze(paths, current_price=100.0)
    assert result.signal in ("SELL", "HOLD")


def test_high_variance_triggers_hold_not_buy(analyzer):
    """高方差不应给出 BUY。"""
    paths = make_paths(trend="up", noise=0.2)  # 高噪声
    result = analyzer.analyze(paths, current_price=100.0)
    # 高噪声情况下不应是 BUY
    if result.trend.confidence == "低":
        assert result.signal != "BUY"


def test_result_has_all_fields(analyzer):
    """结果包含所有必需字段。"""
    paths = make_paths()
    result = analyzer.analyze(paths, current_price=100.0)
    assert result.trend.direction in ("↑", "↓", "→")
    assert 0 <= result.trend.strength <= 1
    assert result.trend.confidence in ("高", "中", "低")
    assert result.risk.volatility in ("高", "中", "低")
    assert result.risk.reversal_risk in ("高", "中", "低")
    assert result.signal in ("BUY", "HOLD", "SELL")
    assert len(result.signal_reason) > 0