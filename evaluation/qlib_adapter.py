"""Qlib 正式回测适配器；所有 Qlib 导入均延迟到真实运行。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.contracts import validate_prediction_frame


@dataclass(frozen=True)
class QlibBacktestConfig:
    """A 股日频正式回测的固定执行约束。"""

    provider_uri: str = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")
    benchmark: str = "SH000300"
    account: float = 100_000_000.0
    topk: int = 10
    n_drop: int = 2
    open_cost: float = 0.0003
    close_cost: float = 0.0013
    min_cost: float = 5.0
    slippage: float = 0.0005
    volume_limit: float = 0.10
    benchmark_is_total_return: bool = False


def build_qlib_signal(predictions: pd.DataFrame) -> pd.Series:
    """把20日预测转换为 Qlib 要求的 execution_date/instrument 信号。"""
    # predictions.csv 同时保留 5/20/60 三个期限用于校准，回测只消费 20 日信号
    frame = predictions[predictions["horizon"].astype(int) == 20].copy()
    checked = validate_prediction_frame(frame, horizon=20)
    instruments = checked["stock_code"].map(_to_qlib_code)
    index = pd.MultiIndex.from_arrays(
        [checked["execution_date"], instruments], names=["datetime", "instrument"]
    )
    signal = pd.Series(checked["score"].astype(float).to_numpy(), index=index, name="score")
    if signal.index.duplicated().any():
        raise ValueError("同一执行日和股票只能有一个正式信号。")
    return signal.sort_index()


class QlibBacktestAdapter:
    """以固定信号文件调用 Qlib executor，不实现自制正式回测器。"""

    def __init__(self, config: QlibBacktestConfig | None = None) -> None:
        self.config = config or QlibBacktestConfig()

    def run(self, predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """执行 Qlib 回测，返回日度报告和持仓对象。"""
        signal = build_qlib_signal(predictions)
        self._validate_environment()
        qlib, backtest, executor, strategy_class = self._imports()
        qlib.init(provider_uri=self.config.provider_uri, region="cn")
        try:
            from qlib.config import C
            C["joblib_backend"] = "sequential"
            C["n_jobs"] = 1
        except Exception:
            pass
        strategy = strategy_class(signal=signal, topk=self.config.topk, n_drop=self.config.n_drop, hold_thresh=1, only_tradable=True)
        simulator = executor.SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True)
        signal_dates = signal.index.get_level_values("datetime")
        start_time = signal_dates.min()
        end_time = signal_dates.max() + pd.Timedelta(days=90)  # 覆盖 60 个交易日的持有期
        try:
            from qlib.data import D
            calendar = pd.DatetimeIndex(D.calendar(start_time=str(signal_dates.max())))
            if len(calendar) >= 2:
                # qlib 步进需要 end_time 之后仍有交易日，截断到倒数第二个
                end_time = min(end_time, calendar[-2])
        except Exception:
            pass
        metrics, indicators = backtest(
            executor=simulator,
            strategy=strategy,
            start_time=start_time,
            end_time=end_time,
            account=self.config.account,
            benchmark=self.config.benchmark,
            exchange_kwargs=self._exchange_kwargs(),
        )
        report, positions = metrics["1day"]
        return report, {"positions": positions, "indicators": indicators}

    def _exchange_kwargs(self) -> dict[str, Any]:
        return {"freq": "day", "limit_threshold": 0.095, "deal_price": "$open", "open_cost": self.config.open_cost, "close_cost": self.config.close_cost, "min_cost": self.config.min_cost, "impact_cost": self.config.slippage, "trade_unit": 100, "volume_threshold": ("current", f"$volume * {self.config.volume_limit}")}

    def _validate_environment(self) -> None:
        if not Path(self.config.provider_uri).exists():
            raise RuntimeError(f"Qlib 中国市场数据目录不存在：{self.config.provider_uri}")

    @staticmethod
    def _imports():
        try:
            import qlib
            from qlib.backtest import backtest, executor
            from qlib.contrib.strategy import TopkDropoutStrategy
        except ImportError as exc:
            raise RuntimeError("未安装 pyqlib，无法执行正式研究回测。") from exc
        return qlib, backtest, executor, TopkDropoutStrategy


def _to_qlib_code(stock_code: str) -> str:
    market = "SH" if str(stock_code).startswith(("5", "6", "9")) else "SZ"
    return f"{market}{str(stock_code).zfill(6)}"
