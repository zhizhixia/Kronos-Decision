"""决策引擎。

编排完整决策流程，始终返回结构化 DecisionReport。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from decision.analyzer import SignalAnalyzer, SignalResult
from decision.config import get_config
from decision.errors import KronosError
from decision.model_manager import get_model_manager
from data.fetcher import DataFetcher


@dataclass
class DecisionReport:
    """决策报告数据类——Engine 的唯一输出格式。"""
    status: Literal["ok", "error", "degraded"]
    error_message: str = ""
    signal: SignalResult | None = None
    prediction: dict | None = None
    stock_code: str = ""
    stock_name: str = ""
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    elapsed_seconds: float = 0.0


class DecisionEngine:
    """决策引擎。

    用法:
        engine = DecisionEngine()
        report = engine.predict_and_analyze("600519", {})
    """

    def __init__(self) -> None:
        cfg = get_config()
        self._pred_cfg = cfg.prediction
        self._analyzer = SignalAnalyzer()
        self._fetcher = DataFetcher()
        self._model_manager = get_model_manager()

    def predict_and_analyze(
        self,
        stock_code: str,
        params: dict | None = None,
    ) -> DecisionReport:
        """执行完整决策流程。

        Args:
            stock_code: 6 位股票代码
            params: 可选参数字典，支持 pred_len / temperature / top_p

        Returns:
            始终返回 DecisionReport（无论成功失败）
        """
        start_time = time.time()
        params = params or {}

        try:
            # ── 1. 获取数据 ──────────────────────
            df = self._fetcher.fetch_daily(stock_code)
            stock_name = self._get_stock_name(stock_code)

            # 最终兜底：确保无 NaN
            required = ["open", "high", "low", "close"]
            optional = ["volume", "amount"]
            for col in required + optional:
                if col in df.columns:
                    df[col] = df[col].fillna(0.0)
            df = df.dropna(subset=required)

            # ── 2. 准备模型输入 ──────────────────
            pred_len = params.get("pred_len", self._pred_cfg.default_pred_len)
            max_context = params.get("max_context", get_config().model.max_context)
            temperature = params.get("temperature", self._pred_cfg.default_temperature)
            top_p = params.get("top_p", self._pred_cfg.default_top_p)

            # 取最近 max_context 行
            recent = df.iloc[-max_context:].copy()
            # x_df 保留 date 列（Kronos 需要 date 不在 index 中）
            x_df = recent[["date"] + required + optional].copy()
            x_df["date"] = pd.to_datetime(x_df["date"])
            x_timestamp = pd.Series(x_df["date"].values)  # 确保是 Series

            # 生成未来时间戳（用 pd.Series 而非 DatetimeIndex）
            y_timestamp = self._generate_future_timestamps(
                pd.Timestamp(df["date"].iloc[-1]), pred_len
            )

            # ── 3. 模型预测 ──────────────────────
            tokenizer, model = self._model_manager.get_predictor()

            pred_df = model.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=temperature,
                top_p=top_p,
                sample_count=self._pred_cfg.default_sample_count,
            )

            # ── 4. 信号分析 ──────────────────────
            current_price = float(df["close"].iloc[-1])
            historical_closes = df["close"].values

            # 从预测结果提取信息供分析器使用
            # pred_df: 行=预测步, 列=OHLCVAMT
            # 模拟多条路径：用预测值与微小噪声生成 20 条路径
            paths = self._simulate_paths(pred_df, n_paths=20)

            signal = self._analyzer.analyze(
                prediction_paths=paths,
                current_price=current_price,
                historical_closes=historical_closes,
            )

            elapsed = time.time() - start_time
            return DecisionReport(
                status="ok",
                signal=signal,
                prediction={
                    "pred_df": pred_df.to_dict(orient="records"),
                    "pred_len": pred_len,
                    "current_price": current_price,
                },
                stock_code=stock_code,
                stock_name=stock_name,
                elapsed_seconds=round(elapsed, 2),
            )

        except KronosError as e:
            elapsed = time.time() - start_time
            return DecisionReport(
                status="error",
                error_message=str(e),
                stock_code=stock_code,
                elapsed_seconds=round(elapsed, 2),
            )
        except Exception as e:
            elapsed = time.time() - start_time
            return DecisionReport(
                status="error",
                error_message=f"未知错误: {e}",
                stock_code=stock_code,
                elapsed_seconds=round(elapsed, 2),
            )

    # ── 私有方法 ──────────────────────────────

    @staticmethod
    def _generate_future_timestamps(
        last_date: pd.Timestamp, pred_len: int
    ) -> pd.Series:
        """生成未来交易日时间戳（跳过周末），返回 pd.Series 避免 .dt 问题。"""
        dates = []
        current = last_date + pd.Timedelta(days=1)
        while len(dates) < pred_len:
            if current.dayofweek < 5:  # 周一到周五
                dates.append(current)
            current += pd.Timedelta(days=1)
        return pd.Series(dates, name="date")

    @staticmethod
    def _simulate_paths(
        pred_df: pd.DataFrame, n_paths: int = 20
    ) -> np.ndarray:
        """从单次预测结果模拟多条采样路径。

        Kronos 的 predict() 返回一条平均路径。
        分析器需要多条路径来计算置信度和 VaR，
        因此用预测值 + 可控噪声生成模拟路径。

        Args:
            pred_df: Kronos 预测输出（pred_len 行 × OHLCVAMT 列）
            n_paths: 模拟路径数

        Returns:
            (n_paths, pred_len, 5) 的 numpy 数组（OHLCV）
        """
        price_cols = ["open", "high", "low", "close", "volume"]
        base = pred_df[price_cols].values.astype(np.float64)  # (pred_len, 5)
        std = np.std(base, axis=0) * 0.05  # 5% 噪声

        paths = np.zeros((n_paths, base.shape[0], 5))
        for i in range(n_paths):
            noise = np.random.randn(*base.shape) * std
            paths[i] = base + noise

        # 确保 high >= low 和 close 合理
        for i in range(n_paths):
            paths[i, :, 1] = np.maximum(paths[i, :, 1], paths[i, :, 2])  # high >= low
            paths[i, :, 2] = np.minimum(paths[i, :, 1], paths[i, :, 2])

        return paths

    @staticmethod
    def _get_stock_name(stock_code: str) -> str:
        """获取股票名称。"""
        try:
            from data.pool import get_hs300_pool
            pool = get_hs300_pool()
            return pool.get(stock_code, "")
        except Exception:
            return ""
