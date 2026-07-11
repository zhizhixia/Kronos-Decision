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

            x_df = df.iloc[-max_context:].set_index("date")
            x_timestamp = df.iloc[-max_context:]["date"]

            # 生成未来时间戳
            y_timestamp = self._generate_future_timestamps(
                df["date"].iloc[-1], pred_len
            )

            # ── 3. 模型预测 ──────────────────────
            tokenizer, predictor = self._model_manager.get_predictor()

            pred_df = predictor.predict(
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

            signal = self._analyzer.analyze(
                prediction_paths=self._extract_paths(pred_df),
                current_price=current_price,
                historical_closes=historical_closes,
            )

            elapsed = time.time() - start_time
            return DecisionReport(
                status="ok",
                signal=signal,
                prediction={"pred_df": pred_df.to_dict()},
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
    ) -> pd.DatetimeIndex:
        """生成未来交易日时间戳（跳过周末）。"""
        dates = []
        current = last_date + pd.Timedelta(days=1)
        while len(dates) < pred_len:
            if current.dayofweek < 5:  # 周一到周五
                dates.append(current)
            current += pd.Timedelta(days=1)
        return pd.DatetimeIndex(dates)

    @staticmethod
    def _extract_paths(pred_df: pd.DataFrame) -> np.ndarray:
        """从预测 DataFrame 提取路径数组。"""
        # Kronos predictor 返回格式需适配，此处做通用转换
        return pred_df.values.reshape(1, -1, 5)

    @staticmethod
    def _get_stock_name(stock_code: str) -> str:
        """获取股票名称。"""
        try:
            from data.pool import get_hs300_pool
            pool = get_hs300_pool()
            return pool.get(stock_code, "")
        except Exception:
            return ""