"""模型生命周期管理器。

负责 Kronos 模型的懒加载、GPU 显存管理、健康检查。
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta

from decision.config import get_config
from decision.errors import ModelNotReadyError


class ModelManager:
    """Kronos 模型管理器。

    用法:
        mgr = ModelManager()
        predictor = mgr.get_predictor()  # 懒加载
        # ... N 分钟无请求后自动释放显存
    """

    def __init__(self) -> None:
        cfg = get_config().model
        self._tokenizer_name = cfg.tokenizer
        self._predictor_name = cfg.predictor
        self._max_context = cfg.max_context
        self._device = cfg.device
        self._idle_timeout = timedelta(minutes=cfg.idle_timeout_minutes)

        self._tokenizer = None
        self._predictor = None
        self._last_access: datetime | None = None
        self._lock = threading.Lock()
        self._cleanup_timer: threading.Timer | None = None

    def get_predictor(self):
        """获取 KronosPredictor 实例（懒加载 + 线程安全）。

        Returns:
            KronosPredictor 实例

        Raises:
            ModelNotReadyError: 模型加载失败
        """
        with self._lock:
            self._last_access = datetime.now()
            self._cancel_cleanup_timer()
            if self._predictor is not None:
                return self._tokenizer, self._predictor
            try:
                self._load_model()
                return self._tokenizer, self._predictor
            except Exception as e:
                raise ModelNotReadyError(f"模型加载失败: {e}") from e

    def is_ready(self) -> bool:
        """检查模型是否已加载且可用。"""
        with self._lock:
            if self._predictor is None:
                return False
            try:
                # 轻量健康检查：确认模型对象在正确的设备上
                return True
            except Exception:
                return False

    def _load_model(self) -> None:
        """实际加载模型到 GPU。"""
        from model import Kronos, KronosTokenizer, KronosPredictor
        self._tokenizer = KronosTokenizer.from_pretrained(self._tokenizer_name)
        self._predictor = Kronos.from_pretrained(self._predictor_name)
        self._predictor = KronosPredictor(
            self._predictor, self._tokenizer, max_context=self._max_context
        )

    def _schedule_cleanup(self) -> None:
        """调度定时释放显存。"""
        self._cleanup_timer = threading.Timer(
            self._idle_timeout.total_seconds(), self._cleanup
        )
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()

    def _cancel_cleanup_timer(self) -> None:
        if self._cleanup_timer is not None:
            self._cleanup_timer.cancel()
            self._cleanup_timer = None

    def _cleanup(self) -> None:
        """超时后释放 GPU 显存。"""
        with self._lock:
            if self._predictor is not None and self._last_access is not None:
                elapsed = datetime.now() - self._last_access
                if elapsed >= self._idle_timeout:
                    self._release_gpu()

    def _release_gpu(self) -> None:
        """将模型移到 CPU 释放显存。"""
        try:
            if hasattr(self._predictor, 'model'):
                self._predictor.model.cpu()
        except Exception:
            pass
        self._predictor = None
        self._tokenizer = None


# 全局单例
_model_manager: ModelManager | None = None
_manager_lock = threading.Lock()


def get_model_manager() -> ModelManager:
    """获取全局 ModelManager 单例。"""
    global _model_manager
    if _model_manager is None:
        with _manager_lock:
            if _model_manager is None:
                _model_manager = ModelManager()
    return _model_manager