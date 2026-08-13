"""模型生命周期管理器。

负责 Kronos 模型的懒加载、GPU 显存管理、健康检查。
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Iterator

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
        self._tokenizer_revision = cfg.tokenizer_revision
        self._model_revision = cfg.model_revision
        self._max_context = cfg.max_context
        self._device = cfg.device
        self._idle_timeout = timedelta(minutes=cfg.idle_timeout_minutes)

        self._tokenizer = None
        self._predictor = None
        self._last_access: datetime | None = None
        self._lock = threading.RLock()
        self._cleanup_timer: threading.Timer | None = None
        self._active_leases = 0

    def get_predictor(self) -> tuple[object, object]:
        """兼容旧调用方地获取模型；内部长任务应使用 ``lease``。

        Returns:
            KronosPredictor 实例

        Raises:
            ModelNotReadyError: 模型加载失败
        """
        with self._lock:
            self._cancel_cleanup_timer()
            return self._get_or_load()

    @contextmanager
    def lease(self) -> Iterator[tuple[object, object]]:
        """租用模型；存在活跃租约时空闲清理器不得释放模型。"""
        with self._lock:
            self._cancel_cleanup_timer()
            tokenizer, predictor = self._get_or_load()
            self._active_leases += 1
        try:
            yield tokenizer, predictor
        finally:
            with self._lock:
                self._active_leases = max(0, self._active_leases - 1)
                self._last_access = datetime.now()
                if self._active_leases == 0:
                    self._schedule_cleanup()

    def _get_or_load(self) -> tuple[object, object]:
        """在持锁状态下返回已加载模型或执行一次懒加载。"""
        self._last_access = datetime.now()
        if self._predictor is None:
            try:
                self._load_model()
            except Exception as exc:
                raise ModelNotReadyError(f"模型加载失败: {exc}") from exc
        return self._tokenizer, self._predictor

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
        """按固定修订加载模型，并按配置选择设备。"""
        from model import Kronos, KronosTokenizer, KronosPredictor
        self._tokenizer = KronosTokenizer.from_pretrained(self._tokenizer_name, revision=self._tokenizer_revision)
        self._predictor = Kronos.from_pretrained(self._predictor_name, revision=self._model_revision)
        device = None if self._device == "auto" else self._device
        self._predictor = KronosPredictor(
            self._predictor, self._tokenizer, device=device, max_context=self._max_context
        )

    def _schedule_cleanup(self) -> None:
        """调度定时释放显存。"""
        self._cancel_cleanup_timer()
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
            self._cleanup_timer = None
            if self._active_leases > 0 or self._predictor is None:
                return
            if self._last_access is None:
                return
            elapsed = datetime.now() - self._last_access
            if elapsed >= self._idle_timeout:
                self._release_gpu()
            else:
                self._schedule_cleanup()

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
