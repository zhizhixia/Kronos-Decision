"""带超时的调用工具：daemon 线程执行，超时抛 TimeoutError 且不阻塞进程退出。"""
from __future__ import annotations

import threading
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def call_with_timeout(func: Callable[[], T], timeout: float, name: str = "call-with-timeout") -> T:
    """在 daemon 线程中执行 func，超过 timeout 秒抛 TimeoutError。

    线程设为 daemon：超时后调用方立即返回，后台线程不会拖住进程退出
    （网络请求自身的超时保护会在后台自然结束它）。
    """
    box: dict[str, Any] = {}
    done = threading.Event()

    def run() -> None:
        try:
            box["value"] = func()
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=run, daemon=True, name=name)
    thread.start()
    if not done.wait(timeout=timeout):
        raise TimeoutError(f"{name} 超过 {timeout} 秒")
    if "error" in box:
        raise box["error"]
    return box["value"]
