"""受限沙箱下 pytest 的临时目录与运行时副作用隔离。

默认 ``tmp_path`` 依赖系统临时目录的 basetemp 遍历，在受限环境中会被
拒绝访问；这里改为工作区内唯一目录，并把配置、缓存和日志重定向到
本次测试会话的临时根目录。
"""
from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path
from typing import Iterator

import pytest

_TMP_ROOT = Path(__file__).parent / ".tmp"
_SESSION_ROOT: Path | None = None


def _reset_runtime_paths() -> None:
    """为当前测试会话重置配置、缓存和日志路径。"""
    if _SESSION_ROOT is None:
        raise RuntimeError("测试会话临时根目录尚未初始化。")

    import decision.config as config_module

    isolated_config = config_module.Config()
    isolated_config.data.cache_dir = str(_SESSION_ROOT / "cache")
    isolated_config.data.snapshot_dir = str(_SESSION_ROOT / "snapshots")
    isolated_config.data.report_db = str(_SESSION_ROOT / "reports" / "decision.sqlite")
    isolated_config.data.portfolio_ledger = str(_SESSION_ROOT / "portfolio.json")
    isolated_config.logging.file = str(_SESSION_ROOT / "logs" / "kronos.log")
    config_module._CONFIG_PATH = _SESSION_ROOT / "config.yaml"
    config_module._config_instance = isolated_config

    pool_module = sys.modules.get("data.pool")
    if pool_module is not None:
        pool_module._CACHE_PATH = Path(isolated_config.data.cache_dir) / "pool_hs300.csv"


def pytest_configure(config) -> None:
    """在收集测试前建立会话级配置与缓存隔离。"""
    del config
    global _SESSION_ROOT
    _SESSION_ROOT = _TMP_ROOT / f"session-{uuid.uuid4().hex}"
    _SESSION_ROOT.mkdir(parents=True, exist_ok=True)
    _reset_runtime_paths()


def pytest_sessionfinish(session, exitstatus) -> None:
    """测试会话结束后仅清理本次创建的运行时目录。"""
    del session, exitstatus
    if _SESSION_ROOT is not None:
        shutil.rmtree(_SESSION_ROOT, ignore_errors=True)


@pytest.fixture(autouse=True)
def _isolate_runtime_paths():
    """在每个测试前恢复隔离配置，避免测试间共享可变单例。"""
    _reset_runtime_paths()
    yield


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    """返回会话隔离且在测试后清理的临时目录。"""
    if _SESSION_ROOT is None:
        raise RuntimeError("测试会话临时根目录尚未初始化。")
    path = _SESSION_ROOT / "tests" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
