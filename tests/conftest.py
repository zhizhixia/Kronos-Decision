"""受限沙箱下 pytest 的临时目录替代实现。

默认 ``tmp_path`` 依赖系统临时目录的 basetemp 遍历，在受限环境中会被
拒绝访问；这里改为工作区内唯一目录，避免干扰正式代码路径。
"""
from __future__ import annotations

import uuid
from pathlib import Path

import pytest

_TMP_ROOT = Path(__file__).parent / ".tmp"


@pytest.fixture
def tmp_path() -> Path:
    path = _TMP_ROOT / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path
