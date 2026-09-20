"""项目测试的导入边界。

测试应使用当前解释器环境中的依赖；被忽略的 vendor 目录不是依赖来源，
避免其中针对其他 Python 版本构建的影子包遮蔽受支持的环境依赖。
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
_VENDOR_ROOT = (_PROJECT_ROOT / "vendor").resolve()


def _is_vendor_path(entry: str) -> bool:
    """判断解释器搜索路径是否指向项目内被忽略的 vendor 目录。"""
    if not entry:
        return False
    try:
        return Path(entry).resolve() == _VENDOR_ROOT
    except (OSError, RuntimeError, ValueError):
        return False


sys.path[:] = [entry for entry in sys.path if not _is_vendor_path(entry)]
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))
