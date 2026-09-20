#!/usr/bin/env python3
"""Kronos WebUI 安全启动入口。"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"
MIN_PYTHON = (3, 10)
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
REQUIRED_MODULES = (
    ("flask", "Flask"),
    ("numpy", "NumPy"),
    ("pandas", "Pandas"),
    ("plotly", "Plotly"),
    ("yaml", "PyYAML"),
    ("sklearn", "scikit-learn"),
)


def log(level: str, message: str) -> None:
    """输出中文启动日志。"""
    print(f"[{level}] {message}")


def _ensure_import_paths() -> None:
    """把脚本目录和项目根目录加入导入路径。"""
    for path in (SCRIPT_DIR, PROJECT_ROOT):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def check_python_version() -> bool:
    """检查解释器是否满足 Python 3.10 及以上要求。"""
    if sys.version_info < MIN_PYTHON:
        current = ".".join(str(part) for part in sys.version_info[:3])
        log("错误", f"需要 Python 3.10 或更高版本，当前版本为 {current}。")
        return False
    log("信息", f"Python 版本检查通过：{sys.version.split()[0]}。")
    return True


def find_missing_dependencies() -> list[str]:
    """返回加载 WebUI 和配置所需的缺失依赖名称。"""
    missing: list[str] = []
    for module_name, display_name in REQUIRED_MODULES:
        try:
            available = importlib.util.find_spec(module_name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            available = False
        if not available:
            missing.append(display_name)
    return missing


def print_manual_install_command() -> None:
    """打印手工安装提示，但不执行安装命令。"""
    log("提示", "启动程序不会自动安装依赖。请在项目根目录手工执行：")
    print("  python -m pip install -r requirements.txt")


def check_dependencies() -> bool:
    """检查运行依赖，缺失时安全失败。"""
    missing = find_missing_dependencies()
    if missing:
        log("错误", f"缺少 WebUI 运行依赖：{', '.join(missing)}。")
        print_manual_install_command()
        return False
    log("信息", "WebUI 运行依赖检查通过。")
    return True


def load_application() -> Any:
    """从项目根目录加载 Flask 应用。"""
    _ensure_import_paths()
    try:
        from webui.app import app
    except Exception as exc:
        raise RuntimeError(f"加载 WebUI 应用失败：{exc}") from exc
    return app


def read_web_config() -> tuple[str, int]:
    """读取并校验配置中的本机监听地址和端口。"""
    try:
        from decision.config import get_config

        web_config = get_config().webui
        host = web_config.host
        port = web_config.port
    except Exception as exc:
        raise RuntimeError(f"读取 WebUI 配置失败：{exc}") from exc
    if host not in LOOPBACK_HOSTS:
        raise RuntimeError("webui.host 只能使用 127.0.0.1、localhost 或 ::1。")
    return host, port


def is_check_only() -> bool:
    """判断是否只执行启动检查而不运行长期服务。"""
    return os.environ.get("KRONOS_STARTUP_CHECK_ONLY") == "1"


def main() -> int:
    """执行启动检查并按配置启动本机 WebUI。"""
    log("信息", "开始检查 Kronos WebUI 启动条件。")
    if not check_python_version() or not check_dependencies():
        return 1
    try:
        application = load_application()
        host, port = read_web_config()
    except RuntimeError as exc:
        log("错误", str(exc))
        return 1

    if is_check_only():
        log("信息", f"启动检查通过：{host}:{port}。仅检查模式未启动服务。")
        return 0

    log("信息", f"正在启动本机 WebUI：http://{host}:{port}")
    try:
        application.run(
            debug=False,
            host=host,
            port=port,
            use_reloader=False,
        )
    except KeyboardInterrupt:
        log("信息", "WebUI 已停止。")
        return 0
    except OSError as exc:
        log("错误", f"WebUI 启动失败，端口可能已被占用：{exc}")
        return 1
    except Exception as exc:
        log("错误", f"WebUI 运行失败：{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
