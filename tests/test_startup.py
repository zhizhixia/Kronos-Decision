"""安全启动入口的最小静态回归测试。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "webui" / "run.py"
LAUNCHER = PROJECT_ROOT / "start.bat"


def test_run_py_can_compile() -> None:
    """启动脚本必须能被当前解释器编译。"""
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(RUNNER)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_run_py_uses_project_root_and_safe_server_defaults() -> None:
    """启动脚本应从项目路径加载应用并关闭调试重载。"""
    source = RUNNER.read_text(encoding="utf-8")
    assert "PROJECT_ROOT = SCRIPT_DIR.parent" in source
    assert "from webui.app import app" in source
    assert "web_config.host" in source
    assert "web_config.port" in source
    assert "debug=False" in source
    assert "use_reloader=False" in source
    assert "0.0.0.0" not in source
    assert "install_dependencies" not in source


def test_start_bat_selects_a_complete_environment() -> None:
    """批处理入口必须选择包含研究页面依赖的环境。"""
    source = LAUNCHER.read_text(encoding="utf-8")
    lower = source.lower()
    assert "%~dp0" in source
    assert "kronos_startup_check_only" in lower
    assert "chcp 65001" in lower
    assert "conda run" in lower
    assert "conda_env_name" in lower
    assert "sklearn" in lower
    assert "set \"path=" not in lower
    assert "0.0.0.0" not in source
    assert "pip install" in lower
    install_lines = [
        line.strip().lower()
        for line in source.splitlines()
        if "pip install" in line.lower()
    ]
    assert install_lines
    assert all(line.startswith("echo ") or line.startswith("rem ") for line in install_lines)
