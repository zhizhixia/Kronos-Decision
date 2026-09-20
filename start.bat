@echo off
setlocal EnableExtensions EnableDelayedExpansion
rem 统一 Windows 控制台为 UTF-8，避免中文启动日志乱码。
chcp 65001 >nul
title Kronos Decision System

rem 仅使用脚本所在目录和现有解释器，不修改 PATH，不下载额外解释器。
set "PROJECT=%~dp0"
set "PYTHON="
set "CONDA_ENV_NAME=%KRONOS_CONDA_ENV%"
if not defined CONDA_ENV_NAME set "CONDA_ENV_NAME=kronos"

echo [信息] 项目目录：!PROJECT!
echo [信息] 检查 Python 3.10 及以上版本...

rem 优先使用项目内已有虚拟环境。
if exist "!PROJECT!.venv\Scripts\python.exe" call :try_python "!PROJECT!.venv\Scripts\python.exe"

rem 如果调用者已经激活 Conda，优先复用当前环境，不执行 activate 或安装。
if not defined PYTHON if defined CONDA_PREFIX if exist "!CONDA_PREFIX!\python.exe" call :try_python "!CONDA_PREFIX!\python.exe"

rem 如果存在 Conda，优先查找指定的完整项目环境。
if not defined PYTHON (
    where conda >nul 2>&1
    if not errorlevel 1 (
        for /f "delims=" %%P in ('call conda run --no-capture-output -n "!CONDA_ENV_NAME!" python -c "import sys; print(sys.executable)" 2^>nul') do if not defined PYTHON call :try_python "%%P"
    )
)

rem 使用 PATH 中已有的完整 python，不安装、不修改 PATH。
if not defined PYTHON (
    for /f "delims=" %%P in ('where python 2^>nul') do if not defined PYTHON call :try_python "%%P"
)

rem 使用已安装的 Python Launcher 选择解释器；不会触发下载。
if not defined PYTHON (
    where py >nul 2>&1
    if not errorlevel 1 (
        for /f "delims=" %%P in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do if not defined PYTHON call :try_python "%%P"
    )
)

if not defined PYTHON (
    echo [失败] 未找到包含 WebUI 和研究页面依赖的 Python 3.10 或更高版本。
    echo        已优先尝试 Conda 环境：!CONDA_ENV_NAME!；本脚本不会安装依赖或修改 PATH。
    echo        请先在项目环境中手工安装 requirements.txt 后重试。
    exit /b 1
)
echo [信息] 使用解释器：!PYTHON!

if not exist "!PROJECT!requirements.txt" (
    echo [失败] 未找到依赖文件：!PROJECT!requirements.txt
    exit /b 1
)

echo [信息] 检查 WebUI 依赖...
"!PYTHON!" -c "import importlib.util, sys; missing = [name for name in ('flask', 'numpy', 'pandas', 'plotly', 'yaml', 'sklearn') if importlib.util.find_spec(name) is None]; print(','.join(missing)); raise SystemExit(1 if missing else 0)"
if errorlevel 1 (
    echo [失败] 缺少 WebUI 或研究页面依赖，未执行任何安装操作。
    echo [提示] 请在项目根目录手工执行：
    echo         "!PYTHON!" -m pip install -r "!PROJECT!requirements.txt"
    exit /b 1
)
echo [信息] WebUI 依赖检查通过。

if /i "!KRONOS_STARTUP_CHECK_ONLY!"=="1" (
    echo [信息] 仅检查模式通过，未启动长期服务。
    exit /b 0
)

echo [信息] 启动本机 WebUI。
echo        地址由 decision/config.yaml 中的 webui.host 和 webui.port 决定。
pushd "!PROJECT!" >nul 2>&1
if errorlevel 1 (
    echo [失败] 无法进入项目目录。
    exit /b 1
)
"!PYTHON!" "!PROJECT!webui\run.py"
set "EXIT_CODE=!ERRORLEVEL!"
popd
exit /b !EXIT_CODE!

:try_python
set "CANDIDATE=%~1"
if not exist "!CANDIDATE!" exit /b 0
"!CANDIDATE!" -c "import importlib.util, sys; required = ('flask', 'numpy', 'pandas', 'plotly', 'yaml', 'sklearn'); raise SystemExit(0 if sys.version_info >= (3, 10) and all(importlib.util.find_spec(name) is not None for name in required) else 1)" >nul 2>&1
if errorlevel 1 exit /b 0
set "PYTHON=!CANDIDATE!"
exit /b 0
