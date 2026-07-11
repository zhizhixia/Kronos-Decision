@echo off
chcp 65001 >nul
title Kronos 决策系统

:: 配置
set PYTHON=D:\miniconda3\envs\kronos\python.exe
set PROJECT=C:\Users\xzh\Desktop\kronos

echo.
echo ╔══════════════════════════════════════════════╗
echo ║        Kronos 日频交易决策系统               ║
echo ╚══════════════════════════════════════════════╝
echo.

:: 检查依赖
echo [1/3] 检查 Python 环境...
%PYTHON% --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 未找到 Python: %PYTHON%
    pause
    exit /b 1
)
echo   ✅ %PYTHON%

:: 检查必需依赖
echo [2/3] 检查依赖包...
%PYTHON% -c "import flask, plotly, torch, akshare, yaml, loguru, pandas" >nul 2>&1
if %errorlevel% neq 0 (
    echo   ⚠️  缺失依赖，正在安装...
    %PYTHON% -m pip install werkzeug flask flask-cors plotly akshare baostock loguru -q
    if %errorlevel% neq 0 (
        echo   ❌ 安装失败，请手动运行: pip install werkzeug flask flask-cors plotly akshare baostock loguru
        pause
        exit /b 1
    )
)
echo   ✅ 所有依赖就绪

:: 启动
echo [3/3] 启动 Web 服务...
echo.
echo   📊 预测页面:  http://localhost:7070
echo   🧭 决策报告:  http://localhost:7070/report
echo.
echo   按 Ctrl+C 停止服务
echo ══════════════════════════════════════════════
echo.

cd /d "%PROJECT%"
%PYTHON% webui/run.py

pause
