@echo off
title Kronos Decision System
set PYTHON=D:\miniconda3\envs\kronos\python.exe
set PROJECT=C:\Users\xzh\Desktop\kronos

echo.
echo ================================================
echo        Kronos Daily Trading Decision System
echo ================================================
echo.

echo [1/3] Checking Python...
%PYTHON% --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [FAIL] Python not found: %PYTHON%
    pause
    exit /b 1
)
echo [ OK ] %PYTHON%

echo [2/3] Checking dependencies...
%PYTHON% -c "import flask, plotly, torch, akshare, yaml, loguru, pandas" >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] Missing deps, installing...
    %PYTHON% -m pip install werkzeug flask flask-cors plotly akshare baostock loguru -q
    if %errorlevel% neq 0 (
        echo [FAIL] Install failed.
        echo Run: pip install werkzeug flask flask-cors plotly akshare baostock loguru
        pause
        exit /b 1
    )
)
echo [ OK ] All dependencies ready

echo [3/3] Starting Web server...
echo.
echo   Prediction : http://localhost:7070
echo   Report     : http://localhost:7070/report
echo.
echo   Press Ctrl+C to stop
echo ================================================
echo.

start http://localhost:7070
cd /d "%PROJECT%"
%PYTHON% webui/run.py

pause
