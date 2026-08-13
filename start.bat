@echo off
setlocal enabledelayedexpansion
title Kronos Decision System
set "PROJECT=%~dp0"

echo.
echo ================================================
echo        Kronos Daily Trading Decision System
echo ================================================
echo.

echo [1/3] Locating Python...
set "PYTHON="

rem 1) Project .venv (validate version)
if exist "%PROJECT%\.venv\Scripts\python.exe" call :try_python "%PROJECT%\.venv\Scripts\python.exe"

rem 2) py launcher: -3.11, then -3.10 (resolve to real path, then validate)
if not defined PYTHON (
    where py >nul 2>&1
    if not errorlevel 1 (
        for /f "usebackq delims=" %%P in (`py -3.11 -c "import sys; print(sys.executable)" 2^>nul`) do if not defined PYTHON call :try_python "%%P"
        for /f "usebackq delims=" %%P in (`py -3.10 -c "import sys; print(sys.executable)" 2^>nul`) do if not defined PYTHON call :try_python "%%P"
    )
)

rem 3) Per-user Programs\Python\Python311 / Python310 (validate version)
if not defined PYTHON if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" call :try_python "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PYTHON if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" call :try_python "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"

rem 4) PATH python (last resort, resolve via where then validate)
if not defined PYTHON (
    for /f "usebackq delims=" %%P in (`where python 2^>nul`) do if not defined PYTHON call :try_python "%%P"
)

rem 5) uv offline: create project .venv from an interpreter already on this machine.
rem    Never select APPDATA bare uv python, never --break-system-packages, never auto-download.
if not defined PYTHON (
    where uv >nul 2>&1
    if not errorlevel 1 (
        set "UV_PY="
        for /f "usebackq delims=" %%P in (`uv python find 3.11 --no-python-downloads 2^>nul`) do if not defined UV_PY set "UV_PY=%%P"
        if not defined UV_PY for /f "usebackq delims=" %%P in (`uv python find 3.10 --no-python-downloads 2^>nul`) do if not defined UV_PY set "UV_PY=%%P"
        if defined UV_PY (
            echo [INFO] Creating project .venv with uv -- offline --
            uv venv --offline --python 3.11 --no-python-downloads --seed "%PROJECT%\.venv" >nul 2>&1
            if errorlevel 1 uv venv --offline --python 3.10 --no-python-downloads --seed "%PROJECT%\.venv" >nul 2>&1
            if not errorlevel 1 if exist "%PROJECT%\.venv\Scripts\python.exe" call :try_python "%PROJECT%\.venv\Scripts\python.exe"
        )
    )
)

if not defined PYTHON (
    echo [FAIL] Python 3.10 or 3.11 not found.
    echo        Install Python 3.11 from python.org and check "Add Python to PATH".
    where uv >nul 2>&1
    if not errorlevel 1 (
        echo        Or install an interpreter for uv, then re-run start.bat:
        echo            uv python install 3.11
    )
    pause
    exit /b 1
)
echo [ OK ] !PYTHON!

echo [2/3] Checking dependencies...
"!PYTHON!" -c "import flask, plotly, torch, akshare, baostock, yaml, loguru, pandas, cvxpy, pypfopt, sklearn" >nul 2>&1
if errorlevel 1 (
    if /i "!KRONOS_STARTUP_CHECK_ONLY!"=="1" (
        echo [FAIL] Dependencies missing -- check-only mode, not installing --
        echo        Run: "!PYTHON!" -m pip install -r "!PROJECT!\requirements.txt"
        pause
        exit /b 1
    )
    echo [WARN] Missing deps, installing...
    "!PYTHON!" -m pip install -r "!PROJECT!\requirements.txt" -q
    if errorlevel 1 (
        echo [FAIL] Install failed.
        echo Run: "!PYTHON!" -m pip install -r "!PROJECT!\requirements.txt"
        pause
        exit /b 1
    )
)
echo [ OK ] All dependencies ready

if /i "!KRONOS_STARTUP_CHECK_ONLY!"=="1" (
    echo [ OK ] Startup checks passed
    exit /b 0
)

echo [3/3] Starting Web server...
echo.
echo   Prediction : http://127.0.0.1:7070
echo   Report     : http://127.0.0.1:7070/report
echo.
echo   Press Ctrl+C to stop
echo ================================================
echo.

cd /d "%PROJECT%"
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:7070"
"!PYTHON!" webui/run.py

pause
goto :eof

rem ---- subroutine: validate a candidate interpreter, set PYTHON if 3.10/3.11 ----
:try_python
set "CAND=%~1"
if not exist "!CAND!" goto :eof
"!CAND!" --version >nul 2>&1
if errorlevel 1 goto :eof
"!CAND!" -c "import sys; raise SystemExit(0 if sys.version_info[:2] in ((3,10),(3,11)) else 1)" >nul 2>&1
if errorlevel 1 goto :eof
set "PYTHON=!CAND!"
goto :eof