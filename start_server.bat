@echo off
title LLM Launcher v2
cd /d "%~dp0"

echo ================================================
echo   LLM Launcher Control Server - Startup
echo ================================================
echo.

echo [1/3] Checking configuration...
if not exist config.json (
    echo ERROR: config.json not found!
    pause
    exit /b 1
)

echo [2/3] Cleaning up any existing server processes...
REM Kill processes on port 8081 (control server)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8081" ^| findstr "LISTENING" 2^>nul') do (
    echo   - Terminating process on port 8081 (PID: %%a)
    taskkill /F /PID %%a >nul 2>&1
)

REM Kill processes on port 8080 (llama-server)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8080" ^| findstr "LISTENING" 2^>nul') do (
    echo   - Terminating process on port 8080 (PID: %%a)
    taskkill /F /PID %%a >nul 2>&1
)

echo [3/3] Starting LLM Launcher Control Server...
echo.
python -u llm_control_server.py
pause
