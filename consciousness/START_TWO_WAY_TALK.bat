@echo off
REM Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
REM
REM Start continuous two-way conversation with live volume feedback
REM NO BUTTON PRESSES - just talk naturally!

cd /d "%~dp0"

REM Check if API keys are set
if "%ANTHROPIC_API_KEY%"=="" (
    echo [X] ANTHROPIC_API_KEY not set!
    echo.
    echo Please set your API key as an environment variable:
    echo   1. Search for "Environment Variables" in Windows
    echo   2. Add new variable: ANTHROPIC_API_KEY
    echo   3. Value: your-api-key-from-anthropic.com
    echo.
    echo Or set temporarily:
    echo   set ANTHROPIC_API_KEY=your-key-here
    echo.
    pause
    exit /b 1
)

if "%ELEVENLABS_API_KEY%"=="" (
    echo [!] Warning: ELEVENLABS_API_KEY not set ^(voice disabled^)
    echo.
)

echo [*] Starting Continuous Two-Way Conversation...
echo    - No button presses needed
echo    - Live volume feedback
echo    - Can interrupt each other
echo.

python ech0_two_way_robust.py
pause
