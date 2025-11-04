@echo off
REM Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
REM
REM AI:OS ONE-CLICK BOOTSTRAP INSTALLER FOR WINDOWS
REM

setlocal enabledelayedexpansion

echo.
echo ========================================================================
echo                          AI:OS INSTALLER
echo              The Agentic Intelligence Operating System
echo ========================================================================
echo.

REM Check for Python
echo [1/7] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [X] Python not found!
    echo.
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Found Python %PYTHON_VERSION%

REM Check for pip
echo [2/7] Checking pip installation...
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [X] pip not found, installing...
    python -m ensurepip --upgrade
)
echo [OK] pip is installed

REM Get script directory
set AIOS_DIR=%~dp0
cd /d "%AIOS_DIR%"

REM Check core files
echo [3/7] Checking AI:OS core files...
if not exist "runtime.py" (
    echo [X] runtime.py not found!
    echo Please ensure you're running this from the AI:OS directory
    pause
    exit /b 1
)
if not exist "config.py" (
    echo [X] config.py not found!
    echo Please ensure you're running this from the AI:OS directory
    pause
    exit /b 1
)
echo [OK] Core files found

REM Install dependencies
echo [4/7] Installing Python dependencies...
echo   Installing numpy...
python -m pip install -q numpy>=1.24.0
echo   Installing scipy...
python -m pip install -q scipy>=1.10.0
echo   Installing torch (optional, may take a while)...
python -m pip install -q torch>=2.0.0 2>nul
echo   Installing psutil...
python -m pip install -q psutil 2>nul
echo   Installing requests...
python -m pip install -q requests 2>nul
echo [OK] Dependencies installed

REM Create launcher
echo [5/7] Creating desktop launcher...
set LAUNCHER_PATH=%USERPROFILE%\Desktop\Launch_AIOS.bat

(
echo @echo off
echo REM AI:OS Quick Launcher
echo cd /d "%AIOS_DIR%"
echo.
echo :menu
echo cls
echo ===============================================================
echo                    AI:OS Quick Launcher
echo ===============================================================
echo.
echo What would you like to do?
echo.
echo   1^) Boot AI:OS ^(verbose mode^)
echo   2^) Run Setup Wizard
echo   3^) Boot with Security Toolkit
echo   4^) Run in Forensic Mode ^(read-only^)
echo   5^) Execute Natural Language Command
echo   6^) View System Status
echo   7^) Exit
echo.
set /p choice="Enter choice [1-7]: "
echo.
if "%%choice%%"=="1" ^(
    python aios -v boot
    pause
    goto menu
^)
if "%%choice%%"=="2" ^(
    python aios wizard
    pause
    goto menu
^)
if "%%choice%%"=="3" ^(
    set AGENTA_SECURITY_SUITE=1
    python aios -v boot
    pause
    goto menu
^)
if "%%choice%%"=="4" ^(
    python aios --forensic -v boot
    pause
    goto menu
^)
if "%%choice%%"=="5" ^(
    set /p cmd="Enter command: "
    python aios -v prompt "%%cmd%%"
    pause
    goto menu
^)
if "%%choice%%"=="6" ^(
    python aios -v metadata
    pause
    goto menu
^)
if "%%choice%%"=="7" ^(
    exit
^)
echo Invalid choice
pause
goto menu
) > "%LAUNCHER_PATH%"

echo [OK] Launcher created: %LAUNCHER_PATH%

REM Test installation
echo [6/7] Testing installation...
python aios --help >nul 2>&1
if %errorlevel% neq 0 (
    echo [X] AI:OS test failed
    python aios --help
    pause
    exit /b 1
)
echo [OK] AI:OS is working correctly!

echo [7/7] Creating desktop shortcut...
powershell -Command "$WS = New-Object -ComObject WScript.Shell; $SC = $WS.CreateShortcut('%USERPROFILE%\Desktop\AIOS.lnk'); $SC.TargetPath = '%LAUNCHER_PATH%'; $SC.WorkingDirectory = '%AIOS_DIR%'; $SC.Description = 'AI:OS Launcher'; $SC.Save()" 2>nul
echo [OK] Desktop shortcut created

REM Print completion
echo.
echo ========================================================================
echo                      INSTALLATION COMPLETE!
echo ========================================================================
echo.
echo Quick Start:
echo   1. Double-click "AIOS.lnk" or "Launch_AIOS.bat" on your Desktop
echo   2. Or run: python aios -v boot
echo.
echo Common Commands:
echo   Boot system:       python aios -v boot
echo   Setup wizard:      python aios wizard
echo   Security mode:     set AGENTA_SECURITY_SUITE=1 ^&^& python aios -v boot
echo   Forensic mode:     python aios --forensic -v boot
echo.
echo Documentation: See CLAUDE.md in the AI:OS directory
echo.
echo Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light)
echo All Rights Reserved. PATENT PENDING.
echo ========================================================================
echo.
pause
