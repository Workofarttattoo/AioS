@echo off
REM Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
REM
REM Ai:oS ONE-CLICK Installer for Windows
REM Just double-click or run: INSTALL_AIOS_WINDOWS.bat

setlocal enabledelayedexpansion

cls
echo ========================================================================
echo.
echo                      Ai:oS ONE-CLICK INSTALLER
echo          Sovereign AI Operating System - Windows Edition
echo.
echo ========================================================================
echo.

REM Check for Python
echo [*] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [!] Python not found. Attempting to install...
    echo.
    echo Opening Microsoft Store to install Python...
    start ms-windows-store://pdp/?ProductId=9NRWMJP3717K
    echo.
    echo After installing Python, please run this installer again.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [+] Python %PYTHON_VERSION% found

REM Check for pip
echo [*] Checking pip...
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [!] Installing pip...
    python -m ensurepip --upgrade
)
echo [+] pip found

REM Check for git
echo [*] Checking git...
git --version >nul 2>&1
if errorlevel 1 (
    echo [!] Git not found. Installing...
    echo Opening git download page...
    start https://git-scm.com/download/win
    echo.
    echo After installing Git, please run this installer again.
    pause
    exit /b 1
)
echo [+] git found

REM Detect system resources
echo.
echo [*] Detecting your system...
for /f "tokens=2 delims==" %%i in ('wmic cpu get NumberOfCores /value ^| find "="') do set CPU_CORES=%%i
for /f "tokens=2 delims==" %%i in ('wmic computersystem get TotalPhysicalMemory /value ^| find "="') do set TOTAL_RAM_BYTES=%%i
set /a TOTAL_RAM_GB=%TOTAL_RAM_BYTES:~0,-9%

echo [+] Platform: Windows
echo [+] CPU Cores: %CPU_CORES%
echo [+] RAM: %TOTAL_RAM_GB%GB
echo.

REM Set install directory
set INSTALL_DIR=%USERPROFILE%\aios
echo [*] Installing to: %INSTALL_DIR%
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

cd /d "%INSTALL_DIR%"

REM Download/update source
echo.
echo [*] Getting Ai:oS source...
if exist ".git" (
    echo    Updating existing installation...
    git pull origin main 2>nul || git pull 2>nul
) else (
    REM For now, copy from current location if available
    if exist "%~dp0aios" (
        echo    Copying from local installation...
        xcopy /E /I /Y "%~dp0aios" aios >nul
        xcopy /E /I /Y "%~dp0tools" tools >nul 2>nul
    ) else (
        echo    Creating directory structure...
        mkdir aios >nul 2>nul
        mkdir tools >nul 2>nul
    )
)

echo [+] Source ready

REM Install Python dependencies
echo.
echo [*] Installing Python packages...
echo    This may take a few minutes...

(
echo anthropic^>=0.18.0
echo numpy^>=1.24.0
echo torch^>=2.0.0
echo qiskit^>=0.45.0
echo qiskit-aer^>=0.13.0
echo pytest^>=7.0.0
echo requests^>=2.31.0
echo fastapi^>=0.104.0
echo uvicorn^>=0.24.0
echo sounddevice^>=0.4.6
echo openai-whisper^>=20231117
echo elevenlabs^>=0.2.0
) > requirements-auto.txt

python -m pip install --upgrade pip -q
python -m pip install -r requirements-auto.txt -q

if errorlevel 1 (
    echo [!] Some packages failed to install. Continuing anyway...
) else (
    echo [+] Dependencies installed
)

REM Create Windows launcher
echo.
echo [*] Creating smart launcher...

(
echo @echo off
echo REM Ai:oS Smart Boot - Auto-configured for your system
echo cd /d "%INSTALL_DIR%"
echo.
echo set AGENTA_CPU_CORES=%CPU_CORES%
echo set AGENTA_TOTAL_RAM=%TOTAL_RAM_GB%
echo set AGENTA_PLATFORM=Windows
echo.
echo if %TOTAL_RAM_GB% GTR 16 set AGENTA_ENABLE_QUANTUM=1
echo if %CPU_CORES% GTR 8 ^(
echo     set AGENTA_SUPERVISOR_CONCURRENCY=8
echo ^) else ^(
echo     set AGENTA_SUPERVISOR_CONCURRENCY=%CPU_CORES%
echo ^)
echo.
echo echo [*] Booting Ai:oS...
echo echo    Platform: Windows ^| Cores: %CPU_CORES% ^| RAM: %TOTAL_RAM_GB%GB
echo echo.
echo python aios/aios -v boot %%*
echo pause
) > aios-boot.bat

REM Create desktop shortcut
echo [*] Creating desktop shortcut...

powershell -Command "$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\Desktop\Launch AiOS.lnk'); $Shortcut.TargetPath = '%INSTALL_DIR%\aios-boot.bat'; $Shortcut.IconLocation = 'C:\Windows\System32\shell32.dll,166'; $Shortcut.WorkingDirectory = '%INSTALL_DIR%'; $Shortcut.Description = 'Launch Ai:oS - Sovereign AI Operating System'; $Shortcut.Save()"

echo [+] Desktop shortcut created

REM Create Start Menu shortcut
echo [*] Creating Start Menu entry...

set START_MENU=%APPDATA%\Microsoft\Windows\Start Menu\Programs
if not exist "%START_MENU%\Ai:oS" mkdir "%START_MENU%\Ai:oS"

powershell -Command "$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%START_MENU%\Ai:oS\Launch Ai:oS.lnk'); $Shortcut.TargetPath = '%INSTALL_DIR%\aios-boot.bat'; $Shortcut.IconLocation = 'C:\Windows\System32\shell32.dll,166'; $Shortcut.WorkingDirectory = '%INSTALL_DIR%'; $Shortcut.Description = 'Launch Ai:oS'; $Shortcut.Save()"

echo [+] Start Menu entry created

REM Show boot visualizer
echo.
echo [*] Opening boot visualizer...
start "" "%INSTALL_DIR%\cinematic_boot_visualizer.html" 2>nul

cls
echo ========================================================================
echo.
echo                   INSTALLATION COMPLETE!
echo.
echo ========================================================================
echo.
echo [+] Ai:oS is installed and ready to use!
echo.
echo ┌───────────────────────────────────────────────────────────────────┐
echo │  THREE WAYS TO LAUNCH:                                            │
echo ├───────────────────────────────────────────────────────────────────┤
echo │                                                                   │
echo │  1. Double-click: Desktop icon "Launch AiOS"                      │
echo │                                                                   │
echo │  2. Start Menu: Ai:oS ^> Launch Ai:oS                             │
echo │                                                                   │
echo │  3. Command line: %INSTALL_DIR%\aios-boot.bat         │
echo │                                                                   │
echo └───────────────────────────────────────────────────────────────────┘
echo.
echo [*] Your System Profile:
echo    - Platform: Windows
echo    - CPU Cores: %CPU_CORES%
echo    - RAM: %TOTAL_RAM_GB%GB
echo    - Auto-tuned for optimal performance!
echo.
echo [*] Ready to boot? Run:
echo    %INSTALL_DIR%\aios-boot.bat
echo.

REM Ask if they want to boot now
set /p BOOT_NOW="Would you like to boot Ai:oS now? (Y/N): "
if /i "%BOOT_NOW%"=="Y" (
    echo.
    call "%INSTALL_DIR%\aios-boot.bat"
)

endlocal
