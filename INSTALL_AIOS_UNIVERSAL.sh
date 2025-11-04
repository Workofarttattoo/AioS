#!/bin/bash
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
#
# Ai:oS UNIVERSAL Installer - Works on macOS, Linux, and Windows (via Git Bash/WSL)
# Detects platform automatically and installs accordingly

set -e

# Detect if running in Windows environment
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ -n "$WINDIR" ]]; then
    echo "=========================================================================="
    echo "Windows Detected"
    echo "=========================================================================="
    echo ""
    echo "For best results on Windows, please run:"
    echo "  INSTALL_AIOS_WINDOWS.bat"
    echo ""
    echo "Or download from:"
    echo "  https://github.com/corporationoflight/aios/releases"
    echo ""
    exit 0
fi

# Run the existing macOS/Linux installer
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/INSTALL_AIOS_ONECLICK.sh"
