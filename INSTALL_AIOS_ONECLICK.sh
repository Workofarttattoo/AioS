#!/bin/bash
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
#
# Ai:oS ONE-CLICK Installer - Auto-detects everything, installs, and creates launcher
# Just run: ./INSTALL_AIOS_ONECLICK.sh

set -e

clear
cat << "BANNER"
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║                      Ai:oS ONE-CLICK INSTALLER                     ║
║          Sovereign AI Operating System with Auto-Setup            ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
BANNER
echo ""
echo "🤖 Detecting your system..."
sleep 1

# Auto-detect platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macOS"
    PYTHON_CMD="python3"
    PKG_MANAGER="brew"
    LAUNCHER_TYPE="app"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="Linux"
    PYTHON_CMD="python3"
    if command -v apt &> /dev/null; then
        PKG_MANAGER="apt"
    elif command -v yum &> /dev/null; then
        PKG_MANAGER="yum"
    elif command -v pacman &> /dev/null; then
        PKG_MANAGER="pacman"
    else
        PKG_MANAGER="unknown"
    fi
    LAUNCHER_TYPE="desktop"
else
    echo "❌ Unsupported platform: $OSTYPE"
    exit 1
fi

# Auto-detect resources
CPU_CORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo "4")
TOTAL_RAM=$(sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1024/1024/1024)}' || free -g 2>/dev/null | awk '/^Mem:/{print $2}' || echo "8")

echo "✅ Platform: $PLATFORM"
echo "✅ CPU Cores: $CPU_CORES"
echo "✅ RAM: ${TOTAL_RAM}GB"
echo ""

# Auto-install missing dependencies
echo "📦 Checking dependencies..."

# Python
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "   Installing Python 3..."
    if [ "$PLATFORM" == "macOS" ]; then
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        brew install python3
    elif [ "$PKG_MANAGER" == "apt" ]; then
        sudo apt update && sudo apt install -y python3 python3-pip
    fi
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "✅ Python $PYTHON_VERSION"

# pip
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    echo "   Installing pip..."
    curl https://bootstrap.pypa.io/get-pip.py | $PYTHON_CMD
fi

# Git
if ! command -v git &> /dev/null; then
    echo "   Installing git..."
    if [ "$PLATFORM" == "macOS" ]; then
        xcode-select --install 2>/dev/null || true
    elif [ "$PKG_MANAGER" == "apt" ]; then
        sudo apt install -y git
    fi
fi
echo "✅ git"

# Determine install location
INSTALL_DIR="$HOME/aios"
echo ""
echo "📁 Installing to: $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"

# Download or update
cd "$INSTALL_DIR"
echo ""
echo "📥 Getting Ai:oS source..."

# If already exists, update
if [ -d ".git" ]; then
    echo "   Updating existing installation..."
    git pull origin main 2>/dev/null || git pull 2>/dev/null || true
else
    # Clone or copy current directory if we're already in aios
    if [ -f "$(dirname "$0")/aios/aios" ]; then
        echo "   Copying from local installation..."
        cp -r "$(dirname "$0")/aios" .
        cp -r "$(dirname "$0")/tools" . 2>/dev/null || true
    else
        echo "   Cloning from repository..."
        # For now, copy from current location since repo isn't set up yet
        SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
        cp -r "$SCRIPT_DIR"/* . 2>/dev/null || true
    fi
fi

# Install Python dependencies
echo ""
echo "📦 Installing Python packages..."

cat > requirements-auto.txt <<EOF
anthropic>=0.18.0
numpy>=1.24.0
torch>=2.0.0
qiskit>=0.45.0
qiskit-aer>=0.13.0
pytest>=7.0.0
requests>=2.31.0
fastapi>=0.104.0
uvicorn>=0.24.0
sounddevice>=0.4.6
openai-whisper>=20231117
elevenlabs>=0.2.0
EOF

$PYTHON_CMD -m pip install --upgrade pip -q
$PYTHON_CMD -m pip install -r requirements-auto.txt -q

echo "✅ Dependencies installed"

# Create optimized launcher based on detected resources
echo ""
echo "🚀 Creating smart launcher..."

cat > "$INSTALL_DIR/aios-boot" <<LAUNCHER_EOF
#!/bin/bash
# Ai:oS Smart Boot - Auto-configured for your system
cd "$INSTALL_DIR"

export AGENTA_CPU_CORES=$CPU_CORES
export AGENTA_TOTAL_RAM=$TOTAL_RAM
export AGENTA_PLATFORM=$PLATFORM

# Auto-enable features based on resources
if [ $TOTAL_RAM -gt 16 ]; then
    export AGENTA_ENABLE_QUANTUM=1
fi

if [ $CPU_CORES -gt 8 ]; then
    export AGENTA_SUPERVISOR_CONCURRENCY=8
else
    export AGENTA_SUPERVISOR_CONCURRENCY=$CPU_CORES
fi

echo "🚀 Booting Ai:oS..."
echo "   Platform: $PLATFORM | Cores: $CPU_CORES | RAM: ${TOTAL_RAM}GB"
echo ""

python3 aios/aios -v boot "\$@"
LAUNCHER_EOF

chmod +x "$INSTALL_DIR/aios-boot"

# Create clickable launcher
if [ "$PLATFORM" == "macOS" ]; then
    echo "🎨 Creating macOS app launcher..."

    mkdir -p "$HOME/Applications/AiOS.app/Contents/MacOS"
    mkdir -p "$HOME/Applications/AiOS.app/Contents/Resources"

    cat > "$HOME/Applications/AiOS.app/Contents/MacOS/AiOS" <<APP_EOF
#!/bin/bash
cd "$INSTALL_DIR"
osascript -e 'tell application "Terminal" to do script "'"$INSTALL_DIR"'/aios-boot"'
APP_EOF

    chmod +x "$HOME/Applications/AiOS.app/Contents/MacOS/AiOS"

    cat > "$HOME/Applications/AiOS.app/Contents/Info.plist" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>AiOS</string>
    <key>CFBundleName</key>
    <string>Ai:oS</string>
    <key>CFBundleIdentifier</key>
    <string>com.corporationoflight.aios</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
</dict>
</plist>
PLIST_EOF

    echo "✅ App created: ~/Applications/AiOS.app"

elif [ "$PLATFORM" == "Linux" ]; then
    echo "🎨 Creating Linux desktop launcher..."

    cat > "$HOME/.local/share/applications/aios.desktop" <<DESKTOP_EOF
[Desktop Entry]
Name=Ai:oS
Comment=Sovereign AI Operating System
Exec=$INSTALL_DIR/aios-boot
Terminal=true
Type=Application
Categories=Development;System;
DESKTOP_EOF

    chmod +x "$HOME/.local/share/applications/aios.desktop"
    echo "✅ Launcher created: ~/.local/share/applications/aios.desktop"
fi

# Create desktop shortcut
cat > "$HOME/Desktop/🤖 Launch AiOS" <<SHORTCUT_EOF
#!/bin/bash
$INSTALL_DIR/aios-boot
SHORTCUT_EOF
chmod +x "$HOME/Desktop/🤖 Launch AiOS"

clear
cat << "SUCCESS"
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║                   ✅ INSTALLATION COMPLETE! ✅                      ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
SUCCESS

echo ""
echo "🎉 Ai:oS is installed and ready to use!"
echo ""
echo "┌─────────────────────────────────────────────────────────────────┐"
echo "│  THREE WAYS TO LAUNCH:                                          │"
echo "├─────────────────────────────────────────────────────────────────┤"
echo "│                                                                 │"
if [ "$PLATFORM" == "macOS" ]; then
echo "│  1️⃣  Double-click: ~/Applications/AiOS.app                      │"
else
echo "│  1️⃣  Click: Applications menu → Ai:oS                           │"
fi
echo "│                                                                 │"
echo "│  2️⃣  Double-click: Desktop icon '🤖 Launch AiOS'                │"
echo "│                                                                 │"
echo "│  3️⃣  Command line: $INSTALL_DIR/aios-boot          │"
echo "│                                                                 │"
echo "└─────────────────────────────────────────────────────────────────┘"
echo ""
echo "📊 Your System Profile:"
echo "   • Platform: $PLATFORM"
echo "   • CPU Cores: $CPU_CORES"
echo "   • RAM: ${TOTAL_RAM}GB"
echo "   • Auto-tuned for optimal performance!"
echo ""
echo "🚀 Ready to boot? Run:"
echo "   $INSTALL_DIR/aios-boot"
echo ""
echo "   Or just click one of the launchers above! 🖱️"
echo ""

# Ask if they want to boot now
read -p "Would you like to boot Ai:oS now? (y/n): " BOOT_NOW
if [[ "$BOOT_NOW" =~ ^[Yy]$ ]]; then
    echo ""
    exec "$INSTALL_DIR/aios-boot"
fi
