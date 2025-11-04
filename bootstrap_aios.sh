#!/bin/bash
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
#
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    AI:OS ONE-CLICK BOOTSTRAP INSTALLER                    ║
# ║                     The Future of Agentic Operating Systems               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# This script installs everything you need to run AI:OS in one click.
# Compatible with: macOS, Linux, Windows (WSL/Git Bash)
#

set -e  # Exit on error

# Colors for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
else
    OS="unknown"
fi

# Print banner
print_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                           ║"
    echo "║                   █████╗ ██╗   ██████╗ ███████╗                         ║"
    echo "║                  ██╔══██╗██║  ██╔═══██╗██╔════╝                         ║"
    echo "║                  ███████║██║  ██║   ██║███████╗                         ║"
    echo "║                  ██╔══██║██║  ██║   ██║╚════██║                         ║"
    echo "║                  ██║  ██║██║  ╚██████╔╝███████║                         ║"
    echo "║                  ╚═╝  ╚═╝╚═╝   ╚═════╝ ╚══════╝                         ║"
    echo "║                                                                           ║"
    echo "║              The Agentic Intelligence Operating System                   ║"
    echo "║           One-Click Installer • Bootstrap Everything • Deploy Fast       ║"
    echo "║                                                                           ║"
    echo "╚═══════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo -e "${PURPLE}Detected OS: ${BOLD}$OS${NC}"
    echo ""
}

# Progress indicator
step_counter=0
total_steps=8

print_step() {
    step_counter=$((step_counter + 1))
    echo -e "${BLUE}[${step_counter}/${total_steps}]${NC} ${BOLD}$1${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${CYAN}ℹ${NC} $1"
}

# Check if running from correct directory
check_directory() {
    print_step "Checking installation directory..."

    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

    if [[ ! -f "$SCRIPT_DIR/runtime.py" ]] || [[ ! -f "$SCRIPT_DIR/config.py" ]]; then
        print_error "AI:OS core files not found in current directory!"
        echo ""
        print_info "This script should be run from the AI:OS directory containing:"
        echo "  - runtime.py"
        echo "  - config.py"
        echo "  - aios (main executable)"
        echo ""
        exit 1
    fi

    AIOS_DIR="$SCRIPT_DIR"
    print_success "Found AI:OS directory: $AIOS_DIR"
}

# Check and install Python
check_python() {
    print_step "Checking Python installation..."

    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        PYTHON_VERSION=$(python3 --version | awk '{print $2}')
        print_success "Found Python $PYTHON_VERSION"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
        PYTHON_VERSION=$(python --version | awk '{print $2}')
        print_success "Found Python $PYTHON_VERSION"
    else
        print_error "Python not found!"
        echo ""
        print_info "Installing Python..."

        if [[ "$OS" == "macos" ]]; then
            if command -v brew &> /dev/null; then
                brew install python3
                PYTHON_CMD="python3"
            else
                print_error "Homebrew not installed. Please install from https://brew.sh"
                exit 1
            fi
        elif [[ "$OS" == "linux" ]]; then
            if command -v apt-get &> /dev/null; then
                sudo apt-get update && sudo apt-get install -y python3 python3-pip
            elif command -v yum &> /dev/null; then
                sudo yum install -y python3 python3-pip
            else
                print_error "Could not detect package manager. Please install Python 3.8+ manually."
                exit 1
            fi
            PYTHON_CMD="python3"
        else
            print_error "Please install Python 3.8+ from https://python.org"
            exit 1
        fi

        print_success "Python installed successfully"
    fi
}

# Check pip
check_pip() {
    print_step "Checking pip installation..."

    if $PYTHON_CMD -m pip --version &> /dev/null; then
        print_success "pip is installed"
    else
        print_warning "pip not found, installing..."
        $PYTHON_CMD -m ensurepip --upgrade
        print_success "pip installed"
    fi
}

# Install dependencies
install_dependencies() {
    print_step "Installing Python dependencies..."

    # Core dependencies
    CORE_DEPS=(
        "numpy>=1.24.0"
        "scipy>=1.10.0"
    )

    # Optional but recommended
    OPTIONAL_DEPS=(
        "torch>=2.0.0"
        "psutil"
        "requests"
    )

    echo ""
    print_info "Installing core dependencies (required)..."
    for dep in "${CORE_DEPS[@]}"; do
        echo -n "  Installing $dep... "
        if $PYTHON_CMD -m pip install -q "$dep"; then
            echo -e "${GREEN}✓${NC}"
        else
            echo -e "${RED}✗${NC}"
            print_error "Failed to install $dep"
        fi
    done

    echo ""
    print_info "Installing optional dependencies (recommended)..."
    for dep in "${OPTIONAL_DEPS[@]}"; do
        echo -n "  Installing $dep... "
        if $PYTHON_CMD -m pip install -q "$dep" 2>/dev/null; then
            echo -e "${GREEN}✓${NC}"
        else
            echo -e "${YELLOW}⚠ skipped${NC}"
        fi
    done

    echo ""
    print_success "Dependencies installed"
}

# Create launcher script
create_launcher() {
    print_step "Creating launcher scripts..."

    # macOS/Linux launcher
    if [[ "$OS" == "macos" ]] || [[ "$OS" == "linux" ]]; then
        LAUNCHER_PATH="$HOME/Desktop/Launch_AIOS.command"

        cat > "$LAUNCHER_PATH" << 'EOF'
#!/bin/bash
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

AIOS_DIR="AIOS_DIR_PLACEHOLDER"

cd "$AIOS_DIR"

# Print banner
echo "════════════════════════════════════════════════════════════"
echo "               AI:OS Quick Launcher"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "What would you like to do?"
echo ""
echo "  1) Boot AI:OS (verbose mode)"
echo "  2) Run Setup Wizard"
echo "  3) Boot with Security Toolkit"
echo "  4) Run in Forensic Mode (read-only)"
echo "  5) Execute Natural Language Command"
echo "  6) Open AI:OS Terminal"
echo "  7) View System Status"
echo "  8) Exit"
echo ""
read -p "Enter choice [1-8]: " choice

case $choice in
    1)
        python3 aios -v boot
        ;;
    2)
        python3 aios wizard
        ;;
    3)
        export AGENTA_SECURITY_SUITE=1
        python3 aios -v boot
        ;;
    4)
        python3 aios --forensic -v boot
        ;;
    5)
        read -p "Enter command: " cmd
        python3 aios -v prompt "$cmd"
        ;;
    6)
        echo "Opening AI:OS terminal..."
        echo "Type 'python3 aios --help' for available commands"
        exec $SHELL
        ;;
    7)
        python3 aios -v metadata
        ;;
    8)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice"
        ;;
esac

echo ""
read -p "Press Enter to close..."
EOF

        # Replace placeholder with actual directory
        sed -i.bak "s|AIOS_DIR_PLACEHOLDER|$AIOS_DIR|g" "$LAUNCHER_PATH"
        rm -f "${LAUNCHER_PATH}.bak"

        chmod +x "$LAUNCHER_PATH"
        print_success "Launcher created: $LAUNCHER_PATH"

        # Also create a simple command-line alias
        SHELL_RC=""
        if [[ -f "$HOME/.zshrc" ]]; then
            SHELL_RC="$HOME/.zshrc"
        elif [[ -f "$HOME/.bashrc" ]]; then
            SHELL_RC="$HOME/.bashrc"
        elif [[ -f "$HOME/.bash_profile" ]]; then
            SHELL_RC="$HOME/.bash_profile"
        fi

        if [[ -n "$SHELL_RC" ]]; then
            if ! grep -q "alias aios=" "$SHELL_RC"; then
                echo "" >> "$SHELL_RC"
                echo "# AI:OS Quick Access" >> "$SHELL_RC"
                echo "alias aios='cd $AIOS_DIR && python3 aios'" >> "$SHELL_RC"
                print_success "Added 'aios' command alias to $SHELL_RC"
            fi
        fi
    fi
}

# Create desktop icon (macOS)
create_desktop_icon_macos() {
    if [[ "$OS" != "macos" ]]; then
        return
    fi

    print_step "Creating desktop application..."

    APP_PATH="$HOME/Desktop/AIOS.app"

    if [[ -d "$APP_PATH" ]]; then
        rm -rf "$APP_PATH"
    fi

    mkdir -p "$APP_PATH/Contents/MacOS"
    mkdir -p "$APP_PATH/Contents/Resources"

    # Create Info.plist
    cat > "$APP_PATH/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>launcher</string>
    <key>CFBundleName</key>
    <string>AI:OS</string>
    <key>CFBundleIdentifier</key>
    <string>com.corporationoflight.aios</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
</dict>
</plist>
EOF

    # Create launcher executable
    cat > "$APP_PATH/Contents/MacOS/launcher" << 'EOF'
#!/bin/bash
AIOS_DIR="AIOS_DIR_PLACEHOLDER"
open -a Terminal "$AIOS_DIR/Launch_AIOS.command"
EOF

    sed -i.bak "s|AIOS_DIR_PLACEHOLDER|$HOME/Desktop|g" "$APP_PATH/Contents/MacOS/launcher"
    rm -f "$APP_PATH/Contents/MacOS/launcher.bak"

    chmod +x "$APP_PATH/Contents/MacOS/launcher"

    print_success "Desktop app created: AIOS.app"
}

# Make main executable
make_executable() {
    print_step "Setting up main executable..."

    if [[ -f "$AIOS_DIR/aios" ]]; then
        chmod +x "$AIOS_DIR/aios"
        print_success "Main executable ready: $AIOS_DIR/aios"
    else
        print_warning "Main executable 'aios' not found"
    fi
}

# Test installation
test_installation() {
    print_step "Testing installation..."

    echo ""
    cd "$AIOS_DIR"

    if $PYTHON_CMD aios --help &> /dev/null; then
        print_success "AI:OS is working correctly!"
    else
        print_error "AI:OS test failed"
        echo ""
        print_info "Trying to diagnose..."
        $PYTHON_CMD aios --help 2>&1 | head -20
        exit 1
    fi
}

# Print completion message
print_completion() {
    echo ""
    echo -e "${GREEN}${BOLD}"
    echo "╔═══════════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                           ║"
    echo "║                    ✨ INSTALLATION COMPLETE ✨                            ║"
    echo "║                                                                           ║"
    echo "╚═══════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    echo -e "${BOLD}🚀 Quick Start:${NC}"
    echo ""

    if [[ "$OS" == "macos" ]] || [[ "$OS" == "linux" ]]; then
        echo -e "  ${CYAN}1.${NC} Double-click ${BOLD}Launch_AIOS.command${NC} on your Desktop"
        if [[ "$OS" == "macos" ]]; then
            echo -e "  ${CYAN}2.${NC} Or double-click the ${BOLD}AIOS.app${NC} icon"
        fi
        echo -e "  ${CYAN}3.${NC} Or type ${BOLD}aios${NC} in your terminal (restart terminal first)"
    else
        echo -e "  ${CYAN}1.${NC} Navigate to: ${BOLD}$AIOS_DIR${NC}"
        echo -e "  ${CYAN}2.${NC} Run: ${BOLD}python aios -v boot${NC}"
    fi

    echo ""
    echo -e "${BOLD}📚 Common Commands:${NC}"
    echo ""
    echo -e "  ${CYAN}Boot the system:${NC}        python3 aios -v boot"
    echo -e "  ${CYAN}Setup wizard:${NC}           python3 aios wizard"
    echo -e "  ${CYAN}Security mode:${NC}          AGENTA_SECURITY_SUITE=1 python3 aios -v boot"
    echo -e "  ${CYAN}Forensic mode:${NC}          python3 aios --forensic -v boot"
    echo -e "  ${CYAN}Natural language:${NC}       python3 aios -v prompt \"your command\""
    echo ""
    echo -e "${BOLD}📖 Documentation:${NC}"
    echo -e "  View ${CYAN}CLAUDE.md${NC} in the AI:OS directory for full documentation"
    echo ""
    echo -e "${BOLD}🌟 Features Included:${NC}"
    echo -e "  ${GREEN}✓${NC} Agentic Control Plane"
    echo -e "  ${GREEN}✓${NC} Sovereign Security Toolkit"
    echo -e "  ${GREEN}✓${NC} ML & Quantum Algorithms"
    echo -e "  ${GREEN}✓${NC} Autonomous Discovery System"
    echo -e "  ${GREEN}✓${NC} Natural Language Interface"
    echo ""
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}Copyright © 2025 Joshua Hendricks Cole (DBA: Corporation of Light)${NC}"
    echo -e "${BOLD}All Rights Reserved. PATENT PENDING.${NC}"
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Main installation flow
main() {
    print_banner

    check_directory
    check_python
    check_pip
    install_dependencies
    create_launcher
    create_desktop_icon_macos
    make_executable
    test_installation

    print_completion
}

# Run main
main
