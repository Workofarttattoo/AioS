#!/bin/bash
# Ai|oS ISO Builder - Prerequisites Verification Script

set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Ai|oS ISO Builder - Prerequisites Check                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

ERRORS=0
WARNINGS=0

# Function to check command
check_command() {
    if command -v "$1" &> /dev/null; then
        echo "✓ $1 found: $(command -v $1)"
    else
        echo "✗ $1 NOT FOUND"
        ((ERRORS++))
    fi
}

# Function to check optional command
check_optional() {
    if command -v "$1" &> /dev/null; then
        echo "✓ $1 found: $(command -v $1)"
    else
        echo "⚠ $1 NOT FOUND (optional)"
        ((WARNINGS++))
    fi
}

# Function to check directory
check_directory() {
    if [ -d "$1" ]; then
        echo "✓ Directory exists: $1"
    else
        echo "✗ Directory NOT FOUND: $1"
        ((ERRORS++))
    fi
}

# Function to check file
check_file() {
    if [ -f "$1" ]; then
        echo "✓ File exists: $1"
    else
        echo "⚠ File NOT FOUND: $1"
        ((WARNINGS++))
    fi
}

# Detect OS
echo "═══ System Information ═══"
OS_TYPE="$(uname -s)"
echo "OS: $OS_TYPE"
echo "Architecture: $(uname -m)"
echo "Kernel: $(uname -r)"
echo ""

# Check required directories
echo "═══ Ai|oS Components ═══"
check_directory "/Users/noone/aios"
check_directory "/Users/noone/aios/agents"
check_directory "/Users/noone/aios/tools"
check_directory "/Users/noone/aios/web"
check_file "/Users/noone/aios/web/aios_launcher.html"
check_file "/Users/noone/aios/web/wolf_icon_head.png"
check_file "/Users/noone/aios/aios"
check_file "/Users/noone/aios/runtime.py"
check_file "/Users/noone/aios/config.py"
echo ""

# Check optional components
echo "═══ Optional Components ═══"
check_directory "/Users/noone/QuLab2.0"
check_directory "/Users/noone/TheGAVLSuite"
check_directory "/Users/noone/aios/red-team-tools"
echo ""

# Check Python
echo "═══ Python Environment ═══"
check_command "python3"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo "  Version: $PYTHON_VERSION"

    # Check required Python packages
    echo "  Checking Python packages..."
    for pkg in fastapi uvicorn numpy qiskit torch; do
        if python3 -c "import $pkg" 2>/dev/null; then
            echo "  ✓ $pkg installed"
        else
            echo "  ⚠ $pkg NOT installed"
            ((WARNINGS++))
        fi
    done
fi
echo ""

# Check build method
if [ "$OS_TYPE" = "Darwin" ]; then
    echo "═══ macOS Build Method ═══"
    echo "Recommended: Docker-based build"
    echo ""
    check_command "docker"
    if command -v docker &> /dev/null; then
        if docker info > /dev/null 2>&1; then
            echo "✓ Docker is running"
            DOCKER_MEM=$(docker info --format '{{.MemTotal}}' 2>/dev/null || echo "0")
            DOCKER_MEM_GB=$((DOCKER_MEM / 1024 / 1024 / 1024))
            echo "  Docker Memory: ${DOCKER_MEM_GB}GB"
            if [ "$DOCKER_MEM_GB" -lt 4 ]; then
                echo "  ⚠ WARNING: Docker has less than 4GB RAM"
                echo "  Recommendation: Increase to 8GB in Docker Desktop settings"
                ((WARNINGS++))
            else
                echo "  ✓ Docker has adequate memory"
            fi
        else
            echo "✗ Docker is NOT running"
            echo "  Please start Docker Desktop and try again"
            ((ERRORS++))
        fi
    fi
    check_optional "docker-compose"
    echo ""

    # Check disk space
    AVAILABLE_GB=$(df -g . | tail -1 | awk '{print $4}')
    echo "Available disk space: ${AVAILABLE_GB}GB"
    if [ "$AVAILABLE_GB" -lt 10 ]; then
        echo "⚠ WARNING: Less than 10GB free. Build may fail."
        ((WARNINGS++))
    else
        echo "✓ Adequate disk space"
    fi

elif [ "$OS_TYPE" = "Linux" ]; then
    echo "═══ Linux Build Method ═══"
    echo "Recommended: Native build"
    echo ""

    # Check Linux build tools
    echo "Required tools for native build:"
    check_command "debootstrap"
    check_command "mksquashfs"
    check_command "xorriso"
    check_command "isolinux"
    echo ""

    if [ "$ERRORS" -gt 0 ]; then
        echo "To install required tools:"
        echo "  sudo apt-get install debootstrap squashfs-tools xorriso \\"
        echo "    isolinux syslinux-efi grub-pc-bin grub-efi-amd64-bin mtools"
    fi

    # Check for sudo
    if sudo -n true 2>/dev/null; then
        echo "✓ sudo access available (passwordless)"
    else
        echo "⚠ sudo requires password (you'll be prompted during build)"
        ((WARNINGS++))
    fi
    echo ""

    # Check disk space
    AVAILABLE_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    echo "Available disk space: ${AVAILABLE_GB}GB"
    if [ "$AVAILABLE_GB" -lt 10 ]; then
        echo "⚠ WARNING: Less than 10GB free. Build may fail."
        ((WARNINGS++))
    else
        echo "✓ Adequate disk space"
    fi
else
    echo "⚠ Unsupported OS: $OS_TYPE"
    echo "Supported: macOS (Darwin), Linux"
    ((WARNINGS++))
fi

echo ""
echo "═══ ISO Builder Files ═══"
check_file "/Users/noone/aios/iso-builder/build-iso.sh"
check_file "/Users/noone/aios/iso-builder/build-iso-macos.sh"
check_file "/Users/noone/aios/iso-builder/Dockerfile.iso"
check_file "/Users/noone/aios/iso-builder/docker-compose.yml"
check_file "/Users/noone/aios/iso-builder/README.md"
check_file "/Users/noone/aios/iso-builder/QUICKSTART.md"
check_directory "/Users/noone/aios/iso-builder/output"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Verification Summary                                    ║"
echo "╠══════════════════════════════════════════════════════════╣"

if [ "$ERRORS" -eq 0 ] && [ "$WARNINGS" -eq 0 ]; then
    echo "║  ✓ All checks passed!                                    ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""
    echo "Ready to build!"
    echo ""
    if [ "$OS_TYPE" = "Darwin" ]; then
        echo "Run: ./build-iso-macos.sh"
    else
        echo "Run: ./build-iso.sh"
    fi
    exit 0
elif [ "$ERRORS" -eq 0 ]; then
    echo "║  ⚠ Passed with $WARNINGS warning(s)                        ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""
    echo "You can proceed, but some optional features may be missing."
    echo ""
    if [ "$OS_TYPE" = "Darwin" ]; then
        echo "Run: ./build-iso-macos.sh"
    else
        echo "Run: ./build-iso.sh"
    fi
    exit 0
else
    echo "║  ✗ Failed with $ERRORS error(s) and $WARNINGS warning(s)    ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""
    echo "Please fix the errors above before building."
    exit 1
fi
