#!/bin/bash
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
#
# Ai:oS One-Click Installer
# Download and run: curl -fsSL https://raw.githubusercontent.com/YOUR_ORG/aios/main/INSTALL.sh | bash

set -e

echo ""
echo "╔═══════════════════════════════════════════╗"
echo "║     Ai:oS - One-Click Installation        ║"
echo "╚═══════════════════════════════════════════╝"
echo ""

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     PLATFORM=Linux;;
    Darwin*)    PLATFORM=macOS;;
    CYGWIN*)    PLATFORM=Windows;;
    MINGW*)     PLATFORM=Windows;;
    *)          PLATFORM="UNKNOWN:${OS}"
esac

echo "🔍 Detected platform: $PLATFORM"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python $PYTHON_VERSION found"

# Check if already in aios directory
if [ -f "aios" ] && [ -f "README.md" ]; then
    echo "✅ Already in aios directory"
    INSTALL_DIR="."
else
    # Clone or use existing
    if [ -d "aios" ]; then
        echo "✅ aios directory exists"
        INSTALL_DIR="aios"
    else
        echo "📥 Cloning aios repository..."
        git clone https://github.com/YOUR_ORG/aios.git
        INSTALL_DIR="aios"
    fi
fi

cd "$INSTALL_DIR"

# Install dependencies
echo ""
echo "📦 Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
    echo "✅ Dependencies installed"
else
    echo "⚠️  No requirements.txt found, skipping"
fi

# Install optional dependencies
echo ""
echo "🔧 Checking optional dependencies..."

# Check for quantum support
if python3 -c "import torch" 2>/dev/null; then
    echo "✅ PyTorch (quantum ML support)"
else
    echo "⚠️  PyTorch not found (optional: quantum ML)"
fi

# Check for Docker
if command -v docker &> /dev/null; then
    echo "✅ Docker (container orchestration)"
else
    echo "⚠️  Docker not found (optional: container support)"
fi

# Check for Ollama
if command -v ollama &> /dev/null || curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "✅ Ollama (autonomous AI agents)"
else
    echo "⚠️  Ollama not found (optional: autonomous agents)"
fi

# Make executable
echo ""
echo "🔧 Setting up executables..."
chmod +x aios
echo "✅ Made aios executable"

# Test installation
echo ""
echo "🧪 Testing installation..."
if ./aios --version &> /dev/null; then
    echo "✅ Installation successful!"
else
    echo "⚠️  Installation complete but test failed. Try running: ./aios --help"
fi

echo ""
echo "╔═══════════════════════════════════════════╗"
echo "║          Installation Complete!           ║"
echo "╚═══════════════════════════════════════════╝"
echo ""
echo "📚 Quick Start:"
echo ""
echo "  # Run the setup wizard:"
echo "  ./aios wizard"
echo ""
echo "  # Boot the system:"
echo "  ./aios boot"
echo ""
echo "  # See all commands:"
echo "  ./aios --help"
echo ""
echo "📖 Documentation: cat README.md"
echo "📖 Quick Start: cat QUICKSTART.md"
echo ""
echo "💚 Ai:oS is ready!"
echo ""
