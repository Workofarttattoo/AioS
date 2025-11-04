#!/bin/bash
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
#
# Ai:oS One-Click Installer
# Download and run: curl -fsSL https://raw.githubusercontent.com/your-username/aios/main/install_aios.sh | bash

set -e

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                   Ai:oS One-Click Installer                       ║"
echo "║       Sovereign AI Operating System with Security Toolkit         ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if running on macOS or Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macOS"
    PYTHON_CMD="python3"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="Linux"
    PYTHON_CMD="python3"
else
    echo -e "${RED}❌ Unsupported platform: $OSTYPE${NC}"
    echo "   Ai:oS currently supports macOS and Linux"
    exit 1
fi

echo -e "${BLUE}🖥️  Platform: $PLATFORM${NC}"
echo ""

# Check for Python 3.8+
echo -e "${YELLOW}🔍 Checking Python...${NC}"
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found!${NC}"
    echo "   Install Python 3.8+ from https://www.python.org/"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version | awk '{print $2}')
echo -e "${GREEN}✅ Python $PYTHON_VERSION found${NC}"

# Check for git
echo -e "${YELLOW}🔍 Checking git...${NC}"
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ git not found!${NC}"
    echo "   Install git from https://git-scm.com/"
    exit 1
fi
echo -e "${GREEN}✅ git found${NC}"

# Ask for installation directory
echo ""
echo -e "${BLUE}📁 Where would you like to install Ai:oS?${NC}"
read -p "   Install path (default: ~/aios): " INSTALL_PATH
INSTALL_PATH=${INSTALL_PATH:-~/aios}
INSTALL_PATH="${INSTALL_PATH/#\~/$HOME}"

# Create directory
mkdir -p "$INSTALL_PATH"
cd "$INSTALL_PATH"

echo ""
echo -e "${YELLOW}📥 Downloading Ai:oS...${NC}"

# Clone repository
if [ -d ".git" ]; then
    echo "   Updating existing installation..."
    git pull
else
    git clone https://github.com/corporationoflight/aios.git .
fi

echo -e "${GREEN}✅ Ai:oS downloaded${NC}"

# Install dependencies
echo ""
echo -e "${YELLOW}📦 Installing dependencies...${NC}"

if [ ! -f "requirements.txt" ]; then
    echo -e "${YELLOW}⚠️  No requirements.txt found, creating minimal setup${NC}"
    cat > requirements.txt <<EOF
anthropic>=0.18.0
numpy>=1.24.0
torch>=2.0.0
qiskit>=0.45.0
pytest>=7.0.0
EOF
fi

$PYTHON_CMD -m pip install --upgrade pip
$PYTHON_CMD -m pip install -r requirements.txt

echo -e "${GREEN}✅ Dependencies installed${NC}"

# Install Sovereign Security Toolkit
echo ""
echo -e "${YELLOW}🔒 Installing Sovereign Security Toolkit...${NC}"
if [ -d "tools" ]; then
    echo "   Security tools available:"
    for tool in tools/*.py; do
        if [ -f "$tool" ]; then
            basename "$tool" .py | sed 's/^/   - /'
        fi
    done
    echo -e "${GREEN}✅ Security toolkit ready${NC}"
fi

# Check for quantum capabilities
echo ""
echo -e "${YELLOW}🔬 Checking quantum capabilities...${NC}"
if $PYTHON_CMD -c "import qiskit" 2>/dev/null; then
    echo -e "${GREEN}✅ Quantum simulation ready (Qiskit found)${NC}"
else
    echo -e "${YELLOW}⚠️  Quantum features require Qiskit${NC}"
    echo "   Install with: pip install qiskit"
fi

# Create launcher script
echo ""
echo -e "${YELLOW}🚀 Creating launcher...${NC}"

cat > "$INSTALL_PATH/aios-launch" <<'LAUNCHER_EOF'
#!/bin/bash
# Ai:oS Launcher
cd "$(dirname "$0")"
python3 -m aios "$@"
LAUNCHER_EOF

chmod +x "$INSTALL_PATH/aios-launch"

# Add to PATH suggestion
echo ""
echo -e "${GREEN}✅ Installation complete!${NC}"
echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                    Ai:oS Quick Start                              ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""
echo "1. Add to your PATH (optional):"
echo "   export PATH=\"$INSTALL_PATH:\$PATH\""
echo ""
echo "2. Boot the system:"
echo "   cd $INSTALL_PATH"
echo "   python3 aios/aios -v boot"
echo ""
echo "3. Run setup wizard:"
echo "   python3 aios/aios wizard"
echo ""
echo "4. Run security tools:"
echo "   python3 -m tools.aurorascan --gui"
echo "   python3 -m tools.cipherspear --gui"
echo ""
echo "5. Check ML algorithms:"
echo "   python3 aios/ml_algorithms.py"
echo ""
echo "6. View documentation:"
echo "   cat $INSTALL_PATH/README.md"
echo ""
echo -e "${BLUE}📖 Full docs: https://github.com/corporationoflight/aios${NC}"
echo -e "${GREEN}🎉 Welcome to Ai:oS - The Sovereign AI Operating System!${NC}"
echo ""
