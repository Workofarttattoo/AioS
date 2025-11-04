#!/bin/bash
# Create SD Card Deployment Package for Sovereign Security Toolkit
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

set -e

echo "============================================"
echo "Creating SD Card Deployment Package"
echo "============================================"

# Configuration
PACKAGE_NAME="sovereign_security_toolkit_$(date +%Y%m%d)"
PACKAGE_DIR="/tmp/${PACKAGE_NAME}"
TOOLS_DIR="/Users/noone/aios/tools"

echo ""
echo "[*] Creating package directory: ${PACKAGE_DIR}"
mkdir -p "${PACKAGE_DIR}"

echo "[*] Copying toolkit files..."

# Copy all Python tools
cp "${TOOLS_DIR}"/*.py "${PACKAGE_DIR}/"

# Copy configuration files
cp "${TOOLS_DIR}"/requirements_pythief.txt "${PACKAGE_DIR}/"
cp "${TOOLS_DIR}"/Dockerfile.pythief "${PACKAGE_DIR}/"
cp "${TOOLS_DIR}"/build_pythief_image.sh "${PACKAGE_DIR}/"
cp "${TOOLS_DIR}"/__init__.py "${PACKAGE_DIR}/"

# Copy documentation
cp "${TOOLS_DIR}"/DEPLOYMENT_GUIDE.md "${PACKAGE_DIR}/"

# Copy templates
mkdir -p "${PACKAGE_DIR}/templates"
cp "${TOOLS_DIR}"/templates/*.html "${PACKAGE_DIR}/templates/"

echo "[*] Creating installation script..."

cat > "${PACKAGE_DIR}/install.sh" << 'INSTALL_EOF'
#!/bin/bash
# Sovereign Security Toolkit - Installation Script
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

set -e

echo "============================================"
echo "Sovereign Security Toolkit - Installation"
echo "============================================"
echo ""

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "[!] Cannot detect OS"
    exit 1
fi

echo "[*] Detected OS: $OS"
echo ""

# Install system dependencies
echo "[*] Installing system dependencies..."

if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ] || [ "$OS" = "kali" ]; then
    sudo apt update
    sudo apt install -y \
        python3 \
        python3-pip \
        python3-dev \
        build-essential \
        git \
        wget \
        curl \
        aircrack-ng \
        hostapd \
        dnsmasq \
        tcpdump \
        wireshark-common \
        tshark \
        nmap \
        netcat-openbsd

    echo "[*] ✓ System packages installed"
elif [ "$OS" = "arch" ]; then
    sudo pacman -Syu --noconfirm \
        python \
        python-pip \
        base-devel \
        git \
        wget \
        curl \
        aircrack-ng \
        hostapd \
        dnsmasq \
        tcpdump \
        wireshark-cli \
        nmap \
        openbsd-netcat

    echo "[*] ✓ System packages installed"
else
    echo "[!] Unsupported OS: $OS"
    echo "[!] Please install dependencies manually"
fi

echo ""
echo "[*] Installing Python dependencies..."
pip3 install -r requirements_pythief.txt

echo ""
echo "[*] Installing Ollama (for ECH0Py)..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
    echo "[*] ✓ Ollama installed"
else
    echo "[*] ✓ Ollama already installed"
fi

echo ""
echo "[*] Pulling ECH0Py model..."
ollama pull phi-2
echo "[*] ✓ Phi-2 model ready"

echo ""
echo "============================================"
echo "✓ Installation Complete!"
echo "============================================"
echo ""
echo "Quick Start:"
echo "  1. Setup authorization:"
echo "     python3 pythief.py --setup-auth"
echo ""
echo "  2. Run ECH0Py agent:"
echo "     python3 ech0py_agent.py --model phi-2"
echo ""
echo "  3. View documentation:"
echo "     cat DEPLOYMENT_GUIDE.md"
echo ""
echo "  4. Health check:"
echo "     python3 -c \"from tools import health_check_all; import json; print(json.dumps(health_check_all(), indent=2))\""
echo ""
INSTALL_EOF

chmod +x "${PACKAGE_DIR}/install.sh"

echo "[*] Creating README..."

cat > "${PACKAGE_DIR}/README.md" << 'README_EOF'
# Sovereign Security Toolkit - SD Card Deployment

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

## ⚠️ AUTHORIZATION WARNING

**ALL TOOLS REQUIRE EXPLICIT AUTHORIZATION FOR USE**

Unauthorized use is ILLEGAL and may result in:
- Federal criminal charges
- Civil liability
- Imprisonment
- Significant fines

## Quick Start

### 1. Installation

```bash
sudo ./install.sh
```

This will install:
- All system dependencies
- Python packages
- Ollama (for ECH0Py)
- Phi-2 model

### 2. Setup Authorization

```bash
python3 pythief.py --setup-auth
```

Follow the prompts to configure your FCC/authorization credentials.

### 3. Run ECH0Py Agent

```bash
python3 ech0py_agent.py --model phi-2
```

ECH0Py is your AI assistant that can operate all tools via natural language.

### 4. Individual Tools

```bash
# Evil Twin WiFi Attack
python3 pythief.py --ssid "Free WiFi" --company "Test Corp" --auth-token TOKEN --engagement-id ENG-001

# WiFi Coconut (14 radios)
python3 wifi_coconut.py --num-radios 14 --duration 300

# Proxmark3 RFID/NFC
python3 proxmark3_toolkit.py --lf-search

# Network Reconnaissance
python3 gpig.py --interface eth0

# Cellular Base Station (REQUIRES FCC LICENSE)
python3 pytower.py --setup-license
python3 pytower.py --license-number WXY1234 --location "Site Alpha" --purpose "Emergency Response"
```

## Tools Included

- **PyThief**: Evil twin WiFi attacks with Marauder and SDR
- **Hak5 Arsenal**: USB Shark, Packet Squirrel, LAN Turtle
- **WiFi Coconut**: 14-radio WiFi analysis
- **Proxmark3 Toolkit**: RFID/NFC/EMV testing
- **Pwnie Revival**: Network implants and assessments
- **gPIG**: Intelligent network reconnaissance
- **ECH0Py**: LLM agent for tool orchestration
- **PyTower**: Portable cellular base station

## Documentation

Full documentation: `DEPLOYMENT_GUIDE.md`

## Hardware Requirements

### Raspberry Pi Deployment
- Raspberry Pi 4 (8GB RAM recommended)
- 32GB+ microSD card (128GB recommended)
- External USB WiFi adapter (Alfa AWUS036ACH)
- Optional: Proxmark3 Easy, ESP32 Marauder, RTL-SDR

### PyTower Deployment
- BeagleBone AI or Odroid N2+ or Intel NUC
- LimeSDR Mini/USB
- 8GB+ RAM, 160GB+ storage
- FCC Part 27/90 license (REQUIRED)

## Legal Compliance

You MUST have:
1. Written authorization for all operations
2. FCC license for PyTower
3. Proper insurance and liability coverage
4. Comprehensive audit logging enabled

## Support

- Documentation: See DEPLOYMENT_GUIDE.md
- Issues: Contact security team
- Updates: Pull latest from repository

---

**Remember: With great power comes great responsibility. Use ethically and legally.**
README_EOF

echo "[*] Creating version info..."

cat > "${PACKAGE_DIR}/VERSION" << VERSION_EOF
Sovereign Security Toolkit
Version: 1.0.0
Build Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Build Host: $(hostname)
Git Commit: $(cd "${TOOLS_DIR}" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")

Components:
- PyThief v1.0
- Hak5 Arsenal v1.0
- WiFi Coconut v1.0
- Proxmark3 Toolkit v1.0
- Pwnie Revival v1.0
- gPIG v1.0
- ECH0Py v1.0
- PyTower v1.0

Total Lines of Code: 15,000+
Tools: 16+
Templates: 3 enterprise themes

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light)
All Rights Reserved. PATENT PENDING.
VERSION_EOF

echo "[*] Creating archive..."
cd /tmp
tar czf "${PACKAGE_NAME}.tar.gz" "${PACKAGE_NAME}"

echo ""
echo "============================================"
echo "✓ Package Created Successfully!"
echo "============================================"
echo ""
echo "Package: ${PACKAGE_NAME}.tar.gz"
echo "Location: /tmp/${PACKAGE_NAME}.tar.gz"
echo "Size: $(du -h /tmp/${PACKAGE_NAME}.tar.gz | cut -f1)"
echo ""
echo "To copy to SD card:"
echo "  1. Insert SD card"
echo "  2. Identify device: lsblk"
echo "  3. Mount: sudo mount /dev/sdX1 /mnt"
echo "  4. Copy: sudo cp /tmp/${PACKAGE_NAME}.tar.gz /mnt/"
echo "  5. Unmount: sudo umount /mnt"
echo ""
echo "On Raspberry Pi:"
echo "  1. Extract: tar xzf ${PACKAGE_NAME}.tar.gz"
echo "  2. Enter: cd ${PACKAGE_NAME}"
echo "  3. Install: sudo ./install.sh"
echo ""
echo "⚠️  Remember: All tools require proper authorization!"
echo ""
