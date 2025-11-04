#!/bin/bash
# Create Bootable Raspberry Pi Image with Sovereign Security Toolkit
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

set -e

echo "============================================"
echo "Raspberry Pi Bootable Image Creator"
echo "Sovereign Security Toolkit"
echo "============================================"
echo ""

# Check if running on macOS or Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "[*] Detected: macOS"
    PLATFORM="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "[*] Detected: Linux"
    PLATFORM="linux"
else
    echo "[!] Unsupported platform: $OSTYPE"
    exit 1
fi

# Configuration
WORK_DIR="/tmp/pi_image_build"
TOOLKIT_DIR="/Users/noone/aios/tools"
OUTPUT_DIR="${HOME}/Desktop"
IMAGE_NAME="SovereignSecurityToolkit_Pi_$(date +%Y%m%d).img"

# Base image URL (Kali Linux ARM for security tools)
BASE_IMAGE_URL="https://kali.download/arm-images/kali-2024.3/kali-linux-2024.3-raspberry-pi-arm64.img.xz"
BASE_IMAGE_NAME="kali-linux-2024.3-raspberry-pi-arm64.img.xz"

echo ""
echo "[*] This script will create a bootable Raspberry Pi image with:"
echo "    - Kali Linux ARM 64-bit (security-focused base)"
echo "    - All Sovereign Security Toolkit tools"
echo "    - Pre-installed dependencies"
echo "    - ECH0Py with Phi-2 model"
echo "    - Auto-configuration scripts"
echo ""
echo "[*] Image size: ~8GB (requires 16GB+ SD card)"
echo "[*] Build time: 30-60 minutes (depending on download speed)"
echo ""

read -p "Continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "[*] Creating work directory..."
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# Download base image
if [ ! -f "${BASE_IMAGE_NAME}" ]; then
    echo "[*] Downloading Kali Linux ARM image (~2GB)..."
    echo "    This may take a while..."
    curl -L -o "${BASE_IMAGE_NAME}" "${BASE_IMAGE_URL}"
    echo "[*] ✓ Download complete"
else
    echo "[*] Using existing base image"
fi

# Extract image
echo "[*] Extracting base image..."
if [ -f "${BASE_IMAGE_NAME%.xz}" ]; then
    echo "[*] Image already extracted"
else
    xz -d -k "${BASE_IMAGE_NAME}"
fi

BASE_IMAGE="${BASE_IMAGE_NAME%.xz}"

echo "[*] Base image: ${BASE_IMAGE}"

# Mount image
echo "[*] Mounting image..."

if [ "$PLATFORM" = "macos" ]; then
    # macOS mounting
    echo "[*] Attaching disk image..."
    DISK_DEV=$(hdiutil attach -imagekey diskimage-class=CRawDiskImage -nomount "${BASE_IMAGE}" | grep -o '/dev/disk[0-9]*' | head -1)

    if [ -z "$DISK_DEV" ]; then
        echo "[!] Failed to attach disk image"
        exit 1
    fi

    echo "[*] Disk device: ${DISK_DEV}"

    # Mount root partition (usually partition 2)
    MOUNT_POINT="/Volumes/kali-root"
    sudo mkdir -p "${MOUNT_POINT}"

    # Identify the root partition
    ROOT_PART="${DISK_DEV}s2"

    echo "[*] Mounting root partition: ${ROOT_PART}"
    sudo mount -t ext4 "${ROOT_PART}" "${MOUNT_POINT}" || {
        echo "[!] Failed to mount root partition"
        echo "[!] You may need to install ext4 support (e.g., via macFUSE and ext4fuse)"
        hdiutil detach "${DISK_DEV}"
        exit 1
    }

elif [ "$PLATFORM" = "linux" ]; then
    # Linux mounting with loop device
    LOOP_DEV=$(sudo losetup -f)
    sudo losetup -P "${LOOP_DEV}" "${BASE_IMAGE}"

    MOUNT_POINT="/mnt/kali-root"
    sudo mkdir -p "${MOUNT_POINT}"

    # Mount root partition (usually partition 2)
    sudo mount "${LOOP_DEV}p2" "${MOUNT_POINT}"
fi

echo "[*] ✓ Image mounted at: ${MOUNT_POINT}"

# Copy toolkit files
echo ""
echo "[*] Installing Sovereign Security Toolkit..."

sudo mkdir -p "${MOUNT_POINT}/opt/sovereign_toolkit"
sudo cp -r "${TOOLKIT_DIR}"/*.py "${MOUNT_POINT}/opt/sovereign_toolkit/"
sudo cp "${TOOLKIT_DIR}"/requirements_pythief.txt "${MOUNT_POINT}/opt/sovereign_toolkit/"
sudo cp "${TOOLKIT_DIR}"/DEPLOYMENT_GUIDE.md "${MOUNT_POINT}/opt/sovereign_toolkit/"
sudo cp -r "${TOOLKIT_DIR}"/templates "${MOUNT_POINT}/opt/sovereign_toolkit/"

echo "[*] ✓ Toolkit files copied"

# Create auto-setup script
echo "[*] Creating auto-setup script..."

sudo tee "${MOUNT_POINT}/opt/sovereign_toolkit/setup.sh" > /dev/null << 'SETUP_EOF'
#!/bin/bash
# Sovereign Security Toolkit - First Boot Setup
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

set -e

echo "============================================"
echo "Sovereign Security Toolkit - First Boot"
echo "============================================"
echo ""

# Update system
echo "[*] Updating system packages..."
apt update
apt upgrade -y

# Install system dependencies
echo "[*] Installing system dependencies..."
apt install -y \
    python3-pip \
    aircrack-ng \
    hostapd \
    dnsmasq \
    tcpdump \
    tshark \
    nmap \
    git \
    curl \
    wget

# Install Python dependencies
echo "[*] Installing Python packages..."
cd /opt/sovereign_toolkit
pip3 install -r requirements_pythief.txt

# Install Ollama
echo "[*] Installing Ollama for ECH0Py..."
curl -fsSL https://ollama.com/install.sh | sh

# Pull Phi-2 model
echo "[*] Downloading Phi-2 model (~1.5GB)..."
ollama pull phi-2

# Create symbolic links
echo "[*] Creating command shortcuts..."
ln -sf /opt/sovereign_toolkit/pythief.py /usr/local/bin/pythief
ln -sf /opt/sovereign_toolkit/ech0py_agent.py /usr/local/bin/ech0py
ln -sf /opt/sovereign_toolkit/gpig.py /usr/local/bin/gpig
ln -sf /opt/sovereign_toolkit/pytower.py /usr/local/bin/pytower
ln -sf /opt/sovereign_toolkit/wifi_coconut.py /usr/local/bin/wifi-coconut
ln -sf /opt/sovereign_toolkit/proxmark3_toolkit.py /usr/local/bin/proxmark3-toolkit
ln -sf /opt/sovereign_toolkit/hak5_arsenal.py /usr/local/bin/hak5
ln -sf /opt/sovereign_toolkit/pwnie_revival.py /usr/local/bin/pwnie

chmod +x /usr/local/bin/pythief
chmod +x /usr/local/bin/ech0py
chmod +x /usr/local/bin/gpig
chmod +x /usr/local/bin/pytower
chmod +x /usr/local/bin/wifi-coconut
chmod +x /usr/local/bin/proxmark3-toolkit
chmod +x /usr/local/bin/hak5
chmod +x /usr/local/bin/pwnie

echo ""
echo "============================================"
echo "✓ Sovereign Security Toolkit Ready!"
echo "============================================"
echo ""
echo "Available commands:"
echo "  pythief         - Evil twin WiFi attacks"
echo "  ech0py          - AI agent for tool orchestration"
echo "  gpig            - Network reconnaissance"
echo "  pytower         - Cellular base station"
echo "  wifi-coconut    - Multi-radio WiFi analysis"
echo "  proxmark3-toolkit - RFID/NFC/EMV testing"
echo "  hak5            - Hak5 Arsenal tools"
echo "  pwnie           - Pwnie Revival tools"
echo ""
echo "Documentation: /opt/sovereign_toolkit/DEPLOYMENT_GUIDE.md"
echo ""
echo "⚠️  REMEMBER: All tools require proper authorization!"
echo ""

# Create desktop shortcut
cat > /home/kali/Desktop/ECH0Py.desktop << 'DESKTOP_EOF'
[Desktop Entry]
Version=1.0
Type=Application
Name=ECH0Py Agent
Comment=AI-powered pentesting tool orchestrator
Exec=lxterminal -e "ech0py --model phi-2"
Icon=utilities-terminal
Terminal=true
Categories=Security;
DESKTOP_EOF

chmod +x /home/kali/Desktop/ECH0Py.desktop
chown kali:kali /home/kali/Desktop/ECH0Py.desktop

# Mark setup as complete
touch /opt/sovereign_toolkit/.setup_complete

echo "Setup complete! Rebooting in 10 seconds..."
sleep 10
reboot
SETUP_EOF

sudo chmod +x "${MOUNT_POINT}/opt/sovereign_toolkit/setup.sh"

# Create systemd service for first boot
echo "[*] Creating first-boot service..."

sudo tee "${MOUNT_POINT}/etc/systemd/system/sovereign-toolkit-setup.service" > /dev/null << 'SERVICE_EOF'
[Unit]
Description=Sovereign Security Toolkit First Boot Setup
After=network-online.target
Wants=network-online.target
ConditionPathExists=!/opt/sovereign_toolkit/.setup_complete

[Service]
Type=oneshot
ExecStart=/opt/sovereign_toolkit/setup.sh
RemainAfterExit=yes
StandardOutput=journal+console
StandardError=journal+console

[Install]
WantedBy=multi-user.target
SERVICE_EOF

# Enable the service
sudo ln -sf "${MOUNT_POINT}/etc/systemd/system/sovereign-toolkit-setup.service" \
    "${MOUNT_POINT}/etc/systemd/system/multi-user.target.wants/sovereign-toolkit-setup.service"

echo "[*] ✓ First-boot setup configured"

# Create MOTD (Message of the Day)
echo "[*] Creating custom MOTD..."

sudo tee "${MOUNT_POINT}/etc/motd" > /dev/null << 'MOTD_EOF'

 ____                          _
/ ___|  _____   _____ _ __ ___(_) __ _ _ __
\___ \ / _ \ \ / / _ \ '__/ _ \ |/ _` | '_ \
 ___) | (_) \ V /  __/ | |  __/ | (_| | | | |
|____/ \___/ \_/ \___|_|  \___|_|\__, |_| |_|
                                 |___/
 ____                      _ _
/ ___|  ___  ___ _   _ _ __(_) |_ _   _
\___ \ / _ \/ __| | | | '__| | __| | | |
 ___) |  __/ (__| |_| | |  | | |_| |_| |
|____/ \___|\___|\__,_|_|  |_|\__|\__, |
                                   |___/
 _____           _ _    _ _
|_   _|__   ___ | | | _(_) |_
  | |/ _ \ / _ \| | |/ / | __|
  | | (_) | (_) | |   <| | |_
  |_|\___/ \___/|_|_|\_\_|\__|

Copyright (c) 2025 Corporation of Light. All Rights Reserved.

⚠️  AUTHORIZED USE ONLY ⚠️

Available Commands:
  ech0py          - AI pentesting agent
  pythief         - Evil twin WiFi attacks
  gpig            - Network reconnaissance
  pytower         - Cellular base station
  wifi-coconut    - Multi-radio WiFi
  proxmark3-toolkit - RFID/NFC/EMV
  hak5            - Hak5 Arsenal
  pwnie           - Pwnie Revival

Documentation: /opt/sovereign_toolkit/DEPLOYMENT_GUIDE.md

Get started: ech0py --model phi-2

MOTD_EOF

echo "[*] ✓ MOTD created"

# Unmount and cleanup
echo ""
echo "[*] Finalizing image..."

sync

if [ "$PLATFORM" = "macos" ]; then
    sudo umount "${MOUNT_POINT}"
    hdiutil detach "${DISK_DEV}"
elif [ "$PLATFORM" = "linux" ]; then
    sudo umount "${MOUNT_POINT}"
    sudo losetup -d "${LOOP_DEV}"
fi

# Copy final image to output directory
echo "[*] Copying final image to: ${OUTPUT_DIR}"
cp "${BASE_IMAGE}" "${OUTPUT_DIR}/${IMAGE_NAME}"

# Calculate checksum
echo "[*] Calculating SHA256 checksum..."
if [ "$PLATFORM" = "macos" ]; then
    CHECKSUM=$(shasum -a 256 "${OUTPUT_DIR}/${IMAGE_NAME}" | cut -d' ' -f1)
else
    CHECKSUM=$(sha256sum "${OUTPUT_DIR}/${IMAGE_NAME}" | cut -d' ' -f1)
fi

echo "${CHECKSUM}" > "${OUTPUT_DIR}/${IMAGE_NAME}.sha256"

# Create instructions file
cat > "${OUTPUT_DIR}/FLASH_INSTRUCTIONS.txt" << INSTR_EOF
Sovereign Security Toolkit - Raspberry Pi Image
Copyright (c) 2025 Corporation of Light. All Rights Reserved.

Image: ${IMAGE_NAME}
Size: $(du -h "${OUTPUT_DIR}/${IMAGE_NAME}" | cut -f1)
SHA256: ${CHECKSUM}
Date: $(date)

Requirements:
- Raspberry Pi 4 (4GB RAM minimum, 8GB recommended)
- 16GB+ microSD card (32GB+ recommended)
- External USB WiFi adapter (for monitor mode)
- Internet connection (for first-boot setup)

Flashing Instructions:

=== macOS ===
1. Insert SD card
2. Identify device: diskutil list
3. Unmount: diskutil unmountDisk /dev/diskX
4. Flash: sudo dd if=${IMAGE_NAME} of=/dev/rdiskX bs=4m
5. Wait for completion (30-60 minutes)
6. Eject: diskutil eject /dev/diskX

=== Linux ===
1. Insert SD card
2. Identify device: lsblk
3. Flash: sudo dd if=${IMAGE_NAME} of=/dev/sdX bs=4M status=progress
4. Sync: sudo sync
5. Eject: sudo eject /dev/sdX

=== Windows ===
1. Download Raspberry Pi Imager: https://www.raspberrypi.com/software/
2. Select "Use custom" image
3. Browse to ${IMAGE_NAME}
4. Select SD card
5. Write

First Boot:
- Default user: kali
- Default password: kali
- Automatic setup will run (requires internet)
- Takes 10-15 minutes
- System will reboot when complete

Getting Started:
1. Login with kali/kali
2. Run: ech0py --model phi-2
3. Ask ECH0Py to show available tools
4. Read documentation: cat /opt/sovereign_toolkit/DEPLOYMENT_GUIDE.md

⚠️  REMEMBER: All tools require proper authorization!

Support: https://aios.is/support
INSTR_EOF

# Cleanup
echo "[*] Cleaning up temporary files..."
cd ~
rm -rf "${WORK_DIR}"

echo ""
echo "============================================"
echo "✓ Raspberry Pi Image Created!"
echo "============================================"
echo ""
echo "Image: ${OUTPUT_DIR}/${IMAGE_NAME}"
echo "Size: $(du -h "${OUTPUT_DIR}/${IMAGE_NAME}" | cut -f1)"
echo "SHA256: ${CHECKSUM}"
echo ""
echo "Instructions: ${OUTPUT_DIR}/FLASH_INSTRUCTIONS.txt"
echo ""
echo "Flash to SD card and boot your Raspberry Pi!"
echo ""
echo "⚠️  First boot requires internet connection for automatic setup"
echo ""
