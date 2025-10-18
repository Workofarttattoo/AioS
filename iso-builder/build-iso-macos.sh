#!/bin/bash
# Ai|oS Bootable ISO Builder for macOS
# Uses Docker to build the ISO on macOS

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIOS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Ai|oS Bootable ISO Builder (macOS)                      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "[info] Building Ai|oS bootable ISO using Docker..."
echo "[info] This may take 20-40 minutes depending on your internet speed."
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "[error] Docker is not running. Please start Docker Desktop and try again."
    exit 1
fi

echo "[info] Checking Docker memory allocation..."
DOCKER_MEM=$(docker info --format '{{.MemTotal}}' 2>/dev/null || echo "0")
DOCKER_MEM_GB=$((DOCKER_MEM / 1024 / 1024 / 1024))
if [ "$DOCKER_MEM_GB" -lt 4 ]; then
    echo "[warn] Docker has less than 4GB RAM allocated. Build may be slow or fail."
    echo "[warn] Increase Docker memory in Docker Desktop > Settings > Resources"
    echo "[warn] Recommended: 8GB or more"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "[info] Starting ISO build process..."
echo "[info] Build context: $AIOS_ROOT"
echo "[info] Output directory: $OUTPUT_DIR"
echo ""

# Change to iso-builder directory for docker-compose
cd "$SCRIPT_DIR"

# Build with docker-compose
echo "[1/3] Building Docker image (this downloads ~500MB)..."
if ! docker-compose build --progress=plain; then
    echo "[error] Docker build failed. Check the logs above."
    exit 1
fi

echo ""
echo "[2/3] Creating ISO filesystem..."
echo "[info] This step takes 15-30 minutes. Please be patient..."
echo ""

# Run the container to export ISO
if ! docker-compose up; then
    echo "[error] ISO creation failed. Check the logs above."
    exit 1
fi

echo ""
echo "[3/3] Verifying ISO..."

if [ ! -f "$OUTPUT_DIR/aios-live.iso" ]; then
    echo "[error] ISO file was not created. Build may have failed."
    exit 1
fi

ISO_SIZE=$(du -h "$OUTPUT_DIR/aios-live.iso" | cut -f1)
echo "[info] ✓ ISO build complete!"
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Build Summary                                           ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  ISO Location: $OUTPUT_DIR/aios-live.iso"
echo "║  ISO Size: $ISO_SIZE"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "[info] Next Steps:"
echo ""
echo "1. Test the ISO in a VM:"
echo "   # Using QEMU (install with: brew install qemu)"
echo "   qemu-system-x86_64 -cdrom $OUTPUT_DIR/aios-live.iso -m 4G -accel hvf"
echo ""
echo "   # Or use VirtualBox, VMware Fusion, or Parallels"
echo ""
echo "2. Write to USB drive:"
echo "   # Find your USB device"
echo "   diskutil list"
echo ""
echo "   # Unmount it (replace diskN with your USB)"
echo "   diskutil unmountDisk /dev/diskN"
echo ""
echo "   # Write ISO (⚠️  WILL ERASE USB!)"
echo "   sudo dd if=$OUTPUT_DIR/aios-live.iso of=/dev/rdiskN bs=4m status=progress"
echo ""
echo "   # Eject"
echo "   diskutil eject /dev/diskN"
echo ""
echo "3. Boot from USB/ISO:"
echo "   - Restart computer"
echo "   - Hold Option/Alt key (Mac) or F12 (PC)"
echo "   - Select USB/CD drive"
echo "   - Choose 'Ai|oS - AI Operating System'"
echo ""
echo "[info] Default login credentials:"
echo "   Username: aios"
echo "   Password: aios"
echo ""
echo "[info] For more information, see: $SCRIPT_DIR/README.md"
