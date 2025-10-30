#!/bin/bash
# Build PyThief Red Team Docker Image
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

set -e

echo "============================================"
echo "Building PyThief Red Team Docker Image"
echo "============================================"

# Configuration
IMAGE_NAME="pythief-redteam"
IMAGE_TAG="latest"
BUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "[*] Build directory: $BUILD_DIR"
echo "[*] Image: $IMAGE_NAME:$IMAGE_TAG"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "[!] Error: Docker is not installed"
    exit 1
fi

echo "[*] Copying required files..."
cd "$BUILD_DIR"

# Ensure all files are present
required_files=(
    "pythief.py"
    "hak5_arsenal.py"
    "wifi_coconut.py"
    "requirements_pythief.txt"
    "Dockerfile.pythief"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "[!] Error: Required file not found: $file"
        exit 1
    fi
done

# Copy Sovereign Security Toolkit tools
echo "[*] Including Sovereign Security Toolkit..."
toolkit_tools=(
    "aurorascan.py"
    "cipherspear.py"
    "skybreaker.py"
    "mythickey.py"
    "spectratrace.py"
    "nemesishydra.py"
    "obsidianhunt.py"
    "vectorflux.py"
)

for tool in "${toolkit_tools[@]}"; do
    if [ ! -f "$tool" ]; then
        echo "[!] Warning: Toolkit tool not found: $tool (skipping)"
    fi
done

# Build Docker image
echo ""
echo "[*] Building Docker image..."
echo ""

docker build \
    -f Dockerfile.pythief \
    -t "$IMAGE_NAME:$IMAGE_TAG" \
    --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
    --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "✓ Build successful!"
    echo "============================================"
    echo ""
    echo "Image: $IMAGE_NAME:$IMAGE_TAG"
    echo ""
    echo "Run with:"
    echo "  docker run --rm -it --privileged --network host $IMAGE_NAME:$IMAGE_TAG"
    echo ""
    echo "⚠  WARNING: This image requires --privileged for WiFi access"
    echo "⚠  AUTHORIZATION REQUIRED: Only use in authorized engagements"
    echo ""
else
    echo ""
    echo "[!] Build failed"
    exit 1
fi

# Optional: Tag with version
if [ -n "$VERSION" ]; then
    docker tag "$IMAGE_NAME:$IMAGE_TAG" "$IMAGE_NAME:$VERSION"
    echo "✓ Tagged as: $IMAGE_NAME:$VERSION"
fi

# Show image size
echo "Image size:"
docker images "$IMAGE_NAME:$IMAGE_TAG" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

echo ""
echo "============================================"
echo "Quick Start"
echo "============================================"
echo ""
echo "1. Setup authorization:"
echo "   docker run --rm -it $IMAGE_NAME:$IMAGE_TAG python3 /opt/pythief/pythief.py --setup-auth"
echo ""
echo "2. Run PyThief evil twin:"
echo "   docker run --rm -it --privileged --network host $IMAGE_NAME:$IMAGE_TAG \\"
echo "     python3 /opt/pythief/pythief.py \\"
echo "     --ssid 'Free WiFi' \\"
echo "     --company 'Acme Corp' \\"
echo "     --auth-token TOKEN \\"
echo "     --engagement-id ENG-001"
echo ""
echo "3. Run Hak5 USB Shark:"
echo "   docker run --rm -it --privileged -v /dev/bus/usb:/dev/bus/usb $IMAGE_NAME:$IMAGE_TAG \\"
echo "     python3 /opt/pythief/hak5_arsenal.py usb-shark"
echo ""
echo "4. Run WiFi Coconut (14 radios):"
echo "   docker run --rm -it --privileged --network host $IMAGE_NAME:$IMAGE_TAG \\"
echo "     python3 /opt/pythief/wifi_coconut.py \\"
echo "     --num-radios 14 \\"
echo "     --duration 300"
echo ""
echo "5. Interactive shell:"
echo "   docker run --rm -it --privileged --network host $IMAGE_NAME:$IMAGE_TAG /bin/bash"
echo ""
