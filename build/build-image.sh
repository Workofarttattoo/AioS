#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build/live-build"
CONFIG_DIR="${BUILD_DIR}/config"
INCLUDES_DIR="${CONFIG_DIR}/includes.chroot/opt/agentaos"
DIST_DIR="${ROOT_DIR}/dist"
OUTPUT_IMAGE="${DIST_DIR}/agentaos.img"

LB_BINARY="${LB_BINARY:-lb}"

if ! command -v "${LB_BINARY}" >/dev/null 2>&1; then
  echo "[error] live-build ('${LB_BINARY}') not found on PATH." >&2
  exit 1
fi

echo "[info] Preparing includes directory: ${INCLUDES_DIR}"
mkdir -p "${INCLUDES_DIR}"

echo "[info] Syncing repository into chroot includes..."
rsync -a --delete \
  --exclude ".git" \
  --exclude ".gitmodules" \
  --exclude ".github" \
  --exclude "dist" \
  --exclude "build/live-build/config/includes.chroot/opt/agentaos" \
  --exclude "__pycache__" \
  --exclude "*.pyc" \
  "${ROOT_DIR}/" "${INCLUDES_DIR}/"

# Preserve marker to make it obvious the directory is managed by the builder.
touch "${INCLUDES_DIR}/.agentaos-image-root"

mkdir -p "${DIST_DIR}"

pushd "${BUILD_DIR}" >/dev/null
  echo "[info] Cleaning previous live-build artifacts (if any)..."
  ${LB_BINARY} clean || true

  echo "[info] Configuring live-build environment..."
  ${LB_BINARY} config \
    --mode debian \
    --distribution bookworm \
    --architectures amd64 \
    --linux-flavours amd64 \
    --binary-images iso-hybrid \
    --chroot-filesystem squashfs \
    --firmware-binary true \
    --firmware-chroot true \
    --apt-secure true

  echo "[info] Building AgentaOS image..."
  ${LB_BINARY} build
popd >/dev/null

if [[ ! -f "${BUILD_DIR}/binary.hybrid.iso" ]]; then
  echo "[error] live-build did not produce binary.hybrid.iso" >&2
  exit 1
fi

echo "[info] Copying image to ${OUTPUT_IMAGE}"
cp -f "${BUILD_DIR}/binary.hybrid.iso" "${OUTPUT_IMAGE}"

echo "[info] Build complete."
echo "       Flash with: sudo dd if=${OUTPUT_IMAGE} of=/dev/sdX bs=4M status=progress oflag=sync"
