#!/usr/bin/env bash
# Raspberry Pi 4/5 installer for Ai:oS.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH="$(uname -m)"
WITH_TORCH=0
SKIP_APT=0
REQUESTED_PREFIX=""

usage() {
  cat <<'EOF'
Ai:oS Raspberry Pi installer

Usage: ./INSTALL_AIOS_RPI.sh [options]

Options:
  --prefix=<path>   Install into a different directory (defaults to repo root or $AIOS_RPI_PREFIX).
  --with-torch      Attempt to install PyTorch wheels for aarch64 (requires 64-bit Pi OS).
  --skip-apt        Skip apt package installation (useful on offline or pre-provisioned systems).
  -h, --help        Show this message.

Environment overrides:
  AIOS_RPI_PREFIX   Default installation directory override.
  AIOS_RPI_WITH_TORCH=1   Same as --with-torch.
EOF
}

for arg in "$@"; do
  case "$arg" in
    --with-torch)
      WITH_TORCH=1
      ;;
    --prefix=*)
      REQUESTED_PREFIX="${arg#*=}"
      ;;
    --skip-apt)
      SKIP_APT=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "${AIOS_RPI_WITH_TORCH:-0}" == "1" ]]; then
  WITH_TORCH=1
fi

INSTALL_DIR="${REQUESTED_PREFIX:-${AIOS_RPI_PREFIX:-$SCRIPT_DIR}}"
INSTALL_DIR="$(realpath -m "$INSTALL_DIR")"

echo "==> Ai:oS Raspberry Pi installer"
echo "    Architecture: $ARCH"
echo "    Source tree : $SCRIPT_DIR"
echo "    Install dir : $INSTALL_DIR"

if [[ "$ARCH" != "aarch64" ]]; then
  echo "!! Warning: Detected architecture '$ARCH'. Raspberry Pi 4/5 64-bit (aarch64) is recommended."
fi

if [[ $SKIP_APT -eq 0 ]] && command -v apt >/dev/null 2>&1; then
  echo "==> Updating apt metadata and installing system packages..."
  sudo apt update
  sudo apt install -y \
    python3 \
    python3-venv \
    python3-dev \
    python3-pip \
    git \
    build-essential \
    libatlas-base-dev \
    libopenblas-dev \
    libffi-dev \
    libssl-dev \
    libsndfile1 \
    portaudio19-dev \
    ffmpeg \
    pkg-config \
    rsync
else
  echo "==> Skipping apt package installation."
fi

mkdir -p "$INSTALL_DIR"

# Copy tree if requested prefix differs from source.
if [[ "$INSTALL_DIR" != "$SCRIPT_DIR" ]]; then
  echo "==> Syncing Ai:oS files to install directory..."
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete \
      --exclude '.git' \
      --exclude '.venv' \
      --exclude '__pycache__' \
      --exclude '.mypy_cache' \
      --exclude '.pytest_cache' \
      "$SCRIPT_DIR"/ "$INSTALL_DIR"/
  else
    echo "rsync not available; falling back to cp -a (no delete)." >&2
    cp -a "$SCRIPT_DIR"/. "$INSTALL_DIR"/
  fi
else
  echo "==> Installing in-place (no file copy)."
fi

cd "$INSTALL_DIR"

VENV_DIR="$INSTALL_DIR/.venv"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "==> Creating Python virtual environment..."
  python3 -m venv "$VENV_DIR"
else
  echo "==> Reusing existing virtual environment."
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip tooling..."
python -m pip install --upgrade pip setuptools wheel >/dev/null

REQ_FILE="requirements-rpi.txt"
if [[ ! -f "$REQ_FILE" ]]; then
  echo "ERROR: $REQ_FILE not found in $INSTALL_DIR" >&2
  exit 1
fi

echo "==> Installing Python dependencies..."
python -m pip install -r "$REQ_FILE"

if [[ $WITH_TORCH -eq 1 ]]; then
  echo "==> Attempting PyTorch install for aarch64..."
  if [[ "$ARCH" == "aarch64" ]]; then
    python -m pip install --extra-index-url https://download.pytorch.org/whl/nightly/cpu torch==2.2.0+cpu -q || \
      echo "!! PyTorch installation failed. You can retry manually with AIOS_RPI_WITH_TORCH=1."
  else
    echo "!! Skipping PyTorch install; unsupported architecture ($ARCH)."
  fi
else
  echo "==> Skipping PyTorch (enable with --with-torch if desired)."
fi

LAUNCHER="$INSTALL_DIR/aios-rpi.sh"
echo "==> Writing launch helper: $LAUNCHER"
cat > "$LAUNCHER" <<'LAUNCH'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"
export AIOS_PROFILE=rpi
cd "$SCRIPT_DIR"
python aios_shell.py "$@"
LAUNCH
chmod +x "$LAUNCHER"

deactivate

cat <<EOF

Ai:oS Raspberry Pi installation complete.

To start the runtime:
  source "$VENV_DIR/bin/activate" && python aios_shell.py boot
or use the helper:
  $LAUNCHER boot

Tips:
  * Re-run this script with --with-torch after enabling a 64-bit Pi OS if you need advanced ML features.
  * Use --skip-apt on air-gapped systems after manually provisioning system packages.
  * Set AIOS_RPI_PREFIX=/opt/aios before running to install into /opt.

EOF
