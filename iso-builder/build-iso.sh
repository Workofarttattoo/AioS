#!/bin/bash
# Ai|oS Bootable ISO Builder Script
# Builds a bootable Linux ISO with Ai|oS pre-installed

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIOS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="/tmp/aios-iso-build"
ISO_OUTPUT="$AIOS_ROOT/aios-live.iso"

echo "[info] Starting Ai|oS ISO build process..."
echo "[info] Ai|oS root: $AIOS_ROOT"
echo "[info] Build directory: $BUILD_DIR"
echo "[info] Output ISO: $ISO_OUTPUT"

# Check for required tools
REQUIRED_TOOLS="debootstrap mksquashfs xorriso"
for tool in $REQUIRED_TOOLS; do
    if ! command -v $tool &> /dev/null; then
        echo "[error] Required tool '$tool' not found. Please install:"
        echo "  sudo apt-get install debootstrap squashfs-tools xorriso isolinux syslinux-efi grub-pc-bin grub-efi-amd64-bin mtools"
        exit 1
    fi
done

# Clean previous build
if [ -d "$BUILD_DIR" ]; then
    echo "[info] Cleaning previous build directory..."
    sudo rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "[info] Creating ISO directory structure..."
mkdir -p iso/{live,boot/grub,EFI/BOOT}

echo "[info] Bootstrapping Debian base system (this may take a while)..."
sudo debootstrap --variant=minbase bookworm chroot http://deb.debian.org/debian/

echo "[info] Configuring base system..."
echo "aios-live" | sudo tee chroot/etc/hostname > /dev/null
echo "127.0.0.1 localhost aios-live" | sudo tee -a chroot/etc/hosts > /dev/null

echo "[info] Installing system packages in chroot..."
sudo chroot chroot /bin/bash -c "
    apt-get update
    apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        python3-tk \
        firefox-esr \
        xorg \
        xinit \
        openbox \
        sudo \
        systemd \
        network-manager \
        linux-image-amd64 \
        live-boot
    rm -rf /var/lib/apt/lists/*
"

echo "[info] Copying Ai|oS files to chroot..."
sudo mkdir -p chroot/opt
sudo cp -r "$AIOS_ROOT" chroot/opt/aios

# Copy QuLab2.0 if it exists
if [ -d "$HOME/QuLab2.0" ]; then
    echo "[info] Copying QuLab2.0..."
    sudo cp -r "$HOME/QuLab2.0" chroot/opt/
fi

# Copy TheGAVLSuite if it exists
if [ -d "$HOME/TheGAVLSuite" ]; then
    echo "[info] Copying TheGAVLSuite..."
    sudo cp -r "$HOME/TheGAVLSuite" chroot/opt/
fi

echo "[info] Installing Python dependencies..."
sudo chroot chroot /bin/bash -c "
    cd /opt/aios
    pip3 install --break-system-packages fastapi uvicorn numpy prometheus-fastapi-instrumentator httpx
    pip3 install --break-system-packages qiskit scipy matplotlib pydantic
    pip3 install --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu
"

echo "[info] Creating auto-start scripts..."
sudo tee chroot/usr/local/bin/aios-autostart.sh > /dev/null << 'EOF'
#!/bin/bash
export PYTHONPATH=/opt/aios:/opt/QuLab2.0:/opt/TheGAVLSuite
export DISPLAY=:0

sleep 3

cd /opt/aios
python3 aios/aios -v boot &

sleep 5

firefox --kiosk /opt/aios/web/aios_launcher.html &

wait
EOF
sudo chmod +x chroot/usr/local/bin/aios-autostart.sh

echo "[info] Creating systemd service..."
sudo tee chroot/etc/systemd/system/aios.service > /dev/null << 'EOF'
[Unit]
Description=Ai|oS Auto-Start Service
After=graphical.target

[Service]
Type=simple
User=aios
Environment="DISPLAY=:0"
Environment="PYTHONPATH=/opt/aios:/opt/QuLab2.0:/opt/TheGAVLSuite"
ExecStart=/usr/local/bin/aios-autostart.sh
Restart=on-failure

[Install]
WantedBy=graphical.target
EOF

echo "[info] Configuring Openbox..."
sudo mkdir -p chroot/etc/xdg/openbox
sudo tee chroot/etc/xdg/openbox/autostart > /dev/null << 'EOF'
#!/bin/bash
/usr/local/bin/aios-autostart.sh &
EOF
sudo chmod +x chroot/etc/xdg/openbox/autostart

echo "[info] Creating aios user..."
sudo chroot chroot /bin/bash -c "
    useradd -m -s /bin/bash -G sudo aios
    echo 'aios:aios' | chpasswd
    echo 'aios ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
"

echo "[info] Configuring auto-login..."
sudo mkdir -p chroot/etc/systemd/system/getty@tty1.service.d
sudo tee chroot/etc/systemd/system/getty@tty1.service.d/autologin.conf > /dev/null << 'EOF'
[Service]
ExecStart=
ExecStart=-/sbin/agetty --autologin aios --noclear %I $TERM
EOF

echo "[info] Configuring X auto-start..."
echo 'startx' | sudo tee -a chroot/home/aios/.bashrc > /dev/null

sudo tee chroot/home/aios/.xinitrc > /dev/null << 'EOF'
#!/bin/bash
exec openbox-session
EOF
sudo chroot chroot chown aios:aios /home/aios/.xinitrc
sudo chroot chroot chmod +x /home/aios/.xinitrc

echo "[info] Enabling aios service..."
sudo chroot chroot systemctl enable aios.service

echo "[info] Creating squashfs filesystem..."
sudo mksquashfs chroot iso/live/filesystem.squashfs -comp xz

echo "[info] Copying kernel and initrd..."
sudo cp chroot/boot/vmlinuz-* iso/live/vmlinuz
sudo cp chroot/boot/initrd.img-* iso/live/initrd.img

echo "[info] Creating GRUB configuration..."
sudo tee iso/boot/grub/grub.cfg > /dev/null << 'EOF'
set default=0
set timeout=5

menuentry "Ai|oS - AI Operating System" {
    linux /live/vmlinuz boot=live quiet splash
    initrd /live/initrd.img
}

menuentry "Ai|oS - Safe Mode" {
    linux /live/vmlinuz boot=live quiet splash single
    initrd /live/initrd.img
}
EOF

echo "[info] Setting up ISOLINUX for BIOS boot..."
sudo cp /usr/lib/ISOLINUX/isolinux.bin iso/boot/ 2>/dev/null || \
    sudo cp /usr/lib/syslinux/isolinux.bin iso/boot/
sudo cp /usr/lib/syslinux/modules/bios/*.c32 iso/boot/ 2>/dev/null || true

sudo tee iso/boot/isolinux.cfg > /dev/null << 'EOF'
DEFAULT aios
TIMEOUT 50
PROMPT 1

LABEL aios
  MENU LABEL Ai|oS - AI Operating System
  LINUX /live/vmlinuz
  APPEND initrd=/live/initrd.img boot=live quiet splash

LABEL safe
  MENU LABEL Ai|oS - Safe Mode
  LINUX /live/vmlinuz
  APPEND initrd=/live/initrd.img boot=live quiet splash single
EOF

echo "[info] Building ISO image..."
sudo xorriso -as mkisofs \
    -iso-level 3 \
    -full-iso9660-filenames \
    -volid "AIOS_LIVE" \
    -output "$ISO_OUTPUT" \
    -eltorito-boot boot/isolinux.bin \
    -eltorito-catalog boot/boot.cat \
    -no-emul-boot \
    -boot-load-size 4 \
    -boot-info-table \
    -isohybrid-mbr /usr/lib/ISOLINUX/isohdpfx.bin \
    iso

echo "[info] Cleaning up build directory..."
sudo rm -rf "$BUILD_DIR"

echo "[info] ✓ ISO build complete!"
echo "[info] Output: $ISO_OUTPUT"
echo ""
echo "[info] To test the ISO:"
echo "  qemu-system-x86_64 -cdrom $ISO_OUTPUT -m 4G -enable-kvm"
echo ""
echo "[info] To write to USB:"
echo "  sudo dd if=$ISO_OUTPUT of=/dev/sdX bs=4M status=progress"
echo "  (Replace /dev/sdX with your USB device)"
