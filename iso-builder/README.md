# Ai|oS Bootable ISO Builder

This directory contains tools to create a bootable Linux ISO with Ai|oS pre-installed.

## Features

The Ai|oS live ISO includes:

- **Debian 12 (Bookworm)** base system
- **Ai|oS** - Full AI Operating System with all meta-agents
- **QuLab2.0** - Quantum computing framework
- **TheGAVLSuite** - GAVL modules and tools
- **Sovereign Security Toolkit** - All red-team tools
- **Auto-Start** - Ai|oS boots automatically with GUI launcher
- **Live Environment** - Run from USB/CD without installation
- **Minimal UI** - Openbox window manager with Firefox kiosk mode

## System Requirements

### Host System (for building)
- Linux system (Debian/Ubuntu recommended)
- 10GB+ free disk space
- Root/sudo access
- Internet connection

### Required Packages
```bash
sudo apt-get install \
    debootstrap \
    squashfs-tools \
    xorriso \
    isolinux \
    syslinux-efi \
    grub-pc-bin \
    grub-efi-amd64-bin \
    mtools \
    dosfstools
```

### Target System (for running the ISO)
- x86_64 CPU (Intel/AMD)
- 4GB+ RAM (8GB recommended)
- BIOS or UEFI boot support
- USB port or CD/DVD drive

## Building the ISO

### Method 1: Shell Script (Recommended)

```bash
cd /Users/noone/aios/iso-builder
./build-iso.sh
```

The script will:
1. Bootstrap a minimal Debian system
2. Install Python, X server, and Firefox
3. Copy Ai|oS, QuLab2.0, and TheGAVLSuite
4. Install all Python dependencies
5. Configure auto-login and auto-start
6. Create squashfs filesystem
7. Build bootable ISO

**Build time:** 20-40 minutes (depending on internet speed)

**Output:** `/Users/noone/aios/aios-live.iso` (~1.5-2.5GB)

### Method 2: Docker Build

```bash
cd /Users/noone/aios/iso-builder
docker build -f Dockerfile.iso -o type=local,dest=. .
```

The ISO will be exported to `./aios-live.iso`

## Testing the ISO

### Using QEMU/KVM

```bash
# Test with 4GB RAM
qemu-system-x86_64 -cdrom aios-live.iso -m 4G -enable-kvm

# Test with GPU passthrough
qemu-system-x86_64 -cdrom aios-live.iso -m 8G -enable-kvm -vga virtio
```

### Using VirtualBox

1. Create new VM with:
   - Type: Linux
   - Version: Debian (64-bit)
   - RAM: 4096MB minimum
   - No hard disk needed (live system)

2. Mount ISO as CD/DVD drive
3. Boot from CD

### Using VMware

1. Create new VM
2. Select "Installer disc image file (iso)"
3. Browse to `aios-live.iso`
4. Allocate 4GB+ RAM
5. Boot

## Writing to USB Drive

⚠️ **WARNING:** This will erase all data on the USB drive!

### Linux

```bash
# Find USB device
lsblk

# Write ISO (replace /dev/sdX with your USB device)
sudo dd if=aios-live.iso of=/dev/sdX bs=4M status=progress oflag=sync

# Or use ddrescue for better error handling
sudo ddrescue -D --force aios-live.iso /dev/sdX
```

### macOS

```bash
# Find USB device
diskutil list

# Unmount (replace diskN with your USB)
diskutil unmountDisk /dev/diskN

# Write ISO
sudo dd if=aios-live.iso of=/dev/rdiskN bs=4m

# Eject
diskutil eject /dev/diskN
```

### Windows

Use one of these tools:
- [Rufus](https://rufus.ie/) (Recommended)
- [Etcher](https://www.balena.io/etcher/)
- [Ventoy](https://www.ventoy.net/)

## Booting the ISO

1. **Insert USB or mount ISO**
2. **Restart computer**
3. **Enter boot menu** (usually F12, F2, ESC, or DEL)
4. **Select USB/CD drive** from boot options
5. **Select "Ai|oS - AI Operating System"** from GRUB menu

The system will:
- Boot into Debian live environment
- Auto-login as user `aios` (password: `aios`)
- Start X server with Openbox
- Launch Ai|oS boot sequence
- Open Firefox in kiosk mode with GUI launcher

## Default Credentials

- **Username:** `aios`
- **Password:** `aios`
- **Root access:** User has passwordless sudo

## Auto-Start Services

The ISO configures the following auto-start sequence:

1. **systemd** boots
2. **getty@tty1** auto-login as `aios`
3. **.bashrc** runs `startx`
4. **X server** starts with Openbox
5. **Openbox autostart** runs `/usr/local/bin/aios-autostart.sh`
6. **aios-autostart.sh:**
   - Sets `PYTHONPATH` for Ai|oS, QuLab2.0, TheGAVLSuite
   - Runs `python3 aios/aios -v boot`
   - Launches Firefox with `aios_launcher.html`

## Customization

### Modify Auto-Start Script

Edit the template in `build-iso.sh`:

```bash
# Find this section:
sudo tee chroot/usr/local/bin/aios-autostart.sh > /dev/null << 'EOF'
#!/bin/bash
# Your custom auto-start commands here
EOF
```

### Add Additional Packages

Edit the package installation section:

```bash
sudo chroot chroot /bin/bash -c "
    apt-get install -y \
        # Add your packages here
        vim \
        htop \
        net-tools
"
```

### Change Window Manager

Replace `openbox` with `i3`, `xfce4`, `lxde`, etc.

### Include Additional Files

Copy files to chroot before creating squashfs:

```bash
sudo cp -r /path/to/files chroot/opt/
```

## Persistent Storage (Optional)

The live ISO runs entirely in RAM. To enable persistence:

### Method 1: Install to Hard Disk

From within the live environment:

```bash
sudo apt-get install debian-installer-launcher
sudo debian-installer-launcher
```

### Method 2: Ventoy Persistence

1. Use [Ventoy](https://www.ventoy.net/) to create USB
2. Add persistence partition
3. Configure persistence in Ventoy settings

## Troubleshooting

### ISO Fails to Boot

- **Check UEFI/BIOS settings:** Disable Secure Boot
- **Try different boot mode:** Switch between UEFI and Legacy BIOS
- **Verify ISO integrity:**
  ```bash
  sha256sum aios-live.iso
  ```
- **Re-burn USB:** Some USB drives have compatibility issues

### Black Screen After Boot

- **Wait 60 seconds:** Initial boot takes time
- **Check display:** Try different display/monitor
- **Safe mode:** Select "Ai|oS - Safe Mode" from GRUB menu
- **TTY access:** Press `Ctrl+Alt+F2` to get console

### Ai|oS Doesn't Start

1. Press `Ctrl+Alt+F2` for console
2. Login as `aios` / `aios`
3. Check logs:
   ```bash
   journalctl -u aios.service
   tail -f /var/log/Xorg.0.log
   ```

### No Network Connection

```bash
# Check interfaces
ip link

# Start NetworkManager
sudo systemctl start NetworkManager

# Connect to WiFi
nmtui
```

### Out of Memory

Increase QEMU RAM or use physical machine with 8GB+

### Python Import Errors

```bash
# Check PYTHONPATH
echo $PYTHONPATH

# Manually install missing package
sudo pip3 install --break-system-packages <package>
```

## ISO Contents

```
aios-live.iso
├── boot/
│   ├── grub/
│   │   └── grub.cfg          # GRUB bootloader config
│   ├── isolinux.bin          # BIOS bootloader
│   ├── isolinux.cfg          # ISOLINUX config
│   └── *.c32                 # SYSLINUX modules
├── live/
│   ├── filesystem.squashfs   # Compressed root filesystem
│   ├── vmlinuz               # Linux kernel
│   └── initrd.img            # Initial ramdisk
└── EFI/
    └── BOOT/
        └── bootx64.efi       # UEFI bootloader
```

## File System Layout (Inside ISO)

```
/
├── opt/
│   ├── aios/                 # Ai|oS installation
│   │   ├── aios/             # Python package
│   │   ├── agents/           # Meta-agents
│   │   ├── tools/            # Security toolkit
│   │   ├── web/              # GUI launcher
│   │   │   ├── aios_launcher.html
│   │   │   └── wolf_icon_head.png
│   │   └── red-team-tools/   # GAVL module callers
│   ├── QuLab2.0/             # Quantum framework
│   └── TheGAVLSuite/         # GAVL suite
├── usr/
│   └── local/
│       └── bin/
│           └── aios-autostart.sh
└── home/
    └── aios/                 # User home directory
```

## Performance

**ISO Size:** ~1.5-2.5GB (depends on included tools)

**RAM Usage:**
- Minimum: 2GB (system only)
- Recommended: 4GB (with Ai|oS)
- Optimal: 8GB+ (for quantum simulations)

**Boot Time:**
- BIOS: 20-40 seconds
- UEFI: 15-30 seconds
- First run: +10 seconds (Ai|oS initialization)

## Security Notes

⚠️ **This is a live/demo environment:**
- Default password is `aios` (insecure)
- Passwordless sudo enabled
- All data stored in RAM (lost on reboot)
- Not recommended for production use

**For production deployments:**
1. Change default passwords
2. Disable passwordless sudo
3. Configure proper firewall
4. Install to disk with encryption
5. Follow security hardening guides

## License

Ai|oS and included components follow their respective licenses.

## Support

- Report issues: https://github.com/anthropics/claude-code/issues
- Documentation: See CLAUDE.md in Ai|oS root
- Quantum tools: See QuLab2.0/README.md

## Build System Details

**Base Distribution:** Debian 12 (Bookworm)

**Included Software:**
- Python 3.11+
- Firefox ESR
- Openbox WM
- X.org Server
- NetworkManager
- systemd

**Python Packages:**
- fastapi, uvicorn
- numpy, scipy, matplotlib
- qiskit (quantum computing)
- torch (CPU-only for quantum ML)
- pydantic (data validation)

**Total Build Time:** 20-40 minutes

**Disk Space Required:**
- Build directory: 5-8GB
- Final ISO: 1.5-2.5GB

## Advanced Usage

### Multi-Boot USB with Ventoy

1. Install Ventoy on USB
2. Copy `aios-live.iso` to USB
3. Boot from USB and select ISO

### Network Boot (PXE)

Extract contents and configure PXE server:

```bash
# Mount ISO
sudo mount -o loop aios-live.iso /mnt

# Copy to PXE server
sudo cp /mnt/live/vmlinuz /tftpboot/
sudo cp /mnt/live/initrd.img /tftpboot/
sudo cp /mnt/live/filesystem.squashfs /var/www/html/

# Configure pxelinux.cfg
```

### Automated Testing

```bash
# Headless QEMU test
qemu-system-x86_64 \
    -cdrom aios-live.iso \
    -m 4G \
    -enable-kvm \
    -nographic \
    -serial stdio

# With VNC
qemu-system-x86_64 \
    -cdrom aios-live.iso \
    -m 4G \
    -enable-kvm \
    -vnc :0
```

## Maintenance

### Updating the ISO

1. Edit `build-iso.sh` with updated package versions
2. Rebuild ISO
3. Test in VM
4. Distribute

### Security Updates

The ISO includes packages from Debian stable. To update:

```bash
# In the build script, add:
sudo chroot chroot /bin/bash -c "
    apt-get update
    apt-get upgrade -y
    apt-get dist-upgrade -y
"
```

## Credits

Built with:
- Debian Live Project
- ISOLINUX/SYSLINUX
- GRUB
- Squashfs-tools
- Xorriso

Ai|oS powered by Claude Code.
