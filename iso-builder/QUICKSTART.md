# Ai|oS Bootable ISO - Quick Start Guide

## Build the ISO

### On macOS (Easiest)

```bash
cd /Users/noone/aios/iso-builder
./build-iso-macos.sh
```

**Requirements:**
- Docker Desktop installed and running
- 8GB+ RAM allocated to Docker
- 10GB free disk space
- 20-40 minutes build time

**Output:** `./output/aios-live.iso`

### On Linux

```bash
cd /Users/noone/aios/iso-builder
./build-iso.sh
```

**Requirements:**
```bash
sudo apt-get install debootstrap squashfs-tools xorriso isolinux \
    syslinux-efi grub-pc-bin grub-efi-amd64-bin mtools dosfstools
```

**Output:** `/Users/noone/aios/aios-live.iso`

### Using Docker (Any Platform)

```bash
cd /Users/noone/aios/iso-builder
docker-compose up
```

**Output:** `./output/aios-live.iso`

## Test the ISO

### QEMU (macOS)

```bash
# Install QEMU
brew install qemu

# Test ISO
qemu-system-x86_64 \
    -cdrom output/aios-live.iso \
    -m 4G \
    -accel hvf
```

### QEMU (Linux)

```bash
qemu-system-x86_64 \
    -cdrom aios-live.iso \
    -m 4G \
    -enable-kvm
```

### VirtualBox

1. New VM → Linux → Debian 64-bit
2. RAM: 4096MB
3. No hard disk
4. Settings → Storage → IDE Controller → Add CD
5. Select `aios-live.iso`
6. Start

### VMware Fusion (macOS)

1. File → New → Install from disc or image
2. Select `aios-live.iso`
3. Debian 11.x 64-bit
4. RAM: 4096MB
5. Start

## Write to USB

### macOS

```bash
# Find USB device
diskutil list

# Unmount (replace diskN)
diskutil unmountDisk /dev/diskN

# Write (⚠️ ERASES USB!)
sudo dd if=output/aios-live.iso of=/dev/rdiskN bs=4m

# Eject
diskutil eject /dev/diskN
```

### Linux

```bash
# Find USB device
lsblk

# Write (⚠️ ERASES USB! Replace sdX)
sudo dd if=aios-live.iso of=/dev/sdX bs=4M status=progress oflag=sync
```

### Windows

Use [Rufus](https://rufus.ie/):
1. Download Rufus
2. Select ISO
3. Select USB drive
4. Click Start

## Boot from USB/ISO

1. **Insert USB** or mount ISO in VM
2. **Restart** computer
3. **Boot menu:**
   - Mac: Hold Option/Alt key
   - PC: Press F12, F2, ESC, or DEL
4. **Select** USB/CD drive
5. **Choose** "Ai|oS - AI Operating System"

## What Happens Next

The system will automatically:

1. ✓ Boot Debian Linux
2. ✓ Auto-login as user `aios`
3. ✓ Start X server with Openbox
4. ✓ Run `python3 aios/aios -v boot`
5. ✓ Launch Firefox with Ai|oS GUI launcher
6. ✓ Display wolf icon menu with glowing eyes

## Default Credentials

- **Username:** `aios`
- **Password:** `aios`
- **Sudo:** Enabled (no password required)

## Troubleshooting

### Black screen after boot?

Press `Ctrl+Alt+F2` for console, login, then:

```bash
# Check Ai|oS logs
journalctl -u aios.service

# Check X server
tail -f /var/log/Xorg.0.log

# Restart Ai|oS manually
/usr/local/bin/aios-autostart.sh
```

### No network?

```bash
# Start NetworkManager
sudo systemctl start NetworkManager

# Connect to WiFi
nmtui
```

### Out of memory?

Use a system with 8GB+ RAM or increase VM memory allocation.

### ISO won't boot?

- Disable Secure Boot in BIOS/UEFI
- Try Legacy BIOS mode instead of UEFI (or vice versa)
- Verify ISO integrity: `sha256sum aios-live.iso`
- Re-burn USB drive with different tool

## Next Steps

Once booted:

1. **Explore Ai|oS:**
   - Click wolf icon menu
   - Browse dashboard
   - Run `python3 aios/aios -v status`

2. **Run Quantum Tools:**
   ```bash
   cd /opt/QuLab2.0
   python3 -m qulab.cli health
   ```

3. **Test Security Toolkit:**
   ```bash
   cd /opt/aios
   python3 -m tools.aurorascan --demo
   ```

4. **Install to Hard Disk:**
   ```bash
   sudo apt-get install debian-installer-launcher
   sudo debian-installer-launcher
   ```

## File Locations

Inside the live environment:

- **Ai|oS:** `/opt/aios/`
- **QuLab2.0:** `/opt/QuLab2.0/`
- **GAVL Suite:** `/opt/TheGAVLSuite/`
- **GUI Launcher:** `/opt/aios/web/aios_launcher.html`
- **Wolf Icon:** `/opt/aios/web/wolf_icon_head.png`
- **Auto-start:** `/usr/local/bin/aios-autostart.sh`

## Performance

- **ISO Size:** ~1.5-2.5GB
- **RAM Usage:** 2-4GB (8GB recommended)
- **Boot Time:** 20-40 seconds
- **Live System:** Runs entirely in RAM

## Need Help?

- Full documentation: `README.md`
- Ai|oS guide: `/opt/aios/CLAUDE.md`
- Report issues: https://github.com/anthropics/claude-code/issues

## What's Included

✓ Debian 12 (Bookworm) base
✓ Python 3.11+
✓ Ai|oS with all meta-agents
✓ QuLab2.0 quantum framework
✓ TheGAVLSuite modules
✓ Sovereign Security Toolkit
✓ Firefox ESR browser
✓ Openbox window manager
✓ Auto-start configured
✓ Network tools
✓ Development tools

**Total:** ~1.5-2.5GB bootable Linux environment
