# Ai|oS Bootable ISO Builder - Index

Complete toolset for creating a bootable Linux ISO with Ai|oS pre-installed.

## Quick Navigation

### 🚀 Getting Started
- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute guide to build and test
- **[verify-prerequisites.sh](verify-prerequisites.sh)** - Check if you're ready to build

### 📚 Documentation
- **[README.md](README.md)** - Comprehensive documentation (10,000+ words)
  - System requirements
  - Build methods
  - Testing procedures
  - Troubleshooting guide
  - Advanced usage

### 🔧 Build Scripts

#### macOS Users (Recommended)
- **[build-iso-macos.sh](build-iso-macos.sh)** - Docker-based build for macOS
- Requires: Docker Desktop with 8GB+ RAM

#### Linux Users
- **[build-iso.sh](build-iso.sh)** - Native Linux build
- Requires: debootstrap, squashfs-tools, xorriso, isolinux

#### Docker (Any Platform)
- **[docker-compose.yml](docker-compose.yml)** - Docker Compose configuration
- **[Dockerfile.iso](Dockerfile.iso)** - Multi-stage ISO builder

### 📦 What Gets Built

The ISO includes:

```
Ai|oS Bootable ISO (~1.5-2.5GB)
├── Debian 12 (Bookworm) base
├── Python 3.11+ environment
├── Ai|oS Framework
│   ├── All meta-agents (kernel, security, networking, storage, etc.)
│   ├── Sovereign Security Toolkit (8 red-team tools)
│   ├── QuantumAgent with QuLab2.0 integration
│   ├── GUI launcher with wolf icon
│   └── Auto-start configuration
├── QuLab2.0 Quantum Framework
│   ├── Quantum teleportation
│   ├── Bayesian evidence ledger
│   ├── Monte Carlo forecasting
│   └── 32 quantum algorithms
├── TheGAVLSuite
│   ├── Agentic ritual engine
│   ├── Boardroom of Light
│   ├── Jiminy Cricket
│   └── GAVL modules (OSINT, Hellfire, Legal, Bayesian)
├── Firefox ESR (kiosk mode)
├── Openbox window manager
└── NetworkManager for connectivity
```

### 🎯 Typical Workflow

```bash
# 1. Verify prerequisites
./verify-prerequisites.sh

# 2. Build ISO (choose one)
./build-iso-macos.sh    # macOS with Docker
./build-iso.sh          # Linux native
docker-compose up       # Any platform with Docker

# 3. Test in VM
qemu-system-x86_64 -cdrom output/aios-live.iso -m 4G -accel hvf

# 4. Write to USB
diskutil list
diskutil unmountDisk /dev/diskN
sudo dd if=output/aios-live.iso of=/dev/rdiskN bs=4m

# 5. Boot and enjoy!
```

## File Structure

```
iso-builder/
├── INDEX.md                    # This file
├── QUICKSTART.md               # Quick start guide
├── README.md                   # Full documentation
├── verify-prerequisites.sh     # Prerequisite checker
├── build-iso-macos.sh         # macOS builder
├── build-iso.sh               # Linux builder
├── docker-compose.yml         # Docker Compose config
├── Dockerfile.iso             # ISO builder Dockerfile
└── output/                    # Build output directory
    └── aios-live.iso          # Final ISO (after build)
```

## Build Time & Requirements

| Platform | Method | Time | Disk | RAM |
|----------|--------|------|------|-----|
| macOS | Docker | 30-40 min | 10GB | 8GB |
| Linux | Native | 20-30 min | 10GB | 4GB |
| Any | Docker | 30-40 min | 10GB | 8GB |

## Default Credentials

Once booted:
- **Username:** `aios`
- **Password:** `aios`
- **Sudo:** Enabled (passwordless)

## Key Features

✅ **Fully Automated** - One command builds everything
✅ **Live Environment** - Runs from RAM, no installation needed
✅ **Auto-Start** - Ai|oS boots automatically
✅ **GUI Ready** - Firefox kiosk with wolf icon launcher
✅ **Network Enabled** - NetworkManager pre-configured
✅ **Quantum Ready** - QuLab2.0 fully integrated
✅ **Security Suite** - All tools pre-installed
✅ **Bootable USB** - Write to USB and boot anywhere
✅ **BIOS & UEFI** - Hybrid boot support
✅ **Safe Testing** - Live mode preserves host system

## Common Use Cases

### Development & Testing
Boot Ai|oS in a clean environment to test changes without affecting your main system.

### Demonstrations
Show Ai|oS capabilities on any x86_64 machine with a bootable USB.

### Recovery & Forensics
Boot into Ai|oS to analyze systems without modifying host disk.

### Training & Education
Provide students with a consistent, pre-configured Ai|oS environment.

### Deployment
Install Ai|oS on new systems from the live environment.

## Support & Troubleshooting

**Prerequisites fail?** → Run `./verify-prerequisites.sh` for detailed diagnostics

**Build errors?** → Check README.md troubleshooting section

**Boot issues?** → Try Safe Mode from GRUB menu

**Network problems?** → Press Ctrl+Alt+F2, login, run `nmtui`

**Black screen?** → Wait 60 seconds, check display/monitor

## Next Steps After Build

1. **Test the ISO** in a VM (QEMU, VirtualBox, VMware)
2. **Verify boot sequence** and auto-start works
3. **Check network connectivity**
4. **Test Ai|oS commands:**
   ```bash
   cd /opt/aios
   python3 aios/aios -v status
   python3 aios/aios -v exec quantum.health_check
   ```
5. **Write to USB** and test on real hardware
6. **Share** or distribute the ISO

## Version Information

**ISO Builder Version:** 1.0
**Ai|oS Version:** Latest (from source)
**Base OS:** Debian 12 (Bookworm)
**Python:** 3.11+
**Kernel:** Linux 6.1+

## License

Ai|oS and components follow their respective licenses. See individual project documentation.

## Credits

**Ai|oS** - AI Operating System with meta-agent architecture
**QuLab2.0** - Quantum computing framework
**TheGAVLSuite** - GAVL module suite
**Debian** - Base Linux distribution
**Powered by** - Claude Code

---

*Built with ❤️ by the Ai|oS team*
