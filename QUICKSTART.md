# AI:OS Quick Start Guide

**Copyright © 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

---

## 🚀 ONE-CLICK INSTALLATION

AI:OS now has a revolutionary one-click installer!

### macOS / Linux
1. **Download** the AI:OS directory
2. **Double-click** `bootstrap_aios.sh` **OR** run:
   ```bash
   cd path/to/aios
   ./bootstrap_aios.sh
   ```

### Windows
1. **Download** the AI:OS directory
2. **Double-click** `bootstrap_aios.bat`

**That's it!** The installer automatically:
- ✅ Checks/installs Python
- ✅ Installs all dependencies
- ✅ Creates desktop launcher
- ✅ Sets up command shortcuts
- ✅ Tests the installation

---

## 🎮 USING AI:OS AFTER INSTALLATION

### Method 1: Desktop Launcher (Easiest!)
After installation, you'll find on your Desktop:
- **macOS/Linux**: `Launch_AIOS.command` or `AIOS.app`
- **Windows**: `AIOS.lnk` shortcut

**Double-click** to open an interactive menu with options:
1. Boot AI:OS
2. Run Setup Wizard
3. Boot with Security Toolkit
4. Run in Forensic Mode (read-only)
5. Execute Natural Language Command
6. View System Status

### Method 2: Terminal Shortcut
After installation, restart your terminal and simply type:
```bash
aios
```
Works from anywhere! (macOS/Linux)

### Method 3: Direct Commands
Navigate to the AI:OS directory:
```bash
python3 aios -v boot              # Boot the system
python3 aios wizard               # Run setup wizard
python3 aios --help               # See all options
```

---

## 📚 Core Commands

```bash
# Setup
./aios wizard              # Interactive setup

# Execution
./aios boot                # Boot the system
./aios exec ACTION         # Run specific action (e.g., security.firewall)
./aios prompt "TEXT"       # Natural language execution

# Information
./aios metadata            # Show all telemetry
./aios --help              # Full command list
./aios --version           # Version info

# Testing
./aios --manifest examples/manifest-security-response.json boot
```

---

## 🎯 What is Ai:oS?

**Ai:oS** (AI Operating System) is an agentic control-plane that coordinates autonomous meta-agents through declarative manifests.

### Meta-Agents:
- **Kernel** - Process management, system initialization
- **Security** - Firewall, encryption, security toolkit
- **Networking** - Network config, DNS, routing
- **Storage** - Volume management, filesystems
- **Applications** - Process/Docker/VM orchestration
- **Scalability** - Load monitoring, cloud providers
- **Orchestration** - Policy engine, telemetry, health
- **User** - User management, authentication
- **GUI** - Display server management

### Key Features:
- ✅ **Declarative Manifests** - Define system as code
- ✅ **Autonomous Agents** - Self-managing subsystems
- ✅ **Multi-Provider** - Docker, AWS, Azure, GCP, QEMU, libvirt
- ✅ **Forensic Mode** - Read-only, safe exploration
- ✅ **Natural Language** - Prompt-based execution
- ✅ **ML/Quantum Ready** - Built-in advanced algorithms

---

## 📦 What's Included

### Core System:
- `/aios` - Main executable
- `/agents/` - Meta-agent implementations
- `/tools/` - Sovereign Security Toolkit (7 tools)
- `/examples/` - Example manifests

### Advanced Features:
- **ML Algorithms** - Mamba, Flow Matching, MCTS, Bayesian inference
- **Quantum ML** - VQE, quantum circuits (1-50 qubits)
- **Autonomous Discovery** - Level 4 self-directed learning agents
- **Security Suite** - AuroraScan, CipherSpear, SkyBreaker, etc.

---

## 🔧 Requirements

### Minimum:
- Python 3.8+
- Linux, macOS, or Windows
- 2GB RAM
- Internet connection

### Optional:
- **Docker** - Container orchestration
- **PyTorch** - Quantum ML support
- **Ollama** - Autonomous AI agents
- **Cloud CLI** - AWS, Azure, GCP integration

---

## 💡 Example Workflows

### Security Audit:
```bash
./aios --env AGENTA_SECURITY_TOOLS=AuroraScan,CipherSpear boot
```

### Natural Language:
```bash
./aios prompt "enable firewall and check container load"
```

### Forensic Mode (Read-Only):
```bash
./aios --forensic boot
```

### Custom Manifest:
```bash
./aios --manifest my-manifest.json boot
```

---

## 📖 Documentation

- **README.md** - Full system overview
- **CLAUDE.md** - Developer guide
- **docs-archive/** - Detailed guides
- **examples/** - Sample manifests

---

## 🆘 Support

**Issues**: https://github.com/YOUR_ORG/aios/issues

**Documentation**: `./aios --help`

---

**🎉 You're ready! Run `./aios wizard` to get started.**
