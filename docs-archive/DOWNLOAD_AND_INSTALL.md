# 🤖 Ai:oS - ONE-CLICK DOWNLOAD & INSTALL

**Sovereign AI Operating System with Auto-Configuration**

## 🚀 Quickest Install (One Command)

```bash
curl -fsSL https://raw.githubusercontent.com/corporationoflight/aios/main/INSTALL_AIOS_ONECLICK.sh | bash
```

That's it! The installer will:
- ✅ Auto-detect your system (macOS/Linux)
- ✅ Install all dependencies automatically
- ✅ Optimize settings for your hardware
- ✅ Create clickable launcher icons
- ✅ Boot Ai:oS when ready

## 📥 Alternative: Download & Run

1. **Download the installer:**
   ```bash
   wget https://raw.githubusercontent.com/corporationoflight/aios/main/INSTALL_AIOS_ONECLICK.sh
   ```

2. **Make it executable:**
   ```bash
   chmod +x INSTALL_AIOS_ONECLICK.sh
   ```

3. **Run it:**
   ```bash
   ./INSTALL_AIOS_ONECLICK.sh
   ```

4. **Or just double-click** the file in your file manager!

## 🎯 What Gets Installed

- **Ai:oS Core** - Agentic control plane with meta-agents
- **Sovereign Security Toolkit** - 8 security assessment tools
- **ML Algorithms Suite** - Mamba, MCTS, Flow Matching, NUTS, etc.
- **Quantum ML Suite** - Quantum-enhanced algorithms
- **Clickable Launchers** - Desktop icon + Applications menu entry

## 🖱️ After Installation

You can launch Ai:oS in **three ways**:

1. **Click the desktop icon**: `🤖 Launch AiOS`
2. **macOS**: Open `~/Applications/AiOS.app`
   **Linux**: Applications menu → Ai:oS
3. **Command line**: `~/aios/aios-boot`

## ⚙️ Auto-Configuration

The installer automatically:

- Detects CPU cores and optimizes concurrency
- Detects RAM and enables/disables quantum features
- Detects platform and installs correct dependencies
- Creates launchers specific to your system
- No manual configuration needed!

## 📦 System Requirements

**Minimum:**
- macOS 10.15+ or Linux (Ubuntu 20.04+, Fedora 35+, etc.)
- Python 3.8+
- 8GB RAM
- 4 CPU cores
- 2GB free disk space

**Recommended:**
- 16GB+ RAM (enables quantum simulation)
- 8+ CPU cores (better parallelization)
- 5GB free disk space

## 🆘 Troubleshooting

**Problem**: "Python not found"
**Solution**: The installer will auto-install Python, but if it fails:
- macOS: `brew install python3`
- Ubuntu/Debian: `sudo apt install python3 python3-pip`
- Fedora: `sudo dnf install python3 python3-pip`

**Problem**: "Permission denied"
**Solution**: Run `chmod +x INSTALL_AIOS_ONECLICK.sh` first

**Problem**: Installer hangs
**Solution**: Some dependencies are large (PyTorch ~1GB). Wait 5-10 minutes on first install.

## 🔄 Updating

To update Ai:oS to the latest version:

```bash
cd ~/aios
git pull
pip install -r requirements-auto.txt --upgrade
```

Or just re-run the one-click installer - it will update automatically!

## 📖 Next Steps

After installation:

1. **Boot the system:**
   ```bash
   ~/aios/aios-boot
   ```

2. **Run setup wizard** (optional):
   ```bash
   cd ~/aios
   python3 aios/aios wizard
   ```

3. **Try security tools:**
   ```bash
   python3 -m tools.aurorascan --gui
   python3 -m tools.cipherspear --gui
   ```

4. **Check ML algorithms:**
   ```bash
   python3 aios/ml_algorithms.py
   python3 aios/quantum_ml_algorithms.py
   ```

5. **Read full docs**: See `~/aios/README.md` or visit the GitHub repo

## 🌟 Features

- **Meta-Agent Architecture**: Declarative manifest-based control plane
- **Sovereign Security Suite**: 8 reimagined security tools
- **Advanced ML**: Mamba/SSM, Flow Matching, MCTS, NUTS HMC, etc.
- **Quantum Computing**: 1-50 qubit simulation with VQE, teleportation
- **Autonomous Discovery**: Level 4 autonomous LLM agents
- **Cloud Integration**: AWS, Azure, GCP, Docker, QEMU/libvirt

## 📄 License

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

## 🤝 Support

- GitHub Issues: https://github.com/corporationoflight/aios/issues
- Documentation: https://github.com/corporationoflight/aios
- Email: support@corporationoflight.com

---

**Made with ❤️ by Corporation of Light**
