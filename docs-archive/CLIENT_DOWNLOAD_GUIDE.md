# AioS Client Download Guide

## What Should Clients Download?

### 🎯 For Different User Types:

---

## 1. **End Users** (Just want to use the tools)

**DON'T download anything!**

Visit the websites:
- **aios.is** - AioS security toolkit web interface
- **thegavl.com** - TheGAVL Suite legal analysis

Everything works in the browser - no downloads needed.

---

## 2. **Developers** (Want to run AioS locally)

**Download the CODE repo:**

```bash
# Clone the aios repository
git clone https://github.com/Workofarttattoo/AioS.git
cd AioS

# Install dependencies
pip install -r requirements.txt

# Run AioS
python aios/aios -v boot
```

**What you get:**
- Full AioS codebase
- All security tools (aurorascan, cipherspear, etc.)
- Agent system
- Quantum algorithms
- ML algorithms
- No website files (those are separate)

---

## 3. **Website Contributors** (Want to edit websites)

**aios.is website:**
```bash
git clone https://github.com/workofarttattoo/aios-website.git
```

**thegavl.com website:**
```bash
git clone https://github.com/Workofarttattoo/thegavl-website.git
```

Edit HTML/CSS/JS and submit pull requests.

---

## 4. **Full Stack Contributors** (Want everything)

```bash
# Code
git clone https://github.com/Workofarttattoo/AioS.git

# Website
git clone https://github.com/workofarttattoo/aios-website.git

# TheGAVL website
git clone https://github.com/Workofarttattoo/thegavl-website.git
```

---

## Repository Structure

### **AioS** (Code - Public Repo)
- **URL:** https://github.com/Workofarttattoo/AioS
- **Purpose:** Core codebase, tools, agents
- **Size:** ~50MB (no website files)
- **License:** See LICENSE file

### **aios-website** (Website - Separate Repo)
- **URL:** https://github.com/workofarttattoo/aios-website
- **Purpose:** aios.is public website
- **Deployed:** GitHub Pages → aios.is

### **thegavl-website** (Website - Separate Repo)
- **URL:** https://github.com/Workofarttattoo/thegavl-website
- **Purpose:** thegavl.com public website
- **Deployed:** GitHub Pages → thegavl.com

---

## Quick Start for Developers

### Minimum Requirements:
- Python 3.9+
- pip
- Git

### Installation:
```bash
# 1. Clone
git clone https://github.com/Workofarttattoo/AioS.git
cd AioS

# 2. Install
pip install -r requirements.txt

# 3. Run
python aios/aios -v boot
```

### Example Usage:
```bash
# Run security suite
python -m tools.aurorascan 192.168.0.0/24 --profile recon

# Start meta-agents
python aios/aios -v exec security.sovereign_suite

# Natural language
python aios/aios -v prompt "scan network and check firewall"
```

---

## Continuous Training (Advanced)

If you want to run the full training pipeline:

```bash
# Run all training labs continuously
python /Users/noone/continuous_training_daemon.py
```

This runs:
- Oracle training (daily)
- Telescope training (daily)
- GAVL ingestion (hourly)
- ECH0 research (continuous)

---

## Support

- **Documentation:** See README.md in each repo
- **Issues:** GitHub Issues in respective repos
- **Website:** aios.is or thegavl.com

---

**Copyright © 2025 Joshua Hendricks Cole (DBA: Corporation of Light)**
**All Rights Reserved. PATENT PENDING.**
