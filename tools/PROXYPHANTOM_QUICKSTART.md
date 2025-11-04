# 🦊 ProxyPhantom Quick Start Guide

**5-Minute Setup & Testing**

---

## Installation

### 1. Check Dependencies
```bash
python -m tools.proxyphantom --health
```

### 2. Install Optional Dependencies (Recommended)
```bash
pip install cryptography aiohttp websockets
```

---

## Basic Usage

### Launch GUI (Easiest)
```bash
python -m tools.proxyphantom --gui
```
Opens stunning orange/amber cyberpunk interface in browser.

### Run Demo
```bash
python -m tools.proxyphantom --demo
```
Demonstrates all features with sample data.

### Start Proxy Server
```bash
python -m tools.proxyphantom --proxy
```
Then configure browser proxy: `127.0.0.1:8080`

### Scan Website
```bash
# Passive scan (safe)
python -m tools.proxyphantom --scan https://example.com

# Active scan (sends payloads)
python -m tools.proxyphantom --scan https://example.com --active
```

### Spider Website
```bash
python -m tools.proxyphantom --spider https://example.com --depth 3
```

---

## Common Commands

```bash
# Health check with JSON output
python -m tools.proxyphantom --health --json

# Custom proxy port
python -m tools.proxyphantom --proxy --port 8888

# Deep spider (depth 5)
python -m tools.proxyphantom --spider https://target.com --depth 5

# Active vulnerability scan with JSON output
python -m tools.proxyphantom --scan https://target.com --active --json
```

---

## Python API

### Quick Scan
```python
from tools.proxyphantom import ProxyPhantom

app = ProxyPhantom()
issues = app.scan_url("https://example.com")

for issue in issues:
    print(f"[{issue.severity.value}] {issue.type.value}")
```

### Decode/Encode
```python
from tools.proxyphantom import Decoder

# Base64
encoded = Decoder.encode("ProxyPhantom", "base64")
decoded = Decoder.decode(encoded, "base64")

# Hashing
hash_value = Decoder.hash("password123", "sha256")
```

### Token Analysis
```python
from tools.proxyphantom import Sequencer

seq = Sequencer()
seq.add_sample("a4f8d2b1c9e7")
seq.add_sample("b5g9e3c2d0f8")

analysis = seq.analyze()
print(f"Entropy: {analysis['entropy']:.2f}")
print(f"Quality: {analysis['quality']}")
```

---

## GUI Modules

| Module     | Purpose                        | Key Feature                    |
|-----------|--------------------------------|--------------------------------|
| Proxy     | Intercept requests/responses   | Toggle intercept on/off        |
| Spider    | Map website structure          | Automatic link discovery       |
| Scanner   | Find vulnerabilities           | Passive + Active scanning      |
| Intruder  | Automated attacks/fuzzing      | 4 attack types                 |
| Repeater  | Manual request manipulation    | Edit & resend                  |
| Decoder   | Encode/decode/hash data        | 8 formats + 4 hash algorithms  |
| Comparer  | Diff responses                 | Visual differences             |
| Sequencer | Token randomness analysis      | Entropy + pattern detection    |

---

## Vulnerability Types Detected

✅ SQL Injection
✅ Cross-Site Scripting (XSS)
✅ Command Injection
✅ Path Traversal
✅ Missing Security Headers
✅ Insecure Cookies
✅ Information Disclosure
✅ Weak Cryptography

---

## Attack Types (Intruder)

| Type          | Description                    | Use Case              |
|--------------|--------------------------------|-----------------------|
| Sniper       | One position at a time         | Parameter testing     |
| Battering Ram| Same payload everywhere        | Password guessing     |
| Pitchfork    | Different payloads per position| Credential stuffing   |
| Cluster Bomb | All combinations               | Brute force           |

---

## Tips & Tricks

### 1. HTTPS Interception
To intercept HTTPS, you must install the ProxyPhantom CA certificate in your browser trust store. Without this, you'll see certificate errors.

### 2. Passive vs Active Scanning
- **Passive**: Safe, analyzes existing traffic, no intrusive requests
- **Active**: Sends attack payloads, use only on authorized targets

### 3. Sequencer Sample Size
Need at least 10-20 token samples for accurate randomness analysis. More samples = more accurate.

### 4. Spider Depth
- Depth 1-3: Fast, surface-level mapping
- Depth 4-5: Thorough, may take time on large sites
- Depth 6+: Very thorough, use for deep reconnaissance

### 5. Intruder Payload Limits
Start with small payload sets (10-50) for testing, then scale up for production scans.

---

## Integration with Ai|oS

### Health Check from SecurityAgent
```python
from tools import run_health_check

result = run_health_check("ProxyPhantom")
print(result["status"])  # "ok"
```

### Enable in Manifest
```json
{
  "meta_agents": {
    "security": {
      "actions": {
        "proxyphantom_scan": {
          "critical": false,
          "description": "Web application security testing"
        }
      }
    }
  }
}
```

---

## Troubleshooting

### Problem: SSL Certificate Errors
**Solution**: Install ProxyPhantom CA certificate in browser trust store

### Problem: No Issues Found
**Solution**: Try enabling active scanning with `--active` flag

### Problem: Slow Spider
**Solution**: Reduce depth with `--depth 2` or limit pages

### Problem: Module not found
**Solution**: Run from project root: `python -m tools.proxyphantom`

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────┐
│  🦊 PROXYPHANTOM QUICK COMMANDS                │
├─────────────────────────────────────────────────┤
│  --gui          Launch web interface            │
│  --proxy        Start proxy server (port 8080)  │
│  --scan URL     Scan for vulnerabilities        │
│  --spider URL   Map website structure           │
│  --demo         Run demonstration               │
│  --health       Check system status             │
│  --json         JSON output format              │
│  --active       Enable active scanning          │
│  --depth N      Spider depth (default: 3)       │
│  --port N       Custom proxy port               │
└─────────────────────────────────────────────────┘
```

---

## Next Steps

1. **Learn More**: Read full `PROXYPHANTOM_README.md`
2. **Practice**: Use `--demo` mode to explore features
3. **Test Safely**: Always get authorization before scanning
4. **Explore GUI**: Launch `--gui` for visual interface
5. **Integrate**: Use Python API for custom workflows

---

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**
