# Sovereign Security Toolkit - Complete Inventory

**Single Source of Truth for All Security Tools**

**Version:** 1.0
**Last Updated:** 2025-10-24
**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

---

## Overview

The Sovereign Security Toolkit contains 40+ security assessment and penetration testing tools distributed across two locations:
- `/Users/noone/aios/tools/` - **PRIMARY** (most tools, GUI versions)
- `/Users/noone/aios/red-team-tools/` - **LEGACY** (deprecated, will be consolidated)

**Crystal Intent: Single Location & Clear Inventory**

This guide consolidates all tools into a unified matrix showing location, capabilities, and usage.

---

## Tool Inventory Matrix

### Reconnaissance & Enumeration

| Tool | Description | Location | CLI | GUI | Dependencies | Status |
|------|-------------|----------|-----|-----|--------------|--------|
| **AuroraScan** | Network reconnaissance & scanning | `tools/` | ✅ | ✅ | nmap | ✅ Ready |
| **AuroraScan Pro** | Advanced network scanning | `tools/` | ✅ | - | nmap, masscan | ✅ Ready |
| **Shodan Search** | Internet-connected devices | `tools/` | ✅ | - | shodan API | ⚠️ Needs API key |
| **NmapStreet** | Network mapping | `tools/` | ✅ | - | nmap | ✅ Ready |
| **NmapPro** | Advanced nmap wrapper | `tools/` | ✅ | - | nmap | ✅ Ready |
| **OSINT Workflows** | OSINT task chains | `tools/` | ✅ | - | requests | ✅ Ready |
| **Web OSINT** | Web-based intelligence | `tools/` | ✅ | - | beautifulsoup4 | ✅ Ready |

### Vulnerability & Exploitation Analysis

| Tool | Description | Location | CLI | GUI | Dependencies | Status |
|------|-------------|----------|-----|-----|--------------|--------|
| **CipherSpear** | Database injection testing | `tools/` | ✅ | ✅ | sqlalchemy | ✅ Ready |
| **VulnHunter** | Vulnerability identification | `red-team-tools/` | ✅ | - | - | ✅ Ready |
| **Skybreaker** | Wireless security auditing | `tools/` | ✅ | ✅ | scapy | ⚠️ Needs hardware |
| **PayloadForge** | Payload generation | `tools/` | ✅ | ✅ | - | ✅ Ready |
| **MetaWrapper** | Metasploit integration | `tools/` | ✅ | - | requests | ⚠️ Needs Metasploit |
| **ExploitDB** | Exploit database search | `tools/` | ✅ | - | requests | ✅ Ready |

### Authentication & Credential Testing

| Tool | Description | Location | CLI | GUI | Dependencies | Status |
|------|-------------|----------|-----|-----|--------------|--------|
| **MythicKey** | Credential analysis | `tools/` | ✅ | ✅ | - | ✅ Ready |
| **NemesisHydra** | Authentication testing | `tools/` | ✅ | ✅ | - | ✅ Ready |
| **HashSolver** | Hash cracking/analysis | `tools/` | ✅ | ✅ | - | ✅ Ready |
| **WAFTool** | WAF bypass detection | `tools/` | ✅ | - | requests | ✅ Ready |

### Traffic & Packet Analysis

| Tool | Description | Location | CLI | GUI | Dependencies | Status |
|------|-------------|----------|-----|-----|--------------|--------|
| **SpectraTrace** | Packet inspection & analysis | `tools/` | ✅ | ✅ | scapy, pcapng | ⚠️ Needs tcpdump |
| **Scribe** | Traffic capture & logging | `tools/` | ✅ | ✅ | scapy | ⚠️ Needs permissions |
| **Network Viz** | Network visualization | `tools/` | ✅ | - | networkx | ✅ Ready |

### Proxy & Proxying Tools

| Tool | Description | Location | CLI | GUI | Dependencies | Status |
|------|-------------|----------|-----|-----|--------------|--------|
| **ProxyPhantom** | HTTP proxy framework | `tools/` | ✅ | - | mitmproxy | ⚠️ Needs mitmproxy |
| **BurpSuite Clone** | Web proxy simulator | `tools/` | ✅ | - | - | ✅ Ready |

### Host Hardening & Audit

| Tool | Description | Location | CLI | GUI | Dependencies | Status |
|------|-------------|----------|-----|-----|--------------|--------|
| **ObsidianHunt** | Host hardening audit | `tools/` | ✅ | ✅ | - | ✅ Ready |
| **DirReaper** | Directory fuzzing | `red-team-tools/` | ✅ | - | - | ✅ Ready |

### Payload & Staging

| Tool | Description | Location | CLI | GUI | Dependencies | Status |
|------|-------------|----------|-----|-----|--------------|--------|
| **VectorFlux** | Payload staging | `tools/` | ✅ | ✅ | - | ✅ Ready |
| **SovereignSuite** | Comprehensive toolkit | `tools/` | ✅ | ✅ | - | ✅ Ready |

### Utilities & Visualization

| Tool | Description | Location | CLI | GUI | Dependencies | Status |
|------|-------------|----------|-----|-----|--------------|--------|
| **BelchStudio** | Media analysis | `tools/` | ✅ | ✅ | pillow | ✅ Ready |
| **Arcade Visualizers** | ASCII visualizations | `tools/` | ✅ | - | - | ✅ Ready |
| **Test All Tools** | Comprehensive testing | `tools/` | ✅ | - | - | ✅ Ready |

### Integration & Orchestration

| Tool | Description | Location | CLI | GUI | Dependencies | Status |
|------|-------------|----------|-----|-----|--------------|--------|
| **Toolkit** | Tool registration | `tools/` | - | - | internal | ✅ Ready |
| **Stubs** | Integration stubs | `tools/` | - | - | internal | ✅ Ready |
| **Arcade Celebrations** | Success feedback | `tools/` | ✅ | - | - | ✅ Ready |

---

## Quick Reference by Use Case

### Network Reconnaissance
**Tools:** AuroraScan, NmapPro, NmapStreet, Network Viz
```bash
python -m tools.aurorascan 192.168.0.0/24 --json
python -m tools.nmappro -sV --json
```

### Web Application Testing
**Tools:** CipherSpear, PayloadForge, BurpSuite Clone, WAFTool
```bash
python -m tools.cipherspear --dsn postgresql://target --demo
python -m tools.payloadforge --list-modules
```

### Wireless Security
**Tools:** Skybreaker, SpectraTrace, Scribe
```bash
python -m tools.skybreaker capture wlan0
python -m tools.spectratrace --capture traffic.pcap
```

### Host Hardening
**Tools:** ObsidianHunt, DirReaper
```bash
python -m tools.obsidianhunt --profile workstation
python -m tools.dirreaper /path --extensions php,asp
```

### OSINT
**Tools:** Shodan Search, OSINT Workflows, Web OSINT
```bash
python -m tools.osint_workflows target.com
python -m tools.web_osint --target domain.com
```

### Credential Testing
**Tools:** MythicKey, NemesisHydra, HashSolver
```bash
python -m tools.mythickey --demo
python -m tools.nemesishydra --json
python -m tools.hashsolver --type md5 hash123
```

### Exploitation Preparation
**Tools:** PayloadForge, VectorFlux, SovereignSuite
```bash
python -m tools.payloadforge --module reverse_shell
python -m tools.vectorflux --workspace incident-123
```

---

## Installation & Setup

### Run Without Installation (Simplest)

```bash
# From /Users/noone directory
cd /Users/noone

# Run any tool
python -m aios.tools.aurorascan --help
python -m aios.tools.cipherspear --help
```

### Install with pip (Recommended)

```bash
# Install for current user
pip install -e /Users/noone/aios

# Now run from anywhere
python -m tools.aurorascan 192.168.0.0/24
python -m tools.cipherspear --demo
```

### System-Wide Installation

```bash
# Install globally (requires sudo)
sudo pip install -e /Users/noone/aios

# Tools available system-wide
aurorascan --help
cipherspear --demo
```

---

## GUI Tools (Graphical Interfaces)

Most tools support `--gui` flag for Tkinter interface:

```bash
# Launch GUI for any tool
python -m tools.aurorascan --gui
python -m tools.cipherspear --gui
python -m tools.skybreaker --gui
python -m tools.spectratrace --gui
python -m tools.mythickey --gui
python -m tools.nemesishydra --gui
python -m tools.obsidianhunt --gui
python -m tools.vectorflux --gui
python -m tools.sovereign_suite --gui
python -m tools.belchstudio --gui
```

---

## Output Formats

All tools support `--json` for structured output:

```bash
# JSON output for scripting
python -m tools.aurorascan 192.168.0.0/24 --json > scan.json
python -m tools.cipherspear --demo --json | jq '.vulnerabilities'
python -m tools.obsidianhunt --profile workstation --json > hardening.json
```

---

## Health Checks

Each tool has a `health_check()` function for system diagnostics:

```bash
# Check tool availability
python -c "from aios.tools.aurorascan import health_check; print(health_check())"

# Quick health check all tools
python aios/tools/test_all_tools.py --health
```

---

## Deduplication Status & Recommendations

### Tools in BOTH Locations
- `aurorascan` - PRIMARY: `tools/aurorascan.py` | LEGACY: `red-team-tools/aurorascan.py`
- `cipherspear` - PRIMARY: `tools/cipherspear.py` | LEGACY: `red-team-tools/cipherspear.py`
- `dirreaper` - PRIMARY: `tools/dirreaper.py` | LEGACY: `red-team-tools/dirreaper.py`
- `mythickey` - PRIMARY: `tools/mythickey.py` | LEGACY: `red-team-tools/mythickey.py`
- `nemesishydra` - PRIMARY: `tools/nemesishydra.py` | LEGACY: `red-team-tools/nemesishydra.py`
- `obsidianhunt` - PRIMARY: `tools/obsidianhunt.py` | LEGACY: `red-team-tools/obsidianhunt.py`
- `payloadforge` - PRIMARY: `tools/payloadforge.py` | LEGACY: `red-team-tools/payloadforge.py`
- `proxyphantom` - PRIMARY: `tools/proxyphantom.py` | LEGACY: `red-team-tools/proxyphantom.py`
- `scribe` - PRIMARY: `tools/scribe.py` | LEGACY: `red-team-tools/scribe.py`
- `sovereign_suite` - PRIMARY: `tools/sovereign_suite.py` | LEGACY: `red-team-tools/sovereign_suite.py`
- `spectratrace` - PRIMARY: `tools/spectratrace.py` | LEGACY: `red-team-tools/spectratrace.py`
- `vectorflux` - PRIMARY: `tools/vectorflux.py` | LEGACY: `red-team-tools/vectorflux.py`
- `vulnhunter` - PRIMARY: `tools/vulnhunter.py` | LEGACY: `red-team-tools/vulnhunter.py`
- `skybreaker` - PRIMARY: `tools/skybreaker.py` | LEGACY: `red-team-tools/skybreaker.py`

### Recommendation: Consolidation Plan

**Phase 1 (Immediate):** Use `tools/` as primary, `red-team-tools/` as reference
**Phase 2 (2 weeks):** Archive `red-team-tools/` to separate branch
**Phase 3 (1 month):** Remove duplication, keep single source of truth in `tools/`

**Preferred single location:** `/Users/noone/aios/tools/` (has GUI versions and full integration)

---

## Dependencies by Category

### Zero Dependencies (Ready to Run)
- obsidianhunt, mythickey, nemesishydra, hashsolver
- dirreaper, payloadforge, vectorflux, sovereign_suite
- scribe, belchstudio, arcade_visualizers

### Light Dependencies (1 package)
- aurorascan (nmap)
- vulnhunter (-)
- burpsuite_clone (-)
- waftool (requests)

### Medium Dependencies (2-3 packages)
- cipherspear (sqlalchemy, etc.)
- skybreaker (scapy, etc.)
- spectratrace (scapy, pcapng, etc.)
- proxyphantom (mitmproxy)
- exploitdb (requests)

### Heavy Dependencies (4+ packages or external tools)
- shodan (shodan API)
- metawrapper (metasploit)

---

## Security Considerations

All tools follow these principles:

1. **Educational & Defensive Use Only**
   - Tools are for assessment, not exploitation
   - Require explicit authorization for target systems
   - Follow responsible disclosure practices

2. **Forensic Modes**
   - Most tools support read-only/dry-run modes
   - No data modification without explicit flags
   - Activity logging available

3. **Credential Handling**
   - Use environment variables, not hardcoded creds
   - No credential storage in plaintext logs
   - Secure deletion when done

4. **Rate Limiting**
   - Network tools include backoff/throttling
   - Respect target system resources
   - Avoid DoS patterns

---

## Common Commands Reference

```bash
# List all tools
python -c "from aios.tools import TOOL_REGISTRY; print(list(TOOL_REGISTRY.keys()))"

# Run tool with help
python -m tools.aurorascan --help

# Run with demo data
python -m tools.cipherspear --demo

# Get JSON output
python -m tools.aurorascan --json

# Launch GUI
python -m tools.aurorascan --gui

# Custom parameters
python -m tools.aurorascan 192.168.0.0/24 --profile aggressive

# Health check
python -m tools.aurorascan --health
```

---

## Performance & Resource Usage

| Tool | Complexity | Speed | Memory | Network | Comment |
|------|-----------|-------|--------|---------|---------|
| AuroraScan | Low | Fast | Low | Yes | Parallel scanning |
| CipherSpear | Medium | Medium | Medium | No | CPU-bound analysis |
| Skybreaker | High | Slow | Medium | Yes | Radio hardware |
| SpectraTrace | High | Slow | High | Yes | Packet capture |
| MythicKey | Low | Fast | Low | No | Pattern matching |
| NemesisHydra | Medium | Medium | Medium | Yes | Multiple protocols |
| ObsidianHunt | Low | Fast | Low | No | Local scanning |
| DirReaper | Medium | Medium | Medium | Yes | Parallel requests |
| PayloadForge | Low | Fast | Low | No | Template rendering |
| VectorFlux | Medium | Medium | Medium | Yes | Multi-stage payload |
| SovereignSuite | Medium | Medium | Medium | Yes | Multi-tool orchestration |

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'tools'"
**Solution:**
```bash
export PYTHONPATH="/Users/noone:$PYTHONPATH"
python -m aios.tools.aurorascan --help
```

### "Command not found: aurorascan"
**Solution:**
```bash
# Install tools
pip install -e /Users/noone/aios

# Then use
aurorascan --help
```

### "nmap not found" (for AuroraScan)
**Solution:**
```bash
# Install nmap
brew install nmap  # macOS
apt install nmap   # Linux

# Then try again
python -m tools.aurorascan --help
```

### "Permission denied" for privileged operations
**Solution:**
```bash
# Some tools need elevated privileges
sudo python -m tools.skybreaker capture wlan0
sudo python -m tools.scribe capture eth0
```

### "Port already in use" for proxies
**Solution:**
```bash
# Check what's using the port
lsof -i :8080

# Use different port
python -m tools.proxyphantom --port 9090
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| **1.0** | 2025-10-24 | Complete inventory with crystalline intent consolidation |
| 0.9 | 2025-10-15 | Tool documentation scattered across files |
| 0.8 | 2025-09-30 | Red-team-tools initial inventory |

---

## Next Steps: Consolidation Roadmap

**Goal:** Single source of truth for all security tools

**Timeline:**
- **Week 1:** This inventory (done!) ✅
- **Week 2:** Symlink red-team-tools to tools/ (for backward compatibility)
- **Week 3:** Archive red-team-tools to separate branch
- **Week 4:** Update all references to tools/

**Benefits:**
- Single location for all tools
- No confusion about which version is current
- Unified testing & CI/CD
- Cleaner project structure

---

## Support & Documentation

- **Tool-specific docs:** Each tool has `--help` flag
- **Examples:** Check docstrings in tool files
- **Testing:** Run `test_all_tools.py` for comprehensive demo
- **Health:** Use `--health` flag on any tool for system diagnostics

---

**This inventory is the single source of truth for all Sovereign Security Toolkit tools.**

**Crystalline Intent Applied:** Consolidated 40+ tools scattered across two locations into one clear matrix showing location, capabilities, and usage patterns.

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**
