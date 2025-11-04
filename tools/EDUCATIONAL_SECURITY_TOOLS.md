# Educational Security Tools - Ai:oS Sovereign Toolkit
## Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

This document describes the educational security tools added to the Ai:oS Sovereign Security Toolkit for authorized security training, research, and penetration testing education.

---

## ⚠️ IMPORTANT DISCLAIMER

**These tools are for EDUCATIONAL and AUTHORIZED SECURITY TESTING purposes ONLY.**

- Use only on systems you own or have explicit written permission to test
- Unauthorized access to computer systems is illegal under the Computer Fraud and Abuse Act (CFAA) and similar laws worldwide
- These tools are designed for learning, security research, and authorized penetration testing
- Misuse of these tools for malicious purposes is strictly prohibited and may result in criminal prosecution

**BY USING THESE TOOLS, YOU ACKNOWLEDGE THAT YOU:**
1. Have proper authorization to test the target systems
2. Understand the legal implications of security testing
3. Will use these tools responsibly and ethically
4. Accept full responsibility for your actions

---

## Tool Overview

### 1. ExploitDB Search Tool (`exploitdb.py`)

**Purpose**: Educational vulnerability research and exploit database exploration

**Inspired by**: Exploit-DB online database (https://www.exploit-db.com)

**Features**:
- Search vulnerabilities by CVE, keyword, or EDB-ID
- Filter by platform (Windows, Linux, PHP, multiple)
- Filter by exploit type (remote, local, webapps, DOS)
- View detailed exploit information and descriptions
- Educational database of well-known exploits (Log4Shell, EternalBlue, BlueKeep, Drupalgeddon2, etc.)

**Usage**:
```bash
# Search for log4j vulnerabilities
python -m tools.exploitdb log4j --json

# Search for Windows remote exploits
python -m tools.exploitdb windows --platform windows --type remote

# Get detailed information
python -m tools.exploitdb CVE-2021-44228 --detailed

# Health check
python -m tools.exploitdb --health
```

**Educational Value**:
- Learn about common vulnerability patterns
- Understand CVE naming conventions
- Study historical security vulnerabilities
- Research defensive strategies against known exploits

---

### 2. Shodan Search Tool (`shodansearch.py`)

**Purpose**: Educational demonstration of internet-connected device research

**Inspired by**: Shodan (https://www.shodan.io)

**Features**:
- Simulated search of internet-exposed devices and services
- Device information including IP, port, service, version, location
- Known vulnerability mapping
- Educational demonstration (does not require actual Shodan API)

**Usage**:
```bash
# Search for Apache servers
python -m tools.shodansearch apache --json

# Search by port
python -m tools.shodansearch "port:22"

# Search by country
python -m tools.shodansearch "country:US"

# Limit results
python -m tools.shodansearch nginx --limit 10

# Health check
python -m tools.shodansearch --health
```

**Educational Value**:
- Understand internet exposure and attack surface
- Learn about common services running on the internet
- Study geographical distribution of services
- Research defense strategies for internet-facing systems

**Note**: This is a simulation using sample data for educational purposes. Real Shodan requires an API key and proper authorization.

---

### 3. NmapPro Network Scanner (`nmappro.py`)

**Purpose**: Advanced network reconnaissance with visualization

**Inspired by**: Nmap (https://nmap.org)

**Features**:
- Full nmap integration (requires nmap binary installed)
- Multiple scan profiles (quick, intense, stealth, comprehensive, ping)
- Interactive HTML/JavaScript GUI with live topology visualization
- Service detection and version enumeration
- OS detection and fingerprinting
- Vulnerability correlation via CVE database
- NSE (Nmap Scripting Engine) support
- Real-time scan progress tracking
- Multiple output formats (JSON, XML, text)

**Usage**:
```bash
# Quick scan of localhost
python -m tools.nmappro 127.0.0.1

# Comprehensive scan of subnet
python -m tools.nmappro 192.168.1.0/24 --profile comprehensive

# Stealth scan with slow timing
python -m tools.nmappro example.com --profile stealth --timing T1

# Custom nmap arguments
python -m tools.nmappro 10.0.0.1 --custom "-sS -sV -O -A"

# Launch interactive GUI
python -m tools.nmappro --gui

# Save results to file
python -m tools.nmappro 192.168.1.1 --json --output scan-results.json

# Health check (verify nmap is installed)
python -m tools.nmappro --health
```

**Scan Profiles**:
- **quick**: Fast scan of common ports (`-T4 -F`)
- **intense**: OS detection, version detection, script scanning (`-T4 -A -v`)
- **stealth**: SYN stealth scan with polite timing (`-sS -T2`)
- **comprehensive**: Full scan with UDP, scripts, and multiple techniques
- **ping**: Ping sweep only, no port scanning (`-sn`)

**Educational Value**:
- Learn network reconnaissance techniques
- Understand TCP/IP protocols and port scanning
- Study service enumeration and fingerprinting
- Practice OS detection methodologies
- Explore network topology mapping
- Research defense against network scans

**Prerequisites**:
- Requires `nmap` binary installed on system
- macOS: `brew install nmap`
- Linux: `apt-get install nmap` or `yum install nmap`
- Windows: Download from https://nmap.org/download.html

---

## Integration with Ai:oS

All tools follow the Ai:oS Sovereign Security Toolkit pattern:

### Health Check System
Every tool provides a `health_check()` function that returns:
```python
{
    "tool": "tool_name",
    "status": "ok" | "warn" | "error",
    "summary": "Human-readable status",
    "details": {
        # Tool-specific metrics
        "latency_ms": float,
        # Additional metadata
    }
}
```

### JSON Output
All tools support `--json` flag for structured output compatible with automation pipelines:
```bash
python -m tools.exploitdb CVE-2021-44228 --json | jq '.results[0].cve'
```

### Metadata Publishing
Tools integrate with Ai:oS ExecutionContext for telemetry:
```python
from tools import exploitdb

health = exploitdb.health_check()
ctx.publish_metadata("security.exploitdb", health)
```

---

## Security Best Practices

### Before Using These Tools

1. **Get Authorization**
   - Obtain written permission from system owners
   - Document the scope of testing
   - Define acceptable testing windows
   - Establish communication protocols

2. **Understand Legal Boundaries**
   - Know your local computer crime laws
   - Understand the Computer Fraud and Abuse Act (CFAA)
   - Be aware of international laws if testing cross-border
   - Consult legal counsel if unsure

3. **Use Safely**
   - Test in isolated lab environments first
   - Use virtualization for practice
   - Never test production systems without proper change management
   - Have rollback plans ready

4. **Document Everything**
   - Keep logs of all testing activities
   - Document findings thoroughly
   - Track authorization and scope
   - Maintain audit trails

### During Testing

1. **Respect Rate Limits**
   - Don't overwhelm target systems
   - Use appropriate timing templates
   - Monitor resource usage
   - Stop if systems become unstable

2. **Minimize Impact**
   - Use non-destructive testing methods first
   - Avoid denial-of-service conditions
   - Clean up test artifacts
   - Restore systems to original state

3. **Protect Data**
   - Handle discovered vulnerabilities responsibly
   - Secure test results and findings
   - Follow responsible disclosure practices
   - Don't share exploitation details publicly

### After Testing

1. **Report Findings**
   - Provide detailed, actionable reports
   - Include remediation recommendations
   - Prioritize vulnerabilities by severity
   - Follow up on fixes

2. **Clean Up**
   - Remove test accounts and data
   - Delete uploaded files
   - Close opened connections
   - Verify no persistent changes

---

## Educational Scenarios

### Scenario 1: Vulnerability Research Lab

**Objective**: Learn about common vulnerabilities and their history

```bash
# Research a specific CVE
python -m tools.exploitdb CVE-2021-44228 --detailed

# Explore platform-specific vulnerabilities
python -m tools.exploitdb --platform windows --type remote

# Study webapps vulnerabilities
python -m tools.exploitdb --type webapps --json | jq '.[] | .title'
```

**Learning Outcomes**:
- Understand vulnerability lifecycle
- Learn CVE naming and tracking
- Study common attack patterns
- Research historical security incidents

### Scenario 2: Network Discovery Training

**Objective**: Practice network reconnaissance techniques

```bash
# Start with ping sweep
python -m tools.nmappro 192.168.1.0/24 --profile ping

# Scan discovered hosts
python -m tools.nmappro 192.168.1.100 --profile quick

# Deep scan with OS detection
python -m tools.nmappro 192.168.1.100 --profile intense --json
```

**Learning Outcomes**:
- Master network mapping
- Understand service enumeration
- Practice OS fingerprinting
- Learn stealth scanning techniques

### Scenario 3: Internet Exposure Assessment

**Objective**: Understand what information is publicly available

```bash
# Research Apache servers
python -m tools.shodansearch apache

# Find SSH servers
python -m tools.shodansearch "port:22"

# Analyze by country
python -m tools.shodansearch "country:US" --limit 20
```

**Learning Outcomes**:
- Understand internet attack surface
- Learn about common exposures
- Study geographic distribution
- Research defensive hardening

---

## Development and Extension

### Adding New Tools

1. Create tool file in `aios/tools/`:
```python
#!/usr/bin/env python3
"""
Tool Name - Description
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

def health_check() -> dict:
    """Return tool health status"""
    return {
        "tool": "toolname",
        "status": "ok",
        "summary": "Tool ready",
        "details": {}
    }

def main(argv=None):
    """Main entry point"""
    pass

if __name__ == "__main__":
    import sys
    sys.exit(main())
```

2. Add documentation to `SOVEREIGN_SECURITY_TOOLKIT.md`

3. Create tests in `aios/tests/`

4. Update tool registry (if encrypted file allows)

### Contributing

Contributions should:
- Follow educational and defensive security focus
- Include comprehensive documentation
- Provide health check functionality
- Support `--json` output
- Include proper copyright headers
- Have clear educational disclaimers

---

## Testing and Validation

### Health Checks

All tools support health checking:
```bash
python -m tools.exploitdb --health
python -m tools.shodansearch --health
python -m tools.nmappro --health
```

### JSON Output Validation

Test structured output:
```bash
python -m tools.exploitdb log4j --json | jq '.'
python -m tools.shodansearch apache --json | jq '.results | length'
python -m tools.nmappro 127.0.0.1 --json | jq '.stats'
```

### Integration Testing

Test with Ai:oS runtime:
```bash
# Boot with security response deck
python aios/aios --manifest aios/examples/manifest-security-response.json \
  --env AGENTA_SECURITY_TOOLS=NmapPro,ExploitDB \
  -v boot

# Execute sovereign suite
python aios/aios -v exec security.sovereign_suite
```

---

## Resources and Further Learning

### Official Documentation
- Nmap: https://nmap.org/book/
- Exploit-DB: https://www.exploit-db.com/docs/
- Shodan: https://help.shodan.io/
- OWASP: https://owasp.org/

### Legal Resources
- Computer Fraud and Abuse Act: https://www.justice.gov/criminal-ccips/ccmanual-109-computer-fraud-and-abuse-act
- Responsible Disclosure: https://cheatsheetseries.owasp.org/cheatsheets/Vulnerability_Disclosure_Cheat_Sheet.html

### Training Platforms
- HackTheBox: https://www.hackthebox.com/
- TryHackMe: https://tryhackme.com/
- PentesterLab: https://pentesterlab.com/
- SANS Cyber Ranges: https://www.sans.org/cyber-ranges/

### Certifications
- Offensive Security Certified Professional (OSCP)
- Certified Ethical Hacker (CEH)
- GIAC Penetration Tester (GPEN)
- Certified Red Team Professional (CRTP)

---

## Support and Community

For questions, issues, or contributions:

1. Review existing documentation in `aios/` directory
2. Check `SOVEREIGN_SECURITY_TOOLKIT.md` for detailed tool information
3. Refer to `CLAUDE.md` for development guidelines
4. Follow responsible disclosure for security issues

---

## License

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

These tools are proprietary and provided for authorized educational and security testing purposes only.

---

## Final Reminder

**ALWAYS:**
- ✅ Get proper authorization
- ✅ Document your testing scope
- ✅ Use ethical testing methods
- ✅ Report vulnerabilities responsibly
- ✅ Learn and improve security posture

**NEVER:**
- ❌ Test systems without permission
- ❌ Use tools maliciously
- ❌ Cause denial of service
- ❌ Share exploitation techniques irresponsibly
- ❌ Violate computer crime laws

---

**Your actions have consequences. Use these tools wisely, legally, and ethically.**
