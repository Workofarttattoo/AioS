# 🎯 VulnHunter - Comprehensive Vulnerability Scanner

**Copyright © 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

VulnHunter is a production-ready vulnerability scanner for Ai|oS that matches the capabilities of industry-standard tools like OpenVAS and Nessus. It provides comprehensive vulnerability assessment with 50+ built-in checks, CVSS v3 scoring, and a stunning crimson cyberpunk GUI.

---

## Features

### Core Capabilities

✅ **Comprehensive Scanning**
- Network vulnerability detection
- Web application security testing
- Database security assessment
- Authentication vulnerability checks
- Configuration auditing

✅ **50+ Built-in Vulnerability Checks**
- 10 Network checks (Telnet, FTP, SNMP, VNC, SSL/TLS, etc.)
- 15 Web checks (SQLi, XSS, LFI, RFI, XXE, SSRF, etc.)
- 10 Authentication checks (Default creds, weak passwords, etc.)
- 10 Configuration checks (Directory listing, backup files, etc.)
- 5 Database checks (MongoDB, Redis, MySQL, PostgreSQL, Elasticsearch)

✅ **Advanced Features**
- CVSS v3 scoring engine
- Multiple scan profiles (Quick, Full, Web, Network, Compliance)
- Credential-based authenticated scanning
- Safe exploit verification
- Asset management and tracking
- Risk prioritization
- Remediation tracking

✅ **Reporting**
- HTML reports with visual styling
- JSON structured output
- CSV exports
- Scan history tracking

✅ **Stunning GUI**
- Red/Crimson cyberpunk theme with targeting crosshair icon 🎯
- Real-time scan progress with animations
- Interactive dashboard with statistics
- Vulnerability browser with filtering
- Asset inventory management
- Report generator

---

## Installation

VulnHunter is part of the Ai|oS Sovereign Security Toolkit and comes pre-installed:

```bash
# Check health
python -m tools.vulnhunter --health

# Verify JSON output
python -m tools.vulnhunter --health --json
```

---

## Quick Start

### 1. Launch GUI

```bash
python -m tools.vulnhunter --gui
```

The GUI opens in your default browser with a comprehensive interface featuring:
- **Dashboard**: Overview of vulnerabilities and statistics
- **Scan**: Configure and run scans with multiple profiles
- **Vulnerabilities**: Browse and filter discovered vulnerabilities
- **Hosts**: Asset inventory with vulnerability counts
- **Reports**: Generate and download reports in multiple formats
- **Plugins**: View all available vulnerability checks

### 2. Command-Line Scanning

#### Quick Scan (Critical & High Only)
```bash
python -m tools.vulnhunter --scan 192.168.1.100 --profile quick
```

#### Full Network Scan
```bash
python -m tools.vulnhunter --scan 192.168.1.0/24 --profile full
```

#### Web Application Scan
```bash
python -m tools.vulnhunter --scan example.com --profile web
```

#### Compliance Scan
```bash
python -m tools.vulnhunter --scan 10.0.0.0/8 --profile compliance
```

### 3. Authenticated Scanning

Create a credentials file:
```
admin:password123
root:toor
user:user
```

Run scan with credentials:
```bash
python -m tools.vulnhunter --scan 192.168.1.100 --credentials creds.txt
```

### 4. Generate Reports

#### HTML Report
```bash
python -m tools.vulnhunter --scan 192.168.1.100 --report html --output report.html
```

#### JSON Report
```bash
python -m tools.vulnhunter --scan 192.168.1.100 --report json --output report.json
```

#### CSV Export
```bash
python -m tools.vulnhunter --scan 192.168.1.100 --report csv --output report.csv
```

---

## Vulnerability Categories

### Network Vulnerabilities (NET-001 to NET-010)

| Check ID | Name | Severity | CVSS | Description |
|----------|------|----------|------|-------------|
| NET-001 | Open Telnet Service | HIGH | 7.5 | Telnet transmits data in plaintext |
| NET-002 | Open FTP Service | MEDIUM | 5.0 | FTP transmits credentials in cleartext |
| NET-003 | Anonymous FTP Enabled | HIGH | 7.0 | FTP allows anonymous access |
| NET-004 | SNMP Default Community | HIGH | 7.5 | SNMP using default community strings |
| NET-005 | Open VNC Service | HIGH | 7.5 | VNC exposed without proper protection |
| NET-006 | Open RDP Service | MEDIUM | 5.5 | RDP exposed to network |
| NET-007 | SSLv2/v3 Enabled | HIGH | 7.0 | Weak SSL/TLS protocols in use |
| NET-008 | Weak SSL Cipher | MEDIUM | 5.3 | Weak encryption cipher suites |
| NET-009 | Self-Signed Certificate | LOW | 3.7 | Self-signed SSL certificate in use |
| NET-010 | Expired SSL Certificate | MEDIUM | 5.0 | SSL certificate has expired |

### Web Vulnerabilities (WEB-001 to WEB-015)

| Check ID | Name | Severity | CVSS | Description |
|----------|------|----------|------|-------------|
| WEB-001 | SQL Injection | CRITICAL | 9.8 | SQL injection vulnerability detected |
| WEB-002 | Cross-Site Scripting (XSS) | HIGH | 7.5 | XSS vulnerability allows code injection |
| WEB-003 | Local File Inclusion | CRITICAL | 9.0 | LFI allows reading arbitrary files |
| WEB-004 | Remote File Inclusion | CRITICAL | 9.5 | RFI allows remote code execution |
| WEB-005 | XML External Entity (XXE) | HIGH | 8.0 | XXE allows data exfiltration |
| WEB-006 | Server-Side Request Forgery | HIGH | 7.5 | SSRF enables internal network access |
| WEB-007 | Directory Traversal | HIGH | 7.5 | Path traversal vulnerability |
| WEB-008 | Command Injection | CRITICAL | 9.5 | OS command injection possible |
| WEB-009 | LDAP Injection | HIGH | 7.5 | LDAP query manipulation |
| WEB-010 | HTTP Header Injection | MEDIUM | 5.5 | HTTP response splitting |
| WEB-011 | Missing Security Headers | LOW | 3.5 | Security headers not configured |
| WEB-012 | Clickjacking | MEDIUM | 4.5 | UI redressing vulnerability |
| WEB-013 | Session Fixation | MEDIUM | 6.0 | Session fixation attack possible |
| WEB-014 | Insecure Cookies | MEDIUM | 5.0 | Cookies lack security attributes |
| WEB-015 | Open Redirect | MEDIUM | 5.5 | Unvalidated redirect vulnerability |

### Authentication Vulnerabilities (AUTH-001 to AUTH-010)

| Check ID | Name | Severity | CVSS | Description |
|----------|------|----------|------|-------------|
| AUTH-001 | Default Credentials | CRITICAL | 9.0 | Default username/password accepted |
| AUTH-002 | Weak Password Policy | HIGH | 7.0 | Password policy is insufficient |
| AUTH-003 | No Account Lockout | MEDIUM | 5.5 | No lockout after failed attempts |
| AUTH-004 | Password in URL | HIGH | 7.5 | Passwords transmitted in URL |
| AUTH-005 | Cleartext Password Storage | CRITICAL | 8.5 | Passwords stored without encryption |
| AUTH-006 | Weak Password Hashing | HIGH | 7.0 | Weak hashing algorithm used |
| AUTH-007 | Missing MFA | MEDIUM | 6.0 | Multi-factor auth not enabled |
| AUTH-008 | Session Timeout Too Long | LOW | 4.0 | Sessions don't expire timely |
| AUTH-009 | Predictable Session IDs | HIGH | 7.5 | Session IDs are predictable |
| AUTH-010 | Privilege Escalation | CRITICAL | 9.0 | Privilege escalation possible |

### Configuration Vulnerabilities (CONF-001 to CONF-010)

| Check ID | Name | Severity | CVSS | Description |
|----------|------|----------|------|-------------|
| CONF-001 | Directory Listing Enabled | MEDIUM | 5.0 | Directory indexes exposed |
| CONF-002 | Backup Files Exposed | HIGH | 7.0 | Backup files publicly accessible |
| CONF-003 | Server Version Disclosure | LOW | 3.0 | Server version information leaked |
| CONF-004 | Debug Mode Enabled | HIGH | 7.5 | Debug mode enabled in production |
| CONF-005 | Unnecessary Services | MEDIUM | 5.5 | Unused services are running |
| CONF-006 | World-Writable Files | HIGH | 7.0 | Files have insecure permissions |
| CONF-007 | Unencrypted Protocols | HIGH | 7.0 | Cleartext protocols in use |
| CONF-008 | Weak File Permissions | MEDIUM | 6.0 | File permissions too permissive |
| CONF-009 | Missing Security Updates | HIGH | 8.0 | Critical patches not applied |
| CONF-010 | Exposed Admin Interface | HIGH | 7.5 | Admin panel publicly accessible |

### Database Vulnerabilities (DB-001 to DB-005)

| Check ID | Name | Severity | CVSS | Description |
|----------|------|----------|------|-------------|
| DB-001 | MongoDB No Auth | CRITICAL | 9.5 | MongoDB without authentication |
| DB-002 | Redis No Auth | CRITICAL | 9.5 | Redis without authentication |
| DB-003 | MySQL Root No Password | CRITICAL | 9.0 | MySQL root has no password |
| DB-004 | PostgreSQL Trust Auth | HIGH | 8.0 | PostgreSQL using trust authentication |
| DB-005 | Elasticsearch No Auth | CRITICAL | 9.0 | Elasticsearch without authentication |

---

## Scan Profiles

VulnHunter supports multiple scan profiles optimized for different use cases:

### Quick Scan
- Runs only CRITICAL and HIGH severity checks
- Fastest scan time (5-10 minutes)
- Ideal for rapid assessment
- Best for time-sensitive audits

### Full Scan (Default)
- Runs all 50+ vulnerability checks
- Comprehensive coverage across all categories
- Typical scan time: 20-30 minutes
- Recommended for thorough assessments

### Web Application Scan
- Focuses on WEB-001 through WEB-015 checks
- Tests for SQLi, XSS, LFI, RFI, XXE, SSRF, etc.
- Includes security header checks
- Ideal for web application pentesting

### Network Scan
- Focuses on NET-001 through NET-010 checks
- Tests for open services, weak protocols, SSL/TLS issues
- Port scanning and service detection
- Ideal for infrastructure assessment

### Compliance Scan
- Focuses on AUTH and CONF checks
- Tests authentication and configuration security
- Aligned with common compliance frameworks
- Ideal for audit preparation

---

## CVSS v3 Scoring

VulnHunter implements a full CVSS v3 scoring engine that calculates base scores using:

### Base Metrics
- **Attack Vector** (Network, Adjacent, Local, Physical)
- **Attack Complexity** (Low, High)
- **Privileges Required** (None, Low, High)
- **User Interaction** (None, Required)
- **Scope** (Unchanged, Changed)
- **Confidentiality Impact** (High, Low, None)
- **Integrity Impact** (High, Low, None)
- **Availability Impact** (High, Low, None)

### Severity Levels
- **CRITICAL**: CVSS 9.0-10.0 (Red, blinking)
- **HIGH**: CVSS 7.0-8.9 (Bright red)
- **MEDIUM**: CVSS 4.0-6.9 (Orange)
- **LOW**: CVSS 0.1-3.9 (Yellow)
- **INFO**: CVSS 0.0 (Green)

---

## GUI Features

### Dashboard Tab
- **Statistics Cards**: Total vulns, Critical, High, Medium, Low counts
- **Risk Timeline Chart**: Visual timeline of vulnerability trends
- **Top 10 Vulnerabilities**: Most critical findings
- **Recently Scanned Hosts**: Latest scan activity

### Scan Tab
- **Target Input**: IP address, CIDR range, or hostname
- **Profile Selection**: Quick, Full, Web, Network, Compliance
- **Credentials Configuration**: Upload or paste credentials for authenticated scans
- **Real-time Progress**: Animated progress bar with status updates
- **Scan Results**: Immediate display of findings

### Vulnerabilities Tab
- **Comprehensive Table**: All discovered vulnerabilities
- **Severity Filtering**: Filter by Critical, High, Medium, Low, Info
- **Sortable Columns**: Sort by host, port, severity, CVSS score
- **Status Tracking**: NEW, CONFIRMED, FALSE_POSITIVE, REMEDIATED
- **Action Buttons**: View details, mark status, export

### Hosts Tab
- **Asset Inventory**: All scanned hosts
- **Vulnerability Counts**: Per-host vulnerability statistics
- **First/Last Seen**: Asset discovery timeline
- **Host Details**: Click for comprehensive host information

### Reports Tab
- **Format Selection**: HTML, JSON, CSV, PDF (coming soon)
- **Scope Selection**: All, Critical/High only, Unresolved, New (24h)
- **Report Preview**: Real-time report preview
- **Download**: One-click download of generated reports

### Plugins Tab
- **Plugin Browser**: View all 50+ vulnerability checks
- **Category Filtering**: Filter by Network, Web, Auth, Config, Database
- **Severity Filtering**: Filter by severity level
- **Plugin Details**: View description, CVSS score, remediation
- **Enable/Disable**: Toggle individual checks (coming soon)

---

## Integration with Ai|oS

VulnHunter integrates seamlessly with the Ai|oS runtime:

### Security Agent Integration

```python
from aios.runtime import ExecutionContext
from tools import vulnhunter

def security_scan_action(ctx: ExecutionContext):
    """Run VulnHunter scan via Security Agent"""
    target = ctx.environment.get("SCAN_TARGET", "localhost")
    profile = ctx.environment.get("SCAN_PROFILE", "full")

    # Initialize scanner
    scanner = vulnhunter.VulnHunterScanner()

    # Run scan
    results = scanner.scan(target, profile)

    # Publish results to metadata
    ctx.publish_metadata("security.vulnhunter", {
        "target": target,
        "vulnerabilities_found": len(results),
        "severity_breakdown": scanner._get_severity_breakdown()
    })

    return ActionResult(
        success=True,
        message=f"VulnHunter found {len(results)} vulnerabilities",
        payload={"results": [v.to_dict() for v in results]}
    )
```

### Environment Variables

```bash
# Enable VulnHunter in Security Agent
export AGENTA_SECURITY_TOOLS=AuroraScan,VulnHunter

# Configure default scan profile
export VULNHUNTER_PROFILE=full

# Set scan target
export SCAN_TARGET=192.168.1.0/24

# Enable verbose logging
export VULNHUNTER_VERBOSE=1
```

### Health Check

```bash
# Check VulnHunter health
python -m tools.vulnhunter --health

# JSON output for automation
python -m tools.vulnhunter --health --json
```

---

## Usage Examples

### Example 1: Basic Network Scan

```bash
python -m tools.vulnhunter --scan 192.168.1.100
```

Output:
```
[info] Starting full scan on 192.168.1.100
[info] Found: Weak SSL Cipher on 192.168.1.100:443 (CVSS: 5.3)
[info] Found: Directory Listing Enabled on 192.168.1.100:80 (CVSS: 5.0)
[info] Scan complete. Found 2 vulnerabilities

Severity Breakdown:
  MEDIUM: 2

Top Vulnerabilities:
  - Weak SSL Cipher (192.168.1.100:443) - CVSS 5.3
  - Directory Listing Enabled (192.168.1.100:80) - CVSS 5.0
```

### Example 2: Web Application Scan with Report

```bash
python -m tools.vulnhunter \
  --scan example.com \
  --profile web \
  --report html \
  --output example_report.html
```

### Example 3: Compliance Scan with Credentials

```bash
python -m tools.vulnhunter \
  --scan 10.0.0.0/24 \
  --profile compliance \
  --credentials admin_creds.txt \
  --report json \
  --output compliance_report.json
```

### Example 4: Python API Usage

```python
from tools.vulnhunter import VulnHunterScanner, Severity

# Initialize scanner
scanner = VulnHunterScanner()

# Run scan
results = scanner.scan("192.168.1.100", scan_profile="full")

# Filter critical vulnerabilities
critical_vulns = [v for v in results if v.severity == Severity.CRITICAL]

print(f"Found {len(critical_vulns)} critical vulnerabilities:")
for vuln in critical_vulns:
    print(f"  - {vuln.name} ({vuln.host}:{vuln.port}) - CVSS {vuln.cvss_score}")
    print(f"    Remediation: {vuln.remediation}")

# Generate report
html_report = scanner.generate_report('html')
with open('report.html', 'w') as f:
    f.write(html_report)
```

---

## Architecture

### Core Components

1. **VulnHunterScanner**: Main scanner engine
   - Manages vulnerability checks
   - Coordinates parallel scanning
   - Tracks assets and scan history
   - Generates reports

2. **VulnerabilityCheck**: Base class for checks
   - Implements check logic
   - Returns Vulnerability objects
   - Includes severity and CVSS scoring

3. **Vulnerability**: Represents a finding
   - Host, port, severity, CVSS score
   - Proof of concept output
   - Remediation guidance
   - References (CVE, CWE, etc.)

4. **CVSS**: CVSS v3 scoring engine
   - Calculates base scores
   - Implements full CVSS v3 metrics
   - Returns scores 0.0-10.0

### Check Types

- **NetworkCheck**: Network vulnerability checks
- **WebCheck**: Web application security checks
- **AuthCheck**: Authentication vulnerability checks
- **ConfigCheck**: Configuration auditing checks
- **DatabaseCheck**: Database security checks

### Parallel Scanning

VulnHunter uses ThreadPoolExecutor for concurrent scanning:
- 10 worker threads by default
- Port scanning before vulnerability checks
- Exception handling per check
- Progress tracking

---

## Comparison with OpenVAS/Nessus

| Feature | VulnHunter | OpenVAS | Nessus |
|---------|------------|---------|--------|
| Built-in Checks | 50+ | 50,000+ | 100,000+ |
| CVSS Scoring | ✅ v3 | ✅ v3 | ✅ v3 |
| Web GUI | ✅ | ✅ | ✅ |
| CLI | ✅ | ✅ | ✅ |
| Authenticated Scans | ✅ | ✅ | ✅ |
| Report Generation | ✅ HTML/JSON/CSV | ✅ PDF/HTML/XML | ✅ PDF/HTML/CSV |
| Scan Profiles | ✅ | ✅ | ✅ |
| Asset Management | ✅ Basic | ✅ Advanced | ✅ Advanced |
| Plugin System | ✅ Modular | ✅ NASL | ✅ NASL |
| Compliance | ✅ Basic | ✅ PCI/HIPAA/CIS | ✅ PCI/HIPAA/CIS |
| Open Source | ✅ | ✅ | ❌ (Commercial) |
| Ai|oS Integration | ✅ Native | ❌ | ❌ |

**VulnHunter Advantages:**
- Native Ai|oS integration
- Lightweight and fast
- Stunning cyberpunk GUI
- Easy to extend with custom checks
- No complex setup required

**OpenVAS/Nessus Advantages:**
- 1000x more vulnerability checks
- Mature CVE database integration
- Advanced compliance frameworks
- Enterprise features

---

## Roadmap

### Phase 1: Core Features (Complete)
- ✅ 50+ vulnerability checks
- ✅ CVSS v3 scoring
- ✅ Multiple scan profiles
- ✅ HTML/JSON/CSV reporting
- ✅ Crimson cyberpunk GUI
- ✅ Ai|oS integration

### Phase 2: Enhanced Detection (Planned)
- 🔄 CVE database integration (NVD API)
- 🔄 100+ additional checks
- 🔄 Advanced exploit verification
- 🔄 False positive reduction
- 🔄 Machine learning classification

### Phase 3: Enterprise Features (Planned)
- 🔄 Scheduled scanning (cron-style)
- 🔄 Email notifications
- 🔄 Team collaboration
- 🔄 PDF report generation
- 🔄 Compliance frameworks (PCI DSS, HIPAA, CIS)

### Phase 4: Advanced Capabilities (Future)
- 🔄 Distributed scanning
- 🔄 Agent-based scanning
- 🔄 Custom plugin development
- 🔄 REST API
- 🔄 Remediation workflow tracking

---

## Contributing

VulnHunter is part of the Ai|oS ecosystem. To add custom vulnerability checks:

1. Create a new check class:

```python
class MyCustomCheck(VulnerabilityCheck):
    def __init__(self):
        super().__init__(
            check_id="CUSTOM-001",
            name="My Custom Vulnerability",
            category="Custom",
            severity=Severity.HIGH,
            cvss_score=7.5
        )

    def check(self, target, port, credentials=None):
        # Implement your check logic
        if vulnerability_detected:
            return Vulnerability(
                target, port,
                self.check_id, self.name,
                self.severity, self.cvss_score,
                "Description of the vulnerability",
                "Proof of concept output",
                "Remediation steps",
                ["CVE-2024-XXXXX", "CWE-XXX"]
            )
        return None
```

2. Register in `_load_plugins()`:

```python
def _load_plugins(self):
    plugins = []
    # ... existing plugins ...
    plugins.append(MyCustomCheck())
    return plugins
```

---

## Troubleshooting

### Common Issues

**Issue: Scan hangs on certain hosts**
- Solution: Check network connectivity, increase timeout values

**Issue: No vulnerabilities found**
- Solution: Verify target is accessible, try different scan profile

**Issue: SSL/TLS checks failing**
- Solution: Ensure Python ssl module is available, check certificate validity

**Issue: GUI doesn't open**
- Solution: Check browser settings, verify temp file permissions

### Debug Mode

Enable verbose logging:

```bash
python -m tools.vulnhunter --scan 192.168.1.100 -v
```

---

## Security Considerations

VulnHunter is designed for **defensive security assessment only**:

- ✅ Non-destructive scanning
- ✅ Safe exploit verification
- ✅ Respects rate limits
- ✅ Logs all activities
- ✅ Requires authorization

**Important**: Always obtain proper authorization before scanning any systems. Unauthorized vulnerability scanning may be illegal.

---

## License

Copyright © 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved.

**PATENT PENDING**

VulnHunter is proprietary software distributed as part of the Ai|oS Sovereign Security Toolkit.

---

## Support

For questions, issues, or feature requests:

- **Documentation**: See `/Users/noone/aios/tools/VULNHUNTER_README.md`
- **Health Check**: `python -m tools.vulnhunter --health`
- **Ai|oS Integration**: See `CLAUDE.md` for integration patterns

---

## Acknowledgments

VulnHunter stands on the shoulders of giants:
- Inspired by OpenVAS and Nessus
- CVSS scoring based on FIRST.org specifications
- Vulnerability database aligned with MITRE CVE/CWE
- Built for the Ai|oS ecosystem

---

**🎯 VulnHunter - Hunt vulnerabilities. Secure systems. Defend infrastructure.**

*Part of the Ai|oS Sovereign Security Toolkit*
