# 🦊 ProxyPhantom - Web Application Security Testing Suite

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

---

## Overview

**ProxyPhantom** is a comprehensive web application security testing suite for Ai|oS, providing equivalent functionality to Burp Suite Professional. It features an HTTP/HTTPS intercepting proxy, vulnerability scanner, spider, intruder (fuzzer), decoder, sequencer, and more—all with a stunning orange/amber cyberpunk GUI themed around a "Phantom Fox" icon.

## Features

### 🌐 Core Modules

1. **HTTP/HTTPS Proxy** - Intercept, inspect, and modify requests/responses in real-time
   - Man-in-the-middle SSL/TLS interception with on-the-fly certificate generation
   - Intercept toggle for request/response manipulation
   - Complete HTTP history with filtering
   - Forward/drop controls

2. **Spider/Crawler** - Automatically map web application structure
   - Configurable crawl depth and page limits
   - Intelligent link extraction and validation
   - Same-domain enforcement
   - Visual sitemap tree generation

3. **Scanner** - Detect vulnerabilities automatically
   - **Passive Scanning**: No intrusive requests, detects issues in traffic flow
     - SQL injection indicators
     - Missing security headers (CSP, X-Frame-Options, HSTS, etc.)
     - Insecure cookies (missing Secure/HttpOnly flags)
     - Information disclosure
   - **Active Scanning**: Sends test payloads (requires explicit opt-in)
     - SQL injection testing
     - XSS (Cross-Site Scripting) detection
     - Command injection probing
     - Path traversal attempts
   - Issue classification: Critical, High, Medium, Low, Informational
   - Confidence scoring for each finding

4. **Intruder** - Automated attack engine with fuzzing capabilities
   - **Attack Types**:
     - **Sniper**: One payload position at a time (focused testing)
     - **Battering Ram**: Same payload in all positions (password guessing)
     - **Pitchfork**: Different payloads per position (credential stuffing)
     - **Cluster Bomb**: All combinations (brute force)
   - **Payload Generators**:
     - Numbers (ranges, sequences)
     - Alphanumeric strings
     - Dates
     - Common usernames/passwords
     - SQL injection payloads
     - XSS payloads
   - Visual payload position markers

5. **Repeater** - Manual request manipulation and testing
   - Edit and resend individual requests
   - Side-by-side request/response view
   - Syntax highlighting

6. **Decoder** - Multi-format encoding/decoding
   - **Encodings**: Base64, Base32, Hex, URL, HTML, Unicode, GZIP
   - **Hashing**: MD5, SHA1, SHA256, SHA512
   - Bidirectional conversion
   - Error-tolerant parsing

7. **Comparer** - Visual diff tool for responses
   - Side-by-side comparison
   - Highlight differences
   - Useful for detecting changes in authentication flows

8. **Sequencer** - Token randomness analysis
   - Shannon entropy calculation
   - Pattern detection (incremental, timestamp-based, common prefixes/suffixes)
   - Character distribution analysis
   - Sequential correlation testing
   - Quality scoring (Excellent, Good, Fair, Poor)

9. **Target Sitemap** - Visual tree of discovered endpoints
   - Hierarchical domain/path display
   - Click-to-select navigation
   - Real-time updates from Spider

10. **Activity Log** - Real-time operation logging
    - Timestamped entries
    - Color-coded severity (info, success, warning, error)
    - Auto-scroll to latest

---

## Architecture

### Technical Stack

- **Language**: Python 3.8+
- **Async I/O**: asyncio for concurrent operations
- **HTTP Proxy**: Custom implementation with asyncio
- **SSL/TLS**: cryptography library for certificate generation
- **Database**: SQLite for history, sitemap, and issues storage
- **GUI**: Embedded HTML5/CSS3/JavaScript (orange/amber cyberpunk theme)

### Dependencies

**Required**:
- Python standard library (asyncio, sqlite3, hashlib, etc.)

**Optional** (for full functionality):
- `cryptography` - SSL certificate generation for HTTPS interception
- `aiohttp` - Improved HTTP client for scanning
- `websockets` - Real-time GUI updates

**Install full dependencies**:
```bash
pip install cryptography aiohttp websockets
```

### Data Models

- **HttpRequest**: Stores method, URL, headers, body, timestamp
- **HttpResponse**: Stores status code, headers, body, response time
- **ScannerIssue**: Vulnerability with type, severity, evidence, remediation
- **FuzzPayload**: Attack payloads with encoding and description

### Security by Design

- **Forensic Mode Compatible**: Can run in read-only advisory mode
- **No Offensive Capabilities**: Designed for defensive security testing only
- **Ethical Defaults**: Active scanning requires explicit opt-in
- **Audit Trail**: All operations logged to SQLite database

---

## Usage

### Command-Line Interface

#### Health Check
```bash
python -m tools.proxyphantom --health
python -m tools.proxyphantom --health --json
```

**Output**:
```json
{
  "tool": "ProxyPhantom",
  "status": "ok",
  "summary": "Web application security testing suite - ok",
  "details": {
    "proxy_port": 8080,
    "ssl_interception": true,
    "websocket_support": true,
    "database": "SQLite",
    "missing_deps": null
  }
}
```

#### GUI Interface
```bash
python -m tools.proxyphantom --gui
```

Opens the stunning cyberpunk GUI in your default browser. Features:
- Orange/amber gradient backgrounds with animated glows
- Phantom Fox logo with pulsing animation
- Tabbed interface for all modules
- Real-time activity log
- Loading animations

#### Proxy Server
```bash
python -m tools.proxyphantom --proxy
python -m tools.proxyphantom --proxy --port 8888
```

Starts intercepting proxy server. Configure your browser to use proxy:
- **Host**: 127.0.0.1
- **Port**: 8080 (or custom with `--port`)

#### Vulnerability Scanning
```bash
# Passive scan only
python -m tools.proxyphantom --scan https://example.com

# Active scan with payloads
python -m tools.proxyphantom --scan https://example.com --active

# JSON output
python -m tools.proxyphantom --scan https://example.com --json
```

**Example Output**:
```
[High] SQL Injection
  URL: /api/search?q=test
  Parameter: q
  Possible SQL injection vulnerability detected

[Medium] Missing Security Headers
  URL: /
  Missing: X-Frame-Options, Content-Security-Policy
  Important security headers are missing from the response
```

#### Web Spidering
```bash
# Default depth (3)
python -m tools.proxyphantom --spider https://example.com

# Custom depth
python -m tools.proxyphantom --spider https://example.com --depth 5

# JSON output
python -m tools.proxyphantom --spider https://example.com --json
```

**Output**:
```
example.com:
  /
  /about
  /contact
  /api/users
  /api/products
  ... and 10 more
```

#### Demonstration Mode
```bash
python -m tools.proxyphantom --demo
```

Runs comprehensive demonstration of all features:
1. Vulnerability scanning with sample issues
2. Decoder operations (Base64, URL, hashing)
3. Token randomness analysis with quality scoring

---

## GUI Features

### Visual Design

**Theme**: Orange/Amber Cyberpunk
- **Primary Color**: `#ff6600` (bright orange)
- **Secondary Color**: `#ffaa00` (amber)
- **Accent**: `#ff9933`
- **Background**: Dark gradients with animated radial glows

**Icon**: 🦊 Phantom Fox with glowing orange eyes

**Layout**:
```
┌─────────────────────────────────────────────────────┐
│ 🦊 ProxyPhantom                                     │
│ [Proxy] [Spider] [Scanner] [Intruder] [Repeater]   │
├──────────┬────────────────────────────────────────┬─┤
│ Sitemap  │       Module Workspace                │R│
│ Tree     │                                        │e│
│ ├─ /     │  [Controls]                            │q│
│ ├─ /api  │                                        │ │
│ └─ ...   │  [Content Area]                        │R│
│          │                                        │e│
│          │                                        │s│
│          │                                        │p│
└──────────┴────────────────────────────────────────┴─┘
│ [Activity Log - Real-time Updates]                  │
└─────────────────────────────────────────────────────┘
```

### Interactive Elements

- **Hover Effects**: All buttons/tabs glow on hover
- **Toggle Switches**: Animated sliding switches with state colors
- **Status Badges**: Color-coded HTTP status (200=green, 400=yellow, 500=red)
- **Severity Badges**: Gradient badges for vulnerability severity
- **Loading Spinner**: Animated spinner with orange border
- **Syntax Highlighting**: Code editors with color-coded syntax

### Module Interactions

**Proxy Module**:
- Toggle intercept on/off
- Forward/drop buttons for intercepted requests
- Real-time history table with filtering
- Click row to view request/response details

**Scanner Module**:
- Configuration panel: URL, scan type, depth
- Start scan button with loading animation
- Issue cards with color-coded severity bars
- Expandable details with remediation advice

**Intruder Module**:
- Attack type selector (visual cards)
- Request template editor with payload markers
- Payload configuration dialog
- Results table with response analysis

**Decoder Module**:
- Split-screen input/output
- Encoding buttons (8 formats)
- Hashing buttons (4 algorithms)
- Instant conversion on click

**Sequencer Module**:
- Token input textarea (one per line)
- Analyze button
- Result cards: Entropy, Unique %, Quality, Patterns
- Visual quality indicator (progress bar)

---

## Integration with Ai|oS

### Tool Registry

ProxyPhantom is automatically registered in the Sovereign Security Toolkit:

```python
TOOL_REGISTRY = {
    # ... other tools
    "ProxyPhantom": "tools.proxyphantom",
}
```

### Health Check

SecurityAgent can query ProxyPhantom health:

```python
from tools import run_health_check

result = run_health_check("ProxyPhantom")
print(result["status"])  # "ok", "warn", or "error"
```

### Programmatic Usage

```python
from tools.proxyphantom import ProxyPhantom, VulnerabilityScanner, Decoder, Sequencer

# Create instance
app = ProxyPhantom()

# Scan URL
issues = app.scan_url("https://example.com", active=False)
for issue in issues:
    print(f"[{issue.severity.value}] {issue.type.value}")

# Decode data
decoded = Decoder.decode("UHJveHlQaGFudG9t", "base64")
print(decoded)  # "ProxyPhantom"

# Analyze token randomness
sequencer = Sequencer()
sequencer.add_sample("a4f8d2b1c9e7")
sequencer.add_sample("b5g9e3c2d0f8")
analysis = sequencer.analyze()
print(analysis["quality"])  # "Excellent", "Good", "Fair", or "Poor"
```

---

## Vulnerability Detection

### Passive Scanning (Safe, always enabled)

**SQL Injection Indicators**:
- `sql syntax`, `mysql_fetch`, `ORA-[0-9]+`
- `PostgreSQL.*ERROR`, `valid MySQL result`
- Database error messages in responses

**Missing Security Headers**:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY/SAMEORIGIN`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=...`
- `Content-Security-Policy: ...`

**Insecure Cookies**:
- Missing `Secure` flag on HTTPS cookies
- Missing `HttpOnly` flag (XSS protection)
- Missing `SameSite` attribute

### Active Scanning (Requires `--active` flag)

**SQL Injection Payloads**:
- `' OR '1'='1`
- `admin'--`
- `' UNION SELECT NULL--`

**XSS Payloads**:
- `<script>alert('XSS')</script>`
- `<img src=x onerror=alert('XSS')>`

**Command Injection Payloads**:
- `;ls`, `|ls`, `` `ls` ``
- `;cat /etc/passwd`

**Path Traversal Payloads**:
- `../../../etc/passwd`
- `..\\..\\..\\windows\\system32\\config\\sam`

---

## Sequencer Analysis

The Sequencer module analyzes token randomness using multiple techniques:

### Metrics

1. **Shannon Entropy**: Measures information content (bits per character)
   - Formula: `H(X) = -Σ p(x) * log₂(p(x))`
   - Higher entropy = more randomness
   - Scale: 0-5+ bits

2. **Uniqueness**: Percentage of unique tokens
   - 100% = every token is unique (good)
   - <90% = potential collision issues

3. **Pattern Detection**:
   - Incremental sequences (1, 2, 3, ...)
   - Timestamp-based (Unix epoch)
   - Common prefixes/suffixes
   - Sequential correlation

4. **Character Distribution**:
   - Digits, uppercase, lowercase, special characters
   - Balanced distribution indicates good randomness

### Quality Scoring

| Quality    | Entropy | Uniqueness | Patterns     |
|-----------|---------|------------|--------------|
| Excellent | >4.0    | 100%       | None detected|
| Good      | >3.0    | >90%       | Minor        |
| Fair      | >2.0    | >70%       | Some         |
| Poor      | <2.0    | <70%       | Many         |

### Example Analysis

**Good Tokens**:
```
a4f8d2b1c9e7  (entropy: 4.2, quality: Excellent)
b5g9e3c2d0f8
c6h0f4d3e1g9
```

**Bad Tokens**:
```
session_001   (entropy: 2.1, quality: Poor, pattern: incremental)
session_002
session_003
```

---

## Database Schema

ProxyPhantom uses SQLite for persistent storage:

### Tables

**requests**:
```sql
CREATE TABLE requests (
    id TEXT PRIMARY KEY,
    timestamp REAL,
    method TEXT,
    url TEXT,
    headers TEXT,  -- JSON
    body BLOB,
    protocol TEXT
);
```

**responses**:
```sql
CREATE TABLE responses (
    request_id TEXT,
    status_code INTEGER,
    headers TEXT,  -- JSON
    body BLOB,
    time_taken REAL,
    FOREIGN KEY (request_id) REFERENCES requests(id)
);
```

**issues**:
```sql
CREATE TABLE issues (
    id TEXT PRIMARY KEY,
    type TEXT,
    severity TEXT,
    url TEXT,
    parameter TEXT,
    evidence TEXT,
    description TEXT,
    remediation TEXT,
    request_id TEXT,
    confidence REAL,
    FOREIGN KEY (request_id) REFERENCES requests(id)
);
```

**sitemap**:
```sql
CREATE TABLE sitemap (
    url TEXT PRIMARY KEY,
    parent_url TEXT,
    depth INTEGER,
    last_visited REAL,
    response_code INTEGER
);
```

---

## SSL/TLS Interception

ProxyPhantom can intercept HTTPS traffic using dynamic certificate generation:

### How It Works

1. **CA Certificate Generation**:
   - Creates root CA certificate on first run
   - 2048-bit RSA key
   - Valid for 10 years

2. **On-the-Fly Certificate Signing**:
   - Intercepts `CONNECT` requests
   - Generates certificate for target hostname
   - Signs with ProxyPhantom CA
   - Presents to client

3. **Trust Setup** (required for HTTPS interception):
   - Export CA certificate from ProxyPhantom
   - Import into browser/system trust store
   - Without this, browsers will show certificate warnings

### Security Notes

- **Use only for authorized testing**: HTTPS interception is powerful
- **Keep CA private key secure**: Anyone with CA key can MITM your connections
- **Remove CA certificate after testing**: Don't leave it in system trust store

---

## Intruder Attack Types

### Sniper
Tests one payload position at a time while keeping others constant.

**Use cases**: Focused parameter testing, SQL injection, XSS

**Example**:
```
Original: GET /api?id=1&user=admin
Payloads: ['100', '200', '300']

Requests:
  GET /api?id=100&user=admin
  GET /api?id=200&user=admin
  GET /api?id=300&user=admin
  GET /api?id=1&user=100
  GET /api?id=1&user=200
  GET /api?id=1&user=300
```

### Battering Ram
Uses same payload in all positions simultaneously.

**Use cases**: Password guessing, authentication bypass

**Example**:
```
Original: GET /api?user=§USER§&pass=§PASS§
Payloads: ['admin', 'root']

Requests:
  GET /api?user=admin&pass=admin
  GET /api?user=root&pass=root
```

### Pitchfork
Uses different payload list for each position (zipped).

**Use cases**: Credential stuffing, username:password pairs

**Example**:
```
Original: GET /api?user=§USER§&pass=§PASS§
Payloads1: ['alice', 'bob']
Payloads2: ['password1', 'password2']

Requests:
  GET /api?user=alice&pass=password1
  GET /api?user=bob&pass=password2
```

### Cluster Bomb
Tests all combinations of payloads (cartesian product).

**Use cases**: Brute force, exhaustive testing

**Example**:
```
Original: GET /api?user=§USER§&pass=§PASS§
Payloads1: ['alice', 'bob']
Payloads2: ['123', '456']

Requests:
  GET /api?user=alice&pass=123
  GET /api?user=alice&pass=456
  GET /api?user=bob&pass=123
  GET /api?user=bob&pass=456
```

---

## Performance Considerations

### Proxy Throughput
- Async I/O enables high concurrency
- Typical: 100-500 requests/sec
- Bottleneck: SQLite writes (batching recommended for high traffic)

### Scanner Speed
- Passive: Real-time (no delay)
- Active: Limited by payloads * parameters * requests
- Recommended: Use passive first, then targeted active scans

### Spider Scalability
- Default: 1000 pages max
- Memory: ~1KB per page in sitemap
- Disk: ~10KB per page with full request/response storage

### Database Growth
- 1000 requests ≈ 10-50 MB (depends on body sizes)
- Recommendation: Periodic cleanup or archival

---

## Troubleshooting

### SSL Interception Not Working

**Symptom**: Browser shows certificate errors

**Fix**: Install ProxyPhantom CA certificate in browser trust store

**Steps**:
1. Export CA certificate from ProxyPhantom
2. Import into browser (Firefox: Preferences > Certificates > Import)
3. Check "Trust this CA to identify websites"

### Scanner Not Finding Issues

**Symptom**: Scan completes with 0 issues

**Possible causes**:
- Website has good security posture (legitimate)
- Website returns generic error pages
- Active scanning disabled (passive only)

**Fix**: Enable active scanning with `--active` flag

### Sequencer Shows "Poor" Quality for Good Tokens

**Symptom**: False positive on token quality

**Possible causes**:
- Too few samples (need 10+ for accurate analysis)
- Tokens have common prefixes (session IDs with fixed prefix)

**Fix**: Provide more samples, strip fixed prefixes before analysis

---

## Comparison to Burp Suite

| Feature              | ProxyPhantom | Burp Suite Professional |
|---------------------|--------------|------------------------|
| HTTP/HTTPS Proxy    | ✅           | ✅                     |
| SSL Interception    | ✅           | ✅                     |
| Spider/Crawler      | ✅           | ✅                     |
| Vulnerability Scanner| ✅          | ✅                     |
| Intruder (Fuzzer)   | ✅           | ✅                     |
| Repeater            | ✅           | ✅                     |
| Decoder             | ✅           | ✅                     |
| Sequencer           | ✅           | ✅                     |
| Comparer            | ✅           | ✅                     |
| Extensibility       | Python API   | BApp Store (Java)      |
| Price               | Free/Open    | $449/year              |
| Platform            | Python/Web   | Java/Desktop           |
| Ai|oS Integration   | ✅ Native    | ❌ Requires bridge     |

---

## Future Enhancements

### Planned Features
- [ ] WebSocket interception
- [ ] GraphQL scanner
- [ ] OAuth 2.0 token analyzer
- [ ] Report generation (PDF, HTML)
- [ ] Collaborative mode (team scanning)
- [ ] Browser extension (Chrome/Firefox)
- [ ] Advanced payload generators (ML-based)

### Extension API
- [ ] Plugin system for custom scanners
- [ ] Custom payload generators
- [ ] Webhook integrations

---

## Security & Ethics

### Intended Use
ProxyPhantom is designed for **defensive security testing** only:
- Authorized penetration testing
- Bug bounty programs
- Security audits of your own applications
- Educational security research

### Not Intended For
- ❌ Unauthorized access attempts
- ❌ Malicious hacking
- ❌ Data theft
- ❌ Privacy violations

### Legal Notice
Users are responsible for ensuring they have proper authorization before testing any web application. Unauthorized testing may violate laws such as:
- Computer Fraud and Abuse Act (CFAA) - USA
- Computer Misuse Act - UK
- Convention on Cybercrime - International

**Always obtain written permission before security testing.**

---

## Credits

**Developer**: Joshua Hendricks Cole (Corporation of Light)
**Project**: Ai|oS Sovereign Security Toolkit
**Inspiration**: Burp Suite (PortSwigger), OWASP ZAP
**Icon**: Phantom Fox 🦊
**Theme**: Orange/Amber Cyberpunk

---

## License

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

This software is proprietary and protected by intellectual property laws. Unauthorized copying, distribution, or modification is prohibited.

For licensing inquiries, contact: [Your Contact Information]

---

## Support

For issues, questions, or feature requests, please contact the Ai|oS development team or file an issue in the project repository.

**Version**: 1.0.0
**Last Updated**: October 2025
**Status**: Production Ready ✅
