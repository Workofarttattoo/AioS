# Algorithm Index by Tool

Quick reference for which algorithms come from which security tools.

---

## 🔍 AuroraScan - Network Reconnaissance (3 algorithms)

### Core Functions
```python
from aios.tools.algorithms import (
    tcp_connect_scan,         # Scan a single TCP port
    parse_port_range,         # Parse port specifications (e.g., "22,80,443,8000-8010")
    resolve_hostname,         # Resolve hostname to IP address
)
```

### Data Structures
- `PortScanResult` - Port scan result with status, timing, service info

### Use Cases
- Network mapping and discovery
- Service enumeration
- Reachability testing

---

## 🗡️ CipherSpear - SQL Injection Analysis (3 algorithms)

### Core Functions
```python
from aios.tools.algorithms import (
    extract_url_parameters,      # Extract params from URLs/query strings
    assess_sql_injection_risk,   # Detect SQL injection patterns
    sanitize_dsn,                # Redact credentials from connection strings
)
```

### Data Structures
- `SQL_INJECTION_PATTERNS` - Pattern database with severity scores

### Use Cases
- Web application security testing
- Input validation auditing
- API security assessment

---

## 📡 SkyBreaker - Wireless Auditing (2 algorithms)

### Core Functions
```python
from aios.tools.algorithms import (
    calculate_channel_congestion,  # Find most congested WiFi channels
    assess_wireless_security,       # Evaluate network security posture
)
```

### Data Structures
- `WirelessNetwork` - Network observation with SSID, BSSID, signal, channel, security

### Use Cases
- Wireless site surveys
- Channel planning
- Security posture assessment

---

## 🔑 MythicKey - Credential Analysis (4 algorithms)

### Core Functions
```python
from aios.tools.algorithms import (
    detect_hash_algorithm,        # Identify hash type from digest
    generate_password_mutations,  # Generate password variations
    crack_password_hash,          # Dictionary attack on hashes
    estimate_quantum_threat,      # Assess quantum computing risk
)
```

### Data Structures
- `HASH_ALGORITHM_BY_LENGTH` - Hash algorithm detection lookup table

### Use Cases
- Password policy auditing
- Credential strength assessment
- Post-quantum cryptography planning

---

## 📊 SpectraTrace - Packet Analysis (5 algorithms)

### Core Functions
```python
from aios.tools.algorithms import (
    parse_pcap_header,                # Parse PCAP file headers
    identify_protocol,                # Map protocol numbers to names
    analyze_traffic_top_talkers,      # Find highest-volume endpoints
    generate_traffic_heatmap,         # Time-series traffic visualization
    classify_ip_traffic_direction,    # Ingress/egress/internal classification
)
```

### Data Structures
- `PacketRecord` - Packet with timestamp, src, dst, protocol, length, info

### Use Cases
- Network forensics
- Traffic analysis
- Performance troubleshooting

---

## 🔓 NemesisHydra - Authentication Testing (2 algorithms)

### Core Functions
```python
from aios.tools.algorithms import (
    estimate_spray_duration,   # Calculate password spray time
    assess_throttle_risk,      # Evaluate lockout risk
)
```

### Data Structures
- `SERVICE_PORTS` - Common service port mappings

### Use Cases
- Authentication security testing
- Account lockout risk assessment
- Credential spray planning

---

## 🛡️ ObsidianHunt - Host Hardening (1 algorithm)

### Core Functions
```python
from aios.tools.algorithms import (
    evaluate_system_control,  # Check presence of security controls
)
```

### Data Structures
- `LINUX_CHECKS`, `MAC_CHECKS`, `WINDOWS_CHECKS` - Platform-specific control lists

### Use Cases
- Baseline security auditing
- Compliance verification
- System hardening validation

---

## ⚛️ QuantumLeapAssessor - Quantum Vulnerability Scanning (1 algorithm)

### Core Functions
```python
from aios.tools.algorithms import (
    scan_file_for_quantum_vulnerabilities,  # Find quantum-vulnerable crypto in source code
)
```

### Data Structures
- `QUANTUM_VULNERABLE_PATTERNS` - Regex patterns for RSA, ECC, DSA, DH

### Use Cases
- Post-quantum migration planning
- Cryptographic inventory
- Security code review

---

## 🎯 Quick Reference by Use Case

### Network Security
- `tcp_connect_scan()` - Port scanning
- `resolve_hostname()` - DNS resolution
- `parse_port_range()` - Port specification parsing
- `calculate_channel_congestion()` - WiFi channel analysis
- `assess_wireless_security()` - Wireless security posture

### Web Application Security
- `extract_url_parameters()` - Parameter extraction
- `assess_sql_injection_risk()` - SQL injection detection
- `sanitize_dsn()` - Credential redaction

### Credential Security
- `detect_hash_algorithm()` - Hash identification
- `generate_password_mutations()` - Password variations
- `crack_password_hash()` - Dictionary attack
- `estimate_spray_duration()` - Spray time estimation
- `assess_throttle_risk()` - Lockout risk

### Network Forensics
- `analyze_traffic_top_talkers()` - Bandwidth analysis
- `generate_traffic_heatmap()` - Traffic visualization
- `classify_ip_traffic_direction()` - Traffic classification
- `identify_protocol()` - Protocol identification
- `parse_pcap_header()` - PCAP parsing

### Cryptographic Assessment
- `estimate_quantum_threat()` - Quantum risk analysis
- `scan_file_for_quantum_vulnerabilities()` - Source code scanning

### System Hardening
- `evaluate_system_control()` - Control verification

---

## 📦 Import Patterns

### Import Everything
```python
from aios.tools import algorithms
result = algorithms.tcp_connect_scan("example.com", 80)
```

### Import Specific Functions
```python
from aios.tools.algorithms import tcp_connect_scan, assess_sql_injection_risk
```

### Import Data Structures
```python
from aios.tools.algorithms import PortScanResult, WirelessNetwork, PacketRecord
```

### Get Catalog
```python
from aios.tools.algorithms import get_algorithm_catalog
catalog = get_algorithm_catalog()
```

---

## 📚 Algorithm Complexity Summary

| Algorithm | Time | Space | Category |
|-----------|------|-------|----------|
| tcp_connect_scan | O(1) | O(1) | Network |
| parse_port_range | O(n) | O(n) | Parsing |
| resolve_hostname | O(1) | O(1) | Network |
| extract_url_parameters | O(n) | O(n) | Parsing |
| assess_sql_injection_risk | O(p×n) | O(p) | Analysis |
| calculate_channel_congestion | O(n) | O(c) | Analysis |
| assess_wireless_security | O(n) | O(n) | Analysis |
| detect_hash_algorithm | O(1) | O(1) | Lookup |
| generate_password_mutations | O(1) | O(1) | Generation |
| crack_password_hash | O(w×m) | O(1) | Cracking |
| estimate_quantum_threat | O(1) | O(1) | Lookup |
| identify_protocol | O(1) | O(1) | Lookup |
| analyze_traffic_top_talkers | O(n log k) | O(n) | Analysis |
| generate_traffic_heatmap | O(n) | O(b) | Visualization |
| classify_ip_traffic_direction | O(1) | O(1) | Classification |
| estimate_spray_duration | O(1) | O(1) | Calculation |
| assess_throttle_risk | O(1) | O(1) | Assessment |
| evaluate_system_control | O(1) | O(1) | Filesystem |
| scan_file_for_quantum_vulnerabilities | O(n×p) | O(f) | Scanning |

**Legend:**
- n = input size (lines, packets, networks, etc.)
- p = number of patterns
- w = wordlist size
- m = mutations per word
- c = number of channels
- k = top-K count
- b = number of buckets
- f = number of findings

---

## 🔧 Common Workflows

### Network Reconnaissance Workflow
```python
from aios.tools.algorithms import resolve_hostname, parse_port_range, tcp_connect_scan

# 1. Resolve target
original, ip = resolve_hostname("example.com")

# 2. Parse ports
ports = parse_port_range("22,80,443,8000-8010")

# 3. Scan ports
for port in ports:
    result = tcp_connect_scan(ip, port)
    if result.status == "open":
        print(f"Open: {port}")
```

### Web Security Assessment Workflow
```python
from aios.tools.algorithms import extract_url_parameters, assess_sql_injection_risk

# 1. Extract parameters from request
params = extract_url_parameters("http://example.com/search?q=test&id=5")

# 2. Assess injection risk
risk = assess_sql_injection_risk("http://example.com/search?q=test' OR 1=1--")

# 3. Take action based on risk
if risk['risk_label'] == 'high':
    print(f"High risk detected: {risk['findings']}")
```

### Password Audit Workflow
```python
from aios.tools.algorithms import detect_hash_algorithm, crack_password_hash

# 1. Identify hash type
algorithm = detect_hash_algorithm(hash_digest)

# 2. Attempt to crack
cracked, plaintext, attempts = crack_password_hash(
    hash_digest, 
    wordlist, 
    profile="cpu"
)

# 3. Report findings
if cracked:
    print(f"Weak password: {plaintext}")
```

### Traffic Analysis Workflow
```python
from aios.tools.algorithms import (
    analyze_traffic_top_talkers,
    generate_traffic_heatmap,
    classify_ip_traffic_direction
)

# 1. Identify top talkers
top_talkers = analyze_traffic_top_talkers(packets)

# 2. Generate heatmap
heatmap = generate_traffic_heatmap(packets, buckets=12)

# 3. Classify directions
for packet in packets:
    direction = classify_ip_traffic_direction(packet.src, packet.dst)
```

---

## 📖 Documentation Files

- **`algorithms.py`** - Core implementation (870 lines, 21 algorithms)
- **`ALGORITHMS.md`** - Comprehensive documentation with examples (1000+ lines)
- **`ALGORITHM_INDEX.md`** - This quick reference guide
- **`README_ALGORITHMS.md`** - Overview and getting started
- **`algorithms_demo.py`** - Live demonstrations of all algorithms

---

## ✅ Verification

Run the demo to verify all algorithms work:
```bash
cd aios/tools
python3 algorithms_demo.py
```

Expected output: ✓ All demonstrations completed successfully!

---

**Last Updated:** 2025-11-05  
**Total Algorithms:** 21 across 8 tools  
**Status:** ✅ Production Ready

