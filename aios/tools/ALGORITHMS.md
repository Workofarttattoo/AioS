# Sovereign Security Toolkit - Core Algorithms

This document catalogs all core algorithms extracted from the Sovereign Security Toolkit. Each algorithm is documented with its purpose, complexity, source tool, and usage examples.

## Overview

The algorithms library (`aios/tools/algorithms.py`) centralizes reusable logic from all security tools, making them accessible for:
- Integration into other systems
- Unit testing and validation
- Educational and research purposes
- Custom tool development

**Total Algorithms:** 22 algorithms across 8 tools

---

## Table of Contents

1. [AuroraScan - Network Reconnaissance](#aurorascan---network-reconnaissance)
2. [CipherSpear - SQL Injection Analysis](#cipherspear---sql-injection-analysis)
3. [SkyBreaker - Wireless Auditing](#skybreaker---wireless-auditing)
4. [MythicKey - Credential Analysis](#mythickey---credential-analysis)
5. [SpectraTrace - Packet Analysis](#spectratrace---packet-analysis)
6. [NemesisHydra - Authentication Testing](#nemesishydra---authentication-testing)
7. [ObsidianHunt - Host Hardening](#obsidianhunt---host-hardening)
8. [QuantumLeapAssessor - Quantum Vulnerability Scanning](#quantumleapassessor---quantum-vulnerability-scanning)
9. [Usage Examples](#usage-examples)

---

## AuroraScan - Network Reconnaissance

### `tcp_connect_scan(host, port, timeout)`

**Purpose:** Performs a TCP connect scan on a single port to determine if it's open.

**Algorithm:** Attempts a full TCP handshake to detect open ports.

**Time Complexity:** O(1) per port

**Parameters:**
- `host`: Target hostname or IP address
- `port`: Port number to scan
- `timeout`: Connection timeout in seconds (default: 1.5)

**Returns:** `PortScanResult` with status ("open", "closed", "filtered"), response time, and optional service information

**Example:**
```python
from aios.tools.algorithms import tcp_connect_scan

result = tcp_connect_scan("example.com", 80, timeout=2.0)
print(f"Port {result.port}: {result.status} ({result.response_time_ms:.2f}ms)")
```

---

### `parse_port_range(port_spec)`

**Purpose:** Parses port specifications like "22,80,443,8000-8010" into a list of port numbers.

**Algorithm:** Split on comma, expand ranges, deduplicate and sort.

**Time Complexity:** O(n) where n is the number of ports specified

**Parameters:**
- `port_spec`: Port specification string (e.g., "22,80,443,8000-8010")

**Returns:** Sorted list of unique port numbers

**Example:**
```python
from aios.tools.algorithms import parse_port_range

ports = parse_port_range("22,80,443,8000-8003")
# Result: [22, 80, 443, 8000, 8001, 8002, 8003]
```

---

### `resolve_hostname(host)`

**Purpose:** Resolves a hostname to an IP address with error handling.

**Algorithm:** DNS lookup with graceful failure handling.

**Time Complexity:** O(1) - DNS lookup time

**Parameters:**
- `host`: Hostname to resolve

**Returns:** Tuple of (original_host, resolved_ip) or (original_host, "unresolved")

**Example:**
```python
from aios.tools.algorithms import resolve_hostname

original, resolved = resolve_hostname("example.com")
print(f"{original} -> {resolved}")
```

---

## CipherSpear - SQL Injection Analysis

### `extract_url_parameters(vector)`

**Purpose:** Extracts parameter key/value pairs from URLs, query strings, or form bodies.

**Algorithm:** Parse URL components and extract query parameters from multiple formats.

**Time Complexity:** O(n) where n is the number of parameters

**Parameters:**
- `vector`: URL, query string, or form data

**Returns:** List of (key, value) tuples

**Example:**
```python
from aios.tools.algorithms import extract_url_parameters

params = extract_url_parameters("http://example.com/search?q=test&page=1")
# Result: [('q', 'test'), ('page', '1')]

params = extract_url_parameters("user=admin&pass=secret")
# Result: [('user', 'admin'), ('pass', 'secret')]
```

---

### `assess_sql_injection_risk(vector, techniques)`

**Purpose:** Assesses SQL injection risk in a given input vector using pattern matching and heuristics.

**Algorithm:** Multi-pattern regex matching with severity-based scoring.

**Time Complexity:** O(p × n) where p is number of patterns, n is vector length

**Parameters:**
- `vector`: Input string to analyze (URL, query, or form data)
- `techniques`: Optional list of techniques to weight (e.g., ['blind', 'time'])

**Returns:** Dictionary with:
- `risk_score`: Numeric risk score (0-∞)
- `risk_label`: "low", "medium", or "high"
- `findings`: List of detected patterns
- `parameter_count`: Number of parameters found
- `recommendation`: Remediation guidance

**Detected Patterns:**
- Boolean tautologies (`' OR 1=1`)
- UNION SELECT statements
- Time-based delays (SLEEP, BENCHMARK)
- SQL comments (`--`, `#`, `/* */`)
- File operations (LOAD_FILE, INTO OUTFILE)
- Command execution (xp_cmdshell)
- Quote injection
- Embedded SQL keywords

**Example:**
```python
from aios.tools.algorithms import assess_sql_injection_risk

result = assess_sql_injection_risk("user=admin' OR 1=1--")
print(f"Risk: {result['risk_label']} (score: {result['risk_score']})")
print(f"Findings: {', '.join(result['findings'])}")
print(f"Recommendation: {result['recommendation']}")
```

---

### `sanitize_dsn(dsn)`

**Purpose:** Sanitizes a database DSN by redacting credentials for safe logging.

**Algorithm:** URL parsing with credential redaction.

**Time Complexity:** O(n) where n is DSN length

**Parameters:**
- `dsn`: Database connection string

**Returns:** Dictionary with sanitized scheme, netloc, and path

**Example:**
```python
from aios.tools.algorithms import sanitize_dsn

clean = sanitize_dsn("postgresql://user:secret@localhost/db")
# Result: {'scheme': 'postgresql', 'netloc': 'user@localhost', 'path': '/db'}
```

---

## SkyBreaker - Wireless Auditing

### `calculate_channel_congestion(networks)`

**Purpose:** Calculates wireless channel congestion from network observations.

**Algorithm:** Bucket networks by channel, find maximum congestion.

**Time Complexity:** O(n) where n is number of networks

**Parameters:**
- `networks`: Iterable of `WirelessNetwork` observations

**Returns:** Dictionary with:
- `max_channel`: Most congested channel
- `max_count`: Number of networks on that channel
- `distribution`: Full channel distribution

**Example:**
```python
from aios.tools.algorithms import WirelessNetwork, calculate_channel_congestion

networks = [
    WirelessNetwork("HomeNet", "aa:bb:cc:dd:ee:01", -45, 6, "WPA2", 0.0, 1.0),
    WirelessNetwork("OfficeNet", "aa:bb:cc:dd:ee:02", -55, 6, "WPA3", 0.0, 1.0),
    WirelessNetwork("GuestNet", "aa:bb:cc:dd:ee:03", -65, 11, "WPA2", 0.0, 1.0),
]

congestion = calculate_channel_congestion(networks)
print(f"Most congested: Channel {congestion['max_channel']} with {congestion['max_count']} networks")
```

---

### `assess_wireless_security(networks)`

**Purpose:** Assesses wireless security posture from network observations.

**Algorithm:** Categorize networks by security level, count vulnerabilities.

**Time Complexity:** O(n) where n is number of networks

**Parameters:**
- `networks`: Iterable of `WirelessNetwork` observations

**Returns:** Dictionary with security statistics and risk assessment

**Example:**
```python
from aios.tools.algorithms import assess_wireless_security

assessment = assess_wireless_security(networks)
print(f"Risk Level: {assessment['risk_level']}")
print(f"Open Networks: {assessment['open_networks']}")
print(f"WPA3 Networks: {assessment['wpa3_networks']}")
```

---

## MythicKey - Credential Analysis

### `detect_hash_algorithm(digest)`

**Purpose:** Detects hash algorithm from digest length.

**Algorithm:** Lookup table by hex digest length.

**Time Complexity:** O(1)

**Parameters:**
- `digest`: Hex digest string

**Returns:** Algorithm name ("md5", "sha1", "sha256", etc.) or "unknown"

**Supported Algorithms:**
- MD5 (32 chars)
- SHA-1 (40 chars)
- SHA-224 (56 chars)
- SHA-256 (64 chars)
- SHA-384 (96 chars)
- SHA-512 (128 chars)

**Example:**
```python
from aios.tools.algorithms import detect_hash_algorithm

algorithm = detect_hash_algorithm("5d41402abc4b2a76b9719d911017c592")
# Result: "md5"
```

---

### `generate_password_mutations(word, profile)`

**Purpose:** Generates common password mutations for dictionary attacks.

**Algorithm:** Rule-based transformations (capitalization, suffixes, leetspeak).

**Time Complexity:** O(1) - fixed number of transformations per word

**Parameters:**
- `word`: Base password word
- `profile`: Processing profile ("cpu" or "gpu-*" for more mutations)

**Returns:** List of password candidates

**Transformations:**
- **CPU Profile:** Capitalization, numeric suffixes (123), exclamation mark
- **GPU Profile:** All CPU transformations plus reverse, @ suffix, leetspeak

**Example:**
```python
from aios.tools.algorithms import generate_password_mutations

mutations = generate_password_mutations("password", "cpu")
# Result: ['password', 'Password', 'password123', 'password!']

mutations_gpu = generate_password_mutations("admin", "gpu-balanced")
# Result: ['admin', 'Admin', 'admin123', 'admin!', 'nimda', 'admin@', '@dmin']
```

---

### `crack_password_hash(digest, wordlist, profile)`

**Purpose:** Attempts to crack a password hash using a wordlist.

**Algorithm:** Dictionary attack with mutations.

**Time Complexity:** O(w × m) where w is wordlist size, m is mutations per word

**Parameters:**
- `digest`: Hash digest to crack
- `wordlist`: Iterable of password candidates
- `profile`: Processing profile for mutation depth

**Returns:** Tuple of (cracked: bool, plaintext: Optional[str], attempts: int)

**Example:**
```python
from aios.tools.algorithms import crack_password_hash

cracked, plaintext, attempts = crack_password_hash(
    "5f4dcc3b5aa765d61d8327deb882cf99",
    ["password", "admin", "welcome"],
    "cpu"
)

if cracked:
    print(f"Cracked! Plaintext: '{plaintext}' in {attempts} attempts")
```

---

### `estimate_quantum_threat(algorithm, key_size)`

**Purpose:** Estimates quantum computing threat to cryptographic keys based on Shor's algorithm.

**Algorithm:** Lookup table based on known quantum complexity estimates.

**Time Complexity:** O(1)

**Parameters:**
- `algorithm`: Cryptographic algorithm ("RSA", "EC", "DSA")
- `key_size`: Key size in bits

**Returns:** Dictionary with:
- `risk`: Risk level ("Low", "Medium", "High", "Critical")
- `qubits`: Estimated qubits required to break
- `notes`: Vulnerability details

**Risk Levels:**
- **RSA:**
  - 4096-bit: Medium (8192 qubits)
  - 2048-bit: High (4096 qubits)
  - <2048-bit: Critical (2048 qubits)
- **EC/ECDSA:**
  - 384-bit: Medium (3072 qubits)
  - 256-bit: High (2330 qubits)
  - <256-bit: Critical (1500 qubits)

**Example:**
```python
from aios.tools.algorithms import estimate_quantum_threat

threat = estimate_quantum_threat("RSA", 2048)
print(f"Risk: {threat['risk']}")
print(f"Qubits needed: {threat['qubits']}")
print(f"Notes: {threat['notes']}")
```

---

## SpectraTrace - Packet Analysis

### `parse_pcap_header(data)`

**Purpose:** Parses PCAP file header to determine endianness.

**Algorithm:** Magic number detection (0xa1b2c3d4 or 0xd4c3b2a1).

**Time Complexity:** O(1)

**Parameters:**
- `data`: First 24 bytes of PCAP file

**Returns:** Tuple of (endian, valid) where endian is ">" or "<"

---

### `identify_protocol(protocol_number)`

**Purpose:** Maps IP protocol number to protocol name.

**Algorithm:** Lookup table of common protocols.

**Time Complexity:** O(1)

**Parameters:**
- `protocol_number`: IP protocol number

**Returns:** Protocol name string

**Supported Protocols:**
- 1: ICMP
- 6: TCP
- 17: UDP
- 50: ESP
- 51: AH
- 58: ICMPv6

**Example:**
```python
from aios.tools.algorithms import identify_protocol

protocol = identify_protocol(6)
# Result: "TCP"
```

---

### `analyze_traffic_top_talkers(packets)`

**Purpose:** Identifies top network talkers by bytes transferred.

**Algorithm:** Hash table aggregation + sorting.

**Time Complexity:** O(n log k) where n is packets, k is top-K (5)

**Parameters:**
- `packets`: Iterable of `PacketRecord` objects

**Returns:** Sorted list of (address, bytes) tuples, top 5

**Example:**
```python
from aios.tools.algorithms import PacketRecord, analyze_traffic_top_talkers

packets = [
    PacketRecord(0.0, "10.0.0.1", "10.0.0.2", "TCP", 1500, "HTTP GET"),
    PacketRecord(0.1, "10.0.0.1", "10.0.0.2", "TCP", 1500, "HTTP Response"),
    # ... more packets
]

top_talkers = analyze_traffic_top_talkers(packets)
for address, bytes_transferred in top_talkers:
    print(f"{address}: {bytes_transferred} bytes")
```

---

### `generate_traffic_heatmap(packets, buckets)`

**Purpose:** Generates a time-series heatmap of traffic volume.

**Algorithm:** Time bucketing with aggregation.

**Time Complexity:** O(n) where n is number of packets

**Parameters:**
- `packets`: Sequence of `PacketRecord` objects
- `buckets`: Number of time buckets (default: 12)

**Returns:** List of bucket dictionaries with packet counts and bytes

**Example:**
```python
from aios.tools.algorithms import generate_traffic_heatmap

heatmap = generate_traffic_heatmap(packets, buckets=10)
for bucket in heatmap:
    print(f"Bucket {bucket['index']}: {bucket['packet_count']} packets, {bucket['bytes']} bytes")
```

---

### `classify_ip_traffic_direction(src, dst)`

**Purpose:** Classifies traffic direction based on private/public IP addresses.

**Algorithm:** IP address type detection using `ipaddress` module.

**Time Complexity:** O(1)

**Parameters:**
- `src`: Source IP address
- `dst`: Destination IP address

**Returns:** Direction string: "ingress", "egress", "internal", "external", or "unknown"

**Example:**
```python
from aios.tools.algorithms import classify_ip_traffic_direction

direction = classify_ip_traffic_direction("10.0.0.1", "8.8.8.8")
# Result: "egress" (private to public)

direction = classify_ip_traffic_direction("10.0.0.1", "192.168.1.1")
# Result: "internal" (private to private)
```

---

## NemesisHydra - Authentication Testing

### `estimate_spray_duration(wordlist_size, rate_limit)`

**Purpose:** Estimates duration for password spray attack.

**Algorithm:** Simple division with safeguards.

**Time Complexity:** O(1)

**Parameters:**
- `wordlist_size`: Number of passwords to try
- `rate_limit`: Attempts per minute allowed

**Returns:** Estimated duration in minutes (float or infinity)

**Example:**
```python
from aios.tools.algorithms import estimate_spray_duration

duration = estimate_spray_duration(120, 12)
# Result: 10.0 minutes
```

---

### `assess_throttle_risk(service, rate_limit)`

**Purpose:** Assesses risk of account lockout based on service and rate.

**Algorithm:** Rule-based risk classification.

**Time Complexity:** O(1)

**Parameters:**
- `service`: Service type (ssh, rdp, http, etc.)
- `rate_limit`: Attempts per minute

**Returns:** Risk level: "low", "medium", or "high"

**Risk Rules:**
- SSH/RDP with >20 attempts/min: HIGH
- Any service >40 attempts/min: HIGH
- Any service >15 attempts/min: MEDIUM
- Otherwise: LOW

**Example:**
```python
from aios.tools.algorithms import assess_throttle_risk

risk = assess_throttle_risk("ssh", 25)
# Result: "high"
```

---

## ObsidianHunt - Host Hardening

### `evaluate_system_control(control_path)`

**Purpose:** Evaluates presence of a security control by path.

**Algorithm:** Filesystem check with fallback to PATH environment variable.

**Time Complexity:** O(1)

**Parameters:**
- `control_path`: Path to control file or binary

**Returns:** Status: "pass", "warn", or "error"

**Example:**
```python
from pathlib import Path
from aios.tools.algorithms import evaluate_system_control

status = evaluate_system_control(Path("/usr/bin/sudo"))
# Result: "pass" (if sudo exists)
```

---

## QuantumLeapAssessor - Quantum Vulnerability Scanning

### `scan_file_for_quantum_vulnerabilities(file_path)`

**Purpose:** Scans a source file for quantum-vulnerable cryptographic algorithms.

**Algorithm:** Line-by-line regex pattern matching.

**Time Complexity:** O(n × p) where n is lines, p is patterns

**Parameters:**
- `file_path`: Path to source file

**Returns:** List of finding dictionaries with line numbers and algorithms

**Detected Algorithms:**
- RSA
- ECC/ECDSA/ECDH
- DSA/Digital Signature Algorithm
- Diffie-Hellman/DH

**Example:**
```python
from pathlib import Path
from aios.tools.algorithms import scan_file_for_quantum_vulnerabilities

findings = scan_file_for_quantum_vulnerabilities(Path("crypto_code.py"))
for finding in findings:
    print(f"{finding['file_path']}:{finding['line']} - {finding['algorithm']}")
    print(f"  {finding['snippet']}")
```

---

## Usage Examples

### Complete Network Scan Example

```python
from aios.tools.algorithms import (
    parse_port_range,
    resolve_hostname,
    tcp_connect_scan
)

# Parse target and ports
target = "example.com"
port_spec = "22,80,443,8080-8085"

# Resolve hostname
original, resolved_ip = resolve_hostname(target)
if resolved_ip == "unresolved":
    print(f"Failed to resolve {target}")
    exit(1)

print(f"Scanning {target} ({resolved_ip})")

# Parse ports
ports = parse_port_range(port_spec)
print(f"Scanning {len(ports)} ports...")

# Scan each port
open_ports = []
for port in ports:
    result = tcp_connect_scan(resolved_ip, port, timeout=2.0)
    if result.status == "open":
        open_ports.append(port)
        print(f"  Port {port}: OPEN ({result.response_time_ms:.2f}ms)")

print(f"\nFound {len(open_ports)} open ports")
```

### SQL Injection Analysis Example

```python
from aios.tools.algorithms import (
    extract_url_parameters,
    assess_sql_injection_risk
)

# Test vectors
test_urls = [
    "https://example.com/search?q=test",
    "https://example.com/login?user=admin' OR 1=1--",
    "https://example.com/api?id=5; DROP TABLE users--",
]

for url in test_urls:
    print(f"\nAnalyzing: {url}")
    
    # Extract parameters
    params = extract_url_parameters(url)
    print(f"  Parameters: {len(params)}")
    
    # Assess risk
    risk = assess_sql_injection_risk(url, techniques=['blind', 'time'])
    print(f"  Risk: {risk['risk_label'].upper()} (score: {risk['risk_score']})")
    
    if risk['findings']:
        print(f"  Findings: {', '.join(risk['findings'])}")
        print(f"  Recommendation: {risk['recommendation']}")
```

### Password Cracking Example

```python
from aios.tools.algorithms import (
    detect_hash_algorithm,
    generate_password_mutations,
    crack_password_hash
)

# Hash to crack (MD5 of "password")
target_hash = "5f4dcc3b5aa765d61d8327deb882cf99"

# Detect algorithm
algorithm = detect_hash_algorithm(target_hash)
print(f"Detected algorithm: {algorithm}")

# Prepare wordlist
base_words = ["password", "admin", "welcome", "letmein"]
print(f"Base wordlist: {len(base_words)} words")

# Show mutations for one word
mutations = generate_password_mutations("password", "cpu")
print(f"Mutations for 'password': {mutations}")

# Attempt to crack
cracked, plaintext, attempts = crack_password_hash(
    target_hash,
    base_words,
    profile="cpu"
)

if cracked:
    print(f"\n✓ Cracked! Plaintext: '{plaintext}' in {attempts} attempts")
else:
    print(f"\n✗ Failed to crack after {attempts} attempts")
```

### Traffic Analysis Example

```python
from aios.tools.algorithms import (
    PacketRecord,
    analyze_traffic_top_talkers,
    generate_traffic_heatmap,
    classify_ip_traffic_direction
)

# Sample packets
packets = [
    PacketRecord(0.0, "10.0.0.1", "8.8.8.8", "UDP", 512, "DNS Query"),
    PacketRecord(0.1, "8.8.8.8", "10.0.0.1", "UDP", 256, "DNS Response"),
    PacketRecord(0.5, "10.0.0.1", "192.168.1.1", "TCP", 1500, "HTTP GET"),
    PacketRecord(1.0, "192.168.1.1", "10.0.0.1", "TCP", 8000, "HTTP Response"),
    PacketRecord(2.0, "10.0.0.1", "203.0.113.5", "TCP", 2048, "HTTPS"),
]

# Analyze top talkers
top_talkers = analyze_traffic_top_talkers(packets)
print("Top Talkers:")
for address, bytes_transferred in top_talkers:
    print(f"  {address}: {bytes_transferred} bytes")

# Generate heatmap
heatmap = generate_traffic_heatmap(packets, buckets=5)
print("\nTraffic Heatmap:")
for bucket in heatmap:
    bars = "█" * (bucket['packet_count'])
    print(f"  Bucket {bucket['index']}: {bars} ({bucket['packet_count']} packets)")

# Classify traffic direction
print("\nTraffic Direction:")
for packet in packets:
    direction = classify_ip_traffic_direction(packet.src, packet.dst)
    print(f"  {packet.src} → {packet.dst}: {direction}")
```

---

## Integration with Ai:oS

All algorithms can be integrated into Ai:oS meta-agents through the execution context:

```python
from aios.runtime import ExecutionContext, ActionResult
from aios.tools.algorithms import assess_sql_injection_risk

def security_scan_action(ctx: ExecutionContext) -> ActionResult:
    """Custom action using centralized algorithms."""
    
    # Get input vectors from environment
    vectors = ctx.environment.get("SCAN_VECTORS", [])
    
    # Analyze each vector
    findings = []
    for vector in vectors:
        risk = assess_sql_injection_risk(vector)
        if risk['risk_label'] in ['medium', 'high']:
            findings.append({
                'vector': vector,
                'risk': risk['risk_label'],
                'score': risk['risk_score']
            })
    
    # Publish results
    ctx.publish_metadata("security.sql_injection_scan", {
        "vectors_scanned": len(vectors),
        "findings": findings
    })
    
    return ActionResult(
        success=True,
        message=f"Scanned {len(vectors)} vectors, found {len(findings)} risks",
        payload={"findings": findings}
    )
```

---

## Performance Characteristics

| Algorithm | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| tcp_connect_scan | O(1) | O(1) | Network I/O bound |
| parse_port_range | O(n) | O(n) | n = number of ports |
| assess_sql_injection_risk | O(p×n) | O(p) | p = patterns, n = input length |
| calculate_channel_congestion | O(n) | O(c) | c = number of channels |
| crack_password_hash | O(w×m) | O(1) | w = wordlist, m = mutations |
| analyze_traffic_top_talkers | O(n log k) | O(n) | k = top-K (5) |
| generate_traffic_heatmap | O(n) | O(b) | b = bucket count |

---

## Contributing New Algorithms

When adding new algorithms to the library:

1. **Document thoroughly** - Include purpose, complexity, parameters, and examples
2. **Use type hints** - All parameters and return types should be typed
3. **Handle errors gracefully** - Catch exceptions and return sensible defaults
4. **Write tests** - Add unit tests to verify correctness
5. **Update catalog** - Add to `get_algorithm_catalog()` function
6. **Update this document** - Document the new algorithm

---

## License

Part of the Sovereign Security Toolkit within the Ai:oS project.

