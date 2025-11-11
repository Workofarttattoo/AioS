"""
Sovereign Security Toolkit - Core Algorithms Library

This module centralizes the core algorithms from all tools in the Sovereign Security
Toolkit, making them accessible for reuse, testing, and integration into other systems.

Each algorithm is documented with its source tool, purpose, and usage examples.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import socket
import struct
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urlparse


# =============================================================================
# AURORASCAN ALGORITHMS - Network Reconnaissance
# =============================================================================

@dataclass
class PortScanResult:
    """Result from a port scan operation."""
    port: int
    status: str  # "open", "closed", "filtered", "error"
    response_time_ms: float
    service: Optional[str] = None
    banner: Optional[str] = None


def tcp_connect_scan(host: str, port: int, timeout: float = 1.5) -> PortScanResult:
    """
    Performs a TCP connect scan on a single port.
    
    Algorithm: Attempts a full TCP handshake to determine if port is open.
    Time complexity: O(1) per port
    
    Source: AuroraScan
    
    Args:
        host: Target hostname or IP address
        port: Port number to scan
        timeout: Connection timeout in seconds
        
    Returns:
        PortScanResult with status and timing information
    """
    start_time = time.time()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    
    try:
        sock.connect((host, port))
        response_time = (time.time() - start_time) * 1000
        sock.close()
        return PortScanResult(
            port=port,
            status="open",
            response_time_ms=response_time
        )
    except socket.timeout:
        return PortScanResult(
            port=port,
            status="filtered",
            response_time_ms=(time.time() - start_time) * 1000
        )
    except (socket.error, OSError):
        return PortScanResult(
            port=port,
            status="closed",
            response_time_ms=(time.time() - start_time) * 1000
        )
    finally:
        sock.close()


def parse_port_range(port_spec: str) -> List[int]:
    """
    Parses port specifications like "22,80,443,8000-8010".
    
    Algorithm: Split on comma, expand ranges, deduplicate
    Time complexity: O(n) where n is number of ports specified
    
    Source: AuroraScan
    
    Args:
        port_spec: Port specification string
        
    Returns:
        Sorted list of unique port numbers
        
    Examples:
        >>> parse_port_range("22,80,443")
        [22, 80, 443]
        >>> parse_port_range("8000-8003,9000")
        [8000, 8001, 8002, 8003, 9000]
    """
    ports = []
    for chunk in port_spec.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            start, end = chunk.split("-", maxsplit=1)
            ports.extend(range(int(start), int(end) + 1))
        elif chunk.isdigit():
            ports.append(int(chunk))
    return sorted(set(ports))


def resolve_hostname(host: str) -> Tuple[str, str]:
    """
    Resolves a hostname to an IP address.
    
    Algorithm: DNS lookup with error handling
    Time complexity: O(1) - DNS lookup time
    
    Source: AuroraScan
    
    Args:
        host: Hostname to resolve
        
    Returns:
        Tuple of (original_host, resolved_ip)
    """
    try:
        resolved = socket.gethostbyname(host)
        return (host, resolved)
    except socket.gaierror:
        return (host, "unresolved")


# =============================================================================
# CIPHERSPEAR ALGORITHMS - SQL Injection Analysis
# =============================================================================

# SQL Injection pattern database with severity scores
SQL_INJECTION_PATTERNS: Sequence[Tuple[re.Pattern[str], str, int]] = [
    (re.compile(r"(?:'|\")\s*(?:or|and)\s+1\s*=\s*1", re.IGNORECASE), "boolean-tautology", 4),
    (re.compile(r"union\s+select", re.IGNORECASE), "union-select", 3),
    (re.compile(r"sleep\(\s*\d+\s*\)", re.IGNORECASE), "time-based-delay", 2),
    (re.compile(r"benchmark\(", re.IGNORECASE), "timing-benchmark", 2),
    (re.compile(r"(?:--|#)\s*$"), "inline-comment", 1),
    (re.compile(r"/\*.*\*/", re.IGNORECASE | re.DOTALL), "multiline-comment", 1),
    (re.compile(r"load_file\s*\(", re.IGNORECASE), "file-read", 3),
    (re.compile(r"into\s+outfile", re.IGNORECASE), "file-write", 3),
    (re.compile(r"xp_cmdshell", re.IGNORECASE), "command-exec", 4),
    (re.compile(r"regexp\s+'[^']*'", re.IGNORECASE), "pattern-match", 1),
]


def extract_url_parameters(vector: str) -> List[Tuple[str, str]]:
    """
    Extracts parameter key/value pairs from URLs, query strings, or form bodies.
    
    Algorithm: Parse URL components and extract query parameters
    Time complexity: O(n) where n is number of parameters
    
    Source: CipherSpear
    
    Args:
        vector: URL, query string, or form data
        
    Returns:
        List of (key, value) tuples
        
    Examples:
        >>> extract_url_parameters("user=admin&pass=123")
        [('user', 'admin'), ('pass', '123')]
        >>> extract_url_parameters("http://example.com/search?q=test&page=1")
        [('q', 'test'), ('page', '1')]
    """
    pairs: List[Tuple[str, str]] = []
    
    # Try parsing as URL
    if "://" in vector:
        parsed = urlparse(vector)
    else:
        parsed = urlparse(f"http://dummy/{vector.lstrip('/')}")
    
    # Extract query parameters
    pairs.extend(parse_qsl(parsed.query, keep_blank_values=True))
    
    # Extract fragment parameters
    fragments = parsed.fragment.split("&") if parsed.fragment else []
    for fragment in fragments:
        if "=" in fragment:
            key, value = fragment.split("=", maxsplit=1)
            pairs.append((key, value))
    
    # Try parsing as raw key=value pairs
    if "=" in vector and not pairs:
        for chunk in vector.split("&"):
            if "=" not in chunk:
                continue
            key, value = chunk.split("=", maxsplit=1)
            pairs.append((key, value))
    
    return pairs


def assess_sql_injection_risk(vector: str, techniques: Sequence[str] = None) -> Dict[str, object]:
    """
    Assesses SQL injection risk in a given input vector.
    
    Algorithm: Pattern matching + heuristic scoring
    Time complexity: O(p * n) where p is number of patterns, n is vector length
    
    Source: CipherSpear
    
    Args:
        vector: Input string to analyze (URL, query, or form data)
        techniques: Optional list of techniques to weight (e.g., ['blind', 'time'])
        
    Returns:
        Dictionary with risk_score, risk_label, findings, and recommendation
        
    Examples:
        >>> assess_sql_injection_risk("user=admin' OR 1=1--")
        {'risk_score': 5, 'risk_label': 'medium', 'findings': [...], ...}
    """
    techniques = techniques or []
    findings: List[str] = []
    score = 0
    
    # Check against known SQL injection patterns
    for pattern, label, severity in SQL_INJECTION_PATTERNS:
        if pattern.search(vector):
            findings.append(label)
            score += severity
    
    # Extract and analyze parameters
    params = extract_url_parameters(vector)
    for _, value in params:
        # Check for quote injection
        if "'" in value or "\"" in value:
            findings.append("quote-injection")
            score += 1
        
        # Check for embedded SQL keywords
        if re.search(r"(select|update|delete)\s+\w+", value, re.IGNORECASE):
            findings.append("embedded-sql")
            score += 2
    
    # Technique-specific weighting
    if any(t in {"blind", "time"} for t in techniques) and "time-based-delay" in findings:
        score += 2
    
    # Determine risk label
    if score >= 8:
        risk_label = "high"
    elif score >= 4:
        risk_label = "medium"
    else:
        risk_label = "low"
    
    # Generate recommendation
    if "file-write" in findings or "command-exec" in findings:
        recommendation = "Immediate mitigation required; high-impact primitives detected."
    elif "union-select" in findings:
        recommendation = "Verify column alignment and sanitise UNION clauses."
    else:
        recommendation = "Review parameterised queries and input validation."
    
    return {
        "risk_score": score,
        "risk_label": risk_label,
        "findings": sorted(set(findings)),
        "parameter_count": len(params),
        "recommendation": recommendation
    }


# =============================================================================
# SKYBREAKER ALGORITHMS - Wireless Auditing
# =============================================================================

@dataclass
class WirelessNetwork:
    """Represents a wireless network observation."""
    ssid: str
    bssid: str
    signal: int  # dBm
    channel: int
    security: str
    first_seen: float
    last_seen: float


def calculate_channel_congestion(networks: Iterable[WirelessNetwork]) -> Dict[str, object]:
    """
    Calculates wireless channel congestion from network observations.
    
    Algorithm: Bucket networks by channel, find maximum congestion
    Time complexity: O(n) where n is number of networks
    
    Source: SkyBreaker
    
    Args:
        networks: Iterable of WirelessNetwork observations
        
    Returns:
        Dictionary with max_channel, max_count, and distribution
        
    Examples:
        >>> networks = [WirelessNetwork(..., channel=6, ...), ...]
        >>> calculate_channel_congestion(networks)
        {'max_channel': 6, 'max_count': 5, 'distribution': {1: 2, 6: 5, 11: 3}}
    """
    bucket: Dict[int, int] = {}
    
    for network in networks:
        bucket[network.channel] = bucket.get(network.channel, 0) + 1
    
    if not bucket:
        return {"max_channel": None, "max_count": 0, "distribution": {}}
    
    max_channel = max(bucket, key=bucket.get)
    
    return {
        "max_channel": max_channel,
        "max_count": bucket[max_channel],
        "distribution": bucket,
    }


def assess_wireless_security(networks: Iterable[WirelessNetwork]) -> Dict[str, object]:
    """
    Assesses wireless security posture from network observations.
    
    Algorithm: Categorize networks by security level, count vulnerabilities
    Time complexity: O(n) where n is number of networks
    
    Source: SkyBreaker
    
    Args:
        networks: Iterable of WirelessNetwork observations
        
    Returns:
        Dictionary with security statistics and risk assessment
    """
    network_list = list(networks)
    open_networks = [n for n in network_list if n.security.upper() in {"OPEN", "WEP"}]
    hidden_networks = [n for n in network_list if not n.ssid]
    wpa3_networks = [n for n in network_list if "WPA3" in n.security.upper()]
    
    risk_level = "low"
    if len(open_networks) > 0:
        risk_level = "high"
    elif len(hidden_networks) > 2:
        risk_level = "medium"
    
    return {
        "total_networks": len(network_list),
        "open_networks": len(open_networks),
        "hidden_networks": len(hidden_networks),
        "wpa3_networks": len(wpa3_networks),
        "risk_level": risk_level,
        "recommendation": "Deploy WPA3 where possible; investigate open networks."
    }


# =============================================================================
# MYTHICKEY ALGORITHMS - Credential Analysis
# =============================================================================

# Hash algorithm detection by digest length
HASH_ALGORITHM_BY_LENGTH = {
    32: "md5",
    40: "sha1",
    56: "sha224",
    64: "sha256",
    96: "sha384",
    128: "sha512",
}


def detect_hash_algorithm(digest: str) -> str:
    """
    Detects hash algorithm from digest length.
    
    Algorithm: Lookup table by hex digest length
    Time complexity: O(1)
    
    Source: MythicKey
    
    Args:
        digest: Hex digest string
        
    Returns:
        Algorithm name or "unknown"
        
    Examples:
        >>> detect_hash_algorithm("5d41402abc4b2a76b9719d911017c592")
        'md5'
        >>> detect_hash_algorithm("aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d")
        'sha1'
    """
    # Check for prefixed format like {SHA256}abc...
    if digest.startswith("{") and "}" in digest:
        return digest[1:digest.index("}")].lower()
    
    return HASH_ALGORITHM_BY_LENGTH.get(len(digest), "unknown")


def generate_password_mutations(word: str, profile: str = "cpu") -> List[str]:
    """
    Generates common password mutations for dictionary attacks.
    
    Algorithm: Rule-based transformations (capitalization, suffixes, leetspeak)
    Time complexity: O(1) - fixed number of transformations per word
    
    Source: MythicKey
    
    Args:
        word: Base password word
        profile: Processing profile ("cpu" or "gpu-*" for more mutations)
        
    Returns:
        List of password candidates
        
    Examples:
        >>> generate_password_mutations("password", "cpu")
        ['password', 'Password', 'password123', 'password!']
        >>> generate_password_mutations("admin", "gpu-balanced")
        ['admin', 'Admin', 'admin123', 'admin!', 'nimda', 'admin@', '@dmin']
    """
    mutations = [word]
    
    if not word:
        return mutations
    
    # Basic mutations
    mutations.append(word.capitalize())
    mutations.append(word + "123")
    mutations.append(word + "!")
    
    # Advanced mutations for GPU profiles
    if profile.startswith("gpu"):
        mutations.append(word[::-1])  # Reverse
        mutations.append(word + "@")
        mutations.append(word.replace("a", "@").replace("o", "0").replace("s", "$"))
    
    return mutations


def crack_password_hash(digest: str, wordlist: Iterable[str], profile: str = "cpu") -> Tuple[bool, Optional[str], int]:
    """
    Attempts to crack a password hash using a wordlist.
    
    Algorithm: Dictionary attack with mutations
    Time complexity: O(w * m) where w is wordlist size, m is mutations per word
    
    Source: MythicKey
    
    Args:
        digest: Hash digest to crack
        wordlist: Iterable of password candidates
        profile: Processing profile for mutation depth
        
    Returns:
        Tuple of (cracked, plaintext, attempts)
        
    Examples:
        >>> crack_password_hash("5f4dcc3b5aa765d61d8327deb882cf99", ["password"], "cpu")
        (True, "password", 1)
    """
    algorithm = detect_hash_algorithm(digest)
    if algorithm == "unknown":
        return (False, None, 0)
    
    attempts = 0
    for word in wordlist:
        for candidate in generate_password_mutations(word, profile):
            attempts += 1
            try:
                h = hashlib.new(algorithm)
                h.update(candidate.encode("utf-8"))
                hashed = h.hexdigest()
                
                if hashed.lower() == digest.lower():
                    return (True, candidate, attempts)
            except ValueError:
                continue
    
    return (False, None, attempts)


def estimate_quantum_threat(algorithm: str, key_size: int) -> Dict[str, object]:
    """
    Estimates quantum computing threat to cryptographic keys.
    
    Algorithm: Lookup table based on Shor's algorithm complexity
    Time complexity: O(1)
    
    Source: MythicKey
    
    Args:
        algorithm: Cryptographic algorithm (RSA, EC, DSA)
        key_size: Key size in bits
        
    Returns:
        Dictionary with risk level, estimated qubits, and notes
        
    Examples:
        >>> estimate_quantum_threat("RSA", 2048)
        {'risk': 'High', 'qubits': 4096, 'notes': "Vulnerable to Shor's algorithm."}
    """
    if algorithm.upper() == "RSA":
        if key_size >= 4096:
            risk, qubits = "Medium", 8192
        elif key_size >= 2048:
            risk, qubits = "High", 4096
        else:
            risk, qubits = "Critical", 2048
        return {"risk": risk, "qubits": qubits, "notes": "Vulnerable to Shor's algorithm."}
    
    elif algorithm.upper() in ["ECDSA", "ECDH", "EC"]:
        if key_size >= 384:
            risk, qubits = "Medium", 3072
        elif key_size >= 256:
            risk, qubits = "High", 2330
        else:
            risk, qubits = "Critical", 1500
        return {"risk": risk, "qubits": qubits, "notes": "Vulnerable to Shor's algorithm."}
    
    elif algorithm.upper() == "DSA":
        risk, qubits = "High", 2048
        return {"risk": risk, "qubits": qubits, "notes": "Vulnerable to Shor's algorithm."}
    
    return {"risk": "Unknown", "qubits": 0, "notes": "Algorithm not recognized or not vulnerable to known quantum attacks."}


# =============================================================================
# SPECTRATRACE ALGORITHMS - Packet Analysis
# =============================================================================

@dataclass
class PacketRecord:
    """Represents a captured network packet."""
    timestamp: float
    src: str
    dst: str
    protocol: str
    length: int
    info: str


def parse_pcap_header(data: bytes) -> Tuple[str, bool]:
    """
    Parses PCAP file header to determine endianness.
    
    Algorithm: Magic number detection
    Time complexity: O(1)
    
    Source: SpectraTrace
    
    Args:
        data: First 24 bytes of PCAP file
        
    Returns:
        Tuple of (endian, valid) where endian is ">" or "<"
    """
    if len(data) < 24:
        return (">", False)
    
    magic = struct.unpack("I", data[:4])[0]
    
    if magic == 0xa1b2c3d4:
        return (">", True)
    elif magic == 0xd4c3b2a1:
        return ("<", True)
    else:
        return (">", False)


def identify_protocol(protocol_number: int) -> str:
    """
    Maps IP protocol number to protocol name.
    
    Algorithm: Lookup table
    Time complexity: O(1)
    
    Source: SpectraTrace
    
    Args:
        protocol_number: IP protocol number
        
    Returns:
        Protocol name string
    """
    mapping = {
        1: "ICMP",
        6: "TCP",
        17: "UDP",
        50: "ESP",
        51: "AH",
        58: "ICMPv6",
    }
    return mapping.get(protocol_number, f"IP_PROTO_{protocol_number}")


def analyze_traffic_top_talkers(packets: Iterable[PacketRecord]) -> List[Tuple[str, int]]:
    """
    Identifies top network talkers by bytes transferred.
    
    Algorithm: Hash table aggregation + sorting
    Time complexity: O(n log k) where n is packets, k is top-K
    
    Source: SpectraTrace
    
    Args:
        packets: Iterable of PacketRecord objects
        
    Returns:
        Sorted list of (address, bytes) tuples, top 5
        
    Examples:
        >>> packets = [PacketRecord(..., src="10.0.0.1", length=1000, ...), ...]
        >>> analyze_traffic_top_talkers(packets)
        [('10.0.0.1', 50000), ('10.0.0.2', 30000), ...]
    """
    talkers: Dict[str, int] = defaultdict(int)
    
    for packet in packets:
        talkers[packet.src] += packet.length
        talkers[packet.dst] += packet.length
    
    top_talkers = sorted(talkers.items(), key=lambda item: item[1], reverse=True)[:5]
    return top_talkers


def generate_traffic_heatmap(packets: Sequence[PacketRecord], buckets: int = 12) -> List[Dict[str, object]]:
    """
    Generates a time-series heatmap of traffic volume.
    
    Algorithm: Time bucketing with aggregation
    Time complexity: O(n) where n is number of packets
    
    Source: SpectraTrace
    
    Args:
        packets: Sequence of PacketRecord objects
        buckets: Number of time buckets to divide traffic into
        
    Returns:
        List of bucket dictionaries with packet counts and bytes
    """
    if not packets or buckets <= 0:
        return []
    
    timestamps = [packet.timestamp for packet in packets]
    min_time = min(timestamps)
    max_time = max(timestamps)
    span = max(max_time - min_time, 1e-6)
    bucket_size = span / buckets
    
    # Initialize buckets
    heatmap: List[Dict[str, object]] = []
    for bucket in range(buckets):
        start = min_time + bucket * bucket_size
        end = start + bucket_size
        heatmap.append({
            "index": bucket,
            "start": start,
            "end": end,
            "packet_count": 0,
            "bytes": 0,
        })
    
    # Populate buckets
    for packet in packets:
        bucket_index = int(((packet.timestamp - min_time) / span) * buckets)
        bucket_index = max(0, min(bucket_index, buckets - 1))
        heatmap[bucket_index]["packet_count"] += 1
        heatmap[bucket_index]["bytes"] += packet.length
    
    return heatmap


# =============================================================================
# NEMESISHYDRA ALGORITHMS - Authentication Testing
# =============================================================================

def estimate_spray_duration(wordlist_size: int, rate_limit: int) -> float:
    """
    Estimates duration for password spray attack.
    
    Algorithm: Simple division with safeguards
    Time complexity: O(1)
    
    Source: NemesisHydra
    
    Args:
        wordlist_size: Number of passwords to try
        rate_limit: Attempts per minute allowed
        
    Returns:
        Estimated duration in minutes
        
    Examples:
        >>> estimate_spray_duration(120, 12)
        10.0
    """
    if rate_limit <= 0:
        return float("inf")
    
    attempts_per_minute = max(1, rate_limit)
    minutes = wordlist_size / attempts_per_minute
    return minutes


def assess_throttle_risk(service: str, rate_limit: int) -> str:
    """
    Assesses risk of account lockout based on service and rate.
    
    Algorithm: Rule-based risk classification
    Time complexity: O(1)
    
    Source: NemesisHydra
    
    Args:
        service: Service type (ssh, rdp, http, etc.)
        rate_limit: Attempts per minute
        
    Returns:
        Risk level: "low", "medium", or "high"
        
    Examples:
        >>> assess_throttle_risk("ssh", 25)
        'high'
        >>> assess_throttle_risk("http", 10)
        'low'
    """
    if service in {"ssh", "rdp"} and rate_limit > 20:
        return "high"
    if rate_limit > 40:
        return "high"
    if rate_limit > 15:
        return "medium"
    return "low"


# =============================================================================
# OBSIDIANHUNT ALGORITHMS - Host Hardening
# =============================================================================

def evaluate_system_control(control_path: Path) -> str:
    """
    Evaluates presence of a security control by path.
    
    Algorithm: Filesystem check with fallback to PATH
    Time complexity: O(1)
    
    Source: ObsidianHunt
    
    Args:
        control_path: Path to control file or binary
        
    Returns:
        Status: "pass", "warn", or "error"
    """
    if control_path.is_dir() or control_path.is_file():
        return "pass"
    
    # Check if binary is in PATH
    import shutil
    if shutil.which(control_path.name):
        return "pass"
    
    return "warn"


# =============================================================================
# QUANTUMLEAPASSESSOR ALGORITHMS - Quantum Vulnerability Scanning
# =============================================================================

# Patterns for quantum-vulnerable algorithms
QUANTUM_VULNERABLE_PATTERNS = {
    "RSA": re.compile(r'\b(RSA)\b', re.IGNORECASE),
    "ECC": re.compile(r'\b(ECC|ECDSA|ECDH)\b', re.IGNORECASE),
    "DSA": re.compile(r'\b(DSA|Digital Signature Algorithm)\b', re.IGNORECASE),
    "Diffie-Hellman": re.compile(r'\b(Diffie-Hellman|DH)\b', re.IGNORECASE),
}


def scan_file_for_quantum_vulnerabilities(file_path: Path) -> List[Dict[str, object]]:
    """
    Scans a source file for quantum-vulnerable cryptographic algorithms.
    
    Algorithm: Line-by-line regex pattern matching
    Time complexity: O(n * p) where n is lines, p is patterns
    
    Source: QuantumLeapAssessor
    
    Args:
        file_path: Path to source file
        
    Returns:
        List of finding dictionaries with line numbers and algorithms
    """
    findings = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                for alg, pattern in QUANTUM_VULNERABLE_PATTERNS.items():
                    if pattern.search(line):
                        findings.append({
                            "file_path": str(file_path),
                            "line": line_num,
                            "algorithm": alg,
                            "snippet": line.strip()
                        })
    except (IOError, OSError):
        pass
    
    return findings


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def sanitize_dsn(dsn: str) -> Dict[str, str]:
    """
    Sanitizes a database DSN by redacting credentials.
    
    Algorithm: URL parsing with credential redaction
    Time complexity: O(n) where n is DSN length
    
    Source: CipherSpear
    
    Args:
        dsn: Database connection string
        
    Returns:
        Dictionary with sanitized components
        
    Examples:
        >>> sanitize_dsn("postgresql://user:secret@localhost/db")
        {'scheme': 'postgresql', 'netloc': 'user@localhost', 'path': '/db'}
    """
    parsed = urlparse(dsn)
    redacted_netloc = parsed.netloc
    
    if "@" in redacted_netloc:
        userinfo, hostinfo = redacted_netloc.split("@", maxsplit=1)
        if ":" in userinfo:
            user, _ = userinfo.split(":", maxsplit=1)
        else:
            user = userinfo
        redacted_netloc = f"{user}@{hostinfo}"
    
    return {
        "scheme": parsed.scheme or "unknown",
        "netloc": redacted_netloc,
        "path": parsed.path or "/",
    }


def classify_ip_traffic_direction(src: str, dst: str) -> str:
    """
    Classifies traffic direction based on private/public IP addresses.
    
    Algorithm: IP address type detection
    Time complexity: O(1)
    
    Source: SpectraTrace
    
    Args:
        src: Source IP address
        dst: Destination IP address
        
    Returns:
        Direction: "ingress", "egress", "internal", or "unknown"
    """
    try:
        src_ip = ipaddress.ip_address(src)
        dst_ip = ipaddress.ip_address(dst)
        
        if src_ip.is_private and dst_ip.is_private:
            return "internal"
        elif src_ip.is_private and not dst_ip.is_private:
            return "egress"
        elif not src_ip.is_private and dst_ip.is_private:
            return "ingress"
        else:
            return "external"
    except ValueError:
        return "unknown"


# =============================================================================
# ALGORITHM CATALOG
# =============================================================================

def get_algorithm_catalog() -> Dict[str, List[str]]:
    """
    Returns a catalog of all available algorithms organized by tool.
    
    Returns:
        Dictionary mapping tool names to lists of algorithm names
    """
    return {
        "AuroraScan": [
            "tcp_connect_scan",
            "parse_port_range",
            "resolve_hostname",
        ],
        "CipherSpear": [
            "extract_url_parameters",
            "assess_sql_injection_risk",
            "sanitize_dsn",
        ],
        "SkyBreaker": [
            "calculate_channel_congestion",
            "assess_wireless_security",
        ],
        "MythicKey": [
            "detect_hash_algorithm",
            "generate_password_mutations",
            "crack_password_hash",
            "estimate_quantum_threat",
        ],
        "SpectraTrace": [
            "parse_pcap_header",
            "identify_protocol",
            "analyze_traffic_top_talkers",
            "generate_traffic_heatmap",
            "classify_ip_traffic_direction",
        ],
        "NemesisHydra": [
            "estimate_spray_duration",
            "assess_throttle_risk",
        ],
        "ObsidianHunt": [
            "evaluate_system_control",
        ],
        "QuantumLeapAssessor": [
            "scan_file_for_quantum_vulnerabilities",
        ],
    }


if __name__ == "__main__":
    # Print algorithm catalog
    print("Sovereign Security Toolkit - Algorithm Catalog")
    print("=" * 60)
    
    catalog = get_algorithm_catalog()
    for tool, algorithms in catalog.items():
        print(f"\n{tool}:")
        for alg in algorithms:
            print(f"  - {alg}()")
    
    print(f"\nTotal: {sum(len(algs) for algs in catalog.values())} algorithms across {len(catalog)} tools")

