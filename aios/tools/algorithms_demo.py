#!/usr/bin/env python3
"""
Sovereign Security Toolkit - Algorithm Demonstrations

This script demonstrates all algorithms from the centralized algorithms library.
Run this to verify that all algorithms are working correctly.

Usage:
    python aios/tools/algorithms_demo.py
"""

import time
from pathlib import Path

# Import all algorithms
from algorithms import (
    # AuroraScan
    tcp_connect_scan,
    parse_port_range,
    resolve_hostname,
    
    # CipherSpear
    extract_url_parameters,
    assess_sql_injection_risk,
    sanitize_dsn,
    
    # SkyBreaker
    WirelessNetwork,
    calculate_channel_congestion,
    assess_wireless_security,
    
    # MythicKey
    detect_hash_algorithm,
    generate_password_mutations,
    crack_password_hash,
    estimate_quantum_threat,
    
    # SpectraTrace
    identify_protocol,
    PacketRecord,
    analyze_traffic_top_talkers,
    generate_traffic_heatmap,
    classify_ip_traffic_direction,
    
    # NemesisHydra
    estimate_spray_duration,
    assess_throttle_risk,
    
    # ObsidianHunt
    evaluate_system_control,
    
    # QuantumLeapAssessor
    scan_file_for_quantum_vulnerabilities,
    
    # Utilities
    get_algorithm_catalog,
)


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


def demo_aurorascan():
    """Demonstrate AuroraScan algorithms."""
    print_section("AuroraScan - Network Reconnaissance")
    
    # Parse port range
    print("\n1. parse_port_range()")
    port_spec = "22,80,443,8000-8003"
    ports = parse_port_range(port_spec)
    print(f"   Input: '{port_spec}'")
    print(f"   Output: {ports}")
    
    # Resolve hostname
    print("\n2. resolve_hostname()")
    test_hosts = ["localhost", "127.0.0.1", "nonexistent.invalid"]
    for host in test_hosts:
        original, resolved = resolve_hostname(host)
        print(f"   {original:25s} -> {resolved}")
    
    # TCP connect scan (only test localhost to be safe)
    print("\n3. tcp_connect_scan()")
    print("   Scanning localhost on common ports...")
    test_ports = [22, 80, 443]
    for port in test_ports:
        result = tcp_connect_scan("127.0.0.1", port, timeout=0.5)
        status_icon = "✓" if result.status == "open" else "✗"
        print(f"   {status_icon} Port {port:5d}: {result.status:8s} ({result.response_time_ms:6.2f}ms)")


def demo_cipherspear():
    """Demonstrate CipherSpear algorithms."""
    print_section("CipherSpear - SQL Injection Analysis")
    
    # Extract URL parameters
    print("\n1. extract_url_parameters()")
    test_vectors = [
        "http://example.com/search?q=test&page=1",
        "user=admin&pass=secret",
        "https://api.example.com/data?id=5&format=json",
    ]
    for vector in test_vectors:
        params = extract_url_parameters(vector)
        print(f"   Input: {vector}")
        print(f"   Params: {params}")
    
    # Assess SQL injection risk
    print("\n2. assess_sql_injection_risk()")
    test_injections = [
        "user=admin&pass=test123",
        "user=admin' OR 1=1--",
        "query=test'; DROP TABLE users--",
        "id=5 UNION SELECT * FROM passwords",
    ]
    for vector in test_injections:
        risk = assess_sql_injection_risk(vector)
        risk_color = {"low": "🟢", "medium": "🟡", "high": "🔴"}
        icon = risk_color.get(risk['risk_label'], "⚪")
        print(f"   {icon} {risk['risk_label'].upper():8s} (score: {risk['risk_score']:2d}) - {vector[:50]}")
        if risk['findings']:
            print(f"      Findings: {', '.join(risk['findings'])}")
    
    # Sanitize DSN
    print("\n3. sanitize_dsn()")
    dsn = "postgresql://user:secret123@localhost:5432/mydb"
    sanitized = sanitize_dsn(dsn)
    print(f"   Original: {dsn}")
    print(f"   Sanitized: {sanitized}")


def demo_skybreaker():
    """Demonstrate SkyBreaker algorithms."""
    print_section("SkyBreaker - Wireless Auditing")
    
    # Create sample wireless networks
    networks = [
        WirelessNetwork("HomeNet", "aa:bb:cc:dd:ee:01", -45, 6, "WPA2-PSK", 0.0, 1.0),
        WirelessNetwork("OfficeNet", "aa:bb:cc:dd:ee:02", -55, 6, "WPA3-SAE", 0.0, 1.0),
        WirelessNetwork("GuestNet", "aa:bb:cc:dd:ee:03", -65, 11, "WPA2-PSK", 0.0, 1.0),
        WirelessNetwork("FreeWiFi", "aa:bb:cc:dd:ee:04", -70, 6, "OPEN", 0.0, 1.0),
        WirelessNetwork("", "aa:bb:cc:dd:ee:05", -75, 1, "WPA2-PSK", 0.0, 1.0),
    ]
    
    # Calculate channel congestion
    print("\n1. calculate_channel_congestion()")
    congestion = calculate_channel_congestion(networks)
    print(f"   Most congested channel: {congestion['max_channel']} ({congestion['max_count']} networks)")
    print(f"   Distribution: {congestion['distribution']}")
    
    # Assess wireless security
    print("\n2. assess_wireless_security()")
    assessment = assess_wireless_security(networks)
    print(f"   Total networks: {assessment['total_networks']}")
    print(f"   Open networks: {assessment['open_networks']}")
    print(f"   Hidden networks: {assessment['hidden_networks']}")
    print(f"   WPA3 networks: {assessment['wpa3_networks']}")
    print(f"   Risk level: {assessment['risk_level'].upper()}")


def demo_mythickey():
    """Demonstrate MythicKey algorithms."""
    print_section("MythicKey - Credential Analysis")
    
    # Detect hash algorithm
    print("\n1. detect_hash_algorithm()")
    test_hashes = [
        ("5d41402abc4b2a76b9719d911017c592", "MD5 of 'hello'"),
        ("aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d", "SHA-1 of 'hello'"),
        ("2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824", "SHA-256 of 'hello'"),
    ]
    for digest, description in test_hashes:
        algorithm = detect_hash_algorithm(digest)
        print(f"   {algorithm:8s} - {description}")
    
    # Generate password mutations
    print("\n2. generate_password_mutations()")
    base_word = "password"
    cpu_mutations = generate_password_mutations(base_word, "cpu")
    gpu_mutations = generate_password_mutations(base_word, "gpu-balanced")
    print(f"   Base word: '{base_word}'")
    print(f"   CPU profile: {cpu_mutations}")
    print(f"   GPU profile: {gpu_mutations}")
    
    # Crack password hash
    print("\n3. crack_password_hash()")
    # MD5 of "password"
    target_hash = "5f4dcc3b5aa765d61d8327deb882cf99"
    wordlist = ["admin", "welcome", "password", "letmein"]
    
    print(f"   Target: {target_hash}")
    print(f"   Wordlist: {wordlist}")
    
    start_time = time.time()
    cracked, plaintext, attempts = crack_password_hash(target_hash, wordlist, "cpu")
    duration = time.time() - start_time
    
    if cracked:
        print(f"   ✓ CRACKED! Plaintext: '{plaintext}'")
        print(f"   Attempts: {attempts}, Duration: {duration*1000:.2f}ms")
    else:
        print(f"   ✗ Failed after {attempts} attempts")
    
    # Estimate quantum threat
    print("\n4. estimate_quantum_threat()")
    test_keys = [
        ("RSA", 2048),
        ("RSA", 4096),
        ("EC", 256),
        ("EC", 384),
    ]
    for algorithm, key_size in test_keys:
        threat = estimate_quantum_threat(algorithm, key_size)
        risk_icons = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}
        icon = risk_icons.get(threat['risk'], "⚪")
        print(f"   {icon} {algorithm} {key_size}-bit: {threat['risk']} ({threat['qubits']} qubits)")


def demo_spectratrace():
    """Demonstrate SpectraTrace algorithms."""
    print_section("SpectraTrace - Packet Analysis")
    
    # Identify protocol
    print("\n1. identify_protocol()")
    test_protocols = [1, 6, 17, 50, 58, 99]
    for proto_num in test_protocols:
        proto_name = identify_protocol(proto_num)
        print(f"   Protocol {proto_num:3d}: {proto_name}")
    
    # Create sample packets
    packets = [
        PacketRecord(0.0, "10.0.0.1", "8.8.8.8", "UDP", 512, "DNS Query"),
        PacketRecord(0.1, "8.8.8.8", "10.0.0.1", "UDP", 256, "DNS Response"),
        PacketRecord(0.5, "10.0.0.1", "192.168.1.1", "TCP", 1500, "HTTP GET"),
        PacketRecord(1.0, "192.168.1.1", "10.0.0.1", "TCP", 8000, "HTTP Response"),
        PacketRecord(2.0, "10.0.0.1", "203.0.113.5", "TCP", 2048, "HTTPS"),
        PacketRecord(3.0, "10.0.0.2", "10.0.0.1", "TCP", 500, "SSH"),
    ]
    
    # Analyze top talkers
    print("\n2. analyze_traffic_top_talkers()")
    top_talkers = analyze_traffic_top_talkers(packets)
    print(f"   Top {len(top_talkers)} talkers:")
    for address, bytes_transferred in top_talkers:
        print(f"      {address:20s}: {bytes_transferred:6d} bytes")
    
    # Generate traffic heatmap
    print("\n3. generate_traffic_heatmap()")
    heatmap = generate_traffic_heatmap(packets, buckets=4)
    print(f"   Traffic distribution over {len(heatmap)} buckets:")
    for bucket in heatmap:
        bars = "█" * bucket['packet_count']
        print(f"      Bucket {bucket['index']}: {bars} ({bucket['packet_count']} packets, {bucket['bytes']} bytes)")
    
    # Classify IP traffic direction
    print("\n4. classify_ip_traffic_direction()")
    test_pairs = [
        ("10.0.0.1", "8.8.8.8"),
        ("8.8.8.8", "10.0.0.1"),
        ("10.0.0.1", "192.168.1.1"),
        ("8.8.8.8", "1.1.1.1"),
    ]
    for src, dst in test_pairs:
        direction = classify_ip_traffic_direction(src, dst)
        arrow = "→"
        print(f"      {src:20s} {arrow} {dst:20s}: {direction}")


def demo_nemesishydra():
    """Demonstrate NemesisHydra algorithms."""
    print_section("NemesisHydra - Authentication Testing")
    
    # Estimate spray duration
    print("\n1. estimate_spray_duration()")
    test_scenarios = [
        (120, 12),
        (500, 20),
        (1000, 50),
    ]
    for wordlist_size, rate_limit in test_scenarios:
        duration = estimate_spray_duration(wordlist_size, rate_limit)
        print(f"   {wordlist_size:4d} passwords @ {rate_limit:2d}/min = {duration:6.1f} minutes")
    
    # Assess throttle risk
    print("\n2. assess_throttle_risk()")
    test_services = [
        ("ssh", 10),
        ("ssh", 25),
        ("http", 30),
        ("rdp", 20),
    ]
    for service, rate_limit in test_services:
        risk = assess_throttle_risk(service, rate_limit)
        risk_icons = {"low": "🟢", "medium": "🟡", "high": "🔴"}
        icon = risk_icons.get(risk, "⚪")
        print(f"   {icon} {service:6s} @ {rate_limit:2d}/min: {risk.upper()}")


def demo_obsidianhunt():
    """Demonstrate ObsidianHunt algorithms."""
    print_section("ObsidianHunt - Host Hardening")
    
    print("\n1. evaluate_system_control()")
    test_paths = [
        Path("/bin/sh"),
        Path("/usr/bin/python3"),
        Path("/nonexistent/path"),
    ]
    for path in test_paths:
        status = evaluate_system_control(path)
        icon = "✓" if status == "pass" else "⚠"
        print(f"   {icon} {str(path):30s}: {status}")


def demo_quantumleap():
    """Demonstrate QuantumLeapAssessor algorithms."""
    print_section("QuantumLeapAssessor - Quantum Vulnerability Scanning")
    
    print("\n1. scan_file_for_quantum_vulnerabilities()")
    
    # Create a temporary test file
    test_file = Path("/tmp/test_crypto.py")
    test_content = """
import rsa
from cryptography.hazmat.primitives.asymmetric import dsa, ec
from Crypto.PublicKey import RSA

# This code uses RSA encryption
key = RSA.generate(2048)

# ECC for signing
private_key = ec.generate_private_key(ec.SECP256R1())

# DSA signature
dsa_key = dsa.generate_private_key(key_size=2048)
"""
    
    test_file.write_text(test_content)
    
    findings = scan_file_for_quantum_vulnerabilities(test_file)
    print(f"   Scanned: {test_file}")
    print(f"   Found {len(findings)} potential vulnerabilities:")
    
    for finding in findings:
        print(f"      Line {finding['line']:3d}: {finding['algorithm']:15s} - {finding['snippet'][:50]}")
    
    # Clean up
    test_file.unlink()


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("  Sovereign Security Toolkit - Algorithm Demonstrations")
    print("="*70)
    
    # Show catalog
    print("\nAlgorithm Catalog:")
    catalog = get_algorithm_catalog()
    total_algorithms = 0
    for tool, algorithms in catalog.items():
        print(f"  {tool}: {len(algorithms)} algorithms")
        total_algorithms += len(algorithms)
    print(f"\nTotal: {total_algorithms} algorithms across {len(catalog)} tools")
    
    # Run demonstrations
    try:
        demo_aurorascan()
        demo_cipherspear()
        demo_skybreaker()
        demo_mythickey()
        demo_spectratrace()
        demo_nemesishydra()
        demo_obsidianhunt()
        demo_quantumleap()
        
        print("\n" + "="*70)
        print("  ✓ All demonstrations completed successfully!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

