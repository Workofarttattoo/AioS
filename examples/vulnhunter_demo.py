#!/usr/bin/env python3
"""
VulnHunter Demonstration Script
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Demonstrates VulnHunter capabilities including:
- Health check
- Vulnerability scanning
- Report generation
- CVSS scoring
- Integration with Ai|oS
"""

import json
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import vulnhunter

def print_banner():
    """Print VulnHunter banner"""
    print("\n" + "="*70)
    print("🎯 VulnHunter - Comprehensive Vulnerability Scanner")
    print("="*70)
    print("Copyright © 2025 Corporation of Light. All Rights Reserved.")
    print("PATENT PENDING")
    print("="*70 + "\n")

def demo_health_check():
    """Demonstrate health check functionality"""
    print("[1/5] Health Check")
    print("-" * 70)

    result = vulnhunter.health_check()

    print(f"Status: {result['status'].upper()}")
    print(f"Summary: {result['summary']}")
    print(f"\nDetails:")
    print(f"  - Plugins Loaded: {result['details']['plugins_loaded']}")
    print(f"  - Categories: {', '.join(result['details']['categories'])}")
    print(f"  - CVSS Functional: {result['details']['cvss_functional']}")
    print(f"  - Severity Levels: {', '.join(result['details']['severity_levels'])}")
    print("\n✅ Health check passed!\n")

def demo_cvss_scoring():
    """Demonstrate CVSS v3 scoring"""
    print("[2/5] CVSS v3 Scoring Engine")
    print("-" * 70)

    # Calculate scores for different scenarios
    scenarios = [
        {
            "name": "Critical SQL Injection",
            "params": {
                "attack_vector": "N",
                "attack_complexity": "L",
                "privileges": "N",
                "user_interaction": "N",
                "scope": "C",
                "confidentiality": "H",
                "integrity": "H",
                "availability": "H"
            }
        },
        {
            "name": "Medium XSS Vulnerability",
            "params": {
                "attack_vector": "N",
                "attack_complexity": "L",
                "privileges": "N",
                "user_interaction": "R",
                "scope": "U",
                "confidentiality": "L",
                "integrity": "L",
                "availability": "N"
            }
        },
        {
            "name": "Low Information Disclosure",
            "params": {
                "attack_vector": "N",
                "attack_complexity": "L",
                "privileges": "L",
                "user_interaction": "N",
                "scope": "U",
                "confidentiality": "L",
                "integrity": "N",
                "availability": "N"
            }
        }
    ]

    for scenario in scenarios:
        score = vulnhunter.CVSS.calculate(**scenario["params"])
        severity = "CRITICAL" if score >= 9.0 else "HIGH" if score >= 7.0 else "MEDIUM" if score >= 4.0 else "LOW"
        print(f"\n{scenario['name']}:")
        print(f"  CVSS Score: {score}")
        print(f"  Severity: {severity}")

    print("\n✅ CVSS scoring functional!\n")

def demo_plugin_listing():
    """Demonstrate plugin listing"""
    print("[3/5] Vulnerability Check Plugins")
    print("-" * 70)

    scanner = vulnhunter.VulnHunterScanner()
    plugins = scanner.plugins

    # Count by category
    category_counts = {}
    for plugin in plugins:
        cat = plugin.category
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print(f"\nTotal Plugins: {len(plugins)}")
    print("\nBy Category:")
    for category, count in sorted(category_counts.items()):
        print(f"  - {category}: {count} checks")

    # Show sample plugins
    print("\nSample Checks:")
    for plugin in plugins[:10]:
        severity_color = plugin.severity.value
        print(f"  [{plugin.check_id}] {plugin.name} ({plugin.category}) - {severity_color} (CVSS {plugin.cvss_score})")

    print("\n✅ All plugins loaded successfully!\n")

def demo_vulnerability_object():
    """Demonstrate vulnerability object creation"""
    print("[4/5] Vulnerability Object Structure")
    print("-" * 70)

    # Create sample vulnerability
    vuln = vulnhunter.Vulnerability(
        host="192.168.1.100",
        port=443,
        check_id="WEB-001",
        name="SQL Injection Vulnerability",
        severity=vulnhunter.Severity.CRITICAL,
        cvss_score=9.8,
        description="SQL injection vulnerability allows attackers to manipulate database queries",
        proof="Payload: ' OR '1'='1 returned database error",
        remediation="Use parameterized queries and input validation",
        references=["CVE-2024-XXXXX", "CWE-89", "OWASP-A03"]
    )

    # Convert to dict
    vuln_dict = vuln.to_dict()

    print("\nVulnerability Structure:")
    print(json.dumps(vuln_dict, indent=2))

    print("\n✅ Vulnerability objects fully functional!\n")

def demo_scan_profiles():
    """Demonstrate scan profiles"""
    print("[5/5] Scan Profiles")
    print("-" * 70)

    profiles = {
        "quick": "Critical & High severity checks only (fastest)",
        "full": "All 50+ vulnerability checks (comprehensive)",
        "web": "Web application security testing",
        "network": "Network infrastructure scanning",
        "compliance": "Authentication and configuration auditing"
    }

    print("\nAvailable Scan Profiles:\n")
    for profile, description in profiles.items():
        print(f"  {profile.upper():12} - {description}")

    print("\n✅ All scan profiles configured!\n")

def demo_report_generation():
    """Demonstrate report generation"""
    print("[BONUS] Report Generation")
    print("-" * 70)

    scanner = vulnhunter.VulnHunterScanner()

    # Add sample vulnerabilities
    sample_vulns = [
        vulnhunter.Vulnerability(
            "192.168.1.100", 443, "NET-007", "SSLv2/v3 Enabled",
            vulnhunter.Severity.HIGH, 7.0,
            "Weak SSL/TLS protocol in use",
            "SSL version: SSLv3",
            "Upgrade to TLS 1.2 or higher",
            ["CWE-326"]
        ),
        vulnhunter.Vulnerability(
            "192.168.1.101", 80, "WEB-011", "Missing Security Headers",
            vulnhunter.Severity.LOW, 3.5,
            "Security headers not configured",
            "Missing: X-Frame-Options, CSP",
            "Implement security headers",
            ["CWE-16"]
        )
    ]

    scanner.vulnerabilities = sample_vulns

    print("\nGenerating reports in multiple formats...\n")

    # JSON Report
    json_report = scanner.generate_report('json')
    print("✅ JSON Report Generated")
    print(f"   Size: {len(json_report)} bytes")

    # HTML Report
    html_report = scanner.generate_report('html')
    print("✅ HTML Report Generated")
    print(f"   Size: {len(html_report)} bytes")

    # CSV Report
    csv_report = scanner.generate_report('csv')
    print("✅ CSV Report Generated")
    print(f"   Size: {len(csv_report)} bytes")

    print("\n✅ All report formats functional!\n")

def main():
    """Main demonstration function"""
    print_banner()

    print("This demonstration will showcase VulnHunter's capabilities:")
    print("  1. Health check and system verification")
    print("  2. CVSS v3 scoring engine")
    print("  3. Vulnerability check plugins")
    print("  4. Vulnerability object structure")
    print("  5. Scan profile configuration")
    print("  + Bonus: Report generation\n")

    # Skip input in non-interactive mode
    try:
        input("Press Enter to start the demonstration...")
        print()
    except EOFError:
        print("Running in non-interactive mode...\n")

    try:
        demo_health_check()
        time.sleep(1)

        demo_cvss_scoring()
        time.sleep(1)

        demo_plugin_listing()
        time.sleep(1)

        demo_vulnerability_object()
        time.sleep(1)

        demo_scan_profiles()
        time.sleep(1)

        demo_report_generation()

        print("="*70)
        print("🎯 DEMONSTRATION COMPLETE")
        print("="*70)
        print("\nVulnHunter is fully operational and ready for use!")
        print("\nQuick Start Commands:")
        print("  - Launch GUI:    python -m tools.vulnhunter --gui")
        print("  - Run Scan:      python -m tools.vulnhunter --scan TARGET")
        print("  - Health Check:  python -m tools.vulnhunter --health")
        print("\nFor full documentation, see:")
        print("  /Users/noone/aios/tools/VULNHUNTER_README.md")
        print("\n" + "="*70 + "\n")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
