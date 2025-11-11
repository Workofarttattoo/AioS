#!/usr/bin/env python3
"""
DirReaper Demonstration Script
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Demonstrates all scanning modes and features of DirReaper.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.dirreaper import DirReaper, WORDLIST_COMMON, WORDLIST_MEDIUM, health_check


async def demo_directory_scan():
    """Demonstrate directory enumeration"""
    print("\n" + "="*60)
    print("DEMO 1: Directory Enumeration")
    print("="*60)

    scanner = DirReaper(
        target="https://httpbin.org",
        wordlist=WORDLIST_COMMON[:20],  # Limit for demo
        mode="dir",
        extensions=[".html", ".json"],
        threads=10,
        status_codes=[200, 301, 302, 401, 403, 404]
    )

    print(f"Target: {scanner.target}")
    print(f"Mode: {scanner.mode}")
    print(f"Wordlist size: {len(scanner.wordlist)} words")
    print(f"Threads: {scanner.threads}")
    print(f"Extensions: {scanner.extensions}")
    print("\nStarting scan...\n")

    results = await scanner.run()

    print(f"\nResults found: {len(results)}")
    for result in results[:5]:  # Show first 5
        print(f"  [{result.status}] {result.url} ({result.size} bytes)")

    return results


async def demo_vhost_discovery():
    """Demonstrate virtual host discovery"""
    print("\n" + "="*60)
    print("DEMO 2: Virtual Host Discovery")
    print("="*60)

    scanner = DirReaper(
        target="https://httpbin.org",
        wordlist=["www", "api", "dev", "test", "staging"],
        mode="vhost",
        threads=5
    )

    print(f"Target: {scanner.target}")
    print(f"Mode: {scanner.mode}")
    print(f"Testing {len(scanner.wordlist)} vhost patterns")
    print("\nStarting scan...\n")

    results = await scanner.run()

    print(f"\nVirtual hosts found: {len(results)}")
    for result in results:
        print(f"  [{result.status}] {result.url}")

    return results


async def demo_dns_enumeration():
    """Demonstrate DNS subdomain enumeration"""
    print("\n" + "="*60)
    print("DEMO 3: DNS Subdomain Enumeration")
    print("="*60)

    scanner = DirReaper(
        target="httpbin.org",
        wordlist=["www", "api", "mail", "ftp", "blog"],
        mode="dns",
        threads=5
    )

    print(f"Domain: {scanner.target}")
    print(f"Mode: {scanner.mode}")
    print(f"Testing {len(scanner.wordlist)} subdomains")
    print("\nStarting scan...\n")

    results = await scanner.run()

    print(f"\nSubdomains found: {len(results)}")
    for result in results:
        print(f"  {result.url} -> {result.redirect}")

    return results


async def demo_s3_discovery():
    """Demonstrate S3 bucket discovery"""
    print("\n" + "="*60)
    print("DEMO 4: S3 Bucket Discovery")
    print("="*60)

    scanner = DirReaper(
        target="httpbin.org",
        mode="s3",
        threads=5
    )

    print(f"Domain: {scanner.target}")
    print(f"Mode: {scanner.mode}")
    print("\nStarting scan...\n")

    results = await scanner.run()

    print(f"\nS3 buckets found: {len(results)}")
    for result in results:
        print(f"  [{result.status}] {result.url}")

    return results


async def demo_fuzzing():
    """Demonstrate parameter fuzzing"""
    print("\n" + "="*60)
    print("DEMO 5: Parameter Fuzzing")
    print("="*60)

    scanner = DirReaper(
        target="https://httpbin.org/get",
        wordlist=["id", "user", "page", "token", "api_key"],
        mode="fuzzing",
        threads=5
    )

    print(f"Target: {scanner.target}")
    print(f"Mode: {scanner.mode}")
    print(f"Testing {len(scanner.wordlist)} parameters")
    print("\nStarting scan...\n")

    results = await scanner.run()

    print(f"\nParameters found: {len(results)}")
    for result in results[:5]:
        print(f"  [{result.status}] {result.url}")

    return results


def demo_health_check():
    """Demonstrate health check"""
    print("\n" + "="*60)
    print("DEMO 0: Health Check")
    print("="*60)

    result = health_check()

    print(f"Tool: {result['tool']}")
    print(f"Status: {result['status']}")
    print(f"Summary: {result['summary']}")
    print("\nDetails:")
    print(f"  Modes: {', '.join(result['details']['modes'])}")
    print(f"  aiohttp version: {result['details']['aiohttp_version']}")
    print(f"  DNS support: {result['details']['dns_support']}")
    print(f"  Max threads: {result['details']['max_threads']}")
    print(f"  Wordlists:")
    for name, size in result['details']['wordlists'].items():
        print(f"    {name}: {size} words")
    print(f"  Latency: {result['details']['latency_ms']:.2f}ms")


async def main():
    """Run all demonstrations"""
    print("\n" + "="*60)
    print("DirReaper - Comprehensive Demonstration")
    print("High-Performance Directory Enumeration Tool")
    print("="*60)

    # Health check
    demo_health_check()

    # Demo 1: Directory scan
    await demo_directory_scan()

    # Demo 2: VHost discovery
    await demo_vhost_discovery()

    # Demo 3: DNS enumeration
    await demo_dns_enumeration()

    # Demo 4: S3 discovery
    await demo_s3_discovery()

    # Demo 5: Fuzzing
    await demo_fuzzing()

    print("\n" + "="*60)
    print("All demonstrations complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run full scan: python -m tools.dirreaper https://example.com")
    print("  2. Launch GUI: python -m tools.dirreaper --gui")
    print("  3. Check health: python -m tools.dirreaper --health-check")
    print("  4. Read docs: cat /Users/noone/aios/tools/DIRREAPER_README.md")
    print("\n💀 DirReaper - Cutting through directories with style! ⚡\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
