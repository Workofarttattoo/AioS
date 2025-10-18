#!/usr/bin/env python3
"""
Shodan Search Tool - Educational Internet-Connected Device Research
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Educational tool demonstrating internet-exposed device research using Shodan-style queries.
FOR AUTHORIZED SECURITY RESEARCH AND EDUCATIONAL PURPOSES ONLY.
"""

import sys
import json
import argparse
import random
from typing import List, Dict

# Simulated Shodan-style results database
SAMPLE_RESULTS = [
    {
        "ip": "185.234.218.142",
        "port": 22,
        "org": "DigitalOcean",
        "country": "United States",
        "city": "New York",
        "product": "OpenSSH",
        "version": "7.4",
        "vulns": ["CVE-2018-15473"],
        "banner": "SSH-2.0-OpenSSH_7.4"
    },
    {
        "ip": "195.123.221.89",
        "port": 80,
        "org": "OVH SAS",
        "country": "France",
        "city": "Paris",
        "product": "Apache httpd",
        "version": "2.4.41",
        "vulns": ["CVE-2021-40438", "CVE-2021-41773"],
        "banner": "Apache/2.4.41 (Ubuntu)"
    },
    {
        "ip": "203.134.55.177",
        "port": 443,
        "org": "Amazon",
        "country": "Singapore",
        "city": "Singapore",
        "product": "nginx",
        "version": "1.18.0",
        "vulns": [],
        "banner": "nginx/1.18.0"
    },
    {
        "ip": "142.250.72.46",
        "port": 8080,
        "org": "Google",
        "country": "United States",
        "city": "Mountain View",
        "product": "Jetty",
        "version": "9.4.35",
        "vulns": ["CVE-2021-28169"],
        "banner": "Jetty(9.4.35.v20201120)"
    },
    {
        "ip": "81.169.241.55",
        "port": 3306,
        "org": "Hetzner Online",
        "country": "Germany",
        "city": "Nuremberg",
        "product": "MySQL",
        "version": "5.7.33",
        "vulns": ["CVE-2021-2194", "CVE-2021-2166"],
        "banner": "5.7.33-0ubuntu0.18.04.1"
    }
]


def search(query: str, limit: int = 50) -> List[Dict]:
    """
    Simulate Shodan search.

    In production, this would use the actual Shodan API.
    This is a demonstration/educational version.
    """
    query_lower = query.lower()

    # Filter results based on query
    results = []
    for result in SAMPLE_RESULTS:
        if (query_lower in str(result.get('port', '')).lower() or
            query_lower in result.get('product', '').lower() or
            query_lower in result.get('country', '').lower() or
            query_lower in result.get('banner', '').lower()):
            results.append(result.copy())

    # If no results match, return random sample
    if not results:
        results = random.sample(SAMPLE_RESULTS, min(len(SAMPLE_RESULTS), 3))

    return results[:limit]


def health_check() -> Dict:
    """Health check for the tool"""
    import time
    start = time.time()

    status = "ok"
    summary = "Shodan Search simulator ready (educational demo)"

    details = {
        "mode": "educational_simulation",
        "sample_results": len(SAMPLE_RESULTS),
        "note": "This is a simulation for educational purposes. Real Shodan requires API key."
    }

    latency = (time.time() - start) * 1000
    details["latency_ms"] = round(latency, 2)

    return {
        "tool": "shodansearch",
        "status": status,
        "summary": summary,
        "details": details
    }


def main(argv=None):
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Shodan Search Tool - Educational Internet device research",
        epilog="FOR AUTHORIZED SECURITY RESEARCH AND EDUCATIONAL PURPOSES ONLY"
    )

    parser.add_argument("query", nargs="?", help="Search query (e.g., 'apache', 'port:22', 'country:US')")
    parser.add_argument("--limit", type=int, default=50, help="Maximum results to return")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--health", action="store_true", help="Run health check")

    args = parser.parse_args(argv)

    # Health check
    if args.health:
        health = health_check()
        if args.json:
            print(json.dumps(health, indent=2))
        else:
            print(f"[{health['status'].upper()}] {health['summary']}")
            print(f"Details: {json.dumps(health['details'], indent=2)}")
        return 0

    # Require query
    if not args.query:
        parser.print_help()
        return 1

    # Educational disclaimer
    if not args.json:
        print("\n" + "="*80)
        print("EDUCATIONAL INTERNET RESEARCH TOOL")
        print("Simulates Shodan-style searches for learning purposes")
        print("FOR AUTHORIZED RESEARCH AND EDUCATIONAL USE ONLY")
        print("="*80 + "\n")

    # Search
    results = search(args.query, args.limit)

    if args.json:
        output = {
            "query": args.query,
            "total": len(results),
            "results": results
        }
        print(json.dumps(output, indent=2))
    else:
        if not results:
            print(f"No results found for query: {args.query}")
            return 0

        print(f"Found {len(results)} result(s):\n")

        for idx, result in enumerate(results, 1):
            print(f"Result #{idx}")
            print(f"  IP: {result['ip']}:{result['port']}")
            print(f"  Organization: {result['org']}")
            print(f"  Location: {result['city']}, {result['country']}")
            print(f"  Service: {result['product']} {result['version']}")
            print(f"  Banner: {result['banner']}")

            if result.get('vulns'):
                print(f"  Vulnerabilities: {', '.join(result['vulns'])}")

            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
