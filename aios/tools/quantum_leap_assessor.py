import argparse
import json
import os
import re
import time
from typing import List, Dict, Any, Set

VULNERABLE_ALGORITHMS = {
    "RSA": re.compile(r'\b(RSA)\b', re.IGNORECASE),
    "ECC": re.compile(r'\b(ECC|ECDSA|ECDH)\b', re.IGNORECASE),
    "DSA": re.compile(r'\b(DSA|Digital Signature Algorithm)\b', re.IGNORECASE),
    "Diffie-Hellman": re.compile(r'\b(Diffie-Hellman|DH)\b', re.IGNORECASE),
}

# Common file extensions to check for cryptographic material
SOURCE_CODE_EXTENSIONS = {'.py', '.java', '.c', '.cpp', '.h', '.js', '.ts', '.go', '.rs'}
CONFIG_EXTENSIONS = {'.json', '.yaml', '.yml', '.xml', '.conf', '.ini', '.toml'}
CERTIFICATE_EXTENSIONS = {'.pem', '.crt', '.cer', '.key'}
INTERESTING_EXTENSIONS = SOURCE_CODE_EXTENSIONS | CONFIG_EXTENSIONS | CERTIFICATE_EXTENSIONS

def scan_file(file_path: str) -> List[Dict[str, Any]]:
    """Scans a single file for vulnerable cryptographic algorithm keywords."""
    findings = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                for alg, pattern in VULNERABLE_ALGORITHMS.items():
                    if pattern.search(line):
                        findings.append({
                            "file_path": file_path,
                            "line": line_num,
                            "algorithm": alg,
                            "snippet": line.strip()
                        })
    except (IOError, OSError):
        # Ignore files we can't read (e.g., permission denied, binary files)
        pass
    return findings

def find_vulnerable_files(scan_path: str) -> List[Dict[str, Any]]:
    """Recursively finds and scans files for vulnerable cryptographic algorithms."""
    all_findings = []
    for root, _, files in os.walk(scan_path):
        for file in files:
            file_path = os.path.join(root, file)
            _, ext = os.path.splitext(file_path)
            if ext in INTERESTING_EXTENSIONS:
                findings = scan_file(file_path)
                if findings:
                    all_findings.extend(findings)
    return all_findings

def health_check() -> Dict[str, Any]:
    """Performs a self-check to ensure the tool is operational."""
    start_time = time.time()
    # In a real scenario, we might check for dependencies or permissions.
    # For now, we'll just return a success status.
    latency = (time.time() - start_time) * 1000
    return {
        "tool": "QuantumLeapAssessor",
        "status": "ok",
        "summary": "QuantumLeapAssessor is operational.",
        "details": {
            "version": "0.1.0",
            "dependencies": ["os", "re", "json"],
            "latency_ms": round(latency)
        }
    }

def main(argv: List[str] = None):
    """Main entry point for the QuantumLeapAssessor tool."""
    parser = argparse.ArgumentParser(description="QuantumLeapAssessor: Scan for quantum-vulnerable cryptography.")
    parser.add_argument("path", nargs="?", default=".", help="The directory or file path to scan. Defaults to the current directory.")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format.")
    
    args = parser.parse_args(argv)

    if not os.path.exists(args.path):
        print(f"[error] Path does not exist: {args.path}")
        return

    start_time = time.time()
    if os.path.isdir(args.path):
        findings = find_vulnerable_files(args.path)
    else:
        findings = scan_file(args.path)
    
    duration = time.time() - start_time

    report = {
        "scan_summary": {
            "target_path": args.path,
            "scan_duration_seconds": round(duration, 2),
            "total_findings": len(findings),
        },
        "findings": findings
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"[info] QuantumLeapAssessor Scan Report")
        print(f"[info] Target: {args.path}")
        print(f"[info] Completed in {duration:.2f} seconds.")
        print(f"[info] Found {len(findings)} potential vulnerabilities.")
        print("-" * 40)
        for finding in findings:
            print(f"  [VULNERABILITY] Algorithm: {finding['algorithm']}")
            print(f"  Location:  {finding['file_path']}:{finding['line']}")
            print(f"  Snippet:   {finding['snippet']}")
            print("-" * 40)

if __name__ == "__main__":
    main()


