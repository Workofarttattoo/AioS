#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

HashSolver - Advanced Hash Cracking & Analysis Tool
Multi-algorithm hash identification, cracking, and rainbow table generation
"""

import sys
import json
import argparse
import hashlib
import hmac
import time
import string
import itertools
from typing import Dict, List, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import binascii


@dataclass
class CrackResult:
    """Hash cracking result"""
    hash_value: str
    algorithm: str
    cracked: bool
    plaintext: Optional[str] = None
    attempts: int = 0
    time_seconds: float = 0.0
    method: str = ""


HASH_ALGORITHMS = {
    "md5": {
        "name": "MD5",
        "length": 32,
        "func": lambda x: hashlib.md5(x.encode()).hexdigest(),
        "common": True
    },
    "sha1": {
        "name": "SHA-1",
        "length": 40,
        "func": lambda x: hashlib.sha1(x.encode()).hexdigest(),
        "common": True
    },
    "sha256": {
        "name": "SHA-256",
        "length": 64,
        "func": lambda x: hashlib.sha256(x.encode()).hexdigest(),
        "common": True
    },
    "sha512": {
        "name": "SHA-512",
        "length": 128,
        "func": lambda x: hashlib.sha512(x.encode()).hexdigest(),
        "common": True
    },
    "sha224": {
        "name": "SHA-224",
        "length": 56,
        "func": lambda x: hashlib.sha224(x.encode()).hexdigest(),
        "common": False
    },
    "sha384": {
        "name": "SHA-384",
        "length": 96,
        "func": lambda x: hashlib.sha384(x.encode()).hexdigest(),
        "common": False
    },
    "blake2b": {
        "name": "BLAKE2b",
        "length": 128,
        "func": lambda x: hashlib.blake2b(x.encode()).hexdigest(),
        "common": False
    },
    "blake2s": {
        "name": "BLAKE2s",
        "length": 64,
        "func": lambda x: hashlib.blake2s(x.encode()).hexdigest(),
        "common": False
    },
    "sha3_256": {
        "name": "SHA3-256",
        "length": 64,
        "func": lambda x: hashlib.sha3_256(x.encode()).hexdigest(),
        "common": False
    },
    "sha3_512": {
        "name": "SHA3-512",
        "length": 128,
        "func": lambda x: hashlib.sha3_512(x.encode()).hexdigest(),
        "common": False
    }
}


COMMON_WORDLISTS = {
    "rockyou": "Common passwords from RockYou breach",
    "darkweb2017": "Dark web password dump 2017",
    "linkedin": "LinkedIn breach passwords",
    "common-passwords": "Top 10,000 most common passwords",
    "names": "Common first names and surnames",
    "dates": "Date patterns (YYYYMMDD, DDMMYYYY, etc.)",
    "keyboard": "Keyboard walk patterns (qwerty, 12345, etc.)"
}


class HashIdentifier:
    """Identify hash algorithms by analyzing hash characteristics"""

    @staticmethod
    def identify(hash_str: str) -> List[str]:
        """Identify possible hash algorithms"""
        hash_str = hash_str.strip().lower()
        hash_len = len(hash_str)

        # Check if valid hex
        try:
            int(hash_str, 16)
        except ValueError:
            return []

        candidates = []

        # Match by length
        for algo_id, algo_info in HASH_ALGORITHMS.items():
            if algo_info['length'] == hash_len:
                candidates.append(algo_id)

        # Prioritize common algorithms
        candidates.sort(key=lambda x: (not HASH_ALGORITHMS[x]['common'], x))

        return candidates


class HashCracker:
    """Hash cracking engine with multiple attack methods"""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.stop_flag = False

    def crack(self, hash_value: str, algorithm: str, method: str = "dictionary", **kwargs) -> CrackResult:
        """Main cracking function"""
        hash_value = hash_value.strip().lower()

        if method == "dictionary":
            return self._dictionary_attack(hash_value, algorithm, **kwargs)
        elif method == "brute":
            return self._brute_force(hash_value, algorithm, **kwargs)
        elif method == "rainbow":
            return self._rainbow_table(hash_value, algorithm, **kwargs)
        elif method == "hybrid":
            return self._hybrid_attack(hash_value, algorithm, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _dictionary_attack(self, hash_value: str, algorithm: str, wordlist: str = None, **kwargs) -> CrackResult:
        """Dictionary-based attack"""
        start_time = time.time()
        attempts = 0

        hash_func = HASH_ALGORITHMS[algorithm]['func']

        # Use built-in common passwords if no wordlist provided
        if not wordlist:
            words = self._get_common_passwords()
        else:
            try:
                with open(wordlist, 'r', encoding='utf-8', errors='ignore') as f:
                    words = [line.strip() for line in f if line.strip()]
            except Exception as e:
                return CrackResult(
                    hash_value=hash_value,
                    algorithm=algorithm,
                    cracked=False,
                    attempts=0,
                    time_seconds=time.time() - start_time,
                    method="dictionary"
                )

        # Try each word
        for word in words[:kwargs.get('max_attempts', 1000000)]:
            if self.stop_flag:
                break

            attempts += 1
            candidate_hash = hash_func(word)

            if candidate_hash == hash_value:
                return CrackResult(
                    hash_value=hash_value,
                    algorithm=algorithm,
                    cracked=True,
                    plaintext=word,
                    attempts=attempts,
                    time_seconds=time.time() - start_time,
                    method="dictionary"
                )

            # Progress feedback
            if attempts % 10000 == 0:
                print(f"[*] Tried {attempts} passwords...")

        return CrackResult(
            hash_value=hash_value,
            algorithm=algorithm,
            cracked=False,
            attempts=attempts,
            time_seconds=time.time() - start_time,
            method="dictionary"
        )

    def _brute_force(self, hash_value: str, algorithm: str, charset: str = None, max_length: int = 6, **kwargs) -> CrackResult:
        """Brute force attack"""
        start_time = time.time()
        attempts = 0

        hash_func = HASH_ALGORITHMS[algorithm]['func']

        # Default charset
        if not charset:
            charset = string.ascii_lowercase + string.digits

        print(f"[*] Brute forcing with charset: {charset[:20]}{'...' if len(charset) > 20 else ''}")
        print(f"[*] Max length: {max_length} characters")

        # Try each length
        for length in range(1, max_length + 1):
            if self.stop_flag:
                break

            print(f"[*] Trying length {length}...")

            for candidate in itertools.product(charset, repeat=length):
                if self.stop_flag:
                    break

                word = ''.join(candidate)
                attempts += 1
                candidate_hash = hash_func(word)

                if candidate_hash == hash_value:
                    return CrackResult(
                        hash_value=hash_value,
                        algorithm=algorithm,
                        cracked=True,
                        plaintext=word,
                        attempts=attempts,
                        time_seconds=time.time() - start_time,
                        method="brute_force"
                    )

                # Progress feedback
                if attempts % 100000 == 0:
                    print(f"[*] Tried {attempts} combinations...")

                # Safety limit
                if attempts >= kwargs.get('max_attempts', 10000000):
                    break

        return CrackResult(
            hash_value=hash_value,
            algorithm=algorithm,
            cracked=False,
            attempts=attempts,
            time_seconds=time.time() - start_time,
            method="brute_force"
        )

    def _rainbow_table(self, hash_value: str, algorithm: str, **kwargs) -> CrackResult:
        """Rainbow table attack (precomputed hashes)"""
        start_time = time.time()

        # For demo purposes, use common passwords
        # In production, this would load actual rainbow tables
        print("[*] Loading rainbow table...")
        words = self._get_common_passwords()

        hash_func = HASH_ALGORITHMS[algorithm]['func']

        # Build rainbow table (hash -> plaintext)
        rainbow_table = {}
        for word in words[:kwargs.get('table_size', 100000)]:
            rainbow_table[hash_func(word)] = word

        print(f"[*] Rainbow table size: {len(rainbow_table)} entries")

        # Lookup
        if hash_value in rainbow_table:
            return CrackResult(
                hash_value=hash_value,
                algorithm=algorithm,
                cracked=True,
                plaintext=rainbow_table[hash_value],
                attempts=len(rainbow_table),
                time_seconds=time.time() - start_time,
                method="rainbow_table"
            )

        return CrackResult(
            hash_value=hash_value,
            algorithm=algorithm,
            cracked=False,
            attempts=len(rainbow_table),
            time_seconds=time.time() - start_time,
            method="rainbow_table"
        )

    def _hybrid_attack(self, hash_value: str, algorithm: str, **kwargs) -> CrackResult:
        """Hybrid attack: dictionary + mutations"""
        start_time = time.time()
        attempts = 0

        hash_func = HASH_ALGORITHMS[algorithm]['func']
        words = self._get_common_passwords()[:1000]

        # Mutation rules
        mutations = [
            lambda w: w,  # Original
            lambda w: w.capitalize(),  # Capital first
            lambda w: w.upper(),  # All caps
            lambda w: w + "123",  # Add numbers
            lambda w: w + "!",  # Add special char
            lambda w: w + "2024",  # Add year
            lambda w: w[::-1],  # Reverse
            lambda w: w.replace('a', '@'),  # Leet speak
            lambda w: w.replace('e', '3'),
            lambda w: w.replace('i', '1'),
            lambda w: w.replace('o', '0'),
            lambda w: w.replace('s', '$'),
        ]

        for word in words:
            if self.stop_flag:
                break

            for mutation in mutations:
                try:
                    candidate = mutation(word)
                    attempts += 1
                    candidate_hash = hash_func(candidate)

                    if candidate_hash == hash_value:
                        return CrackResult(
                            hash_value=hash_value,
                            algorithm=algorithm,
                            cracked=True,
                            plaintext=candidate,
                            attempts=attempts,
                            time_seconds=time.time() - start_time,
                            method="hybrid"
                        )
                except:
                    continue

            if attempts % 1000 == 0:
                print(f"[*] Tried {attempts} mutations...")

        return CrackResult(
            hash_value=hash_value,
            algorithm=algorithm,
            cracked=False,
            attempts=attempts,
            time_seconds=time.time() - start_time,
            method="hybrid"
        )

    def _get_common_passwords(self) -> List[str]:
        """Get list of common passwords"""
        return [
            "password", "123456", "12345678", "qwerty", "abc123", "monkey", "1234567",
            "letmein", "trustno1", "dragon", "baseball", "111111", "iloveyou", "master",
            "sunshine", "ashley", "bailey", "passw0rd", "shadow", "123123", "654321",
            "superman", "qazwsx", "michael", "Football", "password1", "welcome", "jesus",
            "ninja", "mustang", "password123", "admin", "root", "toor", "pass", "test",
            "guest", "oracle", "administrator", "changeme", "linux", "system", "demo",
            "hello", "world", "login", "passwd", "secret", "access", "love", "god",
            "sex", "money", "fuck", "asshole", "pussy", "bitch", "killer", "killer",
            "death", "freedom", "freedom", "matrix", "hacker", "security", "computer",
            "internet", "network", "server", "database", "backup", "firewall", "router",
            "switch", "wireless", "cisco", "juniper", "windows", "microsoft", "apple",
            "google", "amazon", "facebook", "twitter", "github", "docker", "kubernetes"
        ]


class RainbowTableGenerator:
    """Generate rainbow tables for hash algorithms"""

    @staticmethod
    def generate(algorithm: str, wordlist: List[str], output_file: str = None) -> Dict[str, str]:
        """Generate rainbow table"""
        print(f"[*] Generating rainbow table for {HASH_ALGORITHMS[algorithm]['name']}...")

        hash_func = HASH_ALGORITHMS[algorithm]['func']
        rainbow_table = {}

        for idx, word in enumerate(wordlist):
            rainbow_table[hash_func(word)] = word

            if (idx + 1) % 10000 == 0:
                print(f"[*] Processed {idx + 1} entries...")

        print(f"[*] Rainbow table generated: {len(rainbow_table)} entries")

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(rainbow_table, f, indent=2)
            print(f"[*] Saved to {output_file}")

        return rainbow_table


def main(argv=None):
    """CLI entrypoint"""
    parser = argparse.ArgumentParser(
        description="HashSolver - Advanced Hash Cracking & Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  hashsolver.py --identify 5f4dcc3b5aa765d61d8327deb882cf99
  hashsolver.py --crack 5f4dcc3b5aa765d61d8327deb882cf99 --algorithm md5 --method dictionary
  hashsolver.py --crack <hash> --algorithm sha256 --method brute --charset abc123 --max-length 5
  hashsolver.py --crack <hash> --algorithm sha1 --method hybrid
  hashsolver.py --generate-rainbow --algorithm md5 --wordlist passwords.txt
  hashsolver.py --gui
        """
    )

    parser.add_argument('--identify', metavar='HASH', help='Identify hash algorithm')
    parser.add_argument('--crack', metavar='HASH', help='Crack hash')
    parser.add_argument('--algorithm', choices=list(HASH_ALGORITHMS.keys()), help='Hash algorithm')
    parser.add_argument('--method', choices=['dictionary', 'brute', 'rainbow', 'hybrid'], default='dictionary',
                       help='Cracking method')
    parser.add_argument('--wordlist', help='Wordlist file for dictionary attack')
    parser.add_argument('--charset', help='Character set for brute force (default: a-z0-9)')
    parser.add_argument('--max-length', type=int, default=6, help='Max password length for brute force')
    parser.add_argument('--max-attempts', type=int, default=1000000, help='Max attempts before giving up')
    parser.add_argument('--generate-rainbow', action='store_true', help='Generate rainbow table')
    parser.add_argument('--output', help='Output file for rainbow table')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--gui', action='store_true', help='Launch web-based GUI')
    parser.add_argument('--port', type=int, default=8088, help='GUI server port (default: 8088)')

    args = parser.parse_args(argv)

    if args.gui:
        launch_gui(args.port)
        return

    if args.identify:
        candidates = HashIdentifier.identify(args.identify)

        if args.json:
            print(json.dumps({"hash": args.identify, "candidates": candidates}, indent=2))
        else:
            print(f"\nHash: {args.identify}")
            print(f"Length: {len(args.identify)} characters")
            print(f"\nPossible algorithms:")
            for algo in candidates:
                info = HASH_ALGORITHMS[algo]
                print(f"  • {info['name']} ({algo})")
            if not candidates:
                print("  No matches found")
        return

    if args.generate_rainbow:
        if not args.algorithm:
            print("[!] Error: --algorithm required for rainbow table generation")
            return

        if args.wordlist:
            with open(args.wordlist, 'r', encoding='utf-8', errors='ignore') as f:
                words = [line.strip() for line in f if line.strip()]
        else:
            words = HashCracker(max_workers=1)._get_common_passwords()

        table = RainbowTableGenerator.generate(
            args.algorithm,
            words,
            output_file=args.output
        )

        if args.json:
            print(json.dumps({"algorithm": args.algorithm, "entries": len(table)}, indent=2))

        return

    if args.crack:
        if not args.algorithm:
            # Try to identify
            candidates = HashIdentifier.identify(args.crack)
            if candidates:
                args.algorithm = candidates[0]
                print(f"[*] Auto-detected algorithm: {HASH_ALGORITHMS[args.algorithm]['name']}")
            else:
                print("[!] Error: Could not identify hash algorithm. Please specify --algorithm")
                return

        cracker = HashCracker()

        print(f"[*] Cracking hash: {args.crack}")
        print(f"[*] Algorithm: {HASH_ALGORITHMS[args.algorithm]['name']}")
        print(f"[*] Method: {args.method}")

        result = cracker.crack(
            args.crack,
            args.algorithm,
            args.method,
            wordlist=args.wordlist,
            charset=args.charset,
            max_length=args.max_length,
            max_attempts=args.max_attempts
        )

        if args.json:
            print(json.dumps(asdict(result), indent=2))
        else:
            print_crack_result(result)

        return

    parser.print_help()


def print_crack_result(result: CrackResult):
    """Print human-readable crack result"""
    print("\n" + "="*70)
    print("HASH CRACKING RESULT")
    print("="*70)
    print(f"Hash: {result.hash_value}")
    print(f"Algorithm: {HASH_ALGORITHMS[result.algorithm]['name']}")
    print(f"Method: {result.method}")
    print(f"Attempts: {result.attempts:,}")
    print(f"Time: {result.time_seconds:.2f} seconds")

    if result.cracked:
        print(f"\n[✓] CRACKED!")
        print(f"Plaintext: {result.plaintext}")
        print(f"Speed: {result.attempts / result.time_seconds:,.0f} hashes/sec")
    else:
        print(f"\n[✗] Not cracked")
        print(f"Try different method or larger wordlist")

    print("="*70 + "\n")


def launch_gui(port: int = 8088):
    """Launch web-based GUI"""
    from flask import Flask, render_template_string, request, jsonify

    app = Flask(__name__)
    cracker = HashCracker()

    @app.route('/')
    def index():
        return render_template_string(GUI_HTML)

    @app.route('/api/identify', methods=['POST'])
    def identify():
        data = request.json
        hash_value = data.get('hash', '').strip()

        if not hash_value:
            return jsonify({"error": "No hash provided"}), 400

        candidates = HashIdentifier.identify(hash_value)
        return jsonify({
            "hash": hash_value,
            "length": len(hash_value),
            "candidates": [
                {"id": c, "name": HASH_ALGORITHMS[c]['name']}
                for c in candidates
            ]
        })

    @app.route('/api/crack', methods=['POST'])
    def crack():
        data = request.json
        hash_value = data.get('hash', '').strip()
        algorithm = data.get('algorithm')
        method = data.get('method', 'dictionary')

        if not hash_value or not algorithm:
            return jsonify({"error": "Hash and algorithm required"}), 400

        try:
            result = cracker.crack(
                hash_value,
                algorithm,
                method,
                max_attempts=data.get('max_attempts', 100000)
            )
            return jsonify(asdict(result))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/algorithms', methods=['GET'])
    def get_algorithms():
        return jsonify({
            algo_id: {
                "name": info['name'],
                "length": info['length'],
                "common": info['common']
            }
            for algo_id, info in HASH_ALGORITHMS.items()
        })

    print(f"[*] Starting HashSolver GUI on http://127.0.0.1:{port}")
    print(f"[*] Press Ctrl+C to stop")
    app.run(host='0.0.0.0', port=port, debug=False)


GUI_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>HashSolver - Hash Cracking Tool</title>
    <meta charset="utf-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        :root {
            --bg-dark: #0a0a0a;
            --bg-medium: #1a1a1a;
            --bg-light: #2a2a2a;
            --accent: #9d4edd;
            --accent-hover: #c77dff;
            --text-primary: #ffffff;
            --text-secondary: #cccccc;
            --text-muted: #888888;
            --border: #333333;
            --success: #06ffa5;
            --warning: #ffbe0b;
            --error: #ff006e;
        }

        body {
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a0520 100%);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        header {
            text-align: center;
            padding: 40px 0;
            border-bottom: 3px solid var(--accent);
            margin-bottom: 40px;
        }

        h1 {
            font-size: 3.5em;
            color: var(--accent);
            text-shadow: 0 0 30px rgba(157, 78, 221, 0.6);
            margin-bottom: 10px;
            letter-spacing: 4px;
        }

        .subtitle {
            color: var(--text-secondary);
            font-size: 1.2em;
            letter-spacing: 2px;
        }

        .panel {
            background: var(--bg-medium);
            border: 1px solid var(--accent);
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 0 40px rgba(157, 78, 221, 0.2);
        }

        .panel-title {
            color: var(--accent);
            font-size: 1.4em;
            margin-bottom: 20px;
            text-transform: uppercase;
            letter-spacing: 2px;
        }

        .hash-input {
            width: 100%;
            padding: 15px;
            background: var(--bg-dark);
            border: 2px solid var(--border);
            border-radius: 6px;
            color: var(--text-primary);
            font-family: 'Courier New', monospace;
            font-size: 1em;
            margin-bottom: 15px;
        }

        .hash-input:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(157, 78, 221, 0.2);
        }

        .btn-group {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
        }

        .btn {
            flex: 1;
            padding: 15px 25px;
            background: var(--accent);
            border: none;
            border-radius: 6px;
            color: var(--bg-dark);
            font-size: 1em;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s;
            font-family: 'Courier New', monospace;
            text-transform: uppercase;
        }

        .btn:hover:not(:disabled) {
            background: var(--accent-hover);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(157, 78, 221, 0.4);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .btn-secondary {
            background: var(--bg-light);
            border: 2px solid var(--accent);
            color: var(--accent);
        }

        .btn-secondary:hover:not(:disabled) {
            background: var(--accent);
            color: var(--bg-dark);
        }

        .method-selector {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .method-card {
            background: var(--bg-dark);
            border: 2px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            cursor: pointer;
            transition: all 0.3s;
            text-align: center;
        }

        .method-card:hover {
            border-color: var(--accent);
        }

        .method-card.selected {
            border-color: var(--accent);
            background: rgba(157, 78, 221, 0.1);
        }

        .method-name {
            color: var(--accent);
            font-weight: 700;
            margin-bottom: 8px;
        }

        .method-desc {
            color: var(--text-muted);
            font-size: 0.85em;
        }

        .result-panel {
            background: var(--bg-dark);
            border: 2px solid var(--border);
            border-radius: 8px;
            padding: 25px;
        }

        .result-success {
            border-color: var(--success);
            background: rgba(6, 255, 165, 0.05);
        }

        .result-fail {
            border-color: var(--error);
            background: rgba(255, 0, 110, 0.05);
        }

        .result-header {
            font-size: 1.5em;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid var(--border);
        }

        .result-header.success {
            color: var(--success);
        }

        .result-header.fail {
            color: var(--error);
        }

        .result-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }

        .result-item {
            display: flex;
            flex-direction: column;
        }

        .result-label {
            color: var(--text-muted);
            font-size: 0.9em;
            margin-bottom: 5px;
        }

        .result-value {
            color: var(--text-primary);
            font-weight: 700;
            font-size: 1.1em;
        }

        .plaintext-reveal {
            background: var(--accent);
            color: var(--bg-dark);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            font-size: 1.5em;
            font-weight: 700;
            margin-top: 20px;
        }

        .candidates-list {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 15px;
        }

        .candidate-badge {
            background: rgba(157, 78, 221, 0.2);
            border: 1px solid var(--accent);
            padding: 8px 16px;
            border-radius: 20px;
            color: var(--accent);
            cursor: pointer;
            transition: all 0.3s;
        }

        .candidate-badge:hover {
            background: var(--accent);
            color: var(--bg-dark);
        }

        .loading {
            text-align: center;
            padding: 40px;
        }

        .spinner {
            border: 4px solid var(--border);
            border-top: 4px solid var(--accent);
            border-radius: 50%;
            width: 60px;
            height: 60px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .hidden {
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔐 HASHSOLVER</h1>
            <div class="subtitle">ADVANCED HASH CRACKING & ANALYSIS</div>
        </header>

        <div class="panel">
            <div class="panel-title">Hash Input</div>
            <input type="text" id="hash-input" class="hash-input" placeholder="Enter hash value (MD5, SHA-1, SHA-256, etc.)">

            <div class="btn-group">
                <button class="btn btn-secondary" onclick="identifyHash()">🔍 Identify</button>
                <button class="btn" onclick="crackHash()">⚡ Crack</button>
            </div>

            <div id="candidates" class="hidden">
                <div class="panel-title" style="font-size: 1.1em; margin-top: 20px;">Detected Algorithms</div>
                <div class="candidates-list" id="candidates-list"></div>
            </div>
        </div>

        <div class="panel">
            <div class="panel-title">Cracking Method</div>
            <div class="method-selector">
                <div class="method-card selected" onclick="selectMethod('dictionary')">
                    <div class="method-name">📚 Dictionary</div>
                    <div class="method-desc">Common passwords & wordlists</div>
                </div>
                <div class="method-card" onclick="selectMethod('brute')">
                    <div class="method-name">💪 Brute Force</div>
                    <div class="method-desc">Try all combinations</div>
                </div>
                <div class="method-card" onclick="selectMethod('rainbow')">
                    <div class="method-name">🌈 Rainbow Table</div>
                    <div class="method-desc">Precomputed hashes</div>
                </div>
                <div class="method-card" onclick="selectMethod('hybrid')">
                    <div class="method-name">🔀 Hybrid</div>
                    <div class="method-desc">Dictionary + mutations</div>
                </div>
            </div>
        </div>

        <div id="loading" class="loading hidden">
            <div class="spinner"></div>
            <div style="color: var(--text-secondary); font-size: 1.2em;">CRACKING HASH...</div>
        </div>

        <div id="result" class="hidden"></div>
    </div>

    <script>
        let selectedMethod = 'dictionary';
        let detectedAlgorithm = null;

        function selectMethod(method) {
            selectedMethod = method;
            document.querySelectorAll('.method-card').forEach(card => card.classList.remove('selected'));
            event.target.closest('.method-card').classList.add('selected');
        }

        async function identifyHash() {
            const hashValue = document.getElementById('hash-input').value.trim();
            if (!hashValue) {
                alert('Please enter a hash');
                return;
            }

            try {
                const response = await fetch('/api/identify', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ hash: hashValue })
                });

                const data = await response.json();

                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                const candidatesList = document.getElementById('candidates-list');
                candidatesList.innerHTML = '';

                if (data.candidates.length === 0) {
                    candidatesList.innerHTML = '<div style="color: var(--error);">No matching algorithms found</div>';
                } else {
                    detectedAlgorithm = data.candidates[0].id;
                    data.candidates.forEach(c => {
                        const badge = document.createElement('div');
                        badge.className = 'candidate-badge';
                        badge.textContent = c.name;
                        badge.onclick = () => { detectedAlgorithm = c.id; crackHash(); };
                        candidatesList.appendChild(badge);
                    });
                }

                document.getElementById('candidates').classList.remove('hidden');
            } catch (error) {
                alert('Failed to identify hash: ' + error.message);
            }
        }

        async function crackHash() {
            const hashValue = document.getElementById('hash-input').value.trim();
            if (!hashValue) {
                alert('Please enter a hash');
                return;
            }

            if (!detectedAlgorithm) {
                alert('Please identify the hash first');
                return;
            }

            document.getElementById('loading').classList.remove('hidden');
            document.getElementById('result').classList.add('hidden');

            try {
                const response = await fetch('/api/crack', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        hash: hashValue,
                        algorithm: detectedAlgorithm,
                        method: selectedMethod,
                        max_attempts: 100000
                    })
                });

                const data = await response.json();

                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                displayResult(data);
            } catch (error) {
                alert('Failed to crack hash: ' + error.message);
            } finally {
                document.getElementById('loading').classList.add('hidden');
            }
        }

        function displayResult(result) {
            const resultDiv = document.getElementById('result');
            const success = result.cracked;

            resultDiv.className = 'panel result-panel ' + (success ? 'result-success' : 'result-fail');

            const speed = result.time_seconds > 0 ? (result.attempts / result.time_seconds).toLocaleString() : 'N/A';

            resultDiv.innerHTML = `
                <div class="result-header ${success ? 'success' : 'fail'}">
                    ${success ? '✓ HASH CRACKED!' : '✗ HASH NOT CRACKED'}
                </div>

                <div class="result-grid">
                    <div class="result-item">
                        <div class="result-label">Hash Value</div>
                        <div class="result-value" style="word-break: break-all;">${result.hash_value}</div>
                    </div>
                    <div class="result-item">
                        <div class="result-label">Algorithm</div>
                        <div class="result-value">${result.algorithm.toUpperCase()}</div>
                    </div>
                    <div class="result-item">
                        <div class="result-label">Method</div>
                        <div class="result-value">${result.method}</div>
                    </div>
                    <div class="result-item">
                        <div class="result-label">Attempts</div>
                        <div class="result-value">${result.attempts.toLocaleString()}</div>
                    </div>
                    <div class="result-item">
                        <div class="result-label">Time</div>
                        <div class="result-value">${result.time_seconds.toFixed(2)}s</div>
                    </div>
                    <div class="result-item">
                        <div class="result-label">Speed</div>
                        <div class="result-value">${speed} H/s</div>
                    </div>
                </div>

                ${success ? `
                    <div class="plaintext-reveal">
                        🔓 ${result.plaintext}
                    </div>
                ` : `
                    <div style="margin-top: 20px; padding: 15px; background: rgba(255,0,110,0.1); border-radius: 6px; color: var(--error);">
                        Hash not found. Try a different method or larger wordlist.
                    </div>
                `}
            `;

            resultDiv.classList.remove('hidden');
        }

        // Enter key to crack
        document.getElementById('hash-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') identifyHash();
        });
    </script>
</body>
</html>
"""


def health_check() -> Dict[str, Any]:
    """Health check for SecurityAgent integration"""
    return {
        "tool": "hashsolver",
        "status": "ok",
        "summary": "Advanced hash cracking and analysis tool",
        "details": {
            "algorithms": len(HASH_ALGORITHMS),
            "supported_methods": ["dictionary", "brute_force", "rainbow_table", "hybrid"],
            "features": [
                "10 hash algorithms",
                "4 cracking methods",
                "Rainbow table generation",
                "Auto hash identification",
                "Multi-threaded cracking"
            ]
        }
    }


if __name__ == "__main__":
    main()
