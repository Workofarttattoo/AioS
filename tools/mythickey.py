"""
MythicKey — credential resilience analyser.

The CLI inspects password hash inventories, attempts lightweight wordlist
replay rehearsal, and surfaces policy guidance without mutating real systems.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa, ec, dsa

from ._toolkit import (
  Diagnostic,
  build_health_report,
  emit_diagnostic,
  launch_gui,
  summarise_samples,
  synthesise_latency_samples,
)


TOOL_NAME = "MythicKey"

SAMPLE_HASHES: Sequence[str] = (
  hashlib.md5(b"changeme").hexdigest(),
  hashlib.sha1(b"P@ssw0rd").hexdigest(),
  hashlib.sha256(b"winter2024").hexdigest(),
)

SAMPLE_WORDS: Sequence[str] = ("changeme", "password", "winter", "winter2024", "welcome1", "P@ssw0rd")

ALGORITHM_BY_LENGTH = {
  32: "md5",
  40: "sha1",
  56: "sha224",
  64: "sha256",
  96: "sha384",
  128: "sha512",
}


@dataclass
class HashAssessment:
  digest: str
  algorithm: str
  cracked: bool
  plaintext: Optional[str]
  attempts: int

  def as_dict(self) -> Dict[str, object]:
    payload = asdict(self)
    payload["digest_prefix"] = self.digest[:12]
    return payload


@dataclass
class QuantumRiskAssessment:
    file_path: str
    algorithm: str
    key_size: int
    risk_level: str
    estimated_qubits_to_break: int
    notes: str

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="MythicKey credential resilience analyser.")
  parser.add_argument("--hashes", help="Path to newline-delimited password hashes.")
  parser.add_argument("--wordlist", help="Wordlist for rehearsal (one per line).")
  parser.add_argument("--keys", help="Path to a directory of keys/certificates to assess for quantum risk.")
  parser.add_argument("--profile", default="cpu", help="Processing profile hint (e.g. cpu, gpu-balanced).")
  parser.add_argument("--demo", action="store_true", help="Use built-in sample hashes and dictionary.")
  parser.add_argument("--json", action="store_true", help="Emit JSON findings.")
  parser.add_argument("--output", help="Write detailed JSON to path.")
  parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging.")
  parser.add_argument("--gui", action="store_true", help="Launch the MythicKey graphical interface.")
  return parser


def load_hashes(path: Optional[str], demo: bool) -> List[str]:
  hashes: List[str] = []
  if path:
    try:
      hashes.extend(line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())
    except FileNotFoundError:
      pass
  if demo or not hashes:
    hashes.extend(SAMPLE_HASHES)
  logging.debug("Loaded %d hash digest(s)", len(hashes))
  return hashes


def load_wordlist(path: Optional[str], demo: bool) -> List[str]:
  words: List[str] = []
  if path:
    try:
      words.extend(line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())
    except FileNotFoundError:
      pass
  if demo or not words:
    words.extend(SAMPLE_WORDS)
  unique = list(dict.fromkeys(words))
  logging.debug("Loaded %d word(s) after de-duplication", len(unique))
  return unique


def guess_algorithm(digest: str) -> str:
  algorithm = ALGORITHM_BY_LENGTH.get(len(digest), "unknown")
  if digest.startswith("{") and "}" in digest:
    return digest[1:digest.index("}")].lower()
  return algorithm


def estimate_quantum_threat(algorithm: str, key_size: int) -> Dict[str, any]:
    """Provides a rough estimate of resources to break a key with a quantum computer."""
    if algorithm.upper() == "RSA":
        if key_size >= 4096:
            risk = "Medium"
            qubits = 8192
        elif key_size >= 2048:
            risk = "High"
            qubits = 4096
        else:
            risk = "Critical"
            qubits = 2048
        return {"risk": risk, "qubits": qubits, "notes": "Vulnerable to Shor's algorithm."}
    elif algorithm.upper() in ["ECDSA", "ECDH", "EC"]:
        if key_size >= 384:
            risk = "Medium"
            qubits = 3072
        elif key_size >= 256:
            risk = "High"
            qubits = 2330
        else:
            risk = "Critical"
            qubits = 1500
        return {"risk": risk, "qubits": qubits, "notes": "Vulnerable to Shor's algorithm."}
    elif algorithm.upper() == "DSA":
        risk = "High"
        qubits = 2048 # Rough estimate
        return {"risk": risk, "qubits": qubits, "notes": "Vulnerable to Shor's algorithm."}

    return {"risk": "Unknown", "qubits": 0, "notes": "Algorithm is not a primary target for known quantum attacks or is not recognized."}

def analyze_key_file(file_path: Path) -> Optional[QuantumRiskAssessment]:
    """Analyzes a single key or certificate file for quantum risk."""
    try:
        pem_data = file_path.read_bytes()
        
        # Try loading as a certificate first
        try:
            cert = x509.load_pem_x509_certificate(pem_data)
            public_key = cert.public_key()
        except ValueError:
            # If not a cert, try as a private key
            try:
                private_key = serialization.load_pem_private_key(pem_data, password=None)
                public_key = private_key.public_key()
            except (ValueError, TypeError):
                # If not a private key, try as a public key
                public_key = serialization.load_pem_public_key(pem_data)

        if isinstance(public_key, rsa.RSAPublicKey):
            alg = "RSA"
            key_size = public_key.key_size
        elif isinstance(public_key, ec.EllipticCurvePublicKey):
            alg = "EC"
            key_size = public_key.key_size
        elif isinstance(public_key, dsa.DSAPublicKey):
            alg = "DSA"
            key_size = public_key.key_size
        else:
            return None

        threat_info = estimate_quantum_threat(alg, key_size)
        
        return QuantumRiskAssessment(
            file_path=str(file_path),
            algorithm=alg,
            key_size=key_size,
            risk_level=threat_info["risk"],
            estimated_qubits_to_break=threat_info["qubits"],
            notes=threat_info["notes"]
        )

    except Exception:
        # Broad exception to handle various parsing errors or unsupported formats
        return None

def evaluate_keys(scan_path: str) -> List[QuantumRiskAssessment]:
    """Scans a directory for key and certificate files and assesses their quantum risk."""
    assessments = []
    p = Path(scan_path)
    if not p.is_dir():
        return []
    
    key_files = list(p.glob("**/*.pem")) + list(p.glob("**/*.key")) + list(p.glob("**/*.crt"))
    for key_file in key_files:
        assessment = analyze_key_file(key_file)
        if assessment:
            assessments.append(assessment)
            
    return assessments


def mutate_word(word: str, profile: str) -> Iterable[str]:
  yield word
  if not word:
    return
  yield word.capitalize()
  yield word + "123"
  yield word + "!"
  if profile.startswith("gpu"):
    yield word[::-1]
    yield word + "@"
    yield word.replace("a", "@").replace("o", "0").replace("s", "$")


def hash_word(word: str, algorithm: str) -> Optional[str]:
  try:
    h = hashlib.new(algorithm)
  except ValueError:
    return None
  h.update(word.encode("utf-8"))
  return h.hexdigest()


def crack_hash(digest: str, algorithm: str, wordlist: Iterable[str], profile: str) -> HashAssessment:
  attempts = 0
  for word in wordlist:
    for candidate in mutate_word(word, profile):
      attempts += 1
      hashed = hash_word(candidate, algorithm)
      if hashed and hashed.lower() == digest.lower():
        return HashAssessment(digest=digest, algorithm=algorithm, cracked=True, plaintext=candidate, attempts=attempts)
  return HashAssessment(digest=digest, algorithm=algorithm, cracked=False, plaintext=None, attempts=attempts)


def evaluate_hashes(
  digests: Sequence[str],
  wordlist: Sequence[str],
  profile: str,
) -> Tuple[Diagnostic, Dict[str, object]]:
  assessments: List[HashAssessment] = []
  for digest in digests:
    algorithm = guess_algorithm(digest)
    assessments.append(crack_hash(digest, algorithm, wordlist, profile))

  cracked = [item for item in assessments if item.cracked]
  status = "info"
  summary = f"{len(cracked)} of {len(assessments)} hash(es) recovered during rehearsal."
  if cracked:
    status = "warn"
    summary = f"{len(cracked)} hash(es) matched dictionary candidates."

  diagnostic = Diagnostic(status=status, summary=summary, details={
    "hashes": len(assessments),
    "cracked": len(cracked),
    "profile": profile,
  })
  payload = {
    "tool": TOOL_NAME,
    "profile": profile,
    "hashes": [assessment.as_dict() for assessment in assessments],
  }
  logging.debug("Evaluated %d hashes with profile=%s; cracked=%d", len(assessments), profile, len(cracked))
  return diagnostic, payload


def run(args: argparse.Namespace) -> int:
  hashes = load_hashes(args.hashes, args.demo)
  wordlist = load_wordlist(args.wordlist, args.demo)
  profile = args.profile.lower()
  diagnostic, payload = evaluate_hashes(hashes, wordlist, profile)

  key_assessments = []
  if args.keys:
      key_assessments = evaluate_keys(args.keys)
      payload["quantum_risk_assessment"] = [ka.as_dict() for ka in key_assessments]
      logging.debug("Assessed %d key/certificate file(s) for quantum risk", len(key_assessments))

  if args.json and not args.output:
    print(json.dumps(payload, indent=2))
  else:
    emit_diagnostic(TOOL_NAME, diagnostic, json_output=False)
    if key_assessments:
        print("\n[info] Quantum Risk Assessment")
        print("-" * 40)
        for ka in key_assessments:
            print(f"  [RISK] {ka.risk_level}: {ka.algorithm} {ka.key_size}-bit key")
            print(f"  Location:  {ka.file_path}")
            print(f"  Est. Qubits to Break: {ka.estimated_qubits_to_break}")
            print(f"  Notes: {ka.notes}")
            print("-" * 40)

  if args.output:
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
  return 0


def health_check() -> Dict[str, object]:
  samples = synthesise_latency_samples(TOOL_NAME)
  metrics = summarise_samples(samples)
  details = {
    "hash_algorithms": sorted(set(ALGORITHM_BY_LENGTH.values())),
    "latency_profile": metrics,
    "sample_latency": [{"probe": label, "latency_ms": value} for label, value in samples],
  }
  return build_health_report(
    TOOL_NAME,
    "ok",
    "MythicKey dictionary rehearsal engine operational.",
    details,
  )


def main(argv: Optional[Sequence[str]] = None) -> int:
  args_list = list(argv or sys.argv[1:])
  if args_list and args_list[0].lower() == "crack":
    args_list = args_list[1:]
  parser = build_parser()
  args = parser.parse_args(args_list)
  level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
  logging.basicConfig(
    level=level,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
  )
  logging.getLogger("asyncio").setLevel(logging.WARNING)
  logging.debug("Parsed args: %s", vars(args))
  if getattr(args, "gui", False):
    return launch_gui("tools.mythickey_gui")
  return run(args)


if __name__ == "__main__":
  raise SystemExit(main())
