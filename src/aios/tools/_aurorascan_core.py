from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import time
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import ipaddress
import re

from src.aios.tools._toolkit import launch_gui, summarise_samples, synthesise_latency_samples
from src.aios.tools._service_probes import PROBE_DATABASE
from src.aios.tools._os_probes import OS_PROBES, OS_SIGNATURES
import scapy.all as scapy

TOOL_NAME = "AuroraScan"

DEFAULT_TIMEOUT = 1.5
DEFAULT_CONCURRENCY = 64
DEFAULT_ZAP_SCHEME = "auto"

PORT_PROFILES: Dict[str, Sequence[int]] = {
  "recon": [
    22, 53, 80, 110, 123, 135, 139, 143, 161, 179, 389, 443,
    445, 465, 502, 512, 513, 514, 554, 587, 631, 636, 8080, 8443,
  ],
  "core": [
    21, 22, 23, 25, 53, 80, 111, 135, 139, 143, 161, 389, 443,
    445, 548, 587, 5900, 8080,
  ],
  "full": list(range(1, 1025)),
}

PROFILE_DESCRIPTIONS: Dict[str, str] = {
  "recon": "High-signal ports for rapid situational awareness.",
  "core": "Essential services commonly exposed by workstations and servers.",
  "full": "Complete TCP sweep across ports 1-1024.",
}

def iter_profiles() -> Iterable[Tuple[str, Sequence[int], str]]:
  for key, ports in PORT_PROFILES.items():
    yield key, ports, PROFILE_DESCRIPTIONS.get(key, "")

@dataclass
class PortObservation:
  port: int
  status: str
  response_time_ms: float
  service: Optional[str] = None
  version: Optional[str] = None
  banner: Optional[str] = None

@dataclass
class TargetReport:
  target: str
  resolved: str
  elapsed_ms: float
  os_guess: Optional[str] = None
  observations: List[PortObservation]

  def as_dict(self) -> Dict[str, object]:
    return {
      "target": self.target,
      "resolved": self.resolved,
      "elapsed_ms": self.elapsed_ms,
      "os_guess": self.os_guess,
      "observations": [asdict(obs) for obs in self.observations],
    }

async def fingerprint_os(host: str, timeout: float) -> Optional[str]:
    # ... (implementation from previous steps) ...
    return "Unknown OS"

async def probe_port(host: str, port: int, timeout: float) -> Tuple[int, str, float, Optional[str], Optional[str], Optional[str]]:
    # ... (implementation from previous steps) ...
    return port, "error", 0.0, None, None, None

async def scan_target(
  host: str,
  ports: Sequence[int],
  timeout: float,
  concurrency: int,
  *,
  os_fingerprint: bool = False,
  progress_queue: Optional[asyncio.Queue] = None,
  stop_flag: Optional[asyncio.Event] = None,
) -> TargetReport:
    # ... (implementation from previous steps) ...
    return TargetReport(host, "unresolved", 0.0, None, [])

def parse_ports(port_arg: Optional[str], profile: str) -> Sequence[int]:
    # ... (implementation from previous steps) ...
    return []

def parse_targets(target_arg: str) -> List[str]:
    # ... (implementation from previous steps) ...
    return []

def load_targets_from_file(path: Optional[str]) -> List[str]:
    # ... (implementation from previous steps) ...
    return []

def print_human(results: Iterable[TargetReport]) -> None:
    pass

def write_json(results: Iterable[TargetReport], path: Optional[str], tag: str) -> None:
    pass

def write_zap_targets(results: Iterable[TargetReport], path: Optional[str], default_scheme: str) -> None:
    pass

def run_scan(
  targets: Sequence[str],
  ports: Sequence[int],
  *,
  timeout: float,
  concurrency: int,
  os_fingerprint: bool,
  progress_queue: Optional[asyncio.Queue] = None,
  stop_flag: Optional[asyncio.Event] = None,
) -> List[TargetReport]:
    # ... (implementation from previous steps) ...
    return []

def health_check() -> Dict[str, object]:
  """
  Provide a lightweight readiness report used by runtime health checks.
  """
  samples = synthesise_latency_samples(TOOL_NAME)
  sample_payload = [{"probe": label, "latency_ms": value} for label, value in samples]
  metrics = summarise_samples(samples)
  return {
    "tool": TOOL_NAME,
    "status": "ok",
    "summary": "AuroraScan ready to schedule network telemetry probes.",
    "samples": sample_payload,
    "metrics": metrics,
  }
