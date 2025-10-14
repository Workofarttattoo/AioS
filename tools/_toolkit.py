"""
Shared helpers for Sovereign Security Toolkit command-line utilities.

The helper abstractions here keep individual tool CLIs focused on their
domain-specific logic while still emitting consistent diagnostics and supporting
self-test flows leveraged by the runtime security health checks.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, asdict
from importlib import import_module
import sys
from typing import Dict, Iterable, List, Tuple


@dataclass
class Diagnostic:
  """Structured diagnostic payload returned by toolkit utilities."""

  status: str
  summary: str
  details: Dict[str, object]

  def as_dict(self) -> Dict[str, object]:
    payload = asdict(self)
    payload["timestamp"] = time.time()
    return payload


def emit_diagnostic(tool: str, diagnostic: Diagnostic, *, json_output: bool = False) -> None:
  """
  Emit a diagnostic payload in either human or JSON format.

  The wizard and runtime rely on JSON output to capture telemetry, while
  operators benefit from the concise textual layout.
  """

  payload = diagnostic.as_dict()
  payload["tool"] = tool
  if json_output:
    print(json.dumps(payload, indent=2))
    return
  print(f"[{diagnostic.status}] {tool}: {diagnostic.summary}")
  for key, value in payload["details"].items():
    print(f"  - {key}: {value}")


def synthesise_latency_samples(seed: str, *, count: int = 4) -> List[Tuple[str, float]]:
  """
  Produce deterministic pseudo-random latency samples based on a seed.

  The function keeps self-tests lightweight by avoiding real network or
  filesystem dependencies while still returning varied looking metrics.
  """

  random.seed(seed)
  samples: List[Tuple[str, float]] = []
  for idx in range(count):
    label = f"probe-{idx + 1}"
    value = round(random.uniform(12.5, 87.5), 2)
    samples.append((label, value))
  return samples


def summarise_samples(samples: Iterable[Tuple[str, float]]) -> Dict[str, object]:
  values = [latency for _, latency in samples]
  if not values:
    return {"count": 0, "avg": 0.0, "p95": 0.0}
  values_sorted = sorted(values)
  count = len(values_sorted)
  average = sum(values_sorted) / count
  percentile_index = min(count - 1, max(0, int(round(0.95 * count)) - 1))
  return {
    "count": count,
    "avg": round(average, 2),
    "p95": values_sorted[percentile_index],
    "min": values_sorted[0],
    "max": values_sorted[-1],
  }


def build_health_report(tool: str, status: str, summary: str, details: Dict[str, object]) -> Dict[str, object]:
  """
  Compose a standardised health report dictionary for Sovereign tools.

  Using this helper keeps runtime health telemetry consistent and ensures the
  payload mirrors what CLI diagnostics emit.
  """

  payload = Diagnostic(status=status, summary=summary, details=details).as_dict()
  payload["tool"] = tool
  return payload


def launch_gui(module_path: str, *, launcher: str = "launch") -> int:
  """
  Import and launch a toolkit GUI, emitting a friendly message if Tk is unavailable.
  """

  try:
    module = import_module(module_path)
  except ModuleNotFoundError as exc:
    missing = exc.name or ""
    if missing in {"tkinter", "_tkinter"}:
      print(
        "GUI unavailable: Tkinter is not installed on this system. "
        "Run without --gui or install tkinter support.",
        file=sys.stderr
      )
      return 1
    raise

  entrypoint = getattr(module, launcher, None)
  if not callable(entrypoint):
    print(f"GUI module '{module_path}' does not expose callable '{launcher}()'.", file=sys.stderr)
    return 1

  entrypoint()
  return 0
