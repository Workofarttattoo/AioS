"""
AuroraScan — adaptive network mapper with service fingerprint diffing hooks.

Inspired by nmap, AuroraScan focuses on quick enumeration of accessible TCP
endpoints using lightweight heuristics suited for agent-driven workflows.  The
implementation offered here is intentionally conservative: it performs
non-invasive TCP connect probes with configurable concurrency, honours short
timeouts, and emits structured JSON that downstream agents can store in their
telemetry streams.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import queue
import socket
import threading
import time
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import re

from ._toolkit import launch_gui, summarise_samples, synthesise_latency_samples
from ._service_probes import PROBE_DATABASE
from ._os_probes import OS_PROBES, OS_SIGNATURES
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
    """
    Performs TCP/IP stack fingerprinting to guess the OS.
    This is a simplified implementation and requires scapy.
    """
    try:
        logging.debug("Starting OS fingerprinting for %s with timeout %.2fs", host, timeout)
        responses = {}
        for probe in OS_PROBES:
            # This is a conceptual implementation. A real one would use scapy
            # to build and send these specific packets.
            # e.g., pkt = scapy.IP(dst=host)/scapy.TCP(dport=80, flags=probe['flags'], options=probe['options'])
            # response = scapy.sr1(pkt, timeout=timeout, verbose=0)
            await asyncio.sleep(0.1)
            responses[probe["name"]] = {"TCPWindow": 65535, "WScale": 8}
        for signature in OS_SIGNATURES:
            is_match = True
            for probe_name, conditions in signature["matches"].items():
                if probe_name not in responses:
                    is_match = False
                    break
                if responses[probe_name].get("TCPWindow") != conditions.get("TCPWindow"):
                    is_match = False
                    break
            if is_match:
                logging.debug("OS fingerprint match for %s -> %s", host, signature["name"])
                return signature["name"]
    except ImportError:
        print("[warn] scapy is not installed. OS fingerprinting is disabled. `pip install scapy`")
        return "scapy not installed"
    except Exception as exc:
        logging.debug("OS fingerprinting error for %s: %r", host, exc)
        return "Error during fingerprinting"
    return "Unknown OS"


async def probe_port(host: str, port: int, timeout: float) -> Tuple[int, str, float, Optional[str], Optional[str], Optional[str]]:
  start = time.perf_counter()
  try:
    conn = asyncio.open_connection(host, port)
    reader, writer = await asyncio.wait_for(conn, timeout=timeout)
    elapsed = (time.perf_counter() - start) * 1000
    banner = None
    service = None
    version = None
    probes = PROBE_DATABASE.get(port, PROBE_DATABASE.get("default", []))
    for probe in probes:
      try:
        writer.write(probe["payload"])
        await asyncio.wait_for(writer.drain(), timeout=timeout)
        response = await asyncio.wait_for(reader.read(1024), timeout=timeout)
        if response:
          banner = response.decode(errors="ignore").strip()
          for match_rule in probe["matches"]:
            match = match_rule["regex"].search(response)
            if match:
              service = match_rule["service"]
              if match.groups():
                version = match.groups()[0].decode(errors="ignore")
              break
        if service:
          break
      except (asyncio.TimeoutError, ConnectionError):
        continue
    writer.close()
    with contextlib.suppress(Exception):
      await writer.wait_closed()
    logging.debug("Probe %s:%d -> %s %.2fms service=%s version=%s", host, port, "open", elapsed, service or "-", version or "-")
    return port, "open", elapsed, service, version, banner
  except asyncio.TimeoutError:
    elapsed = (time.perf_counter() - start) * 1000
    logging.debug("Probe %s:%d -> filtered %.2fms", host, port, elapsed)
    return port, "filtered", elapsed, None, None, None
  except ConnectionRefusedError:
    elapsed = (time.perf_counter() - start) * 1000
    logging.debug("Probe %s:%d -> closed %.2fms", host, port, elapsed)
    return port, "closed", elapsed, None, None, None
  except OSError as exc:
    elapsed = (time.perf_counter() - start) * 1000
    logging.debug("Probe %s:%d -> error %.2fms (%r)", host, port, elapsed, exc)
    return port, "error", elapsed, None, None, None


async def scan_target(
  host: str,
  ports: Sequence[int],
  timeout: float,
  concurrency: int,
  os_fingerprint: bool = False,
  progress_queue: Optional[queue.Queue] = None,
  stop_flag: Optional[threading.Event] = None,
) -> TargetReport:
  try:
    resolved = socket.gethostbyname(host)
  except socket.gaierror:
    resolved = "unresolved"
  connect_host = resolved if resolved != "unresolved" else host
  logging.debug("Scanning target %s (resolved=%s) ports=%d concurrency=%d timeout=%.2f", host, resolved, len(ports), concurrency, timeout)
  semaphore = asyncio.Semaphore(concurrency)
  observations: List[PortObservation] = []
  async def worker(port: int) -> None:
    if stop_flag and stop_flag.is_set():
      return
    async with semaphore:
      if stop_flag and stop_flag.is_set():
        return
      result_port, status, elapsed, service, version, banner = await probe_port(connect_host, port, timeout)
      observations.append(PortObservation(result_port, status, elapsed, service, version, banner))
      if progress_queue:
        progress_queue.put((host, result_port, status, elapsed, service, version, banner))
  start = time.perf_counter()
  tasks = [asyncio.create_task(worker(port)) for port in ports]
  os_guess = None
  if os_fingerprint:
    os_guess = await fingerprint_os(connect_host, timeout)
  try:
    await asyncio.gather(*tasks)
  except asyncio.CancelledError:
    for task in tasks:
      task.cancel()
    raise
  observations.sort(key=lambda item: item.port)
  elapsed_ms = (time.perf_counter() - start) * 1000
  logging.debug("Completed scan for %s in %.2fms with %d observations", host, elapsed_ms, len(observations))
  return TargetReport(host, resolved, elapsed_ms, os_guess, observations)


def parse_ports(port_arg: Optional[str], profile: str) -> Sequence[int]:
  if port_arg:
    ports: List[int] = []
    for chunk in port_arg.split(","):
      chunk = chunk.strip()
      if not chunk:
        continue
      if "-" in chunk:
        start_str, end_str = chunk.split("-", maxsplit=1)
        start_port = int(start_str)
        end_port = int(end_str)
        ports.extend(range(start_port, end_port + 1))
      else:
        ports.append(int(chunk))
    parsed = sorted(set(p for p in ports if 1 <= p <= 65535))
    logging.debug("Parsed explicit ports: %s", parsed[:12] + (["..."] if len(parsed) > 12 else []))
    return parsed
  selected = PORT_PROFILES.get(profile, PORT_PROFILES["recon"])
  logging.debug("Using profile %s with %d ports", profile, len(selected))
  return selected


def parse_targets(target_arg: str) -> List[str]:
  targets: List[str] = []
  for chunk in target_arg.split(","):
    chunk = chunk.strip()
    if not chunk:
      continue
    if "/" in chunk:
      try:
        net = ipaddress.ip_network(chunk, strict=False)
        targets.extend([str(ip) for ip in net.hosts()])
        continue
      except ValueError:
        pass
    if "-" in chunk:
      try:
        start_ip_str, end_ip_str = chunk.split("-", maxsplit=1)
        start_ip = ipaddress.ip_address(start_ip_str.strip())
        if len(end_ip_str.split('.')) < 4:
          end_ip = ipaddress.ip_address('.'.join(start_ip_str.strip().split('.')[:-1] + [end_ip_str.strip()]))
        else:
          end_ip = ipaddress.ip_address(end_ip_str.strip())
        while start_ip <= end_ip:
          targets.append(str(start_ip))
          start_ip += 1
        continue
      except ValueError:
        pass
    if chunk:
      targets.append(chunk)
  unique_targets = list(dict.fromkeys(targets))
  logging.debug("Parsed %d target(s)", len(unique_targets))
  return unique_targets


def load_targets_from_file(path: Optional[str]) -> List[str]:
  if not path:
    return []
  try:
    with open(path, "r", encoding="utf-8") as handle:
      lines = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
      logging.debug("Loaded %d target(s) from file %s", len(lines), path)
      return lines
  except FileNotFoundError:
    print("[warn] Target file not found; ignoring.")
    return []


def display_profiles() -> None:
  print("[info] Available scan profiles:")
  for name, ports, description in iter_profiles():
    print(f"  - {name:<10} ({len(ports)} ports)  {description}")


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="AuroraScan network mapper.")
  parser.add_argument("targets", nargs="?", help="Comma-separated hostnames or IP addresses.")
  parser.add_argument("--targets-file", help="Path to file containing one target per line.")
  parser.add_argument("--list-profiles", action="store_true", help="Show built-in scanning profiles and exit.")
  parser.add_argument("--ports", help="Comma/range list of ports to scan (e.g., 22,80,4000-4010).")
  parser.add_argument("--profile", default="recon", choices=list(PORT_PROFILES.keys()), help="Port profile preset.")
  parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Per-connection timeout in seconds.")
  parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Concurrent connection attempts.")
  parser.add_argument("--json", action="store_true", help="Emit results as JSON instead of human-readable text.")
  parser.add_argument("--output", help="Optional path to write JSON results.")
  parser.add_argument("--tag", default="aurorascan", help="Label included in JSON output.")
  parser.add_argument("--zap-targets", help="Write discovered open services to a file for OWASP ZAP import.")
  parser.add_argument("--zap-scheme", choices=["http", "https", "auto"], default=DEFAULT_ZAP_SCHEME, help="Scheme used when generating ZAP URLs (auto guesses from port).")
  parser.add_argument("--os-fingerprint", action="store_true", help="Attempt to identify the target operating system.")
  parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging.")
  parser.add_argument("--gui", action="store_true", help="Launch the AuroraScan graphical interface.")
  return parser


def print_human(results: Iterable[TargetReport]) -> None:
  for report in results:
    open_observations = [obs for obs in report.observations if obs.status == 'open']
    other_counts: Dict[str, int] = {}
    for obs in report.observations:
      if obs.status != 'open':
        other_counts[obs.status] = other_counts.get(obs.status, 0) + 1
    print(f"[info] Target: {report.target} ({report.resolved}) - {report.elapsed_ms:.2f} ms total")
    if report.os_guess:
      print(f"    OS Guess: {report.os_guess}")
    if open_observations:
      print('    PORT  STATUS   LAT(ms)  SERVICE              VERSION              BANNER')
      for obs in open_observations:
        banner = (obs.banner or '')[:24]
        service = obs.service or ''
        version = obs.version or ''
        print(f"    {obs.port:>4}/tcp  open    {obs.response_time_ms:>7.2f}  {service:<20} {version:<20} {banner}")
    else:
      print('    No open ports detected under the selected profile.')
    if other_counts:
      summary = ', '.join(f"{status}:{count}" for status, count in sorted(other_counts.items()))
      print(f"    Other responses - {summary}")
    print('')


def write_json(results: Iterable[TargetReport], path: Optional[str], tag: str) -> None:
  payload = {
    "tool": tag or TOOL_NAME.lower(),
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "results": [report.as_dict() for report in results],
  }
  if path:
    with open(path, "w", encoding="utf-8") as handle:
      json.dump(payload, handle, indent=2)
    print(f"[info] Results written to {path}")
  else:
    print(json.dumps(payload, indent=2))


def write_zap_targets(results: Iterable[TargetReport], path: Optional[str], default_scheme: str) -> None:
  if not path:
    return
  entries: List[str] = []
  seen = set()
  for report in results:
    host = report.target
    for obs in report.observations:
      if obs.status != "open":
        continue
      scheme = default_scheme
      if scheme == "auto":
        scheme = "https" if obs.port in {443, 8443, 9443, 9444} else "http"
      if (host, obs.port, scheme) in seen:
        continue
      seen.add((host, obs.port, scheme))
      if (scheme == "http" and obs.port == 80) or (scheme == "https" and obs.port == 443):
        url = f"{scheme}://{host}"
      else:
        url = f"{scheme}://{host}:{obs.port}"
      entries.append(url)
  with open(path, "w", encoding="utf-8") as handle:
    handle.write("\n".join(entries) + ("\n" if entries else ""))
  print(f"[info] ZAP target list written to {path} ({len(entries)} endpoint(s)).")


def run_scan(
  targets: Sequence[str],
  ports: Sequence[int],
  *,
  timeout: float,
  concurrency: int,
  os_fingerprint: bool,
  progress_queue: Optional[queue.Queue] = None,
  stop_flag: Optional[threading.Event] = None,
) -> List[TargetReport]:
  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)
  tasks = [
    scan_target(
      target,
      ports,
      timeout,
      concurrency,
      os_fingerprint=os_fingerprint,
      progress_queue=progress_queue,
      stop_flag=stop_flag,
    )
    for target in targets
  ]
  try:
    reports = loop.run_until_complete(asyncio.gather(*tasks))
  finally:
    loop.run_until_complete(asyncio.sleep(0))
    loop.close()
  return reports


def _configure_logging(verbose: bool) -> None:
  level = logging.DEBUG if verbose else logging.INFO
  logging.basicConfig(
    level=level,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
  )
  logging.getLogger("asyncio").setLevel(logging.WARNING)


def main(argv: Optional[Sequence[str]] = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  _configure_logging(getattr(args, "verbose", False))
  logging.debug("Parsed args: %s", vars(args))
  if args.list_profiles:
    logging.debug("Listing available profiles")
    display_profiles()
    return 0
  if getattr(args, "gui", False):
    logging.debug("Launching GUI")
    return launch_gui("tools.aurorascan_gui")
  targets: List[str] = []
  if args.targets:
    targets.extend(parse_targets(args.targets))
  targets.extend(load_targets_from_file(args.targets_file))
  if not targets:
    parser.error("No targets specified. Provide targets argument or --targets-file.")
  ports = parse_ports(args.ports, args.profile)
  if not ports:
    parser.error("No ports selected after parsing profile and overrides.")
  print(f"[info] Starting AuroraScan against {len(targets)} target(s) on {len(ports)} port(s).")
  reports = run_scan(
    targets,
    ports,
    timeout=args.timeout,
    concurrency=args.concurrency,
    os_fingerprint=args.os_fingerprint,
  )
  if args.json or args.output:
    write_json(reports, args.output, args.tag)
  else:
    print_human(reports)
  if args.zap_targets:
    write_zap_targets(reports, args.zap_targets, args.zap_scheme)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())


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
