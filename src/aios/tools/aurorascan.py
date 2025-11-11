import argparse
import asyncio
from typing import Optional, Sequence

from ._toolkit import launch_gui
from ._aurorascan_core import (
    TOOL_NAME,
    PORT_PROFILES,
    DEFAULT_TIMEOUT,
    DEFAULT_CONCURRENCY,
    DEFAULT_ZAP_SCHEME,
    iter_profiles,
    parse_ports,
    parse_targets,
    load_targets_from_file,
    run_scan,
    write_json,
    print_human,
    write_zap_targets,
    health_check,
)

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
  parser.add_argument("--gui", action="store_true", help="Launch the AuroraScan graphical interface.")
  return parser

def main(argv: Optional[Sequence[str]] = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)

  if args.list_profiles:
    display_profiles()
    return 0

  if getattr(args, "gui", False):
    return launch_gui("tools.aurorascan_gui")

  targets = []
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


