#!/usr/bin/env python3
"""
Lightweight watcher for Ai:oS shell processes.

Continuously scans the host process table for commands that include the target
pattern (defaults to `aios_shell.py`) and logs when matching processes appear
or exit.  Designed for operational awareness without needing full supervisor
integration.
"""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict

try:  # Prefer psutil when available (better Windows support).
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore

PSUTIL_USABLE = psutil is not None
PSUTIL_WARNING_EMITTED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor the host for active aios-shell processes."
    )
    parser.add_argument(
        "--pattern",
        default="aios_shell.py",
        help="Substring to match against running commands (default: %(default)s).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Polling interval in seconds between checks (default: %(default)s).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Optional log file to append monitor events.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Take a single snapshot and exit with code 0 if matches are found.",
    )
    return parser.parse_args()


def emit(log_handle, level: str, message: str) -> None:
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{level}] {timestamp} {message}"
    print(line)
    if log_handle is not None:
        log_handle.write(line + "\n")
        log_handle.flush()


def collect_processes(pattern: str) -> Dict[int, str]:
    pattern_lower = pattern.lower()
    global PSUTIL_USABLE, PSUTIL_WARNING_EMITTED
    if PSUTIL_USABLE and psutil is not None:
        matches: Dict[int, str] = {}
        try:
            for proc in psutil.process_iter(attrs=["pid", "cmdline", "name"]):
                try:
                    cmdline = " ".join(proc.info.get("cmdline") or [])
                    candidate = cmdline or proc.info.get("name") or ""
                    if pattern_lower in candidate.lower():
                        matches[int(proc.info["pid"])] = candidate.strip()
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
        except PermissionError:
            PSUTIL_USABLE = False
            if not PSUTIL_WARNING_EMITTED:
                print(
                    "[warn] Permission denied when using psutil; falling back to ps.",
                    file=sys.stderr,
                )
                PSUTIL_WARNING_EMITTED = True
        except Exception:
            PSUTIL_USABLE = False
            if not PSUTIL_WARNING_EMITTED:
                print(
                    "[warn] psutil unavailable; falling back to ps.",
                    file=sys.stderr,
                )
                PSUTIL_WARNING_EMITTED = True
        else:
            return matches

    if platform.system() == "Windows":
        raise RuntimeError(
            "psutil is required on Windows hosts. Install it via 'pip install psutil'."
        )

    command = ["ps", "-axo", "pid=,command="]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except PermissionError as exc:
        raise RuntimeError(
            "ps command not permitted; re-run with elevated privileges."
        ) from exc
    if result.returncode not in (0, 1):
        raise RuntimeError(f"ps exited with status {result.returncode}: {result.stderr}")

    matches: Dict[int, str] = {}
    for line in result.stdout.splitlines():
        entry = line.strip()
        if not entry:
            continue
        parts = entry.split(None, 1)
        if len(parts) != 2:
            continue
        pid_str, command_line = parts
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        if pattern_lower in command_line.lower():
            matches[pid] = command_line.strip()
    return matches


def run_once(pattern: str, log_handle) -> int:
    matches = collect_processes(pattern)
    if matches:
        emit(
            log_handle,
            "info",
            f"Found {len(matches)} process(es) matching '{pattern}'.",
        )
        for pid, command_line in sorted(matches.items()):
            emit(log_handle, "info", f"  pid={pid} cmd={command_line}")
        return 0
    emit(log_handle, "warn", f"No processes matched '{pattern}'.")
    return 1


def monitor(pattern: str, interval: float, log_handle) -> int:
    psutil_state = "yes" if PSUTIL_USABLE and psutil is not None else "no"
    emit(
        log_handle,
        "info",
        f"Monitoring for '{pattern}' every {interval:.1f}s "
        f"(psutil={psutil_state}).",
    )
    previous: Dict[int, str] = {}
    seen_active = False
    try:
        while True:
            matches = collect_processes(pattern)
            new_pids = set(matches) - set(previous)
            terminated_pids = set(previous) - set(matches)

            for pid in sorted(new_pids):
                emit(
                    log_handle,
                    "info",
                    f"Process detected (pid={pid}): {matches[pid]}",
                )

            for pid in sorted(terminated_pids):
                emit(
                    log_handle,
                    "warn",
                    f"Process exited (pid={pid}): {previous[pid]}",
                )

            currently_active = bool(matches)
            if currently_active and not seen_active:
                emit(
                    log_handle,
                    "info",
                    f"{len(matches)} matching process(es) currently active.",
                )
            elif not currently_active and seen_active:
                emit(log_handle, "warn", "No matching processes remain.")
            seen_active = currently_active
            previous = matches

            sleep_interval = interval if interval > 0 else 1.0
            time.sleep(max(0.5, sleep_interval))
    except KeyboardInterrupt:
        emit(log_handle, "info", "Monitor interrupted by user.")
        return 130


def main() -> int:
    args = parse_args()
    log_handle = None
    try:
        if args.log_file:
            args.log_file.parent.mkdir(parents=True, exist_ok=True)
            log_handle = args.log_file.open("a", encoding="utf-8")
        if args.once:
            return run_once(args.pattern, log_handle)
        return monitor(args.pattern, args.interval, log_handle)
    except RuntimeError as exc:
        emit(log_handle, "error", str(exc))
        return 2
    finally:
        if log_handle is not None:
            log_handle.close()


if __name__ == "__main__":
    sys.exit(main())
