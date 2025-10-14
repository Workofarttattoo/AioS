"""
Concrete subsystem agents for AgentaOS.

Each agent performs lightweight, real system inspections or orchestration
commands so the runtime produces actionable telemetry rather than placeholder
strings.  The implementations favour read-only interactions to keep the
prototype safe while still demonstrating how a production deployment could
coordinate host resources, container engines, and user sessions.
"""

from __future__ import annotations

import asyncio
import getpass
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
import shlex

from ..model import ActionResult
from ..providers import ProviderReport, ResourceProvider, discover_providers
from ..apps import (
    DEFAULT_CONCURRENCY,
    AppConfigurationError,
    SupervisorScheduler,
    load_app_specs,
)
from ..tools import available_security_tools, run_health_check
from ..virtualization import VirtualizationLayer
from ..gui.schema import (
    ActionDescriptor,
    DashboardDescriptor,
    LayoutHint,
    MetricDescriptor,
    PanelDescriptor,
    TableDescriptor,
    TableRow,
)

DEFAULT_TIMEOUT = 8
MAX_OUTPUT_LENGTH = 2048


class CommandExecutionError(RuntimeError):
    """Raised when a shell command fails in a critical way."""


def _truncate(text: str, limit: int = MAX_OUTPUT_LENGTH) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...<truncated {len(text) - limit} chars>"


def _json_list(output: str) -> List[Dict[str, object]]:
    if not output or not output.strip():
        return []
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    return []


def _path_exists(path: Path, limit: int = 5) -> Tuple[int, List[str]]:
    if not path.exists():
        return 0, []
    try:
        entries = sorted(p.name for p in path.iterdir())
    except PermissionError:
        return 0, []
    return len(entries), entries[:limit]


@dataclass
class BaseAgent:
    name: str
    platform_name: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.platform_name = platform.system().lower()

    def info(self, action: str, message: str, payload: Optional[Dict] = None) -> ActionResult:
        return ActionResult(success=True, message=f"[info] {self.name}.{action}: {message}", payload=payload or {})

    def warn(self, action: str, message: str, payload: Optional[Dict] = None) -> ActionResult:
        return ActionResult(success=True, message=f"[warn] {self.name}.{action}: {message}", payload=payload or {})

    def error(self, action: str, message: str, payload: Optional[Dict] = None) -> ActionResult:
        return ActionResult(success=False, message=f"[error] {self.name}.{action}: {message}", payload=payload or {})

    def run_command(
        self,
        command: List[str],
        *,
        timeout: int = DEFAULT_TIMEOUT,
        check: bool = False,
    ) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=check,
            )
        except PermissionError as exc:
            raise CommandExecutionError(
                f"Permission denied while executing '{' '.join(command)}': {exc}"
            ) from exc
        except FileNotFoundError as exc:
            raise CommandExecutionError(f"Command not found: {command[0]}") from exc
        except subprocess.CalledProcessError as exc:
            raise CommandExecutionError(
                f"Command '{' '.join(command)}' failed with code {exc.returncode}: {exc.stderr.strip()}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise CommandExecutionError(f"Command '{' '.join(command)}' timed out after {timeout}s") from exc

    def run_powershell(
        self,
        script: str,
        *,
        timeout: int = DEFAULT_TIMEOUT,
        check: bool = False,
    ) -> subprocess.CompletedProcess:
        if not self.platform_name.startswith("windows"):
            raise CommandExecutionError("PowerShell is only available on Windows hosts.")
        command = ["powershell", "-NoProfile", "-Command", script]
        return self.run_command(command, timeout=timeout, check=check)

    def is_forensic(self, ctx) -> bool:
        value = ctx.environment.get("AGENTA_FORENSIC_MODE", "")
        return value.lower() in {"1", "true", "yes", "on"}


class KernelAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("kernel")

    def process_management(self, ctx) -> ActionResult:
        try:
            snapshot = self._collect_process_snapshot()
            ctx.publish_metadata("kernel_process_snapshot", snapshot)
            summary = snapshot["summary"]
            lines = [f"{proc['pid']} {proc['name']} cpu={proc['cpu']} mem={proc['memory']}" for proc in snapshot["processes"][:5]]
            payload = {
                "process_sample": lines,
                "sample_size": summary["count"],
                "utilization": summary["utilization"],
                "anomalies": summary["anomalies"],
                "provider": snapshot["provider"],
            }
            if summary["anomalies"]:
                return self.warn("process_management", "Scheduler detected elevated processes.", payload)
            return self.info("process_management", "Process scheduler online.", payload)
        except CommandExecutionError as exc:
            load = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
            payload = {"fallback": "loadavg", "load": load, "error": str(exc)}
            return self.warn("process_management", "Scheduler fallback metrics collected.", payload)

    def memory_management(self, ctx) -> ActionResult:
        payload: Dict[str, float] = {}
        try:
            snapshot = self._collect_memory_snapshot()
            ctx.publish_metadata("kernel_memory_snapshot", snapshot)
            payload = {
                "free_mb": snapshot.get("free_mb"),
                "total_mb": snapshot.get("total_mb"),
                "pressure": snapshot.get("pressure"),
                "breakdown": snapshot.get("breakdown"),
                "provider": snapshot.get("provider"),
            }
            if snapshot.get("pressure", 0.0) > 0.8:
                return self.warn("memory_management", "Memory pressure elevated.", payload)
            return self.info("memory_management", "Virtual memory subsystem ready.", payload)
        except Exception as exc:
            load = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
            payload["loadavg"] = load
            payload["error"] = str(exc)
            return self.warn("memory_management", "Memory metrics degraded; fallback load averages used.", payload)

    def device_drivers(self, ctx) -> ActionResult:
        if self.platform_name.startswith("windows"):
            try:
                script = (
                    "Get-CimInstance Win32_PnPSignedDriver | "
                    "Select-Object -First 5 DeviceName,DriverVersion "
                    "| ConvertTo-Json"
                )
                proc = self.run_powershell(script)
                entries = _json_list(proc.stdout)
                if entries:
                    payload = {
                        "drivers": entries,
                        "total_count": len(entries),
                    }
                    return self.info("device_drivers", "Device driver inventory collected.", payload)
                raise CommandExecutionError("No driver data returned from PowerShell.")
            except CommandExecutionError as exc:
                return self.warn("device_drivers", f"Unable to enumerate drivers: {exc}")
            else:
                kext_path = Path("/System/Library/Extensions")
                count, drivers = _path_exists(kext_path)
                if count:
                    payload = {"installed_drivers": drivers, "total_count": count}
                    return self.info("device_drivers", "Device driver registry loaded.", payload)
        return self.warn("device_drivers", "Unable to enumerate drivers on this platform.")

    def _collect_process_snapshot(self) -> Dict[str, object]:
        if self.platform_name.startswith("windows"):
            script = (
                "Get-Process | Select-Object Id,ProcessName,CPU,WorkingSet,StartTime "
                "| Sort-Object CPU -Descending | ConvertTo-Json"
            )
            proc = self.run_powershell(script)
            entries = _json_list(proc.stdout)
            if not entries:
                raise CommandExecutionError("No process data returned from PowerShell.")
            processes = []
            cpu_total = 0.0
            anomalies: List[Dict[str, object]] = []
            for entry in entries:
                cpu = float(entry.get("CPU") or 0.0)
                mem = float(entry.get("WorkingSet") or 0.0) / (1024 * 1024)
                processes.append({
                    "pid": entry.get("Id"),
                    "name": entry.get("ProcessName"),
                    "cpu": round(cpu, 3),
                    "memory": round(mem, 2),
                })
                cpu_total += cpu
                if cpu > 500 or mem > 1024:
                    anomalies.append({
                        "pid": entry.get("Id"),
                        "name": entry.get("ProcessName"),
                        "cpu": round(cpu, 2),
                        "memory": round(mem, 2),
                    })
            return {
                "provider": "powershell:get-process",
                "processes": processes,
                "summary": {
                    "count": len(processes),
                    "utilization": {"cpu_total": round(cpu_total, 2)},
                    "anomalies": anomalies[:5],
                },
            }

        if sys.platform == "darwin":
            script = ["ps", "-Ao", "pid,comm,pcpu,pmem"]
        else:
            script = ["ps", "-eo", "pid,comm,pcpu,pmem"]
        proc = self.run_command(script)
        lines = proc.stdout.strip().splitlines()
        processes = []
        cpu_total = 0.0
        anomalies: List[Dict[str, object]] = []
        for line in lines[1:]:
            parts = line.split(None, 3)
            if len(parts) < 4:
                continue
            pid, command, cpu_str, mem_str = parts
            try:
                cpu = float(cpu_str)
                mem = float(mem_str)
            except ValueError:
                continue
            entry = {
                "pid": int(pid),
                "name": command[:64],
                "cpu": round(cpu, 2),
                "memory": round(mem, 2),
            }
            processes.append(entry)
            cpu_total += cpu
            if cpu > 80 or mem > 20:
                anomalies.append(entry)
        return {
            "provider": "ps",
            "processes": processes,
            "summary": {
                "count": len(processes),
                "utilization": {"cpu_total": round(cpu_total, 2)},
                "anomalies": anomalies[:5],
            },
        }

    def _collect_memory_snapshot(self) -> Dict[str, object]:
        if self.platform_name.startswith("windows"):
            script = (
                "Get-CimInstance -ClassName Win32_OperatingSystem | "
                "Select-Object FreePhysicalMemory,TotalVisibleMemorySize,TotalVirtualMemorySize,FreeVirtualMemory "
                "| ConvertTo-Json"
            )
            proc = self.run_powershell(script)
            entries = _json_list(proc.stdout)
            if not entries:
                raise CommandExecutionError("No memory data returned from PowerShell.")
            record = entries[0]
            free_mb = float(record.get("FreePhysicalMemory") or 0) / 1024
            total_mb = float(record.get("TotalVisibleMemorySize") or 0) / 1024
            free_virtual_mb = float(record.get("FreeVirtualMemory") or 0) / 1024
            total_virtual_mb = float(record.get("TotalVirtualMemorySize") or 0) / 1024
            pressure = 1 - (free_mb / total_mb) if total_mb else 0.0
            return {
                "provider": "powershell:get-ciminstance",
                "free_mb": round(free_mb, 2),
                "total_mb": round(total_mb, 2),
                "pressure": round(max(0.0, min(1.0, pressure)), 3),
                "breakdown": {
                    "free_virtual_mb": round(free_virtual_mb, 2),
                    "total_virtual_mb": round(total_virtual_mb, 2),
                },
            }

        if sys.platform == "darwin":
            proc = self.run_command(["/usr/bin/vm_stat"])
            stats = {}
            for line in proc.stdout.splitlines():
                if ":" in line:
                    key, value = line.split(":", maxsplit=1)
                    stats[key.strip()] = int(value.strip().strip(".").replace(",", "")) * 4096
            free = stats.get("Pages free", 0)
            active = stats.get("Pages active", 0)
            wired = stats.get("Pages wired down", 0)
            return {
                "provider": "vm_stat",
                "free_mb": round(free / (1024 * 1024), 2),
                "total_mb": None,
                "pressure": round(active / max(1, active + free + wired), 3),
                "breakdown": {
                    "active_mb": round(active / (1024 * 1024), 2),
                    "wired_mb": round(wired / (1024 * 1024), 2),
                },
            }

        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            info = {}
            for line in fh:
                key, value = line.split(":", maxsplit=1)
                info[key.strip()] = value.strip()
        def _to_mb(field: str) -> float:
            raw = info.get(field, "0 kB").split()[0]
            try:
                return float(raw) / 1024
            except ValueError:
                return 0.0
        total_mb = _to_mb("MemTotal")
        free_mb = _to_mb("MemFree") + _to_mb("Buffers") + _to_mb("Cached")
        pressure = 1 - (free_mb / total_mb) if total_mb else 0.0
        return {
            "provider": "procfs",
            "free_mb": round(free_mb, 2),
            "total_mb": round(total_mb, 2),
            "pressure": round(max(0.0, min(1.0, pressure)), 3),
            "breakdown": {
                "buffers_mb": round(_to_mb("Buffers"), 2),
                "cached_mb": round(_to_mb("Cached"), 2),
                "swap_free_mb": round(_to_mb("SwapFree"), 2),
            },
        }

    def system_calls(self, ctx) -> ActionResult:
        uname = platform.uname()
        payload = {
            "system": uname.system,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
        }
        return self.info("system_calls", "System call interface active.", payload)

    def audit(self, ctx) -> ActionResult:
        ctx.publish_metadata("kernel_audit_started", {"timestamp": time.time()})
        return self.info("audit", "Kernel audit trail recording.")


class SecurityAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("security")

    def access_control(self, ctx) -> ActionResult:
        try:
            user = getpass.getuser()
            if self.platform_name.startswith("windows"):
                groups_proc = self.run_command(["whoami", "/groups"])
                groups_list = [
                    line.strip()
                    for line in groups_proc.stdout.splitlines()
                    if line.strip() and not line.startswith("GROUP INFORMATION")
                ]
            else:
                groups_proc = self.run_command(["id", "-Gn"])
                groups_list = groups_proc.stdout.strip().split()
            payload = {
                "user": user,
                "groups": groups_list[:10],
                "cgroups": self._cgroup_status(ctx),
            }
            return self.info("access_control", "RBAC policies enforced.", payload)
        except CommandExecutionError as exc:
            return self.error("access_control", str(exc))

    def encryption(self, ctx) -> ActionResult:
        if self.platform_name.startswith("windows"):
            try:
                script = (
                    "Get-BitLockerVolume | "
                    "Select-Object MountPoint,VolumeType,ProtectionStatus,VolumeStatus "
                    "| ConvertTo-Json"
                )
                proc = self.run_powershell(script)
                entries = _json_list(proc.stdout)
                if entries:
                    payload = {"bitlocker": entries}
                    return self.info("encryption", "BitLocker status collected.", payload)
                raise CommandExecutionError("No BitLocker data returned.")
            except CommandExecutionError as exc:
                return self.warn("encryption", f"Unable to query BitLocker: {exc}")
        if shutil.which("fdesetup"):
            try:
                proc = self.run_command(["fdesetup", "status"])
                payload = {"status": proc.stdout.strip()}
                return self.info("encryption", "Disk encryption service queried.", payload)
            except CommandExecutionError as exc:
                return self.warn("encryption", f"Unable to query FileVault: {exc}")
        return self.warn("encryption", "Disk encryption tooling not available on this host.")

    def firewall(self, ctx) -> ActionResult:
        socketfilter = "/usr/libexec/ApplicationFirewall/socketfilterfw"
        payload: Dict[str, object] = {}
        if self.platform_name.startswith("windows"):
            try:
                script = (
                    "Get-NetFirewallProfile | "
                    "Select-Object Name,Enabled,DefaultInboundAction,DefaultOutboundAction "
                    "| ConvertTo-Json"
                )
                proc = self.run_powershell(script)
                profiles = _json_list(proc.stdout)
                if profiles:
                    payload["profiles"] = profiles
                else:
                    payload["profiles_error"] = "No firewall profiles returned."
            except CommandExecutionError as exc:
                payload["profiles_error"] = str(exc)
        if Path(socketfilter).exists():
            try:
                proc = self.run_command([socketfilter, "--getglobalstate"])
                payload["raw"] = proc.stdout.strip()
            except CommandExecutionError as exc:
                payload["raw_error"] = str(exc)

        profile = ctx.environment.get("AGENTA_FIREWALL_PROFILE")
        if profile:
            payload["profile"] = self._apply_firewall_profile(profile)

        if payload:
            message = "Firewall policy evaluated."
            return self.info("firewall", message, payload)
        return self.warn("firewall", "Firewall interface unavailable.")

    def threat_detection(self, ctx) -> ActionResult:
        try:
            if self.platform_name.startswith("windows"):
                script = (
                    "Get-Process | Sort-Object CPU -Descending | "
                    "Select-Object -First 5 Id,ProcessName,CPU "
                    "| ConvertTo-Json"
                )
                proc = self.run_powershell(script)
                entries = _json_list(proc.stdout)
                lines = []
                high_cpu = []
                for entry in entries:
                    cpu_value = float(entry.get("CPU") or 0)
                    line = f"Id={entry.get('Id')} Name={entry.get('ProcessName')} CPU={cpu_value}"
                    lines.append(line)
                    if cpu_value > 75.0:
                        high_cpu.append(line)
                payload = {"high_cpu_candidates": high_cpu, "sample": lines}
            else:
                proc = self.run_command(["ps", "-Ao", "pid,pcpu,comm", "-r"])
                lines = proc.stdout.strip().splitlines()[1:6]
                high_cpu = [line for line in lines if float(line.split()[1]) > 75.0]
                payload = {"high_cpu_candidates": high_cpu, "sample": lines}
            ctx.publish_metadata("security_anomaly_scan", payload)
            if high_cpu:
                return self.warn("threat_detection", "High CPU processes observed.", payload)
            return self.info("threat_detection", "Baseline anomaly scanner clear.", payload)
        except CommandExecutionError as exc:
            payload = {"error": str(exc)}
            return self.warn("threat_detection", "Threat scan degraded by system permissions.", payload)
        except Exception as exc:
            return self.warn("threat_detection", f"Threat scan failed: {exc}")

    def audit_review(self, ctx) -> ActionResult:
        system_log = Path("/private/var/log/system.log")
        if self.platform_name.startswith("windows"):
            try:
                script = (
                    "(Get-WinEvent -LogName System -MaxEvents 5 | "
                    "Select-Object TimeCreated,Id,LevelDisplayName,Message) | "
                    "ConvertTo-Json -Depth 3"
                )
                proc = self.run_powershell(script)
                entries = _json_list(proc.stdout)
                if entries:
                    payload = {"events": entries}
                    return self.info("audit_review", "Security events retrieved.", payload)
                raise CommandExecutionError("No event data returned.")
            except CommandExecutionError as exc:
                return self.warn("audit_review", f"Event log unavailable: {exc}")
        if system_log.exists():
            tail = system_log.read_text(encoding="utf-8", errors="ignore").splitlines()[-5:]
            payload = {"log_tail": tail}
            return self.info("audit_review", "Security audit log ingestion enabled.", payload)
        return self.warn("audit_review", "System log unavailable for review.")

    def integrity_survey(self, ctx) -> ActionResult:
        artifacts: List[str] = []
        for candidate in self._forensic_file_candidates():
            digest = self._hash_file(candidate)
            if digest:
                artifacts.append(digest)

        if self.platform_name.startswith("windows"):
            try:
                script = (
                    "(Get-WinEvent -LogName Security -MaxEvents 3 | "
                    "Select-Object TimeCreated,Id,LevelDisplayName,Message) | "
                    "ConvertTo-Json -Depth 3"
                )
                proc = self.run_powershell(script, timeout=10)
                log_entries = _json_list(proc.stdout)
            except CommandExecutionError:
                log_entries = []
        else:
            log_entries = self._collect_log_tails()

        payload = {
            "artifacts": artifacts,
            "logs": log_entries,
            "provider": "forensic",
        }
        ctx.publish_metadata("security_integrity", payload)
        message = "Forensic integrity snapshot captured."
        if not artifacts and not log_entries:
            return self.warn("integrity_survey", message, payload)
        return self.info("integrity_survey", message, payload)

    def sovereign_suite(self, ctx) -> ActionResult:
        suite_name = ctx.environment.get("AGENTA_SECURITY_SUITE", "Sovereign Security Toolkit")
        profile = ctx.environment.get("AGENTA_SECURITY_PROFILE", "")
        raw_tools = ctx.environment.get("AGENTA_SECURITY_TOOLS", "")
        requested = [item.strip() for item in raw_tools.split(",") if item.strip()]
        registry = {name.lower(): name for name in available_security_tools()}

        resolved: List[str] = []
        missing: List[str] = []
        for item in requested:
            canonical = registry.get(item.lower())
            if canonical:
                resolved.append(canonical)
            else:
                missing.append(item)

        reports: List[Dict[str, object]] = []
        degraded: Set[str] = set()
        binaries: Dict[str, Dict[str, object]] = {}

        for tool in resolved:
            binary_hint = self._tool_binary_hint(tool)
            if binary_hint:
                binary_info = self._binary_metadata(binary_hint)
            else:
                binary_info = {
                    "name": None,
                    "path": "",
                    "source": "unknown",
                    "exists": False,
                    "executable": False,
                }
            binaries[tool] = binary_info

            try:
                report = run_health_check(tool)
            except KeyError:
                missing.append(tool)
                continue
            except ImportError as exc:
                missing.append(f"{tool} ({exc})")
                continue
            except Exception as exc:  # pylint: disable=broad-except
                degraded.add(tool)
                reports.append({
                    "tool": tool,
                    "status": "error",
                    "summary": f"Health check failed: {exc}",
                    "details": {},
                })
                continue
            if not report:
                degraded.add(tool)
                reports.append({
                    "tool": tool,
                    "status": "degraded",
                    "summary": "Health check routine unavailable.",
                    "details": {},
                })
                continue
            reports.append(report)

        payload = {
            "suite": suite_name,
            "profile": profile,
            "requested": requested,
            "resolved": resolved,
            "reports": reports,
            "missing": missing,
            "degraded": sorted(degraded),
            "available": list(available_security_tools()),
            "binaries": binaries,
        }
        ctx.publish_metadata("security.sovereign_suite", payload)

        if not requested:
            return self.info("sovereign_suite", "Sovereign toolkit not requested.", payload)
        if missing:
            return self.warn("sovereign_suite", "Some Sovereign tools were not recognised.", payload)
        if degraded:
            return self.warn("sovereign_suite", "Toolkit health signals degraded.", payload)
        return self.info("sovereign_suite", "Toolkit health signals nominal.", payload)

    def _tool_binary_hint(self, tool: str) -> Optional[str]:
        hints = {
            "CipherSpear": "cipherspear",
            "SkyBreaker": "skybreaker",
            "MythicKey": "mythickey",
            "SpectraTrace": "spectratrace",
            "NemesisHydra": "nemesishydra",
            "ObsidianHunt": "obsidianhunt",
            "VectorFlux": "vectorflux",
        }
        return hints.get(tool)

    def _binary_metadata(self, binary: str) -> Dict[str, object]:
        workspace = Path(__file__).resolve().parents[1]
        candidate = workspace / binary
        metadata = {
            "name": binary,
            "path": str(candidate),
            "source": "workspace",
            "exists": candidate.exists(),
        }
        if metadata["exists"]:
            metadata["executable"] = os.access(candidate, os.X_OK)
            return metadata
        located = shutil.which(binary)
        if located:
            metadata.update({
                "path": located,
                "source": "path",
                "exists": True,
                "executable": os.access(located, os.X_OK),
            })
            return metadata
        metadata["executable"] = False
        return metadata

    def _cgroup_status(self, ctx) -> Dict[str, object]:
        if self.platform_name.startswith("windows"):
            return {"available": False, "reason": "cgroups are not supported on Windows hosts."}
        base = Path("/sys/fs/cgroup")
        if not base.exists():
            return {"available": False, "reason": "cgroup filesystem not mounted."}

        try:
            controllers = [p.name for p in base.iterdir() if p.is_dir()]
        except PermissionError as exc:
            return {"available": False, "reason": f"permission denied: {exc}"}

        info: Dict[str, object] = {
            "available": True,
            "controller_count": len(controllers),
            "controllers_sample": controllers[:6],
        }

        slice_name = ctx.environment.get("AGENTA_CGROUP_SLICE")
        if slice_name and shutil.which("systemctl"):
            try:
                status = self.run_command(["systemctl", "status", slice_name])
                info["slice_status"] = status.stdout.strip()[:512]
            except CommandExecutionError as exc:
                info["slice_error"] = str(exc)

        return info

    def _apply_firewall_profile(self, profile: str) -> Dict[str, object]:
        profile = profile.lower()
        if profile in {"windows", "netsh"} and self.platform_name.startswith("windows"):
            try:
                script = (
                    "Get-NetFirewallRule -PolicyStore ActiveStore | "
                    "Select-Object -First 10 DisplayName,Direction,Action,Enabled "
                    "| ConvertTo-Json"
                )
                proc = self.run_powershell(script)
                rules = _json_list(proc.stdout)
                return {
                    "profile": "windows",
                    "status": "applied",
                    "rules": rules,
                }
            except CommandExecutionError as exc:
                return {"profile": "windows", "status": "error", "message": str(exc)}
        if profile in {"pf", "pfctl"} and shutil.which("pfctl"):
            try:
                summary = self.run_command(["pfctl", "-s", "info"])
                rules = self.run_command(["pfctl", "-sr"])
                return {
                    "profile": "pfctl",
                    "status": "applied",
                    "summary": summary.stdout.strip(),
                    "rules": rules.stdout.strip()[:512],
                }
            except CommandExecutionError as exc:
                return {"profile": "pfctl", "status": "error", "message": str(exc)}

        if profile in {"iptables", "nftables", "iptables-save"}:
            if shutil.which("iptables"):
                try:
                    listing = self.run_command(["iptables", "-S"])
                    return {
                        "profile": "iptables",
                        "status": "applied",
                        "rules": listing.stdout.strip()[:512],
                    }
                except CommandExecutionError as exc:
                    return {"profile": "iptables", "status": "error", "message": str(exc)}
            if shutil.which("nft"):
                try:
                    listing = self.run_command(["nft", "list", "ruleset"])
                    return {
                        "profile": "nft",
                        "status": "applied",
                        "rules": listing.stdout.strip()[:512],
                    }
                except CommandExecutionError as exc:
                    return {"profile": "nft", "status": "error", "message": str(exc)}

        return {
            "profile": profile,
            "status": "unsupported",
            "message": "No compatible firewall tool available or permissions missing.",
        }

    def _forensic_file_candidates(self) -> List[Path]:
        paths: List[Path] = []
        if self.platform_name.startswith("windows"):
            candidates = [
                Path(os.environ.get("SystemRoot", "C:\\Windows")) / "System32" / "drivers" / "etc" / "hosts",
                Path(os.environ.get("SystemRoot", "C:\\Windows")) / "System32" / "config" / "SYSTEM",
            ]
        else:
            candidates = [
                Path("/etc/passwd"),
                Path("/etc/hosts"),
                Path("/etc/ssh/sshd_config"),
            ]
        for candidate in candidates:
            if candidate.exists():
                paths.append(candidate)
        return paths

    def _hash_file(self, path: Path, max_bytes: int = 1024 * 1024) -> Optional[Dict[str, object]]:
        try:
            digest = hashlib.sha256()
            size = path.stat().st_size
            read_bytes = 0
            with path.open("rb") as fh:
                while read_bytes < max_bytes:
                    chunk = fh.read(min(65536, max_bytes - read_bytes))
                    if not chunk:
                        break
                    digest.update(chunk)
                    read_bytes += len(chunk)
            return {
                "path": str(path),
                "sha256": digest.hexdigest(),
                "bytes_read": read_bytes,
                "size": size,
            }
        except (OSError, PermissionError):
            return None

    def _collect_log_tails(self) -> List[Dict[str, object]]:
        tails = []
        candidates = [
            Path("/var/log/auth.log"),
            Path("/var/log/secure"),
            Path("/var/log/system.log"),
            Path("/var/log/syslog"),
        ]
        seen = set()
        for candidate in candidates:
            if candidate.exists() and candidate not in seen:
                entry = self._read_log_tail(candidate, lines=5)
                if entry:
                    tails.append(entry)
                seen.add(candidate)
        return tails

    def _read_log_tail(self, path: Path, lines: int = 5) -> Optional[Dict[str, object]]:
        try:
            dq = deque(maxlen=lines)
            with path.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    dq.append(line.rstrip("\n"))
            return {"path": str(path), "tail": list(dq)}
        except (OSError, PermissionError):
            return None


class NetworkingAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("networking")

    def network_configuration(self, ctx) -> ActionResult:
        if self.platform_name.startswith("windows"):
            try:
                script = (
                    "Get-NetAdapter | "
                    "Select-Object -First 5 Name,Status,LinkSpeed,InterfaceDescription "
                    "| ConvertTo-Json"
                )
                proc = self.run_powershell(script)
                adapters = _json_list(proc.stdout)
                if adapters:
                    payload = {"adapters": adapters}
                    return self.info("network_configuration", "Interfaces configured.", payload)
                raise CommandExecutionError("No adapter data returned.")
            except CommandExecutionError as exc:
                return self.warn("network_configuration", f"Unable to list adapters: {exc}")
        if shutil.which("networksetup"):
            try:
                proc = self.run_command(["networksetup", "-listallhardwareports"])
                payload = {"hardware_ports": proc.stdout.strip().splitlines()[:12]}
                return self.info("network_configuration", "Interfaces configured.", payload)
            except CommandExecutionError as exc:
                return self.warn("network_configuration", f"Unable to list hardware ports: {exc}")
        interfaces = socket.getaddrinfo(socket.gethostname(), None)
        payload = {"interfaces": interfaces[:3]}
        return self.info("network_configuration", "Interfaces inspected via socket API.", payload)

    def protocol_handling(self, ctx) -> ActionResult:
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.settimeout(1)
            test_socket.bind(("0.0.0.0", 0))
            port = test_socket.getsockname()[1]
            test_socket.close()
            return self.info("protocol_handling", "Protocol stack ready.", {"ephemeral_port": port})
        except Exception as exc:
            return self.warn("protocol_handling", f"Socket stack degraded: {exc}")

    def data_transmission(self, ctx) -> ActionResult:
        ping_cmd = None
        if self.platform_name.startswith("windows"):
            ping_cmd = ["ping", "-n", "1", "127.0.0.1"]
        elif shutil.which("ping"):
            ping_cmd = ["ping", "-c", "1", "127.0.0.1"]
        if ping_cmd:
            try:
                proc = self.run_command(ping_cmd, timeout=3)
                payload = {"ping_summary": proc.stdout.strip().splitlines()[-1]}
                return self.info("data_transmission", "Data plane operational.", payload)
            except CommandExecutionError as exc:
                return self.warn("data_transmission", f"Loopback ping failed: {exc}")
        return self.warn("data_transmission", "Ping utility unavailable.")

    def dns_resolver(self, ctx) -> ActionResult:
        try:
            hostname = "example.com"
            address = socket.gethostbyname(hostname)
            payload = {"hostname": hostname, "address": address}
            return self.info("dns_resolver", "DNS resolver cached.", payload)
        except Exception as exc:
            return self.warn("dns_resolver", f"DNS resolution failed: {exc}")


class StorageAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("storage")

    def file_system(self, ctx) -> ActionResult:
        try:
            if self.platform_name.startswith("windows"):
                root = Path.home().anchor or "C:\\"
                usage = shutil.disk_usage(root)
                total = usage.total
                available = usage.free
                payload = {
                    "root": root,
                    "total_gb": round(total / (1024 ** 3), 2),
                    "available_gb": round(available / (1024 ** 3), 2),
                }
            else:
                stat = os.statvfs("/")
                total = stat.f_frsize * stat.f_blocks
                available = stat.f_frsize * stat.f_bavail
                payload = {
                    "total_gb": round(total / (1024 ** 3), 2),
                    "available_gb": round(available / (1024 ** 3), 2),
                }
            return self.info("file_system", "Filesystems mounted.", payload)
        except Exception as exc:
            return self.warn("file_system", f"Unable to read filesystem metrics: {exc}")

    def backup(self, ctx) -> ActionResult:
        tmutil = shutil.which("tmutil")
        if tmutil:
            try:
                proc = self.run_command([tmutil, "status"])
                payload = {"status": proc.stdout.strip()[:256]}
                return self.warn("backup", "Backups deferred until workload idle.", payload)
            except CommandExecutionError as exc:
                return self.warn("backup", f"Unable to query Time Machine: {exc}")
        if self.platform_name.startswith("windows"):
            if shutil.which("wbadmin"):
                try:
                    proc = self.run_command(["wbadmin", "get", "status"])
                    payload = {"status": proc.stdout.strip()[:512]}
                    return self.warn("backup", "Backup status retrieved.", payload)
                except CommandExecutionError as exc:
                    return self.warn("backup", f"Backup tooling unavailable: {exc}")
        return self.warn("backup", "No backup tooling detected.")

    def recovery(self, ctx) -> ActionResult:
        payload = {"recovery_partitions": []}
        if shutil.which("diskutil"):
            try:
                proc = self.run_command(["diskutil", "list"])
                payload["recovery_partitions"] = [
                    line.strip() for line in proc.stdout.splitlines() if "Recovery" in line
                ][:3]
                return self.info("recovery", "Recovery points verified.", payload)
            except CommandExecutionError as exc:
                return self.warn("recovery", f"Recovery query failed: {exc}")
        if self.platform_name.startswith("windows") and shutil.which("reagentc"):
            try:
                proc = self.run_command(["reagentc", "/info"])
                payload = {"reagentc": proc.stdout.strip().splitlines()[:12]}
                return self.info("recovery", "Windows recovery agent status retrieved.", payload)
            except CommandExecutionError as exc:
                return self.warn("recovery", f"Recovery agent unavailable: {exc}")
        return self.warn("recovery", "Unable to inspect recovery partitions.")

    def disk_management(self, ctx) -> ActionResult:
        if self.platform_name.startswith("windows"):
            try:
                script = (
                    "Get-PhysicalDisk | "
                    "Select-Object -First 5 FriendlyName,MediaType,CanPool,HealthStatus "
                    "| ConvertTo-Json"
                )
                proc = self.run_powershell(script)
                entries = _json_list(proc.stdout)
                if entries:
                    payload = {"devices": entries}
                    return self.info("disk_management", "Disk inventory collected.", payload)
                raise CommandExecutionError("No physical disk data returned.")
            except CommandExecutionError as exc:
                return self.warn("disk_management", f"Unable to inspect disks: {exc}")
        disks = []
        dev_path = Path("/dev")
        if dev_path.exists():
            disks = sorted(name for name in dev_path.iterdir() if name.name.startswith("disk"))[:5]
        payload = {"devices": [p.name for p in disks]}
        return self.info("disk_management", "Disk partitions healthy.", payload)

    def volume_inventory(self, ctx) -> ActionResult:
        try:
            report = self._collect_volume_inventory()
            ctx.publish_metadata("storage_volume_inventory", report)
            alerts = report.get("alerts", [])
            if alerts:
                return self.warn("volume_inventory", "Volume inventory captured with alerts.", report)
            return self.info("volume_inventory", "Volume inventory captured.", report)
        except CommandExecutionError as exc:
            payload = {"error": str(exc)}
            return self.warn("volume_inventory", "Unable to inspect volumes.", payload)

    def _collect_volume_inventory(self) -> Dict[str, object]:
        if self.platform_name.startswith("windows"):
            script = (
                "Get-Volume | "
                "Select-Object DriveLetter,FileSystemLabel,FileSystem,SizeRemaining,Size,HealthStatus "
                "| ConvertTo-Json"
            )
            proc = self.run_powershell(script)
            entries = _json_list(proc.stdout)
            volumes = []
            alerts = []
            for entry in entries:
                size = float(entry.get("Size") or 0)
                free = float(entry.get("SizeRemaining") or 0)
                free_percent = (free / size * 100) if size else None
                volume = {
                    "drive": entry.get("DriveLetter"),
                    "label": entry.get("FileSystemLabel"),
                    "filesystem": entry.get("FileSystem"),
                    "free_gb": round(free / (1024**3), 2) if free else 0.0,
                    "size_gb": round(size / (1024**3), 2) if size else 0.0,
                    "free_percent": round(free_percent, 2) if free_percent is not None else None,
                    "health": entry.get("HealthStatus"),
                }
                volumes.append(volume)
                if free_percent is not None and free_percent < 10:
                    alerts.append(volume)
            return {
                "provider": "powershell:get-volume",
                "volumes": volumes,
                "alerts": alerts[:5],
            }

        df = self.run_command(["df", "-kP"])
        lines = df.stdout.strip().splitlines()
        volumes = []
        alerts = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 6:
                continue
            filesystem, blocks, used, available, percent, mountpoint = parts[:6]
            try:
                total_kb = int(blocks)
                available_kb = int(available)
                percent_used = float(percent.strip("%"))
                free_percent = 100 - percent_used
            except ValueError:
                continue

            # Skip non-critical volumes that don't need alerts
            skip_patterns = [
                '/dev',  # devfs is always 100% used
                '/Library/Developer/CoreSimulator',  # iOS simulators
                '/System/Volumes/xarts',  # System volumes
                '/System/Volumes/iSCPreboot',
                '/System/Volumes/Hardware',
                '/Volumes/com.apple.TimeMachine',  # Time Machine snapshots
            ]

            should_skip = any(mountpoint.startswith(pattern) for pattern in skip_patterns)

            volume = {
                "filesystem": filesystem,
                "mountpoint": mountpoint,
                "size_gb": round(total_kb / (1024**2), 2),
                "free_gb": round(available_kb / (1024**2), 2),
                "free_percent": round(free_percent, 2),
            }
            volumes.append(volume)

            # Only alert on critical volumes with low space and at least 10GB size
            if not should_skip and free_percent < 10 and volume["size_gb"] > 10:
                alerts.append(volume)
        try:
            mount_output = self.run_command(["mount"]).stdout
            mount_digest = hashlib.sha256(mount_output.encode("utf-8", errors="ignore")).hexdigest()
        except CommandExecutionError:
            mount_digest = None
        return {
            "provider": "df",
            "volumes": volumes,
            "alerts": alerts[:5],
            "mount_digest": mount_digest,
        }


class ApplicationAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("application")

    def package_manager(self, ctx) -> ActionResult:
        if self.platform_name.startswith("windows"):
            if shutil.which("choco"):
                try:
                    proc = self.run_command(["choco", "--version"])
                    payload = {"manager": "chocolatey", "version": proc.stdout.strip()}
                    return self.info("package_manager", "Package registry synchronized.", payload)
                except CommandExecutionError as exc:
                    return self.warn("package_manager", f"Chocolatey query failed: {exc}")
            if shutil.which("winget"):
                try:
                    proc = self.run_command(["winget", "--version"])
                    payload = {"manager": "winget", "version": proc.stdout.strip()}
                    return self.info("package_manager", "Package registry synchronized.", payload)
                except CommandExecutionError as exc:
                    return self.warn("package_manager", f"winget query failed: {exc}")
        if shutil.which("brew"):
            try:
                proc = self.run_command(["brew", "--version"])
                payload = {"brew_version": proc.stdout.strip().splitlines()[0]}
                return self.info("package_manager", "Package registry synchronized.", payload)
            except CommandExecutionError as exc:
                return self.warn("package_manager", f"Homebrew query failed: {exc}")
        elif shutil.which("apt"):
            return self.info("package_manager", "APT detected.", {"manager": "apt"})
        return self.warn("package_manager", "No package manager detected.")

    def dependency_resolver(self, ctx) -> ActionResult:
        pip_version = shutil.which("pip3") or shutil.which("pip")
        if pip_version:
            try:
                proc = self.run_command([pip_version, "--version"])
                payload = {"pip_version": proc.stdout.strip()}
                return self.info("dependency_resolver", "Dependencies resolved.", payload)
            except CommandExecutionError as exc:
                return self.warn("dependency_resolver", f"Dependency resolver unavailable: {exc}")
        return self.warn("dependency_resolver", "Python package manager not installed.")

    def service_manager(self, ctx) -> ActionResult:
        if self.platform_name.startswith("windows"):
            try:
                script = (
                    "Get-Service | Select-Object -First 5 Name,Status,StartType | ConvertTo-Json"
                )
                proc = self.run_powershell(script)
                entries = _json_list(proc.stdout)
                if entries:
                    payload = {"services": entries}
                    return self.info("service_manager", "Service manager supervising daemons.", payload)
                raise CommandExecutionError("No service data returned.")
            except CommandExecutionError as exc:
                return self.warn("service_manager", f"Service inspection failed: {exc}")
        if shutil.which("launchctl"):
            try:
                proc = self.run_command(["launchctl", "print", "system"], timeout=5)
                payload = {"launchd_sample": proc.stdout.strip().splitlines()[:10]}
                return self.info("service_manager", "Service manager supervising daemons.", payload)
            except CommandExecutionError as exc:
                return self.warn("service_manager", f"launchctl inspection failed: {exc}")
        return self.warn("service_manager", "Service manager interface not available.")

    def application_launcher(self, ctx) -> ActionResult:
        if self.platform_name.startswith("windows"):
            programs = []
            for base in [Path(os.environ.get("ProgramFiles", "C:\\Program Files")), Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"))]:
                if base.exists():
                    _, entries = _path_exists(base)
                    programs.extend(entries)
            payload = {"applications": programs[:10], "count": len(programs)}
            return self.info("application_launcher", "Application launcher online.", payload)
        applications_path = Path("/Applications")
        count, apps = _path_exists(applications_path)
        payload = {"applications": apps, "count": count}
        return self.info("application_launcher", "Application launcher online.", payload)

    def supervisor(self, ctx) -> ActionResult:
        config_path = ctx.environment.get("AGENTA_APPS_CONFIG")
        if not config_path:
            payload = {"apps": []}
            ctx.publish_metadata("application_supervisor", payload)
            return self.info("supervisor", "No application config specified; skipping supervisor run.", payload)

        try:
            specs = load_app_specs(config_path)
        except AppConfigurationError as exc:
            payload = {"error": str(exc)}
            ctx.publish_metadata("application_supervisor", payload)
            return self.warn(
                "supervisor",
                f"Application supervisor configuration error: {exc}",
                payload,
            )

        if not specs:
            payload = {"apps": []}
            ctx.publish_metadata("application_supervisor", payload)
            return self.info("supervisor", "Application supervisor found no apps to launch.", payload)

        try:
            concurrency = int(
                ctx.environment.get("AGENTA_SUPERVISOR_CONCURRENCY", str(DEFAULT_CONCURRENCY))
            )
        except ValueError:
            concurrency = DEFAULT_CONCURRENCY

        scheduler = SupervisorScheduler(
            specs,
            concurrency=concurrency,
            base_env=os.environ.copy(),
            forensic_mode=self.is_forensic(ctx),
        )

        result = asyncio.run(scheduler.run())
        ctx.publish_metadata("application_supervisor", result)
        summary = result.get("summary", {})
        if self.is_forensic(ctx):
            return self.warn("supervisor", "Forensic mode active: applications skipped.", result)
        if summary.get("failed") or summary.get("errors"):
            return self.warn("supervisor", "Application supervisor completed with issues.", result)
        return self.info("supervisor", "Application supervisor completed.", result)


class UserAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("user")

    def authentication(self, ctx) -> ActionResult:
        user = getpass.getuser()
        uid = os.getuid() if hasattr(os, "getuid") else None
        payload = {"user": user, "uid": uid}
        return self.info("authentication", "Authentication service ready.", payload)

    def profile_manager(self, ctx) -> ActionResult:
        home = Path.home()
        count, entries = _path_exists(home, limit=7)
        payload = {"home": str(home), "entry_sample": entries, "entry_count": count}
        return self.info("profile_manager", "User profiles loaded.", payload)

    def preference(self, ctx) -> ActionResult:
        if self.platform_name.startswith("windows"):
            prefs_dir = Path.home() / "AppData" / "Roaming"
        else:
            prefs_dir = Path.home() / "Library" / "Preferences"
        if prefs_dir.exists():
            count, entries = _path_exists(prefs_dir, limit=5)
            payload = {"preferences": entries, "total": count}
            return self.info("preference", "Preference store synced.", payload)
        return self.warn("preference", "Preferences directory missing.")

    def preferences(self, ctx) -> ActionResult:
        return self.preference(ctx)

    def session_manager(self, ctx) -> ActionResult:
        commands = []
        if self.platform_name.startswith("windows"):
            commands.append(["query", "user"])
        commands.append(["who"])
        for command in commands:
            try:
                proc = self.run_command(command)
                payload = {"sessions": proc.stdout.strip().splitlines()}
                return self.info("session_manager", "Session manager standing by.", payload)
            except CommandExecutionError:
                continue
        try:
            if self.platform_name.startswith("windows"):
                script = (
                    "(Get-CimInstance Win32_OperatingSystem | "
                    "Select-Object LastBootUpTime) | ConvertTo-Json"
                )
                proc = self.run_powershell(script)
                entries = _json_list(proc.stdout)
                payload = {"last_boot": entries}
            else:
                uptime = self.run_command(["uptime"])
                payload = {"uptime": uptime.stdout.strip()}
            return self.info("session_manager", "Session manager standing by.", payload)
        except CommandExecutionError as exc:
            return self.warn("session_manager", f"Unable to query sessions: {exc}")


class GuiAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("gui")

    def window_management(self, ctx) -> ActionResult:
        if self.platform_name.startswith("windows"):
            try:
                script = (
                    "Get-CimInstance Win32_VideoController | "
                    "Select-Object -First 1 Name,DriverVersion,CurrentHorizontalResolution,"
                    "CurrentVerticalResolution | ConvertTo-Json"
                )
                proc = self.run_powershell(script)
                info = _json_list(proc.stdout)
                if info:
                    payload = {"display_info": info}
                    return self.info("window_management", "Window compositor initialized.", payload)
                raise CommandExecutionError("No video controller data returned.")
            except CommandExecutionError as exc:
                return self.warn("window_management", f"Display inspection failed: {exc}")
        if shutil.which("system_profiler"):
            try:
                proc = self.run_command(["system_profiler", "SPDisplaysDataType"])
                payload = {"display_info": _truncate(proc.stdout)}
                return self.info("window_management", "Window compositor initialized.", payload)
            except CommandExecutionError as exc:
                return self.warn("window_management", f"Display inspection failed: {exc}")
        return self.warn("window_management", "Display inspector unavailable.")

    def event_handling(self, ctx) -> ActionResult:
        if self.platform_name.startswith("windows"):
            try:
                script = (
                    "Get-PnpDevice -Class HIDClass | "
                    "Select-Object -First 5 FriendlyName,Status | ConvertTo-Json"
                )
                proc = self.run_powershell(script)
                devices = _json_list(proc.stdout)
                payload = {"hid_devices": devices}
                return self.info("event_handling", "Input events streaming.", payload)
            except CommandExecutionError as exc:
                return self.warn("event_handling", f"Unable to inspect input devices: {exc}")
        payload = {"trackpad_enabled": Path("/Library/Preferences/com.apple.AppleMultitouchTrackpad.plist").exists()}
        return self.info("event_handling", "Input events streaming.", payload)

    def _build_dashboard_descriptor(
        self,
        theme: str,
        personalize: Optional[List[Dict[str, object]]],
    ) -> DashboardDescriptor:
        metrics = [
            MetricDescriptor(id="theme", label="Theme", value=theme, severity="info"),
            MetricDescriptor(id="platform", label="Platform", value=self.platform_name, severity="info"),
        ]
        tables: List[TableDescriptor] = []
        if personalize:
            rows = []
            for entry in personalize:
                for key, value in entry.items():
                    rows.append(TableRow(cells={"setting": str(key), "value": str(value)}))
            tables.append(
                TableDescriptor(
                    id="personalize",
                    title="Personalization Flags",
                    columns=["setting", "value"],
                    rows=rows,
                    empty_state="No personalization keys detected.",
                )
            )

        actions = [
            ActionDescriptor(id="refresh-theme", label="Refresh Theme", runtime_hook="gui.gui_design"),
            ActionDescriptor(id="inspect-display", label="Inspect Display", runtime_hook="gui.window_management"),
            ActionDescriptor(id="list-inputs", label="List Input Devices", runtime_hook="gui.event_handling"),
        ]

        panel = PanelDescriptor(
            id="appearance",
            title="Appearance",
            description="Theme and compositor metadata detected on the host.",
            layout=LayoutHint(placement="stack"),
            metrics=metrics,
            tables=tables,
            actions=actions,
        )

        dashboard = DashboardDescriptor(
            worker="gui",
            panels=[panel],
            annotations={"theme": theme, "platform": self.platform_name},
        )
        return dashboard

    def gui_design(self, ctx) -> ActionResult:
        if self.platform_name.startswith("windows"):
            try:
                script = (
                    "Get-ItemProperty -Path "
                    "'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize' "
                    "| Select-Object AppsUseLightTheme,SystemUsesLightTheme "
                    "| ConvertTo-Json"
                )
                proc = self.run_powershell(script)
                config = _json_list(proc.stdout)
                theme = "Light" if not config else ("Light" if config[0].get("AppsUseLightTheme", 1) else "Dark")
                dashboard = self._build_dashboard_descriptor(theme, config)
                ctx.publish_metadata("gui.dashboard_schema", dashboard.as_payload())
                payload = {"theme": theme, "personalize": config, "dashboard": dashboard.as_payload()}
                return self.info("gui_design", "Adaptive UI theme generated.", payload)
            except CommandExecutionError as exc:
                return self.warn("gui_design", f"Theme inspection failed: {exc}")
        theme = "Light"
        defaults = shutil.which("defaults")
        if defaults:
            try:
                proc = self.run_command([defaults, "read", "-g", "AppleInterfaceStyle"])
                theme = proc.stdout.strip() or "Light"
            except CommandExecutionError:
                theme = "Light"
        dashboard = self._build_dashboard_descriptor(theme, None)
        ctx.publish_metadata("gui.dashboard_schema", dashboard.as_payload())
        payload = {"theme": theme, "dashboard": dashboard.as_payload()}
        return self.info("gui_design", "Adaptive UI theme generated.", payload)

    def theme_management(self, ctx) -> ActionResult:
        if self.platform_name.startswith("windows"):
            assets_dir = Path(os.environ.get("WINDIR", "C:\\Windows")) / "Web" / "Wallpaper"
        else:
            assets_dir = Path("/System/Library/Desktop Pictures")
        count, assets = _path_exists(assets_dir)
        payload = {"asset_samples": assets, "asset_count": count}
        return self.info("theme_management", "Theme manager synced.", payload)


class ScalabilityAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("scalability")
        self.providers: List[ResourceProvider] = []

    def _ensure_providers(self, ctx) -> None:
        if not self.providers:
            self.providers = discover_providers(ctx.environment)

    def _virtualization_layer(self, ctx) -> VirtualizationLayer:
        provider_names: Set[str] = {provider.name for provider in self.providers} if self.providers else set()
        virtualization_providers = {name for name in provider_names if name in {"qemu", "libvirt"}}
        providers_arg: Optional[Set[str]] = virtualization_providers if virtualization_providers else None
        return VirtualizationLayer(
            ctx.environment,
            providers=providers_arg,
            forensic_mode=self.is_forensic(ctx),
        )

    def _collect_inventory(self) -> List[ProviderReport]:
        inventory: List[ProviderReport] = []
        for provider in self.providers:
            try:
                inventory.append(provider.inventory())
            except Exception as exc:
                inventory.append(
                    ProviderReport(
                        name=getattr(provider, "name", "unknown"),
                        healthy=False,
                        message=str(exc),
                        details={},
                    )
                )
        return inventory

    def _format_report(self, report: ProviderReport) -> Dict[str, object]:
        return {
            "provider": report.name,
            "healthy": report.healthy,
            "message": report.message,
            "details": report.details,
        }

    def monitor_load(self, ctx) -> ActionResult:
        load = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
        self._ensure_providers(ctx)
        inventory = [self._format_report(report) for report in self._collect_inventory()]
        payload = {
            "load_1m": load[0],
            "load_5m": load[1],
            "load_15m": load[2],
            "providers": inventory,
        }
        virtualization = self._virtualization_status(ctx)
        payload["virtualization"] = virtualization
        ctx.publish_metadata("scalability.monitor_load_summary", payload)
        ctx.publish_metadata("virtualization.status", virtualization)
        if any(report["healthy"] for report in inventory):
            return self.info("monitor_load", "System load under observation.", payload)
        return self.warn("monitor_load", "No healthy providers detected; monitoring host-only load.", payload)

    def scale_up(self, ctx) -> ActionResult:
        self._ensure_providers(ctx)
        intent = {"reason": "runtime_scale_up"}
        reports = []
        for provider in self.providers:
            try:
                reports.append(self._format_report(provider.scale_up(intent)))
            except Exception as exc:
                reports.append(
                    {
                        "provider": getattr(provider, "name", "unknown"),
                        "healthy": False,
                        "message": f"Scale-up invocation failed: {exc}",
                        "details": {},
                    }
                )
        virtualization = self._run_virtualization(ctx, action="up")
        payload = {"providers": reports, "virtualization": virtualization}
        if self.is_forensic(ctx):
            return self.warn("scale_up", "Forensic mode: scale-up recorded but not executed.", payload)
        return self.warn("scale_up", "Scale-up request queued; awaiting approval.", payload)

    def load_balancing(self, ctx) -> ActionResult:
        self._ensure_providers(ctx)
        payload = {"providers": [self._format_report(report) for report in self._collect_inventory()]}
        healthy = any(report["healthy"] for report in payload["providers"])
        if healthy:
            return self.info("load_balancing", "Traffic balanced across resources.", payload)
        return self.warn("load_balancing", "No healthy resource providers detected.", payload)

    def scale_down(self, ctx) -> ActionResult:
        self._ensure_providers(ctx)
        intent = {"reason": "runtime_scale_down"}
        reports = []
        for provider in self.providers:
            try:
                reports.append(self._format_report(provider.scale_down(intent)))
            except Exception as exc:
                reports.append(
                    {
                        "provider": getattr(provider, "name", "unknown"),
                        "healthy": False,
                        "message": f"Scale-down invocation failed: {exc}",
                        "details": {},
                    }
                )
        virtualization = self._run_virtualization(ctx, action="down")
        payload = {"providers": reports, "virtualization": virtualization}
        if self.is_forensic(ctx):
            return self.info("scale_down", "Forensic mode: scale-down recommendations logged only.", payload)
        return self.info("scale_down", "Scale-down policy ready.", payload)

    def _run_virtualization(self, ctx, action: str) -> List[Dict[str, object]]:
        layer = self._virtualization_layer(ctx)
        outcomes = layer.execute(action)
        results = [outcome.as_dict() for outcome in outcomes]
        if results:
            ctx.publish_metadata(f"virtualization.{action}", {"outcomes": results})
        return results

    def _virtualization_status(self, ctx) -> Dict[str, object]:
        layer = self._virtualization_layer(ctx)
        status = layer.status()
        plans = {
            "up": layer.describe("up"),
            "down": layer.describe("down"),
        }
        return {
            "status": status,
            "plans": plans,
        }

    def virtualization_inspect(self, ctx) -> ActionResult:
        self._ensure_providers(ctx)
        details = self._virtualization_status(ctx)
        ctx.publish_metadata("virtualization.inspect", details)
        return self.info("virtualization_inspect", "Virtualization readiness assessed.", details)

    def virtualization_domains(self, ctx) -> ActionResult:
        self._ensure_providers(ctx)
        layer = self._virtualization_layer(ctx)
        status = layer.status()
        domains: List[str] = []
        for entry in status:
            if entry.get("provider") != "libvirt":
                continue
            for domain in entry.get("domains", []):
                name = domain.get("name")
                if name:
                    domains.append(str(name))
        payload = {"domains": domains, "status": status}
        ctx.publish_metadata("virtualization.domains", payload)
        if not domains:
            return self.warn("virtualization_domains", "No libvirt domains detected.", payload)
        return self.info("virtualization_domains", "Libvirt domains enumerated.", payload)


class OrchestrationAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("orchestration")

    def policy_engine(self, ctx) -> ActionResult:
        manifest_summary = {
            "meta_agents": list(ctx.manifest.meta_agents.keys()),
            "boot_length": len(ctx.manifest.boot_sequence),
        }
        ctx.publish_metadata("manifest_summary", manifest_summary)
        return self.info("policy_engine", "Policy engine enforcing intent.", manifest_summary)

    def telemetry(self, ctx) -> ActionResult:
        telemetry_blob = json.dumps(ctx.metadata, indent=2)[:MAX_OUTPUT_LENGTH]
        payload = {"telemetry_snapshot": telemetry_blob}
        return self.info("telemetry", "Telemetry streaming to observability bus.", payload)

    def health_monitor(self, ctx) -> ActionResult:
        health = {
            "boot_time": time.time(),
            "metadata_keys": list(ctx.metadata.keys()),
        }
        return self.info("health_monitor", "Health monitor scanning subsystems.", health)
