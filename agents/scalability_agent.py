"""
ScalabilityAgent - Load Monitoring & Resource Scaling

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import logging
import psutil
import time
from typing import Dict, List, Optional, Any
from collections import deque
from pathlib import Path
import sys
import platform

# Add parent directory to path for imports (align with other agents)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import ActionResult with fallback (mirrors security_agent pattern)
try:
    from src.aios.runtime import ActionResult, ExecutionContext  # type: ignore
except ImportError:
    from dataclasses import dataclass, field

    @dataclass
    class ActionResult:
        success: bool
        message: str
        payload: Dict[str, Any] = field(default_factory=dict)
        latency_ms: float = 0.0
        cost_usd: float = 0.0
        error: Optional[str] = None
        error_type: Optional[str] = None
        metadata: Dict[str, Any] = field(default_factory=dict)

    class ExecutionContext:
        def __init__(self, manifest=None, environment=None):
            self.manifest = manifest
            self.environment = environment or {}
            self.metadata = {}
            self.forensic_mode = False

# Virtualization manager (quantum backend)
try:
    from src.aios.virtualization import VirtualizationManager  # type: ignore
except Exception:
    VirtualizationManager = None  # type: ignore

LOG = logging.getLogger(__name__)


class ScalabilityAgent:
    """
    Meta-agent for monitoring system load and orchestrating scaling.

    Responsibilities:
    - Real-time load monitoring
    - Automatic scaling decisions
    - Provider management (Docker, cloud, VM)
    - Performance optimization
    - Cost monitoring
    """

    def __init__(self, window_size: int = 10):
        self.name = "scalability"
        self.window_size = window_size
        self.cpu_history = deque(maxlen=window_size)
        self.memory_history = deque(maxlen=window_size)
        self.disk_history = deque(maxlen=window_size)
        self.scaling_decisions = []
        LOG.info("ScalabilityAgent initialized")

    def get_current_load(self) -> Dict:
        """Get current system load metrics."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        network = psutil.net_io_counters()

        # Record history
        self.cpu_history.append(cpu_percent)
        self.memory_history.append(memory.percent)
        self.disk_history.append(disk.percent)

        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_mb": memory.available / 1024 / 1024,
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / (1024**3),
            "network_bytes_sent": network.bytes_sent,
            "network_bytes_recv": network.bytes_recv,
        }

    def get_average_load(self) -> Dict:
        """Get average load over the history window."""
        return {
            "avg_cpu_percent": sum(self.cpu_history) / len(self.cpu_history) if self.cpu_history else 0,
            "avg_memory_percent": sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0,
            "avg_disk_percent": sum(self.disk_history) / len(self.disk_history) if self.disk_history else 0,
            "sample_count": len(self.cpu_history),
        }

    def should_scale_up(self, threshold: float = 80.0) -> bool:
        """Determine if we should scale up resources."""
        current = self.get_current_load()
        avg = self.get_average_load()

        # Scale up if average CPU or memory is high
        should_scale = (avg["avg_cpu_percent"] > threshold or
                       avg["avg_memory_percent"] > threshold)

        if should_scale:
            LOG.warning(f"Scale-up triggered: CPU={avg['avg_cpu_percent']:.1f}%, Memory={avg['avg_memory_percent']:.1f}%")

        return should_scale

    def should_scale_down(self, threshold: float = 30.0) -> bool:
        """Determine if we should scale down resources."""
        current = self.get_current_load()
        avg = self.get_average_load()

        # Scale down if average CPU and memory are low
        should_scale = (avg["avg_cpu_percent"] < threshold and
                       avg["avg_memory_percent"] < threshold)

        if should_scale:
            LOG.info(f"Scale-down possible: CPU={avg['avg_cpu_percent']:.1f}%, Memory={avg['avg_memory_percent']:.1f}%")

        return should_scale

    def estimate_resource_needs(self) -> Dict:
        """Estimate required resources based on current usage."""
        current = self.get_current_load()
        avg = self.get_average_load()

        # Estimate needed CPU cores
        needed_cpu_cores = max(1, int(avg["avg_cpu_percent"] / 20))  # ~20% per core

        # Estimate needed RAM
        current_memory = psutil.virtual_memory().total / (1024**3)
        needed_memory_gb = current_memory * (avg["avg_memory_percent"] / 100) * 1.2  # 20% headroom

        return {
            "estimated_cpu_cores": needed_cpu_cores,
            "estimated_memory_gb": needed_memory_gb,
            "current_memory_gb": current_memory,
            "headroom_percent": 20,
        }

    def predict_resource_exhaustion(self) -> Dict:
        """Predict when resources might be exhausted."""
        if len(self.memory_history) < 3:
            return {"can_predict": False, "reason": "insufficient history"}

        # Simple linear trend prediction
        memory_trend = (
            list(self.memory_history)[-1] - list(self.memory_history)[0]
        ) / max(1, len(self.memory_history) - 1)

        current_memory = list(self.memory_history)[-1]

        if memory_trend > 0:
            # Estimate when we'll hit 95% memory
            samples_to_95 = (95 - current_memory) / memory_trend if memory_trend > 0 else float('inf')
            minutes_to_exhaustion = samples_to_95  # Each sample is ~1 second for our purposes

            return {
                "can_predict": True,
                "current_memory_percent": current_memory,
                "memory_trend_percent_per_sample": memory_trend,
                "estimated_minutes_to_95_percent": minutes_to_exhaustion,
                "action_needed": minutes_to_exhaustion < 5,
            }
        else:
            return {
                "can_predict": True,
                "current_memory_percent": current_memory,
                "memory_trend": "stable or decreasing",
                "action_needed": False,
            }

    def recommend_provider(self) -> Dict:
        """Recommend which provider to use for scaling."""
        current = self.get_current_load()
        avg = self.get_average_load()

        recommendations = {
            "providers": [],
            "reasoning": [],
        }

        # Prefer native Apple virtualization on macOS for GUI workloads
        if platform.system() == "Darwin":
            recommendations["providers"].append("apple_virtualization")
            recommendations["reasoning"].append("macOS host - native Virtualization.framework available")

        # Docker for container workloads
        if avg["avg_cpu_percent"] > 50:
            recommendations["providers"].append("docker")
            recommendations["reasoning"].append("High CPU - consider containerization")

        # Cloud for variable load
        if abs(list(self.cpu_history)[-1] - (sum(self.cpu_history) / len(self.cpu_history))) > 20:
            recommendations["providers"].append("aws_ec2")
            recommendations["reasoning"].append("Volatile load - cloud autoscaling recommended")

        # VM for stable workloads
        if avg["avg_cpu_percent"] > 60 and avg["avg_memory_percent"] > 60:
            recommendations["providers"].append("qemu_libvirt")
            recommendations["reasoning"].append("Sustained high load - dedicated VM")

        # Default fallback
        if not recommendations["providers"]:
            recommendations["providers"].append("local")
            recommendations["reasoning"].append("Load is normal - no scaling needed")

        return recommendations

    def record_scaling_decision(self, decision: Dict) -> None:
        """Record a scaling decision for audit trail."""
        decision["timestamp"] = time.time()
        self.scaling_decisions.append(decision)
        LOG.info(f"Recorded scaling decision: {decision}")

    def get_scaling_history(self, limit: int = 10) -> List[Dict]:
        """Get recent scaling decisions."""
        return self.scaling_decisions[-limit:]

    # ═══════════════════════════════════════════════════════════════════════
    # VIRTUALIZATION ACTIONS (Quantum-backed virtualization layer)
    # ═══════════════════════════════════════════════════════════════════════
    def _bool_env(self, ctx: ExecutionContext, key: str, default: bool = False) -> bool:
        val = str(ctx.environment.get(key, "1" if default else "0")).strip().lower()
        return val in {"1", "true", "yes", "on"}

    def virtualization_inspect(self, ctx: ExecutionContext, payload: Dict[str, Any]) -> ActionResult:
        """
        Assess virtualization readiness and ensure an OS domain exists on the selected backend.
        Uses the quantum virtualization backend by default or when explicitly selected.
        """
        try:
            if VirtualizationManager is None:
                return ActionResult(
                    success=False,
                    message="[error] Virtualization manager unavailable",
                    error="virtualization module not found"
                )

            manager = VirtualizationManager(ctx.environment)
            status = manager.inspect()

            domain_name = str(ctx.environment.get("AGENTA_OS_DOMAIN", "aios-os")).strip() or "aios-os"
            autostart = self._bool_env(ctx, "AGENTA_QUANTUM_AUTOSTART", default=True)
            if ctx.forensic_mode:
                autostart = False

            domain, start_result = manager.ensure_os_domain(
                name=domain_name,
                autostart=autostart,
                forensic_mode=ctx.forensic_mode,
            )

            domains_snapshot = [d.to_dict() for d in manager.list_domains()]
            result_payload = {
                "backend": status.get("backend"),
                "available": status.get("available"),
                "qubits": status.get("qubits"),
                "engine_mode": status.get("engine_mode"),
                "autostart": autostart,
                "forensic": ctx.forensic_mode,
                "domain": domain.to_dict(),
                "domains": domains_snapshot,
            }
            if start_result is not None:
                result_payload["start"] = start_result

            message = (
                f"[info] virtualization: backend={status.get('backend')} "
                f"qubits={status.get('qubits')} domains={len(domains_snapshot)}"
            )

            return ActionResult(
                success=bool(status.get("backend") == "quantum"),
                message=message,
                payload=result_payload
            )
        except Exception as exc:
            LOG.exception("virtualization_inspect failed: %s", exc)
            return ActionResult(
                success=False,
                message=f"[error] virtualization_inspect failed: {exc}",
                error=str(exc),
                error_type=type(exc).__name__
            )

    def virtualization_domains(self, ctx: ExecutionContext, payload: Dict[str, Any]) -> ActionResult:
        """
        List available virtualization domains on the active backend.
        """
        try:
            if VirtualizationManager is None:
                return ActionResult(
                    success=False,
                    message="[error] Virtualization manager unavailable",
                    error="virtualization module not found"
                )

            manager = VirtualizationManager(ctx.environment)
            # Ensure default domain exists so listing reflects expected OS domain
            default_domain = str(ctx.environment.get("AGENTA_OS_DOMAIN", "aios-os")).strip() or "aios-os"
            manager.ensure_os_domain(name=default_domain, autostart=False, forensic_mode=ctx.forensic_mode)
            domains = [d.to_dict() for d in manager.list_domains()]
            names = [d["name"] for d in domains]
            msg = f"[info] virtualization domains: {len(domains)} found"
            return ActionResult(
                success=True,
                message=msg,
                payload={"domains": names, "details": domains}
            )
        except Exception as exc:
            LOG.exception("virtualization_domains failed: %s", exc)
            return ActionResult(
                success=False,
                message=f"[error] virtualization_domains failed: {exc}",
                error=str(exc),
                error_type=type(exc).__name__
            )
