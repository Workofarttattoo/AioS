"""
Quantum-backed Virtualization Layer for Ai:oS.

Provides a simple virtualization facade with a Quantum backend that simulates
guest environments atop a 30-qubit (configurable) quantum state engine.

Key features:
- Provision "domains" (guest OS instances) on the quantum simulator
- Start/stop lifecycle with forensic-mode safeguards
- Capability inspection and domain listing
- Telemetry-friendly, JSON-serializable payloads
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import sys
import platform

# Ensure repository root is importable (matches pattern used by runtime.py)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

LOG = logging.getLogger(__name__)


# Optional quantum engine import with graceful degradation
try:
    from quantum_ml_algorithms import QuantumStateEngine  # type: ignore
    QUANTUM_AVAILABLE = True
except Exception as exc:
    QUANTUM_AVAILABLE = False
    QuantumStateEngine = None  # type: ignore
    LOG.warning("QuantumStateEngine unavailable: %s", exc)

# Optional Apple Virtualization backend import
try:
    from .apple_virtualization import AppleVirtualizationBackend  # type: ignore
    APPLE_BACKEND_SUPPORTED = AppleVirtualizationBackend.is_supported()
except Exception as exc:
    AppleVirtualizationBackend = None  # type: ignore
    APPLE_BACKEND_SUPPORTED = False
    LOG.debug("Apple virtualization backend not available: %s", exc)


@dataclass
class VirtualMachineDomain:
    """
    Represents a provisioned guest "domain" (OS instance) on the virtualization layer.
    """
    name: str
    qubits: int
    status: str  # provisioned | running | stopped | error
    created_at: float
    backend: str = "quantum"
    details: Dict[str, Any] = None  # Additional backend-specific info

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Normalize None for JSON stability
        data["details"] = data.get("details") or {}
        return data


class VirtualizationBackend:
    """
    Backend interface for virtualization providers.

    Subclasses should implement:
      - inspect()
      - provision_os()
      - start()
      - shutdown()
      - list_domains()
    """

    def __init__(self, environment: Optional[Dict[str, str]] = None):
        self.environment = environment or {}

    # Interface methods
    def inspect(self) -> Dict[str, Any]:
        raise NotImplementedError

    def provision_os(self, name: str, forensic_mode: bool = False) -> VirtualMachineDomain:
        raise NotImplementedError

    def start(self, name: str, forensic_mode: bool = False) -> Dict[str, Any]:
        raise NotImplementedError

    def shutdown(self, name: str, forensic_mode: bool = False) -> Dict[str, Any]:
        raise NotImplementedError

    def list_domains(self) -> List[VirtualMachineDomain]:
        raise NotImplementedError


class QuantumVirtualizationBackend(VirtualizationBackend):
    """
    Virtualization backend backed by a quantum state simulator.

    Uses QuantumStateEngine with:
      - 1-20 qubits: exact statevector
      - 20-40 qubits: tensor network approximation
      - 40-50 qubits: MPS compression
    """

    def __init__(self, environment: Optional[Dict[str, str]] = None):
        super().__init__(environment)
        self._domains: Dict[str, VirtualMachineDomain] = {}
        self._engines: Dict[str, Any] = {}  # name -> QuantumStateEngine

    # Internal helpers
    def _resolve_qubits(self) -> int:
        try:
            return int(self.environment.get("AGENTA_QUANTUM_QUBITS", "30"))
        except ValueError:
            return 30

    def _engine_backend_for_qubits(self, qubits: int) -> str:
        if qubits <= 20:
            return "statevector"
        if qubits <= 40:
            return "tensor_network"
        return "mps"

    # Interface implementations
    def inspect(self) -> Dict[str, Any]:
        qubits = self._resolve_qubits()
        available = QUANTUM_AVAILABLE
        backend_mode = self._engine_backend_for_qubits(qubits)
        info = {
            "backend": "quantum",
            "available": available,
            "qubits": qubits,
            "engine_mode": backend_mode,
            "domains": [d.to_dict() for d in self._domains.values()],
        }
        return info

    def provision_os(self, name: str, forensic_mode: bool = False) -> VirtualMachineDomain:
        if name in self._domains:
            return self._domains[name]

        qubits = self._resolve_qubits()

        if forensic_mode:
            # Do not allocate engines in forensic mode; record advisory-only domain.
            domain = VirtualMachineDomain(
                name=name,
                qubits=qubits,
                status="provisioned",
                created_at=time.time(),
                details={"forensic": True, "advisory": "provision deferred by forensic mode"},
            )
            self._domains[name] = domain
            return domain

        if not QUANTUM_AVAILABLE or QuantumStateEngine is None:
            domain = VirtualMachineDomain(
                name=name,
                qubits=qubits,
                status="error",
                created_at=time.time(),
                details={"error": "Quantum engine unavailable (requires PyTorch)"},
            )
            self._domains[name] = domain
            return domain

        # Allocate engine and create domain
        try:
            engine = QuantumStateEngine(num_qubits=qubits)
            self._engines[name] = engine
            domain = VirtualMachineDomain(
                name=name,
                qubits=qubits,
                status="provisioned",
                created_at=time.time(),
                details={"engine_mode": getattr(engine, "backend", self._engine_backend_for_qubits(qubits))},
            )
            self._domains[name] = domain
            return domain
        except Exception as exc:
            LOG.exception("Failed to provision quantum OS domain '%s': %s", name, exc)
            domain = VirtualMachineDomain(
                name=name,
                qubits=qubits,
                status="error",
                created_at=time.time(),
                details={"error": str(exc)},
            )
            self._domains[name] = domain
            return domain

    def start(self, name: str, forensic_mode: bool = False) -> Dict[str, Any]:
        domain = self._domains.get(name)
        if not domain:
            return {"success": False, "error": f"domain '{name}' not found"}

        if forensic_mode:
            return {
                "success": True,
                "forensic": True,
                "advisory": f"would start domain '{name}'",
                "domain": domain.to_dict(),
            }

        if domain.status == "running":
            return {"success": True, "message": "already running", "domain": domain.to_dict()}

        # Engine should exist for non-forensic domains; tolerate missing engine
        engine = self._engines.get(name)
        if engine is None and QUANTUM_AVAILABLE and QuantumStateEngine is not None:
            try:
                engine = QuantumStateEngine(num_qubits=domain.qubits)
                self._engines[name] = engine
            except Exception as exc:
                LOG.warning("Lazy engine init failed for '%s': %s", name, exc)

        domain.status = "running"
        return {"success": True, "domain": domain.to_dict()}

    def shutdown(self, name: str, forensic_mode: bool = False) -> Dict[str, Any]:
        domain = self._domains.get(name)
        if not domain:
            return {"success": False, "error": f"domain '{name}' not found"}

        if forensic_mode:
            return {
                "success": True,
                "forensic": True,
                "advisory": f"would shutdown domain '{name}'",
                "domain": domain.to_dict(),
            }

        domain.status = "stopped"
        # We keep the engine allocated for quick restart; could be freed if needed
        return {"success": True, "domain": domain.to_dict()}

    def list_domains(self) -> List[VirtualMachineDomain]:
        return list(self._domains.values())


class VirtualizationManager:
    """
    Simple manager to resolve and operate the selected virtualization backend.
    """

    def __init__(self, environment: Optional[Dict[str, str]] = None):
        self.environment = environment or {}
        self.backend_name = self._resolve_backend_name()
        self.backend = self._create_backend(self.backend_name)

    def _resolve_backend_name(self) -> str:
        # Explicit override takes precedence
        explicit = (self.environment.get("AGENTA_VIRTUALIZATION_BACKEND") or "").strip().lower()
        if explicit:
            return explicit

        # Provider list hint
        providers = (self.environment.get("AGENTA_PROVIDER") or "").lower()
        if "apple" in providers or "apple_virtualization" in providers or "apple_hv" in providers:
            return "apple"
        if "quantum" in providers or "quantum_vm" in providers:
            return "quantum"

        # Default: prefer native Apple virtualization on macOS when supported
        if platform.system() == "Darwin" and APPLE_BACKEND_SUPPORTED:
            return "apple"

        # Fallback to quantum backend to enable simulation-first behavior
        return "quantum"

    def _create_backend(self, name: str) -> VirtualizationBackend:
        if name == "quantum":
            return QuantumVirtualizationBackend(self.environment)
        if name == "apple":
            if AppleVirtualizationBackend is not None:
                return AppleVirtualizationBackend(self.environment)  # type: ignore
            LOG.warning("Requested backend 'apple' not available on this platform; falling back to 'quantum'.")
            return QuantumVirtualizationBackend(self.environment)
        if name in {"qemu", "libvirt", "docker"}:
            LOG.warning("Requested backend '%s' not yet implemented; falling back to 'quantum'.", name)
            return QuantumVirtualizationBackend(self.environment)
        LOG.warning("Unknown virtualization backend '%s'; falling back to 'quantum'.", name)
        return QuantumVirtualizationBackend(self.environment)

    def inspect(self) -> Dict[str, Any]:
        return self.backend.inspect()

    def ensure_os_domain(
        self,
        name: str,
        autostart: bool = False,
        forensic_mode: bool = False
    ) -> Tuple[VirtualMachineDomain, Optional[Dict[str, Any]]]:
        """
        Ensure that a domain exists; optionally autostart it.
        Returns (domain, start_result).
        """
        domain = self.backend.provision_os(name=name, forensic_mode=forensic_mode)
        start_result = None
        if autostart:
            start_result = self.backend.start(name=name, forensic_mode=forensic_mode)
        return domain, start_result

    def list_domains(self) -> List[VirtualMachineDomain]:
        return self.backend.list_domains()


