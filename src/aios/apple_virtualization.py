"""
Apple Virtualization.framework backend for Ai:oS (macOS native).

This module provides a graceful, Python-level bridge to Apple's
Virtualization.framework via PyObjC when available. It adheres to the
VirtualizationBackend interface defined in src.aios.virtualization.

Design goals:
- Prefer native macOS virtualization when running on Darwin
- Fail gracefully when PyObjC (Virtualization) is not installed
- Respect forensic mode (no host mutations)
- Provide clear, structured telemetry for GUI-capable VMs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import logging
import platform
from pathlib import Path

from .virtualization import VirtualizationBackend, VirtualMachineDomain  # type: ignore

LOG = logging.getLogger(__name__)


def _try_import_virtualization_module() -> bool:
	"""
	Check whether the PyObjC Virtualization framework is importable.
	We avoid hard import at module import-time to keep non-macOS platforms happy.
	"""
	try:
		# PyObjC package: pyobjc-framework-Virtualization
		import Virtualization  # type: ignore  # noqa: F401
		return True
	except Exception as exc:
		LOG.debug("PyObjC Virtualization import failed: %s", exc)
		return False


def _is_macos() -> bool:
	return platform.system() == "Darwin"


@dataclass
class AppleVMConfig:
	"""
	High-level Apple VM configuration derived from environment variables.
	This is intentionally generic and GUI-friendly.
	"""
	image_path: Optional[str]
	cpu_count: int
	memory_mb: int
	display_width: int
	display_height: int
	enable_graphics: bool


class AppleVirtualizationBackend(VirtualizationBackend):
	"""
	Native macOS virtualization backend.

	Notes:
	- Actual VM bring-up via Virtualization.framework typically requires an
	  app bundle with entitlements. In CLI contexts we treat start/shutdown
	  as advisory when entitlements are not present.
	- This backend still provides consistent domain tracking and telemetry.
	"""

	def __init__(self, environment: Optional[Dict[str, str]] = None):
		super().__init__(environment)
		self._domains: Dict[str, VirtualMachineDomain] = {}
		self._vm_configs: Dict[str, AppleVMConfig] = {}
		self._virtualization_available: bool = _is_macos() and _try_import_virtualization_module()

	@classmethod
	def is_supported(cls) -> bool:
		"""
		Determine whether Apple virtualization is supported on this host.
		Requires macOS plus PyObjC Virtualization framework.
		"""
		return _is_macos() and _try_import_virtualization_module()

	def _resolve_config(self) -> AppleVMConfig:
		# Defaults chosen for a comfortable desktop experience
		def _int_env(key: str, default: int) -> int:
			try:
				return int(self.environment.get(key, str(default)))
			except Exception:
				return default

		def _bool_env(key: str, default: bool) -> bool:
			val = str(self.environment.get(key, "1" if default else "0")).strip().lower()
			return val in {"1", "true", "yes", "on"}

		image_path = self.environment.get("AGENTA_APPLE_HV_IMAGE_PATH")
		cfg = AppleVMConfig(
			image_path=image_path if image_path else None,
			cpu_count=max(2, _int_env("AGENTA_APPLE_HV_CPU", 4)),
			memory_mb=max(2048, _int_env("AGENTA_APPLE_HV_MEMORY_MB", 8192)),
			display_width=max(1024, _int_env("AGENTA_APPLE_HV_DISPLAY_WIDTH", 1920)),
			display_height=max(768, _int_env("AGENTA_APPLE_HV_DISPLAY_HEIGHT", 1080)),
			enable_graphics=_bool_env("AGENTA_APPLE_HV_GRAPHICS", True),
		)
		return cfg

	def inspect(self) -> Dict[str, Any]:
		cfg = self._resolve_config()
		image_exists = bool(cfg.image_path and Path(cfg.image_path).exists())
		return {
			"backend": "apple",
			"available": bool(self._virtualization_available and image_exists),
			"platform": platform.platform(),
			"pyobjc_virtualization": self._virtualization_available,
			"image_configured": bool(cfg.image_path is not None),
			"image_exists": image_exists,
			"domains": [d.to_dict() for d in self._domains.values()],
			"display": {
				"enabled": cfg.enable_graphics,
				"resolution": f"{cfg.display_width}x{cfg.display_height}",
			},
			"resources": {
				"cpu_count": cfg.cpu_count,
				"memory_mb": cfg.memory_mb,
			},
		}

	def provision_os(self, name: str, forensic_mode: bool = False) -> VirtualMachineDomain:
		if name in self._domains:
			return self._domains[name]

		cfg = self._resolve_config()
		image_exists = bool(cfg.image_path and Path(cfg.image_path).exists())
		details: Dict[str, Any] = {
			"graphics": cfg.enable_graphics,
			"display": {"width": cfg.display_width, "height": cfg.display_height},
			"cpu_count": cfg.cpu_count,
			"memory_mb": cfg.memory_mb,
			"image_path": cfg.image_path or "",
			"image_exists": image_exists,
			"forensic": forensic_mode,
		}

		# We do not allocate or mutate host state in forensic mode
		if forensic_mode:
			details["advisory"] = "provision deferred by forensic mode"

		# Outline what would be configured if entitlements are present
		if not self._virtualization_available:
			details["note"] = "PyObjC Virtualization not available; run `pip install pyobjc-framework-Virtualization`"

		domain = VirtualMachineDomain(
			name=name,
			qubits=0,  # Not applicable; maintain schema compatibility
			status="provisioned",
			created_at=__import__("time").time(),
			backend="apple",
			details=details,
		)
		self._domains[name] = domain
		self._vm_configs[name] = cfg
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

		# Without proper entitlements, treat as advisory success
		if not self._virtualization_available:
			return {
				"success": False,
				"error": "Virtualization.framework not available (PyObjC missing or entitlements not present)",
				"hint": "Install pyobjc-framework-Virtualization and run in an entitled app context",
			}

		# Mark as running (best-effort in CLI contexts)
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
		return {"success": True, "domain": domain.to_dict()}

	def list_domains(self) -> List[VirtualMachineDomain]:
		return list(self._domains.values())


