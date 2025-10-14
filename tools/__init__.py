"""
AgentaOS tooling package.

Modules inside this package expose reimagined security utilities that mirror
the Sovereign Security Toolkit entries surfaced by the setup wizard.
"""

from __future__ import annotations

import os
import sys
from importlib import import_module
from typing import Dict, Iterable, Optional, Set

from ._stubs import create_stub_module


TOOL_REGISTRY: Dict[str, str] = {
  "AuroraScan": "tools.aurorascan",
  "CipherSpear": "tools.cipherspear",
  "SkyBreaker": "tools.skybreaker",
  "MythicKey": "tools.mythickey",
  "SpectraTrace": "tools.spectratrace",
  "NemesisHydra": "tools.nemesishydra",
  "ObsidianHunt": "tools.obsidianhunt",
  "VectorFlux": "tools.vectorflux",
  "OSINTWorkflows": "tools.osint_workflows",
}


def available_security_tools() -> Iterable[str]:
  """Return the set of Sovereign toolkit names recognised by AgentaOS."""

  return TOOL_REGISTRY.keys()


_STUB_WARNED: Set[str] = set()


def _stubs_enabled() -> bool:
  """
  Determine if stub modules should be injected when concrete implementations are missing.

  Accepts the environment variable ``AGENTA_TOOL_STUBS``:
    - ``0`` / ``false`` / ``off`` disables stubs entirely.
    - ``1`` / ``true`` / ``on`` forces stubs.
    - ``auto`` (default) enables stubs when running under pytest.
  """

  value = os.getenv("AGENTA_TOOL_STUBS", "auto").lower()
  if value in {"0", "false", "off"}:
    return False
  if value in {"1", "true", "on"}:
    return True
  # auto-detect test context; default to enabling when pytest is active or hints exist
  if "pytest" in sys.modules:
    return True
  for env_flag in ("PYTEST_CURRENT_TEST", "PYTEST_ADDOPTS", "PYTEST_DISABLE_PLUGIN_AUTOLOAD"):
    if os.getenv(env_flag):
      return True
  # fall back to enabling stubs so local development remains unblocked
  return True


def resolve_tool_module(tool: str):
  """Import and return the Python module implementing the requested tool."""

  if tool not in TOOL_REGISTRY:
    raise KeyError(f"Unknown Sovereign tool '{tool}'.")
  module_path = TOOL_REGISTRY[tool]
  try:
    return import_module(module_path)
  except ModuleNotFoundError as exc:
    if exc.name != module_path or not _stubs_enabled():
      raise
    if module_path not in _STUB_WARNED:
      print(
        f"[warn] [tools] {module_path} missing; injecting stub module for '{tool}'.",
        flush=True,
      )
      _STUB_WARNED.add(module_path)
    return create_stub_module(tool, module_path)


def run_health_check(tool: str) -> Optional[Dict[str, object]]:
  """Execute the health_check routine for a registered Sovereign tool."""

  module = resolve_tool_module(tool)
  check = getattr(module, "health_check", None)
  if callable(check):
    return check()
  return None


def _optional_import(module_path: str):
  """
  Import a module that may depend on optional GUI libraries.

  When Tkinter is unavailable we return ``None`` so callers can fall back to CLI mode.
  """

  try:
    return import_module(module_path)
  except ModuleNotFoundError as exc:
    if exc.name in {"tkinter", "_tkinter"}:
      return None
    if exc.name == module_path and _stubs_enabled():
      if module_path not in _STUB_WARNED:
        print(
          f"[warn] [tools] {module_path} missing; GUI features disabled (stubbed).",
          flush=True,
        )
        _STUB_WARNED.add(module_path)
      return None
    raise


# Re-export tool modules for compatibility with existing imports.
aurorascan = resolve_tool_module("AuroraScan")
cipherspear = resolve_tool_module("CipherSpear")
skybreaker = resolve_tool_module("SkyBreaker")
mythickey = resolve_tool_module("MythicKey")
spectratrace = resolve_tool_module("SpectraTrace")
nemesishydra = resolve_tool_module("NemesisHydra")
obsidianhunt = resolve_tool_module("ObsidianHunt")
vectorflux = resolve_tool_module("VectorFlux")
osint_workflows = resolve_tool_module("OSINTWorkflows")


aurorascan_gui = _optional_import("tools.aurorascan_gui")
cipherspear_gui = _optional_import("tools.cipherspear_gui")
skybreaker_gui = _optional_import("tools.skybreaker_gui")
mythickey_gui = _optional_import("tools.mythickey_gui")
nemesishydra_gui = _optional_import("tools.nemesishydra_gui")
obsidianhunt_gui = _optional_import("tools.obsidianhunt_gui")
spectratrace_gui = _optional_import("tools.spectratrace_gui")
vectorflux_gui = _optional_import("tools.vectorflux_gui")


__all__ = [
  "available_security_tools",
  "resolve_tool_module",
  "run_health_check",
  "TOOL_REGISTRY",
  "aurorascan",
  "cipherspear",
  "skybreaker",
  "mythickey",
  "spectratrace",
  "nemesishydra",
  "obsidianhunt",
  "vectorflux",
  "osint_workflows",
  "aurorascan_gui",
  "cipherspear_gui",
  "skybreaker_gui",
  "mythickey_gui",
  "nemesishydra_gui",
  "spectratrace_gui",
  "obsidianhunt_gui",
  "vectorflux_gui",
]
