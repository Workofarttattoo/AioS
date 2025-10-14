"""
Lightweight stubs for Sovereign Security Toolkit modules.

The production implementations live in encrypted packages. During local testing
or documentation builds those modules may be absent, so we inject predictable
stub modules that satisfy importer expectations without executing real tooling.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Dict, Callable


def create_stub_module(tool_name: str, module_path: str) -> ModuleType:
  """Register and return a placeholder module for ``tool_name``."""

  module = ModuleType(module_path)
  module.__dict__.update(
    {
      "__doc__": (
        f"Stub module for {tool_name}. "
        "Injected automatically when the concrete implementation is "
        "unavailable so tests can import the toolkit package."
      ),
      "TOOL_NAME": tool_name,
      "IS_STUB": True,
      "health_check": _build_health_check(tool_name),
      "run": _build_run(tool_name),
      "list_modules": _build_list_modules(),
    }
  )
  module.__getattr__ = _build_attr_proxy(tool_name)  # type: ignore[attr-defined]
  sys.modules[module_path] = module
  return module


def _build_health_check(tool_name: str):
    def health_check() -> Dict[str, str]:
        return {
            "status": "unavailable",
            "tool": tool_name,
            "summary": "Stubbed toolkit component used for tests.",
        }

    return health_check


def _build_run(tool_name: str):
  def run(*_args, **_kwargs):
    raise RuntimeError(
      f"The {tool_name} toolkit component is stubbed for this environment."
    )

  return run


def _build_list_modules():
  def list_modules() -> Dict[str, str]:
    return {}

  return list_modules


def _build_attr_proxy(tool_name: str) -> Callable[[str], Callable[..., object]]:
  def proxy(attr: str):
    def _stub(*_args, **_kwargs):
      raise RuntimeError(
        f"The attribute '{attr}' on {tool_name} is unavailable in stub mode."
      )

    return _stub

  return proxy


__all__ = ["create_stub_module"]
