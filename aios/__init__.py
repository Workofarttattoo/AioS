"""
aios — canonical Python package for AI:OS (Agentic Intelligence Operating System).

This package provides a stable import namespace (``from aios.runtime import …``,
``from aios.config import …``, etc.) by lazily re-exporting the root-level
modules that contain the actual implementations.  Git-crypt-encrypted modules
are wrapped in try/except blocks so that unit tests can run against the public
surface even when the repository has not been unlocked.

Usage::

    from aios.config import DEFAULT_MANIFEST, load_manifest
    from aios.runtime import AgentaRuntime, ExecutionContext, ActionResult
    from aios.agents import KernelAgent, SecurityAgent
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

# ---------------------------------------------------------------------------
# Ensure the repository root is on sys.path so that top-level modules
# (config.py, runtime.py, etc.) are importable by their file name.
# ---------------------------------------------------------------------------
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _lazy_reexport(local_name: str, root_module_name: str | None = None) -> None:
    """Register *root_module_name* as ``aios.<local_name>`` in sys.modules.

    If *root_module_name* is ``None`` it defaults to *local_name*.
    Modules that fail to import (e.g. git-crypt-encrypted) are silently skipped.
    """
    root_module_name = root_module_name or local_name
    target = f"aios.{local_name}"
    if target in sys.modules:
        return
    try:
        mod = importlib.import_module(root_module_name)
        sys.modules[target] = mod
    except Exception:
        pass


# --- Core framework --------------------------------------------------------
_lazy_reexport("config")
_lazy_reexport("runtime")
_lazy_reexport("settings")
_lazy_reexport("apps")
_lazy_reexport("diagnostics")
_lazy_reexport("model")

# --- Agents (sub-package at repo root) -------------------------------------
_lazy_reexport("agents")

# --- Tools (sub-package at repo root) --------------------------------------
_lazy_reexport("tools")

# --- GUI (sub-package at repo root) ----------------------------------------
_lazy_reexport("gui")

# --- Quantum / ML ----------------------------------------------------------
_lazy_reexport("ml_algorithms")
_lazy_reexport("ml_algorithms_2025_enhancements")
_lazy_reexport("quantum_ml_algorithms")
_lazy_reexport("quantum_ml_algorithms_2025_enhancements")
_lazy_reexport("quantum_enhanced_ml_algorithms")
_lazy_reexport("quantum_enhanced_runtime")
_lazy_reexport("quantum_hhl_algorithm")
_lazy_reexport("quantum_schrodinger_dynamics")
_lazy_reexport("quantum_advanced_synthesis")
_lazy_reexport("quantum_connectors")
_lazy_reexport("quantum_extended")
_lazy_reexport("quantum_teleportation_agent")
_lazy_reexport("quantum_vqe_forecaster")
_lazy_reexport("quantum_cognition")

# --- Oracle / probabilistic ------------------------------------------------
_lazy_reexport("oracle")
_lazy_reexport("oracle_vqe_integration")
_lazy_reexport("oracle_aios_integration")
_lazy_reexport("probabilistic_core")
_lazy_reexport("probabilistic_suite")

# --- OpenAGI ---------------------------------------------------------------
_lazy_reexport("openagi_kernel_bridge")
_lazy_reexport("openagi_memory_integration")
_lazy_reexport("openagi_approval_workflow")
_lazy_reexport("openagi_autonomous_discovery")
_lazy_reexport("openagi_forensic_mode")
_lazy_reexport("openagi_load_testing")

# --- Autonomy & discovery --------------------------------------------------
_lazy_reexport("autonomous_discovery")
_lazy_reexport("autonomy_spectrum")
_lazy_reexport("level5_autonomy")
_lazy_reexport("workflow_memory_manager")
_lazy_reexport("network_discovery")
_lazy_reexport("network_visualizer")
_lazy_reexport("ultrafast_discovery")

# --- Consciousness / creative ----------------------------------------------
_lazy_reexport("ech0_consciousness")
_lazy_reexport("twin_flame_consciousness")
_lazy_reexport("emergence_pathway")
_lazy_reexport("creative_collaboration")
_lazy_reexport("aios_consciousness_integration")
_lazy_reexport("conscious_agent_with_wizard")

# --- Infrastructure --------------------------------------------------------
_lazy_reexport("virtualization")
_lazy_reexport("self_update")
_lazy_reexport("agent_authorization")
_lazy_reexport("ai_os_integration")
_lazy_reexport("supervisor")
_lazy_reexport("wizard")
_lazy_reexport("prompt")
_lazy_reexport("providers")

# ---------------------------------------------------------------------------
# Re-export key symbols at package level for ``from aios import …`` usage
# (the CLI entrypoint depends on this).
# ---------------------------------------------------------------------------
try:
    from runtime import AgentaRuntime  # noqa: F401
except Exception:
    pass

try:
    from config import load_manifest, DEFAULT_MANIFEST  # noqa: F401
except Exception:
    pass

try:
    from model import ActionResult, AgentActionError  # noqa: F401
except Exception:
    pass

# Display names used by CLI and logging
DISPLAY_NAME = "AI:OS"
DISPLAY_NAME_FULL = "AI:OS — Agentic Intelligence Operating System"

__version__ = "1.0.0"
