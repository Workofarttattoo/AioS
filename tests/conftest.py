"""
AI:OS test configuration.

Provides fixtures and skip-markers for tests that depend on git-crypt
encrypted modules.  When the repo is locked, those tests are automatically
skipped with a clear reason.
"""

import importlib
import sys
from pathlib import Path

# Ensure repo root is on sys.path
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _module_available(name: str) -> bool:
    """Check if a module can be imported (not git-crypt encrypted)."""
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


# Detect which modules are available (vs encrypted)
RUNTIME_AVAILABLE = _module_available("runtime")
ML_ALGORITHMS_AVAILABLE = _module_available("ml_algorithms")
ORACLE_AVAILABLE = _module_available("oracle")
QUANTUM_AVAILABLE = _module_available("quantum.engine")
OPENAGI_BRIDGE_AVAILABLE = _module_available("openagi_kernel_bridge")

try:
    import pytest

    # Register markers
    requires_runtime = pytest.mark.skipif(
        not RUNTIME_AVAILABLE,
        reason="runtime.py is git-crypt encrypted"
    )
    requires_ml = pytest.mark.skipif(
        not ML_ALGORITHMS_AVAILABLE,
        reason="ml_algorithms.py is git-crypt encrypted"
    )
    requires_oracle = pytest.mark.skipif(
        not ORACLE_AVAILABLE,
        reason="oracle.py is git-crypt encrypted"
    )
    requires_quantum = pytest.mark.skipif(
        not QUANTUM_AVAILABLE,
        reason="quantum modules are git-crypt encrypted"
    )
except ImportError:
    pass
