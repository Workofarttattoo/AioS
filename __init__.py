"""
Compatibility package that mirrors the legacy ``Ai:oS`` namespace.

Many integration tests and third-party extensions still reference
``Ai:oS.virtualization``. The canonical implementation now lives in
``aios.virtualization``; this shim keeps old imports working while the codebase
converges on the new layout.
"""

from __future__ import annotations

try:
    from . import virtualization  # noqa: F401
except ImportError:
    pass

__all__ = ["virtualization"]
