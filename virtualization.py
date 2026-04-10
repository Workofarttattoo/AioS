"""
Shim module that re-exports ``aios.virtualization`` under the historical
``Ai:oS`` namespace.

Falls back gracefully if the canonical aios package or the underlying
virtualization module is not available (e.g. git-crypt encrypted).
"""

from __future__ import annotations

try:
    from aios.virtualization import *  # noqa: F401,F403
except ImportError:
    pass
