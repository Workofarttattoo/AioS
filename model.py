"""
Shared data structures for AgentaOS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any


class AgentActionError(RuntimeError):
    """Raised when a meta-agent cannot fulfill a requested action."""


@dataclass
class ActionResult:
    success: bool
    message: str
    payload: Optional[Dict[str, Any]] = None
    elapsed: float = 0.0

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return self.success
