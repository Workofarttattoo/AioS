"""
AI:OS control-plane package.

This module exposes helpers for constructing the AI:OS runtime from the
default manifest or a user-provided configuration bundle.  The CLI entrypoint
(`aios/aios`) imports from here.
"""

# Branding
DISPLAY_NAME = "Ai|oS"
DISPLAY_NAME_FULL = "Ai|oS - Agentic Intelligence Operating System"
VERSION = "0.1.0"

# Core imports
from .config import load_manifest

try:
    from .probabilistic_core import agentaos_load
except Exception:
    agentaos_load = None

try:
    from .prompt import IntentMatch, PromptRouter
except Exception:
    IntentMatch = None
    PromptRouter = None

try:
    from .runtime import AgentaRuntime
except Exception:
    AgentaRuntime = None

try:
    from .settings import settings
except Exception:
    settings = None

__all__ = [
    "DISPLAY_NAME",
    "DISPLAY_NAME_FULL",
    "VERSION",
    "AgentaRuntime",
    "agentaos_load",
    "load_manifest",
    "PromptRouter",
    "IntentMatch",
    "settings"
]
