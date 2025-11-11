# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""Ai:oS - Agentic Control Plane"""

import json
from pathlib import Path

# Import runtime and config components
try:
    from .runtime import AgentaRuntime
except ImportError:
    # Fallback for when runtime.py is encrypted or unavailable
    class AgentaRuntime:
        """Mock runtime for import compatibility"""
        def __init__(self, *args, **kwargs):
            self.context = type('Context', (), {'environment': {}})()
            self.manifest = DEFAULT_MANIFEST

        def boot(self):
            return {"success": False, "message": "Runtime not available"}

        def status(self):
            return {
                "status": "mock",
                "message": "Runtime module encrypted - using mock implementation"
            }

        def shutdown(self):
            return {"success": True, "message": "Mock shutdown"}

        def metadata_snapshot(self):
            return {}

try:
    from .config import load_manifest, DEFAULT_MANIFEST
except ImportError:
    # Fallback for when config.py is encrypted or unavailable
    DEFAULT_MANIFEST = {
        "name": "Ai:oS Default",
        "version": "1.0.0",
        "platform": "universal",
        "meta_agents": {},
        "boot_sequence": [],
        "shutdown_sequence": []
    }

    def load_manifest(path=None):
        """Fallback manifest loader"""
        if path and Path(path).exists():
            with open(path) as f:
                return json.load(f)
        return DEFAULT_MANIFEST

# Display constants
DISPLAY_NAME = "Ai:oS"
DISPLAY_NAME_FULL = "Ai:oS Agentic Control Plane"

# Export all public APIs
__all__ = [
    "AgentaRuntime",
    "DISPLAY_NAME",
    "DISPLAY_NAME_FULL",
    "load_manifest",
    "DEFAULT_MANIFEST"
]

__version__ = "1.0.0"