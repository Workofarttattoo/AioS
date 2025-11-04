"""
FastAPI endpoints for QuLab.

Provides REST API endpoints for teleportation, simulation, governance,
and encoding operations with comprehensive documentation and validation.
"""

from .teleport import router as teleport_router
from .simulate import router as simulate_router
from .governance import router as governance_router
from .encoding import router as encoding_router

__all__ = [
    "teleport_router",
    "simulate_router", 
    "governance_router",
    "encoding_router",
]
