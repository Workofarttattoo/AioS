"""
Concrete subsystem meta-agents for AgentaOS.
"""

from .system import (
    KernelAgent,
    SecurityAgent,
    NetworkingAgent,
    StorageAgent,
    ApplicationAgent,
    UserAgent,
    GuiAgent,
    ScalabilityAgent,
    OrchestrationAgent,
    QuantumAgent,
)

try:
    from .ai_os_agent import AIOperatingSystemAgent
    AI_OS_AGENT_AVAILABLE = True
except ImportError:
    AI_OS_AGENT_AVAILABLE = False
    AIOperatingSystemAgent = None

__all__ = [
    "KernelAgent",
    "SecurityAgent",
    "NetworkingAgent",
    "StorageAgent",
    "ApplicationAgent",
    "UserAgent",
    "GuiAgent",
    "ScalabilityAgent",
    "OrchestrationAgent",
    "QuantumAgent",
]

if AI_OS_AGENT_AVAILABLE:
    __all__.append("AIOperatingSystemAgent")
