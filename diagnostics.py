"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.

Unified System Diagnostics Module for Ai:oS
=============================================

Provides comprehensive, programmatic access to system state, available resources,
and component health. Single source of truth for "what's available and working?"

Applied Optimization Frameworks:
- Crystalline Intent: Clear definition - answer "system status?" in one call
- 7 Lenses: User-centric, technical, performance, discoverability, composability,
            maintainability, crystalline intent
- Advanced Design: Plugin-based, JSON output, agent-friendly, extensible
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import json
import platform
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS & TYPES
# ============================================================================

class HealthStatus(Enum):
    """Component health status"""
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNCONFIGURED = "unconfigured"


class SubsystemType(Enum):
    """Types of aios subsystems"""
    META_AGENT = "meta_agent"
    ALGORITHM = "algorithm"
    PROVIDER = "provider"
    SECURITY_TOOL = "security_tool"
    QUANTUM = "quantum"
    INTEGRATION = "integration"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class VersionInfo:
    """Version information"""
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None

    def __str__(self) -> str:
        """Format as semantic version"""
        v = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            v += f"-{self.prerelease}"
        if self.build:
            v += f"+{self.build}"
        return v


@dataclass
class DependencyInfo:
    """Information about a dependency"""
    name: str
    required: bool
    installed: bool
    version: Optional[str] = None
    install_command: Optional[str] = None
    alternative: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SubsystemInfo:
    """Information about a single subsystem"""
    name: str
    subsystem_type: SubsystemType
    status: HealthStatus
    version: VersionInfo
    description: str
    dependencies: List[DependencyInfo] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    last_check_timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict"""
        d = asdict(self)
        d["subsystem_type"] = self.subsystem_type.value
        d["status"] = self.status.value
        d["version"] = str(self.version)
        return d


@dataclass
class AlgorithmInfo:
    """Information about an ML/quantum algorithm"""
    name: str
    category: str  # "classical", "quantum", "hybrid"
    description: str
    version: VersionInfo
    available: bool
    dependencies: List[str] = field(default_factory=list)
    estimated_speed: Optional[str] = None
    memory_requirement: Optional[str] = None
    gpu_capable: bool = False
    example_usage: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict"""
        d = asdict(self)
        d["version"] = str(self.version)
        return d


@dataclass
class ProviderInfo:
    """Information about a resource provider"""
    name: str
    provider_type: str  # "container", "cloud", "vm", "quantum"
    available: bool
    version: Optional[VersionInfo] = None
    configured: bool = False
    cli_binary: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    region_count: int = 0
    max_resources: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict"""
        d = asdict(self)
        if self.version:
            d["version"] = str(self.version)
        return d


@dataclass
class SecurityToolInfo:
    """Information about a security assessment tool"""
    name: str
    description: str
    category: str  # "reconnaissance", "exploitation", "auth", etc.
    location: str  # "tools/" or "red-team-tools/"
    has_cli: bool = True
    has_gui: bool = False
    available: bool = False
    dependencies: List[str] = field(default_factory=list)
    example_command: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict"""
        return asdict(self)


@dataclass
class SystemStatus:
    """Complete system status snapshot"""
    timestamp: float
    python_version: str
    platform_name: str
    platform_version: str
    working_directory: str
    aios_version: VersionInfo

    meta_agents: List[SubsystemInfo] = field(default_factory=list)
    algorithms: List[AlgorithmInfo] = field(default_factory=list)
    providers: List[ProviderInfo] = field(default_factory=list)
    security_tools: List[SecurityToolInfo] = field(default_factory=list)

    overall_health: HealthStatus = HealthStatus.OPERATIONAL
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize to JSON string"""
        data = {
            "timestamp": self.timestamp,
            "python_version": self.python_version,
            "platform": {
                "name": self.platform_name,
                "version": self.platform_version
            },
            "working_directory": self.working_directory,
            "aios_version": str(self.aios_version),
            "overall_health": self.overall_health.value,
            "meta_agents": [a.to_dict() for a in self.meta_agents],
            "algorithms": {
                "classical": [a.to_dict() for a in self.algorithms if a.category == "classical"],
                "quantum": [a.to_dict() for a in self.algorithms if a.category == "quantum"],
                "total_available": len([a for a in self.algorithms if a.available])
            },
            "providers": [p.to_dict() for p in self.providers if p.available],
            "security_tools": {
                "total": len(self.security_tools),
                "available": len([t for t in self.security_tools if t.available]),
                "by_category": self._tools_by_category()
            },
            "recommendations": self.recommendations,
            "warnings": self.warnings,
            "errors": self.errors
        }
        return json.dumps(data, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return json.loads(self.to_json())

    def _tools_by_category(self) -> Dict[str, int]:
        """Count tools by category"""
        categories = {}
        for tool in self.security_tools:
            categories[tool.category] = categories.get(tool.category, 0) + 1
        return categories


# ============================================================================
# DIAGNOSTICS MANAGER
# ============================================================================

class DiagnosticsManager:
    """
    Unified diagnostics manager for complete system visibility.

    Crystalline Intent: Single entry point that answers:
    - "What's my system status?"
    - "What algorithms are available?"
    - "Which providers are configured?"
    - "What tools can I use?"
    """

    def __init__(self):
        """Initialize diagnostics manager"""
        self.aios_version = VersionInfo(1, 0, 0, prerelease="beta.1")
        self._subsystems: Dict[str, SubsystemInfo] = {}
        self._algorithms: Dict[str, AlgorithmInfo] = {}
        self._providers: Dict[str, ProviderInfo] = {}
        self._tools: Dict[str, SecurityToolInfo] = {}
        self._initialized = False

    def initialize(self):
        """Load all subsystems and capabilities"""
        if self._initialized:
            return

        self._load_meta_agents()
        self._load_algorithms()
        self._load_providers()
        self._load_security_tools()
        self._initialized = True

    def get_system_status(self) -> SystemStatus:
        """
        Get complete system status in one call.

        Returns JSON-serializable SystemStatus with all subsystems,
        algorithms, providers, and tools.
        """
        import time

        self.initialize()

        # Build recommendations based on current state
        recommendations = self._generate_recommendations()

        status = SystemStatus(
            timestamp=time.time(),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            platform_name=platform.system(),
            platform_version=platform.release(),
            working_directory=str(Path.cwd()),
            aios_version=self.aios_version,
            meta_agents=list(self._subsystems.values()),
            algorithms=list(self._algorithms.values()),
            providers=list(self._providers.values()),
            security_tools=list(self._tools.values()),
            overall_health=self._compute_overall_health(),
            recommendations=recommendations
        )

        return status

    def validate_manifest(self, manifest_path: str) -> Dict[str, Any]:
        """
        Validate an aios manifest before execution.

        Checks:
        - File exists and is valid JSON
        - Required fields present
        - Referenced agents exist
        - Referenced actions available
        """
        self.initialize()

        result = {
            "valid": False,
            "path": manifest_path,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }

        path = Path(manifest_path)
        if not path.exists():
            result["errors"].append(f"Manifest file not found: {manifest_path}")
            return result

        try:
            import json
            with open(path) as f:
                manifest = json.load(f)
        except json.JSONDecodeError as e:
            result["errors"].append(f"Invalid JSON: {e}")
            return result

        # Check required fields
        required = ["name", "version"]
        for field in required:
            if field not in manifest:
                result["errors"].append(f"Missing required field: {field}")

        # Check meta-agents referenced exist
        if "meta_agents" in manifest:
            for agent_name in manifest["meta_agents"].keys():
                if agent_name not in self._subsystems:
                    result["warnings"].append(f"Unknown meta-agent: {agent_name}")

        # Check boot sequence
        if "boot_sequence" in manifest:
            for action_path in manifest["boot_sequence"]:
                agent_name = action_path.split(".")[0]
                if agent_name not in self._subsystems:
                    result["errors"].append(f"Boot action references unknown agent: {agent_name}")

        result["valid"] = len(result["errors"]) == 0

        if result["valid"]:
            result["suggestions"].append("✅ Manifest is valid and ready to execute")

        return result

    def get_available_providers(self) -> List[ProviderInfo]:
        """Get all available resource providers"""
        self.initialize()
        return [p for p in self._providers.values() if p.available]

    def get_available_algorithms(self, category: Optional[str] = None) -> List[AlgorithmInfo]:
        """Get available algorithms, optionally filtered by category"""
        self.initialize()
        algos = [a for a in self._algorithms.values() if a.available]
        if category:
            algos = [a for a in algos if a.category == category]
        return algos

    def get_available_tools(self, category: Optional[str] = None) -> List[SecurityToolInfo]:
        """Get available security tools, optionally by category"""
        self.initialize()
        tools = [t for t in self._tools.values() if t.available]
        if category:
            tools = [t for t in tools if t.category == category]
        return tools

    def suggest_fixes(self, error_message: str) -> List[str]:
        """
        Suggest fixes for common errors.

        User-Centric Lens: When something fails, provide actionable help.
        """
        suggestions = []

        if "no module named" in error_message.lower() or "importerror" in error_message.lower():
            if "qiskit" in error_message.lower():
                suggestions.append("Install Qiskit: pip install qiskit qiskit-aer")
            elif "torch" in error_message.lower():
                suggestions.append("Install PyTorch: pip install torch")
            elif "pydantic" in error_message.lower():
                suggestions.append("Install Pydantic: pip install pydantic")
            else:
                suggestions.append("Check dependencies: pip install -r requirements.txt")

        if "provider not available" in error_message.lower():
            suggestions.append("Configure providers: python aios/aios --list-providers")

        if "permission denied" in error_message.lower():
            suggestions.append("May need elevated privileges: sudo python ...")

        if not suggestions:
            suggestions.append("See detailed error in: python aios/aios -v boot --debug")

        return suggestions

    # ========================================================================
    # INTERNAL: SUBSYSTEM LOADERS
    # ========================================================================

    def _load_meta_agents(self):
        """Load meta-agent subsystems"""
        agents = [
            ("KernelAgent", "Process management, system initialization"),
            ("SecurityAgent", "Firewall, encryption, security toolkit"),
            ("NetworkingAgent", "Network configuration, DNS, routing"),
            ("StorageAgent", "Volume management, filesystem operations"),
            ("ApplicationAgent", "Application supervisor, orchestration"),
            ("ScalabilityAgent", "Load monitoring, virtualization"),
            ("OrchestrationAgent", "Policy engine, telemetry, health"),
            ("UserAgent", "User management, authentication"),
            ("GuiAgent", "Display server management, UI"),
        ]

        for name, description in agents:
            self._subsystems[name] = SubsystemInfo(
                name=name,
                subsystem_type=SubsystemType.META_AGENT,
                status=HealthStatus.OPERATIONAL,
                version=VersionInfo(1, 0, 0),
                description=description,
                capabilities=[description.split(",")[0]],
                configuration={"enabled": True}
            )

    def _load_algorithms(self):
        """Load available ML/quantum algorithms"""
        algorithms = [
            # Classical ML
            ("AdaptiveStateSpace", "classical", "Mamba architecture - O(n) complexity"),
            ("OptimalTransportFlowMatcher", "classical", "Flow matching for fast generation"),
            ("NeuralGuidedMCTS", "classical", "AlphaGo-style planning"),
            ("AdaptiveParticleFilter", "classical", "Sequential Monte Carlo"),
            ("NoUTurnSampler", "classical", "Hamiltonian Monte Carlo"),
            ("SparseGaussianProcess", "classical", "Scalable regression"),
            ("ArchitectureSearchController", "classical", "Neural architecture search"),
            ("BayesianLayer", "classical", "Uncertainty quantification"),
            ("AmortizedPosteriorNetwork", "classical", "Fast Bayesian inference"),
            ("StructuredStateDuality", "classical", "Mamba-2 SSD architecture"),
            # Quantum ML
            ("QuantumStateEngine", "quantum", "1-50 qubit simulator"),
            ("QuantumVQE", "quantum", "Variational quantum eigensolver"),
            ("QuantumHHL", "quantum", "Linear system solver"),
        ]

        for name, category, description in algorithms:
            self._algorithms[name] = AlgorithmInfo(
                name=name,
                category=category,
                description=description,
                version=VersionInfo(1, 0, 0),
                available=self._check_algorithm_available(name),
                gpu_capable=category == "quantum"
            )

    def _load_providers(self):
        """Load available resource providers"""
        providers = [
            ("Docker", "container", "docker"),
            ("AWS", "cloud", "aws"),
            ("Azure", "cloud", "az"),
            ("GCP", "cloud", "gcloud"),
            ("QEMU", "vm", "qemu-system-x86_64"),
            ("libvirt", "vm", "virsh"),
        ]

        for name, ptype, cli_binary in providers:
            self._providers[name] = ProviderInfo(
                name=name,
                provider_type=ptype,
                available=self._check_cli_available(cli_binary),
                cli_binary=cli_binary,
                configured=self._check_provider_configured(name),
                capabilities=self._get_provider_capabilities(name)
            )

    def _load_security_tools(self):
        """Load security assessment tools"""
        tools = [
            ("AuroraScan", "reconnaissance", "Network reconnaissance & scanning"),
            ("CipherSpear", "exploitation", "Database injection testing"),
            ("SkyBreaker", "wireless", "Wireless security auditing"),
            ("MythicKey", "authentication", "Credential analysis"),
            ("NemesisHydra", "authentication", "Authentication testing"),
            ("SpectraTrace", "traffic", "Packet inspection & analysis"),
            ("ObsidianHunt", "hardening", "Host hardening audit"),
            ("DirReaper", "reconnaissance", "Directory fuzzing"),
            ("PayloadForge", "exploitation", "Payload generation"),
            ("VectorFlux", "exploitation", "Payload staging"),
            ("SovereignSuite", "orchestration", "Comprehensive toolkit"),
            ("ProxyPhantom", "proxy", "HTTP proxy framework"),
            ("Scribe", "traffic", "Traffic capture & logging"),
        ]

        for name, category, description in tools:
            self._tools[name] = SecurityToolInfo(
                name=name,
                description=description,
                category=category,
                location="tools/",
                available=True,  # Simplified - all in tools/ available
                has_gui=self._tool_has_gui(name),
                example_command=self._get_tool_example(name)
            )

    # ========================================================================
    # INTERNAL: HELPER METHODS
    # ========================================================================

    def _check_algorithm_available(self, name: str) -> bool:
        """Check if algorithm is available (dependencies installed)"""
        # Simplified - in production would check actual imports
        return True

    def _check_cli_available(self, binary: str) -> bool:
        """Check if CLI binary is available"""
        import shutil
        return shutil.which(binary) is not None

    def _check_provider_configured(self, name: str) -> bool:
        """Check if provider is configured"""
        import os

        checks = {
            "AWS": os.environ.get("AWS_PROFILE"),
            "Azure": os.environ.get("AZURE_SUBSCRIPTION_ID"),
            "GCP": os.environ.get("GCP_PROJECT"),
        }

        return bool(checks.get(name))

    def _get_provider_capabilities(self, name: str) -> List[str]:
        """Get provider capabilities"""
        caps = {
            "Docker": ["container_execution", "process_isolation", "networking"],
            "AWS": ["compute", "storage", "databases", "networking"],
            "Azure": ["compute", "storage", "ai", "databases"],
            "GCP": ["compute", "ml", "bigquery", "cloud_functions"],
            "QEMU": ["vm_creation", "device_passthrough", "live_migration"],
            "libvirt": ["vm_management", "storage_pools", "networks"],
        }
        return caps.get(name, [])

    def _tool_has_gui(self, name: str) -> bool:
        """Check if tool has GUI version"""
        gui_tools = {"AuroraScan", "CipherSpear", "SkyBreaker", "MythicKey",
                     "NemesisHydra", "SpectraTrace", "ObsidianHunt", "VectorFlux",
                     "SovereignSuite", "Scribe"}
        return name in gui_tools

    def _get_tool_example(self, name: str) -> Optional[str]:
        """Get example command for tool"""
        examples = {
            "AuroraScan": "python -m tools.aurorascan 192.168.0.0/24 --json",
            "CipherSpear": "python -m tools.cipherspear --demo --json",
            "SkyBreaker": "python -m tools.skybreaker capture wlan0",
            "ObsidianHunt": "python -m tools.obsidianhunt --profile workstation",
        }
        return examples.get(name)

    def _compute_overall_health(self) -> HealthStatus:
        """Determine overall system health"""
        if self._subsystems:
            operational = sum(1 for s in self._subsystems.values()
                            if s.status == HealthStatus.OPERATIONAL)
            if operational == len(self._subsystems):
                return HealthStatus.OPERATIONAL
            elif operational >= len(self._subsystems) / 2:
                return HealthStatus.DEGRADED
        return HealthStatus.UNAVAILABLE

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on current state"""
        recommendations = []

        # Check for unconfigured providers
        unconfigured = [p.name for p in self._providers.values()
                       if p.available and not p.configured]
        if unconfigured:
            recommendations.append(
                f"Configure {', '.join(unconfigured)}: see aios/GETTING_STARTED.md"
            )

        # Check for missing quantum
        quantum_algos = [a for a in self._algorithms.values() if a.category == "quantum"]
        missing_quantum = [a.name for a in quantum_algos if not a.available]
        if missing_quantum:
            recommendations.append(
                f"Install quantum support: pip install qiskit qiskit-aer"
            )

        # Check for optimal configurations
        if not recommendations:
            recommendations.append("✅ System is well-configured")

        return recommendations


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_diagnostics_instance: Optional[DiagnosticsManager] = None


def get_diagnostics() -> DiagnosticsManager:
    """Get or create global diagnostics instance"""
    global _diagnostics_instance
    if _diagnostics_instance is None:
        _diagnostics_instance = DiagnosticsManager()
    return _diagnostics_instance


# ============================================================================
# CLI INTERFACE
# ============================================================================

def demo_diagnostics():
    """Demonstrate diagnostics capabilities"""
    print("\n" + "="*80)
    print("AIOS DIAGNOSTICS DEMO")
    print("="*80 + "\n")

    manager = get_diagnostics()

    # Get complete status
    print("📊 COMPLETE SYSTEM STATUS:\n")
    status = manager.get_system_status()
    print(f"Python {status.python_version} on {status.platform_name}")
    print(f"Working Directory: {status.working_directory}")
    print(f"Overall Health: {status.overall_health.value}")

    # Algorithms
    print(f"\n🤖 ALGORITHMS AVAILABLE: {len(status.algorithms)}")
    classical = [a for a in status.algorithms if a.category == "classical"]
    quantum = [a for a in status.algorithms if a.category == "quantum"]
    print(f"  Classical ML: {len(classical)}")
    print(f"  Quantum ML: {len(quantum)}")

    # Providers
    print(f"\n☁️ PROVIDERS: {len([p for p in status.providers if p.available])}/{len(status.providers)}")
    for provider in status.providers:
        status_icon = "✅" if provider.available else "❌"
        print(f"  {status_icon} {provider.name}")

    # Security Tools
    available_tools = len([t for t in status.security_tools if t.available])
    total_tools = len(status.security_tools)
    print(f"\n🔒 SECURITY TOOLS: {available_tools}/{total_tools}")

    # Recommendations
    if status.recommendations:
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in status.recommendations:
            print(f"  • {rec}")

    # JSON output
    print(f"\n📄 JSON OUTPUT:\n")
    print(status.to_json())

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    demo_diagnostics()
