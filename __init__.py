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

from .config import load_manifest
from .probabilistic_core import agentaos_load
from .prompt import IntentMatch, PromptRouter
from .runtime import AgentaRuntime
from .settings import settings

# Quantum Computing Suite (23 algorithms)
try:
    from .quantum_ml_algorithms import (
        QuantumStateEngine,
        QuantumVQE,
        QuantumQAOA,
        QuantumKernelML,
        QuantumNeuralNetwork,
        QuantumGAN,
        QuantumBoltzmannMachine,
        QuantumReinforcementLearning,
        QuantumCircuitLearning,
        QuantumAmplitudeEstimation,
        QuantumBayesianInference,
    )
    from .quantum_hhl_algorithm import HHLQuantumLinearSolver
    from .quantum_schrodinger_dynamics import (
        SchrodingerTimeEvolution,
        AdiabaticQuantumComputing,
    )
    from .quantum_advanced_synthesis import (
        QuantumTemporalLinearSolver,
        VariationalQuantumDynamics,
        QuantumKalmanFilter,
        QuantumHamiltonianNeuralNetwork,
        QuantumOptimalControl,
        TemporalQuantumKernel,
        QuantumPolicyGradient,
        MetaHamiltonianLearning,
        QuantumNeuralODE,
        StochasticQuantumLinearSolver,
    )
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    # Graceful degradation if PyTorch/SciPy not available

# Extended Quantum Framework (Claude Teleport Brainstorm)
try:
    from .quantum_extended import (
        QUANTUM_EXTENDED_AVAILABLE,
        check_quantum_extended,
    )
    # Import key classes if available
    if QUANTUM_EXTENDED_AVAILABLE:
        from .quantum_extended import (
            EntanglementMeasures,
            DecoherenceEngine,
            BellTest,
            SchrodingerEquation as SchrodingerEquationSolver,
            design_teleportation_protocol,
        )
except ImportError:
    QUANTUM_EXTENDED_AVAILABLE = False

__all__ = [
    "DISPLAY_NAME",
    "DISPLAY_NAME_FULL",
    "VERSION",
    "AgentaRuntime",
    "agentaos_load",
    "load_manifest",
    "PromptRouter",
    "IntentMatch",
    "settings",
    "QUANTUM_AVAILABLE",
    "QUANTUM_EXTENDED_AVAILABLE",
    "check_quantum_extended",
]

# Add quantum exports if available
if QUANTUM_AVAILABLE:
    __all__.extend([
        # Quantum ML (11 algorithms)
        "QuantumStateEngine",
        "QuantumVQE",
        "QuantumQAOA",
        "QuantumKernelML",
        "QuantumNeuralNetwork",
        "QuantumGAN",
        "QuantumBoltzmannMachine",
        "QuantumReinforcementLearning",
        "QuantumCircuitLearning",
        "QuantumAmplitudeEstimation",
        "QuantumBayesianInference",
        # HHL Algorithm (1)
        "HHLQuantumLinearSolver",
        # Schrödinger Dynamics (2)
        "SchrodingerTimeEvolution",
        "AdiabaticQuantumComputing",
        # Advanced Synthesis (10 novel frameworks)
        "QuantumTemporalLinearSolver",
        "VariationalQuantumDynamics",
        "QuantumKalmanFilter",
        "QuantumHamiltonianNeuralNetwork",
        "QuantumOptimalControl",
        "TemporalQuantumKernel",
        "QuantumPolicyGradient",
        "MetaHamiltonianLearning",
        "QuantumNeuralODE",
        "StochasticQuantumLinearSolver",
    ])

# Add extended quantum exports if available
if QUANTUM_EXTENDED_AVAILABLE:
    __all__.extend([
        # Extended Quantum Framework (Claude Teleport Brainstorm)
        "EntanglementMeasures",
        "DecoherenceEngine",
        "BellTest",
        "SchrodingerEquationSolver",
        "design_teleportation_protocol",
    ])
