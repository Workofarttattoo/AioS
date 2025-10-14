#!/usr/bin/env python3
"""
Extended Quantum Mechanics Framework for aios
==============================================

This module provides access to the comprehensive quantum mechanics framework
developed in Claude Teleport Brainstorm, including:

- Quantum fundamentals (complementarity, decoherence, entanglement, etc.)
- Quantum experiments (Bell, double-slit, Stern-Gerlach, etc.)
- Quantum equations (Schrödinger, Dirac, Klein-Gordon, etc.)
- Quantum formulations (Heisenberg, Path Integral, Wigner, etc.)
- Quantum interpretations (Copenhagen, Many-Worlds, Pilot Wave, etc.)
- Quantum teleportation (novel protocols for scaling)

Integration Status:
- ✅ Available if scipy is installed
- ✅ Compatible with existing aios quantum modules
- ✅ Can be used by any meta-agent

Usage:
    from aios.quantum_extended import (
        EntanglementMeasures,
        DecoherenceEngine,
        BellTest,
        SchrodingerEquation,
        design_teleportation_protocol
    )
"""

import sys
import os

# Add Claude Teleport Brainstorm to path
brainstorm_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Claude Teleport Brainstorm")
if os.path.exists(brainstorm_path) and brainstorm_path not in sys.path:
    sys.path.insert(0, brainstorm_path)

# Try to import extended quantum modules
QUANTUM_EXTENDED_AVAILABLE = False

try:
    # Quantum fundamentals
    from quantum_fundamentals import (
        ComplementarityFramework,
        DecoherenceEngine,
        DecoherenceModel,
        EntanglementMeasures,
        EnergyLevelCalculator,
        QuantumMeasurement,
        compute_decoherence_time,
        compute_entanglement_resource,
    )

    # Quantum experiments
    from quantum_experiments import (
        BellTest,
        DoubleSlit,
        SternGerlach,
        QuantumEraser,
        InteractionFreeMeasurement,
        LeggettGargTest,
        SchrodingerCat,
        run_bell_test,
        run_double_slit,
        run_stern_gerlach_sequence,
    )

    # Quantum equations
    from quantum_equations import (
        SchrodingerEquation,
        DiracEquation,
        KleinGordonEquation,
        PauliEquation,
        RydbergFormula,
        solve_quantum_system,
    )

    # Quantum formulations
    from quantum_formulations import (
        SchrodingerPicture,
        HeisenbergPicture,
        InteractionPicture,
        MatrixMechanics,
        PhaseSpaceFormulation,
        PathIntegralFormulation,
        compare_pictures,
    )

    # Quantum interpretations
    from quantum_interpretations import (
        QuantumInterpretation,
        CopenhagenInterpretation,
        ManyWorldsInterpretation,
        PilotWaveTheory,
        ObjectiveCollapseTheory,
        InterpretationComparison,
        get_interpretation_summary,
        simulate_measurement_different_interpretations,
    )

    # Quantum teleportation (novel protocols)
    from quantum_teleportation import (
        TeleportationRegime,
        DecoherenceParameters,
        TeleportationResult,
        HierarchicalQuantumTeleportation,
        DecoherenceBudgetOptimizer,
        TemporalEntanglementMultiplexer,
        TopologicalStateTransfer,
        MacroscopicTeleportationFramework,
        teleport_single_qubit,
        design_teleportation_protocol,
        analyze_teleportation_feasibility,
    )

    QUANTUM_EXTENDED_AVAILABLE = True

except ImportError as e:
    # Graceful degradation if modules not available
    import warnings
    warnings.warn(f"Quantum extended modules not available: {e}")
    QUANTUM_EXTENDED_AVAILABLE = False


# Export availability flag and functions
__all__ = [
    "QUANTUM_EXTENDED_AVAILABLE",
]

if QUANTUM_EXTENDED_AVAILABLE:
    __all__.extend([
        # Fundamentals
        "ComplementarityFramework",
        "DecoherenceEngine",
        "DecoherenceModel",
        "EntanglementMeasures",
        "EnergyLevelCalculator",
        "QuantumMeasurement",
        "compute_decoherence_time",
        "compute_entanglement_resource",
        # Experiments
        "BellTest",
        "DoubleSlit",
        "SternGerlach",
        "QuantumEraser",
        "InteractionFreeMeasurement",
        "LeggettGargTest",
        "SchrodingerCat",
        "run_bell_test",
        "run_double_slit",
        "run_stern_gerlach_sequence",
        # Equations
        "SchrodingerEquation",
        "DiracEquation",
        "KleinGordonEquation",
        "PauliEquation",
        "RydbergFormula",
        "solve_quantum_system",
        # Formulations
        "SchrodingerPicture",
        "HeisenbergPicture",
        "InteractionPicture",
        "MatrixMechanics",
        "PhaseSpaceFormulation",
        "PathIntegralFormulation",
        "compare_pictures",
        # Interpretations
        "QuantumInterpretation",
        "CopenhagenInterpretation",
        "ManyWorldsInterpretation",
        "PilotWaveTheory",
        "ObjectiveCollapseTheory",
        "InterpretationComparison",
        "get_interpretation_summary",
        "simulate_measurement_different_interpretations",
        # Teleportation
        "TeleportationRegime",
        "DecoherenceParameters",
        "TeleportationResult",
        "HierarchicalQuantumTeleportation",
        "DecoherenceBudgetOptimizer",
        "TemporalEntanglementMultiplexer",
        "TopologicalStateTransfer",
        "MacroscopicTeleportationFramework",
        "teleport_single_qubit",
        "design_teleportation_protocol",
        "analyze_teleportation_feasibility",
    ])


def check_quantum_extended() -> dict:
    """
    Check availability and version of quantum extended modules.

    Returns:
        Dict with availability status and module info
    """
    return {
        "available": QUANTUM_EXTENDED_AVAILABLE,
        "modules": {
            "quantum_fundamentals": QUANTUM_EXTENDED_AVAILABLE,
            "quantum_experiments": QUANTUM_EXTENDED_AVAILABLE,
            "quantum_equations": QUANTUM_EXTENDED_AVAILABLE,
            "quantum_formulations": QUANTUM_EXTENDED_AVAILABLE,
            "quantum_interpretations": QUANTUM_EXTENDED_AVAILABLE,
            "quantum_teleportation": QUANTUM_EXTENDED_AVAILABLE,
        },
        "location": brainstorm_path if QUANTUM_EXTENDED_AVAILABLE else None,
        "capabilities": [
            "Decoherence simulation (Lindblad master equation)",
            "Entanglement measures (entropy, concurrence, negativity)",
            "Bell tests and CHSH inequality",
            "Double-slit and quantum eraser experiments",
            "Schrödinger/Dirac/Klein-Gordon equation solvers",
            "Heisenberg/Path Integral formulations",
            "Multiple quantum interpretations (Copenhagen, Many-Worlds, etc.)",
            "Novel teleportation protocols (HQT, DBO, TEM, TPST)",
        ] if QUANTUM_EXTENDED_AVAILABLE else []
    }


if __name__ == "__main__":
    import json
    print(json.dumps(check_quantum_extended(), indent=2))
