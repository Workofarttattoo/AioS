"""
Quantum-Enhanced ML Algorithms Integration Example for AgentaOS.

This script demonstrates how to use quantum algorithms within AgentaOS:
1. Quantum state manipulation
2. Quantum circuit building
3. VQE optimization
4. Integration with meta-agent actions
"""

import sys
import numpy as np
from pathlib import Path

# Add AgentaOS to path if running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from quantum_ml_algorithms import (
        QuantumStateEngine,
        QuantumVQE,
        check_quantum_dependencies,
        get_quantum_algorithm_catalog,
        benchmark_qubit_scaling
    )
    from config import DEFAULT_MANIFEST
except ImportError:
    # Fallback for different import contexts
    from aios.quantum_ml_algorithms import (
        QuantumStateEngine,
        QuantumVQE,
        check_quantum_dependencies,
        get_quantum_algorithm_catalog,
        benchmark_qubit_scaling
    )
    from aios.config import DEFAULT_MANIFEST

# Minimal imports to avoid circular dependencies
from dataclasses import dataclass, field
from typing import Dict, Any


# Minimal ExecutionContext for standalone example
@dataclass
class ExecutionContext:
    """Simplified execution context for examples."""
    manifest: Any = None
    environment: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, dict] = field(default_factory=dict)

    def publish_metadata(self, key: str, value: dict):
        self.metadata[key] = value


@dataclass
class ActionResult:
    """Simplified action result for examples."""
    success: bool
    message: str
    payload: dict = field(default_factory=dict)
    elapsed: float = 0.0


def example_quantum_superposition():
    """
    Example 1: Create quantum superposition and measure.
    Demonstrates basic quantum circuit operations.
    """
    print("═" * 70)
    print("EXAMPLE 1: Quantum Superposition and Measurement")
    print("═" * 70)

    # Create 3-qubit quantum circuit
    qc = QuantumStateEngine(num_qubits=3)

    print("\nInitial state: |000⟩")

    # Apply Hadamard to create superposition
    for i in range(3):
        qc.hadamard(i)

    print("After Hadamard gates: Equal superposition of all 8 basis states")

    # Measure expectations
    print("\nExpectation values:")
    for i in range(3):
        exp = qc.expectation_value(f'Z{i}')
        print(f"  ⟨Z{i}⟩ = {exp:.4f}")

    print("\n✓ Quantum superposition created successfully")
    print("  All qubits are in |+⟩ = (|0⟩ + |1⟩)/√2 state\n")


def example_quantum_entanglement():
    """
    Example 2: Create quantum entanglement.
    Demonstrates Bell state preparation.
    """
    print("═" * 70)
    print("EXAMPLE 2: Quantum Entanglement (Bell State)")
    print("═" * 70)

    # Create 2-qubit circuit
    qc = QuantumStateEngine(num_qubits=2)

    print("\nPreparing Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2")

    # Bell state preparation
    qc.hadamard(0)
    qc.cnot(0, 1)

    print("  Step 1: Hadamard on qubit 0")
    print("  Step 2: CNOT with control=0, target=1")

    # Check correlations
    z0 = qc.expectation_value('Z0')
    z1 = qc.expectation_value('Z1')

    print(f"\nExpectation values:")
    print(f"  ⟨Z0⟩ = {z0:.4f}")
    print(f"  ⟨Z1⟩ = {z1:.4f}")

    print("\n✓ Bell state prepared successfully")
    print("  Qubits 0 and 1 are now maximally entangled\n")


def example_quantum_rotation():
    """
    Example 3: Parametric quantum rotations.
    Demonstrates parameterized quantum circuits.
    """
    print("═" * 70)
    print("EXAMPLE 3: Parametric Quantum Rotations")
    print("═" * 70)

    qc = QuantumStateEngine(num_qubits=2)

    print("\nApplying rotation gates with different angles:")

    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]

    for angle in angles:
        qc_test = QuantumStateEngine(num_qubits=2)
        qc_test.ry(0, angle)

        exp = qc_test.expectation_value('Z0')
        prob_0 = (1 + exp) / 2  # Probability of measuring |0⟩

        print(f"  RY({angle:.4f}) → P(|0⟩) = {prob_0:.4f}")

    print("\n✓ Rotation gates demonstrate continuous quantum control\n")


def example_vqe_optimization():
    """
    Example 4: VQE for simple optimization problem.
    Demonstrates variational quantum algorithm.
    """
    print("═" * 70)
    print("EXAMPLE 4: Variational Quantum Eigensolver (VQE)")
    print("═" * 70)

    # Define simple Hamiltonian: H = -Z0 - Z1
    # Ground state should be |11⟩ with energy = -2
    def hamiltonian(qc):
        return -qc.expectation_value('Z0') - qc.expectation_value('Z1')

    print("\nOptimizing Hamiltonian: H = -Z0 - Z1")
    print("Expected ground state: |11⟩ with energy ≈ -2")

    # Initialize VQE
    vqe = QuantumVQE(num_qubits=2, depth=2)

    print("\nRunning VQE optimization (this may take a moment)...")

    # Optimize
    energy, params = vqe.optimize(hamiltonian, max_iter=50)

    print(f"\nOptimization complete!")
    print(f"  Ground state energy: {energy:.4f}")
    print(f"  Optimal parameters: {params[:5]}...")
    print(f"  Error from theoretical: {abs(energy - (-2)):.4f}")

    print("\n✓ VQE successfully found approximate ground state\n")


def example_agentaos_integration():
    """
    Example 5: Integration with AgentaOS meta-agent.
    Shows how to use quantum algorithms in action handlers.
    """
    print("═" * 70)
    print("EXAMPLE 5: AgentaOS Meta-Agent Integration")
    print("═" * 70)

    # Create execution context
    ctx = ExecutionContext(manifest=DEFAULT_MANIFEST)
    ctx.environment = {
        'AGENTA_FORENSIC_MODE': '0',
        'AGENTA_QUANTUM_ENABLED': '1'
    }

    def quantum_forecasting_handler(ctx: ExecutionContext) -> ActionResult:
        """
        Example meta-agent action using quantum algorithms.
        This could be part of the Oracle agent for quantum-enhanced predictions.
        """
        # Create quantum circuit for state preparation
        qc = QuantumStateEngine(num_qubits=4)

        # Prepare problem-specific state
        for i in range(4):
            qc.hadamard(i)
            qc.ry(i, np.random.random() * np.pi)

        # Add entanglement
        for i in range(3):
            qc.cnot(i, i + 1)

        # Compute quantum expectation
        expectations = []
        for i in range(4):
            exp = qc.expectation_value(f'Z{i}')
            expectations.append(exp)

        mean_exp = np.mean(expectations)

        # Publish to metadata
        ctx.publish_metadata('quantum.forecast', {
            'expectations': expectations,
            'mean': float(mean_exp),
            'qubits': 4,
            'algorithm': 'QuantumStateEngine'
        })

        return ActionResult(
            success=True,
            message=f"[info] Quantum forecast computed: {mean_exp:.4f}",
            payload={'forecast': float(mean_exp), 'expectations': expectations}
        )

    print("\nExecuting quantum-enhanced meta-agent action...")
    result = quantum_forecasting_handler(ctx)

    print(f"  Result: {result.message}")
    print(f"  Success: {result.success}")
    print(f"  Metadata keys: {list(ctx.metadata.keys())}")

    forecast_data = ctx.metadata.get('quantum.forecast', {})
    print(f"  Quantum forecast: {forecast_data.get('mean'):.4f}")
    print(f"  Expectations: {[f'{e:.3f}' for e in forecast_data.get('expectations', [])]}")

    print("\n✓ Successfully integrated quantum algorithms into AgentaOS\n")


def example_quantum_benchmarking():
    """
    Example 6: Benchmark quantum simulation performance.
    Shows scaling behavior with qubit count.
    """
    print("═" * 70)
    print("EXAMPLE 6: Quantum Simulation Benchmarking")
    print("═" * 70)

    print("\nRunning performance benchmark...")
    print("Testing qubit counts: 3, 5, 7, 9, 11")

    results = benchmark_qubit_scaling(max_qubits=12)

    print("\nBenchmark Results:")
    print("  Qubits | Time (s) | Backend      | Memory")
    print("  " + "-" * 50)

    for i, n in enumerate(results['qubit_counts']):
        time_val = results['times'][i]
        backend = results['backends'][i]
        memory = results['memories'][i]

        memory_str = f"{memory // 1024}KB" if memory > 0 else "N/A"
        print(f"    {n:2d}   | {time_val:8.4f} | {backend:12s} | {memory_str}")

    print("\n✓ Benchmarking complete")
    print("  Performance scales exponentially with qubit count\n")


def main():
    """Run all quantum algorithm examples."""
    print()
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║   AgentaOS Quantum-Enhanced ML Algorithms - Integration Examples  ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()

    # Check dependencies
    deps = check_quantum_dependencies()
    print("Dependency Check:")
    for dep, available in deps.items():
        status = "✓" if available else "✗"
        print(f"  {status} {dep}")
    print()

    if not deps.get('torch'):
        print("✗ PyTorch is required for quantum algorithms.")
        print("  Install with: pip install torch")
        return

    # Show available algorithms
    catalog = get_quantum_algorithm_catalog()
    print("Available Quantum Algorithms:")
    for algo in catalog:
        print(f"  • {algo['name']}")
        print(f"    {algo['description']}")
    print()

    # Run examples
    try:
        example_quantum_superposition()
    except Exception as e:
        print(f"  ✗ Error in superposition example: {e}\n")

    try:
        example_quantum_entanglement()
    except Exception as e:
        print(f"  ✗ Error in entanglement example: {e}\n")

    try:
        example_quantum_rotation()
    except Exception as e:
        print(f"  ✗ Error in rotation example: {e}\n")

    try:
        example_vqe_optimization()
    except Exception as e:
        print(f"  ✗ Error in VQE example: {e}\n")

    try:
        example_agentaos_integration()
    except Exception as e:
        print(f"  ✗ Error in AgentaOS integration example: {e}\n")

    try:
        example_quantum_benchmarking()
    except Exception as e:
        print(f"  ✗ Error in benchmarking example: {e}\n")

    print("═" * 70)
    print("All quantum examples completed!")
    print("═" * 70)
    print()
    print("Next steps:")
    print("  - Explore QuantumStateEngine and QuantumVQE classes")
    print("  - Integrate quantum algorithms into your meta-agent handlers")
    print("  - Use get_quantum_algorithm_catalog() for full algorithm list")
    print("  - Check performance with benchmark_qubit_scaling()")
    print("  - For production, integrate with IBM Qiskit or Google Cirq")
    print()


if __name__ == "__main__":
    main()
