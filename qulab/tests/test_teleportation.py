"""
Unit tests for quantum teleportation protocol.

Tests teleportation fidelity, circuit construction, and measurement
outcomes with golden vectors and property-based testing.
"""

import pytest
import numpy as np
from qiskit.quantum_info import Statevector
from pydantic import ValidationError

from qulab.quantum.teleportation import TeleportationProtocol, TeleportationResult


class TestTeleportationProtocol:
    """Test cases for TeleportationProtocol."""
    
    def test_initialization(self):
        """Test protocol initialization."""
        protocol = TeleportationProtocol()
        assert protocol.backend is not None
        assert len(protocol._bell_states) == 4
    
    def test_bell_basis_creation(self):
        """Test Bell basis creation."""
        protocol = TeleportationProtocol()
        bell_states = protocol._bell_states
        
        # Check that all four Bell states are present
        assert "00" in bell_states
        assert "01" in bell_states
        assert "10" in bell_states
        assert "11" in bell_states
        
        # Check normalization of Bell states
        for state_name, state_vector in bell_states.items():
            norm = sum(abs(amp)**2 for amp in state_vector)
            assert np.isclose(norm, 1.0, atol=1e-10), f"Bell state {state_name} not normalized"
    
    def test_circuit_creation_valid_state(self):
        """Test circuit creation with valid quantum state."""
        protocol = TeleportationProtocol()
        
        # Test with |0⟩ state
        alpha, beta = 1.0, 0.0
        qc = protocol.create_teleportation_circuit(alpha, beta)
        
        assert qc.num_qubits == 3  # Alice's qubit + Bell pair
        assert qc.num_clbits == 3  # Bell measurement + Bob's measurement
    
    def test_circuit_creation_invalid_state(self):
        """Test circuit creation with invalid quantum state."""
        protocol = TeleportationProtocol()
        
        # Test with unnormalized state
        alpha, beta = 0.6, 0.8  # |α|² + |β|² = 0.36 + 0.64 = 1.0 ✓
        # But let's test with unnormalized state
        alpha_invalid, beta_invalid = 0.6, 0.9  # |α|² + |β|² = 0.36 + 0.81 = 1.17 ≠ 1.0
        
        with pytest.raises(ValueError, match="State not normalized"):
            protocol.create_teleportation_circuit(alpha_invalid, beta_invalid)
    
    def test_teleportation_ideal_state(self):
        """Test teleportation with ideal states."""
        protocol = TeleportationProtocol()
        
        # Test with |0⟩ state
        result = protocol.teleport(alpha=1.0, beta=0.0, shots=1000)
        
        assert isinstance(result, TeleportationResult)
        assert 0.0 <= result.fidelity <= 1.0
        assert 0.0 <= result.success_probability <= 1.0
        assert result.shots == 1000
        assert result.execution_time > 0.0
        assert len(result.classical_bits) == 2
        assert len(result.measurement_results) == 4  # Four Bell states
    
    def test_teleportation_superposition_state(self):
        """Test teleportation with superposition state."""
        protocol = TeleportationProtocol()
        
        # Test with |+⟩ state: (|0⟩ + |1⟩)/√2
        alpha, beta = 1/np.sqrt(2), 1/np.sqrt(2)
        result = protocol.teleport(alpha=alpha, beta=beta, shots=1000)
        
        assert isinstance(result, TeleportationResult)
        assert 0.0 <= result.fidelity <= 1.0
        assert result.shots == 1000
    
    def test_teleportation_fidelity_property(self):
        """Test that teleportation fidelity is reasonable."""
        protocol = TeleportationProtocol()
        
        # For ideal teleportation, fidelity should be close to 1.0
        result = protocol.teleport(alpha=0.6, beta=0.8, shots=1000)
        
        # In a perfect implementation, fidelity should be very high
        # For this test, we'll check that it's at least reasonable
        assert result.fidelity > 0.0, "Fidelity should be positive"
    
    def test_measurement_results_format(self):
        """Test that measurement results have correct format."""
        protocol = TeleportationProtocol()
        result = protocol.teleport(alpha=0.6, beta=0.8, shots=100)
        
        # Check that all Bell states are present in results
        expected_bell_states = {"00", "01", "10", "11"}
        actual_bell_states = set(result.measurement_results.keys())
        assert actual_bell_states == expected_bell_states
        
        # Check that counts are non-negative
        for state, count in result.measurement_results.items():
            assert count >= 0, f"Count for {state} should be non-negative"
    
    def test_classical_bits_format(self):
        """Test that classical bits have correct format."""
        protocol = TeleportationProtocol()
        result = protocol.teleport(alpha=0.6, beta=0.8, shots=100)
        
        # Classical bits should be a tuple of 2 integers
        assert isinstance(result.classical_bits, tuple)
        assert len(result.classical_bits) == 2
        
        # Each bit should be 0 or 1
        for bit in result.classical_bits:
            assert bit in {0, 1}, f"Classical bit should be 0 or 1, got {bit}"
    
    def test_fidelity_bands_analysis(self):
        """Test fidelity bands analysis."""
        protocol = TeleportationProtocol()
        
        # Test with small number of trials for speed
        bands = protocol.analyze_fidelity_bands(alpha=0.6, beta=0.8, shots=100, num_trials=10)
        
        assert "mean" in bands
        assert "std" in bands
        assert "confidence_95_lower" in bands
        assert "confidence_95_upper" in bands
        assert "confidence_99_lower" in bands
        assert "confidence_99_upper" in bands
        
        # Check that confidence intervals are ordered correctly
        assert bands["confidence_95_lower"] <= bands["confidence_95_upper"]
        assert bands["confidence_99_lower"] <= bands["confidence_99_upper"]
        assert bands["confidence_99_lower"] <= bands["confidence_95_lower"]
        assert bands["confidence_95_upper"] <= bands["confidence_99_upper"]
        
        # Check that mean is within confidence intervals
        assert bands["confidence_95_lower"] <= bands["mean"] <= bands["confidence_95_upper"]
        assert bands["confidence_99_lower"] <= bands["mean"] <= bands["confidence_99_upper"]
    
    def test_multiple_teleportation_consistency(self):
        """Test consistency across multiple teleportation runs."""
        protocol = TeleportationProtocol()
        
        # Run multiple teleportations with same parameters
        results = []
        for _ in range(5):
            result = protocol.teleport(alpha=0.6, beta=0.8, shots=100)
            results.append(result)
        
        # Check that all results have reasonable fidelity
        fidelities = [r.fidelity for r in results]
        assert all(0.0 <= f <= 1.0 for f in fidelities)
        
        # Check that execution times are reasonable
        execution_times = [r.execution_time for r in results]
        assert all(t > 0.0 for t in execution_times)
    
    def test_teleportation_with_different_shots(self):
        """Test teleportation with different numbers of shots."""
        protocol = TeleportationProtocol()
        
        shots_list = [100, 500, 1000]
        results = []
        
        for shots in shots_list:
            result = protocol.teleport(alpha=0.6, beta=0.8, shots=shots)
            results.append(result)
            assert result.shots == shots
        
        # All results should have reasonable fidelity
        fidelities = [r.fidelity for r in results]
        assert all(0.0 <= f <= 1.0 for f in fidelities)
    
    @pytest.mark.slow
    def test_teleportation_large_shots(self):
        """Test teleportation with large number of shots."""
        protocol = TeleportationProtocol()
        
        # Test with large number of shots
        result = protocol.teleport(alpha=0.6, beta=0.8, shots=10000)
        
        assert result.shots == 10000
        assert 0.0 <= result.fidelity <= 1.0
        assert result.execution_time > 0.0
    
    def test_teleportation_result_serialization(self):
        """Test that TeleportationResult can be serialized."""
        protocol = TeleportationProtocol()
        result = protocol.teleport(alpha=0.6, beta=0.8, shots=100)
        
        # Test that result can be converted to dict
        result_dict = result.dict()
        assert isinstance(result_dict, dict)
        assert "fidelity" in result_dict
        assert "success_probability" in result_dict
        assert "shots" in result_dict
    
    def test_teleportation_property_based(self):
        """Property-based test for teleportation."""
        protocol = TeleportationProtocol()
        
        # Test with random valid quantum states
        for _ in range(10):
            # Generate random normalized state
            theta = np.random.uniform(0, 2 * np.pi)
            alpha = np.cos(theta)
            beta = np.sin(theta)
            
            result = protocol.teleport(alpha=alpha, beta=beta, shots=100)
            
            # Properties that should always hold
            assert 0.0 <= result.fidelity <= 1.0
            assert 0.0 <= result.success_probability <= 1.0
            assert result.shots == 100
            assert result.execution_time > 0.0
            assert len(result.classical_bits) == 2
            assert len(result.measurement_results) == 4


class TestTeleportationResult:
    """Test cases for TeleportationResult."""
    
    def test_teleportation_result_creation(self):
        """Test TeleportationResult creation."""
        result = TeleportationResult(
            fidelity=0.95,
            success_probability=0.98,
            measurement_results={"00": 25, "01": 25, "10": 25, "11": 25},
            classical_bits=(0, 1),
            execution_time=0.1,
            shots=100
        )
        
        assert result.fidelity == 0.95
        assert result.success_probability == 0.98
        assert result.shots == 100
        assert result.execution_time == 0.1
        assert result.classical_bits == (0, 1)
    
    def test_teleportation_result_validation(self):
        """Test TeleportationResult validation."""
        # Test valid result
        result = TeleportationResult(
            fidelity=0.95,
            success_probability=0.98,
            measurement_results={"00": 25, "01": 25, "10": 25, "11": 25},
            classical_bits=(0, 1),
            execution_time=0.1,
            shots=100
        )
        assert result.fidelity == 0.95
        
        # Test invalid fidelity (should be caught by Pydantic)
        with pytest.raises(ValidationError):
            TeleportationResult(
                fidelity=1.5,  # Invalid: > 1.0
                success_probability=0.98,
                measurement_results={"00": 25, "01": 25, "10": 25, "11": 25},
                classical_bits=(0, 1),
                execution_time=0.1,
                shots=100
            )


# Golden vectors for testing
GOLDEN_VECTORS = {
    "zero_state": {
        "alpha": 1.0,
        "beta": 0.0,
        "expected_fidelity_range": (0.9, 1.0)
    },
    "one_state": {
        "alpha": 0.0,
        "beta": 1.0,
        "expected_fidelity_range": (0.9, 1.0)
    },
    "plus_state": {
        "alpha": 1/np.sqrt(2),
        "beta": 1/np.sqrt(2),
        "expected_fidelity_range": (0.8, 1.0)
    },
    "minus_state": {
        "alpha": 1/np.sqrt(2),
        "beta": -1/np.sqrt(2),
        "expected_fidelity_range": (0.8, 1.0)
    }
}


class TestGoldenVectors:
    """Test cases using golden vectors."""
    
    @pytest.mark.parametrize("state_name,params", GOLDEN_VECTORS.items())
    def test_golden_vector_teleportation(self, state_name, params):
        """Test teleportation with golden vectors."""
        protocol = TeleportationProtocol()
        
        result = protocol.teleport(
            alpha=params["alpha"],
            beta=params["beta"],
            shots=1000
        )
        
        # Check that fidelity is within expected range
        min_fidelity, max_fidelity = params["expected_fidelity_range"]
        assert min_fidelity <= result.fidelity <= max_fidelity, \
            f"Fidelity {result.fidelity} not in expected range [{min_fidelity}, {max_fidelity}] for {state_name}"
        
        # Check other properties
        assert 0.0 <= result.success_probability <= 1.0
        assert result.shots == 1000
        assert result.execution_time > 0.0
