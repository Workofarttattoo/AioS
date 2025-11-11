"""Thermal sampling algorithm that treats noise as a computational resource.

The sampler interleaves unitary evolution with phenomenological thermal
steps consistent with the configured environment on the QuantumState.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from ..state_vector import QuantumState
from ..gates import H, CNOT


@dataclass
class ThermalSampler:
    temperature_kelvin: float = 300.0
    spin_frequency_hz: float = 2.87e9  # NV-like reference

    def sample(self, qs: QuantumState, steps: int, measure_qubits: Sequence[int]) -> Dict[str, int]:
        qs.set_thermal_environment(self.temperature_kelvin, self.spin_frequency_hz)
        # Simple demonstration circuit: generate entanglement, then walk with noise
        if qs.num_qubits >= 2:
            qs.apply(H, [0])
            qs.apply(CNOT, [0, 1])
        # Interleave noise and passive evolution
        for _ in range(max(1, steps)):
            qs.noise_step()
        return qs.measure(measure_qubits, shots=1024)


