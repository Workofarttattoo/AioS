# Bio-Q SDK (Phase 1)

Foundational SDK for biological/room-temperature quantum computing research.

This package provides:

- `QuantumState` — a readable state-vector simulator with thermal/noise models
- Standard quantum gates (NumPy) and parametrized rotations
- A thermal sampling algorithm that treats ambient noise as a resource
- Stubs for an accelerated Rust backend via `pyo3` (optional)

The code favors clarity and explicitness over micro-optimizations to establish
a solid base for future extensions (density matrices, GPU backends, NV-center
Hamiltonians, etc.).

## Layout

```
bio_q_sdk/
  ├── README.md
  ├── __init__.py
  ├── gates.py
  ├── state_vector.py
  ├── algorithms/
  │   └── thermal_sampling.py
  └── src/
      └── lib.rs            # Optional Rust acceleration (not required to run)
```

## Quickstart

```python
from bio_q_sdk.state_vector import QuantumState
from bio_q_sdk.gates import H, CNOT

qs = QuantumState(num_qubits=2)
qs.apply(H, targets=[0])
qs.apply(CNOT, targets=[0, 1])
counts = qs.measure([0,1], shots=1000)
print(counts)  # ~{'00': 500, '11': 500}
```

## Thermal Environment

```python
qs.set_thermal_environment(temperature_kelvin=300.0, spin_frequency_hz=2.87e9)
qs.noise_step()  # applies amplitude damping + dephasing consistent with kT
```

## Thermal Sampler

```python
from bio_q_sdk.algorithms.thermal_sampling import ThermalSampler
sampler = ThermalSampler(temperature_kelvin=300.0)
samples = sampler.sample(qs, steps=100, measure_qubits=[0,1])
print(samples)
```

## License

MIT (research preview). See repository’s root license if present.


