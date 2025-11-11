"""Readable state-vector simulator with thermal/noise support.

This module intentionally optimizes for clarity. It is appropriate for
algorithm development, educational use, and reproducible benchmarks.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Complex = np.complex128
Array = np.ndarray

KB = 1.380649e-23  # Boltzmann constant (J/K)
H = 6.62607015e-34  # Planck constant (J*s)


def kron_n(ops: List[Array]) -> Array:
    out = np.array([[1.0 + 0.0j]], dtype=Complex)
    for op in ops:
        out = np.kron(out, op.astype(Complex))
    return out


def amplitude_damping_kraus(p: float) -> List[Array]:
    p = float(np.clip(p, 0.0, 1.0))
    k0 = np.array([[1, 0], [0, np.sqrt(1 - p)]], dtype=Complex)
    k1 = np.array([[0, np.sqrt(p)], [0, 0]], dtype=Complex)
    return [k0, k1]


def dephasing_kraus(p: float) -> List[Array]:
    p = float(np.clip(p, 0.0, 1.0))
    k0 = np.sqrt(1 - p) * np.eye(2, dtype=Complex)
    k1 = np.sqrt(p) * np.array([[1, 0], [0, -1]], dtype=Complex)
    return [k0, k1]


def expand_operator(op: Array, targets: Sequence[int], num_qubits: int) -> Array:
    """Expand a k-qubit operator to the full Hilbert space.
    targets are 0-indexed (little-endian by convention here).
    """
    k = int(round(math.log2(op.shape[0])))
    assert op.shape == (2**k, 2**k), "Operator must be 2^k square"
    targets = list(targets)
    assert len(targets) == k, "Targets must match operator arity"
    # Build in order from most-significant to least-significant qubit
    ops = []
    for q in range(num_qubits):
        if q in targets:
            # Insert a placeholder to be replaced after construction
            ops.append(None)  # type: ignore
        else:
            ops.append(np.eye(2, dtype=Complex))
    # Replace the k consecutive None slots with op via reshape/permutation
    # Compute permutation matrix that maps tensor basis to (targets first)
    perm = targets + [q for q in range(num_qubits) if q not in targets]
    # Full identity then swap axes via permutation on kron factors
    full = kron_n([np.eye(2, dtype=Complex) for _ in range(num_qubits)])
    # Construct selection projector to embed `op`
    # For clarity, use slow but explicit embedding
    full_op = np.zeros_like(full)
    for i in range(2**num_qubits):
        for j in range(2**num_qubits):
            # Extract bits for targets and non-targets
            def bits(x: int, idxs: Sequence[int]) -> int:
                out = 0
                for b, q in enumerate(idxs[::-1]):
                    out |= ((x >> q) & 1) << b
                return out
            it = bits(i, targets)
            inon = bits(i, [q for q in range(num_qubits) if q not in targets])
            jt = bits(j, targets)
            jnon = bits(j, [q for q in range(num_qubits) if q not in targets])
            if inon == jnon:
                full_op[i, j] = op[it, jt]
    return full_op


class QuantumState:
    """State-vector with thermal/noise environment."""

    def __init__(self, num_qubits: int, state: Optional[Array] = None) -> None:
        self.num_qubits = int(num_qubits)
        dim = 2 ** self.num_qubits
        if state is None:
            self.state = np.zeros(dim, dtype=Complex)
            self.state[0] = 1.0 + 0.0j
        else:
            state = np.asarray(state, dtype=Complex)
            assert state.shape == (dim,)
            self.state = state / np.linalg.norm(state)
        # Thermal env params
        self.temperature_K = 0.0
        self.spin_freq_hz = 0.0
        self.gamma_amp = 0.0
        self.gamma_phi = 0.0

    @property
    def dim(self) -> int:
        return self.state.shape[0]

    def set_thermal_environment(self, temperature_kelvin: float, spin_frequency_hz: float) -> None:
        """Configure ambient thermal rates using a simple BE occupancy model.

        We map mean occupation n_th to amplitude damping/dephasing rates.
        These are phenomenological and suitable for algorithm exploration.
        """
        self.temperature_K = float(temperature_kelvin)
        self.spin_freq_hz = float(spin_frequency_hz)
        if self.temperature_K <= 0 or self.spin_freq_hz <= 0:
            self.gamma_amp = 0.0
            self.gamma_phi = 0.0
            return
        hv = H * self.spin_freq_hz
        n_th = 1.0 / (np.exp(hv / (KB * self.temperature_K)) - 1.0)
        # Map to probabilities per "step"
        self.gamma_amp = float(np.clip(n_th / (n_th + 1.0) * 0.01, 0.0, 0.05))
        self.gamma_phi = float(np.clip(n_th * 0.02, 0.0, 0.1))

    # --- Unitary and noisy evolution -------------------------------------------------
    def apply(self, op: Array, targets: Sequence[int]) -> None:
        full = expand_operator(np.asarray(op, dtype=Complex), targets, self.num_qubits)
        self.state = full @ self.state
        self.state /= np.linalg.norm(self.state)

    def apply_kraus(self, kraus: Sequence[Array], targets: Sequence[int]) -> None:
        density = np.outer(self.state, np.conjugate(self.state))
        fulls = [expand_operator(k, targets, self.num_qubits) for k in kraus]
        new_rho = sum(K @ density @ K.conj().T for K in fulls)
        # Sample a pure state from the density (largest eigenvector) for continued SV sim
        vals, vecs = np.linalg.eigh(new_rho)
        self.state = vecs[:, np.argmax(vals)]
        self.state /= np.linalg.norm(self.state)

    def noise_step(self) -> None:
        if self.gamma_amp > 0:
            self.apply_kraus(amplitude_damping_kraus(self.gamma_amp), targets=[0])
        if self.gamma_phi > 0:
            self.apply_kraus(dephasing_kraus(self.gamma_phi), targets=[0])

    # --- Measurement ----------------------------------------------------------------
    def measure(self, qubits: Sequence[int], shots: int = 1, rng: Optional[np.random.Generator] = None) -> Dict[str, int]:
        qubits = list(qubits)
        rng = np.random.default_rng() if rng is None else rng
        probs = np.abs(self.state) ** 2
        outcomes = rng.choice(self.dim, size=shots, p=probs)
        counts: Dict[str, int] = {}
        for idx in outcomes:
            bitstring = "".join(str((idx >> q) & 1) for q in qubits[::-1])
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts


