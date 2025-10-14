"""Quantum state simulation helpers for AgentaOS.

This module provides a compact statevector simulator that supports a modest
number of qubits and exposes convenience methods for common single and
two-qubit gates.  The implementation intentionally favours readability and
portability over raw performance so it can run in the CI environment without
specialised hardware.  Additional backends (tensor networks, MPS, compressed
states) are acknowledged via the :class:`SimulationBackend` enum but are not
implemented yet; attempts to use them raise :class:`NotImplementedError`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

LOG = logging.getLogger("AgentaOS.quantum")


class SimulationBackend(Enum):
  """Enumeration of supported simulation strategies."""

  STATEVECTOR = "statevector"
  TENSOR_NETWORK = "tensor_network"
  MPS = "mps"
  COMPRESSED = "compressed"


@dataclass
class SimulationMetrics:
  """Basic telemetry describing simulator resource usage."""

  backend: SimulationBackend
  num_qubits: int
  memory_mb: float
  fidelity: float
  gate_count: int


class QuantumStateEngine:
  """Minimal statevector simulator with a consistent interface.

  The engine currently executes statevector simulations for up to 12 qubits by
  default.  Larger systems automatically fall back to the same backend but log
  a warning so callers know that tensor network and MPS implementations are
  still pending.
  """

  MAX_SUPPORTED_QUBITS = 12

  def __init__(
    self,
    num_qubits: int,
    *,
    backend: Optional[SimulationBackend] = None,
    use_gpu: bool = False,
    max_bond_dim: int = 64,
    target_fidelity: float = 0.99,
    rng: Optional[np.random.Generator] = None,
  ) -> None:
    if num_qubits < 1:
      raise ValueError("QuantumStateEngine requires at least one qubit.")
    self.num_qubits = int(num_qubits)
    self.backend = backend or self._select_backend()
    self.use_gpu = bool(use_gpu)  # Placeholder for future GPU support.
    self.max_bond_dim = max_bond_dim
    self.target_fidelity = target_fidelity
    self.rng = rng or np.random.default_rng()

    if self.backend != SimulationBackend.STATEVECTOR:
      LOG.warning(
        "Backend %s not fully implemented. Falling back to statevector simulation.",
        self.backend.value,
      )
      self.backend = SimulationBackend.STATEVECTOR

    if self.num_qubits > self.MAX_SUPPORTED_QUBITS:
      LOG.warning(
        "Statevector backend is optimised for <=%d qubits. Current: %d.",
        self.MAX_SUPPORTED_QUBITS,
        self.num_qubits,
      )

    self.state = self._init_statevector()
    self.metrics = SimulationMetrics(
      backend=self.backend,
      num_qubits=self.num_qubits,
      memory_mb=self._estimate_memory(),
      fidelity=1.0,
      gate_count=0,
    )

  # ---------------------------------------------------------------------------
  # Initialisation helpers
  # ---------------------------------------------------------------------------
  def _select_backend(self) -> SimulationBackend:
    if self.num_qubits <= self.MAX_SUPPORTED_QUBITS:
      return SimulationBackend.STATEVECTOR
    if self.num_qubits <= 36:
      return SimulationBackend.TENSOR_NETWORK
    if self.num_qubits <= 54:
      return SimulationBackend.MPS
    return SimulationBackend.COMPRESSED

  def _estimate_memory(self) -> float:
    # 16 bytes per complex128 amplitude.
    return (2 ** self.num_qubits) * 16 / (1024 ** 2)

  def _init_statevector(self) -> np.ndarray:
    state = np.zeros(2 ** self.num_qubits, dtype=np.complex128)
    state[0] = 1.0
    return state

  # ---------------------------------------------------------------------------
  # Public helpers
  # ---------------------------------------------------------------------------
  def reset(self) -> None:
    """Reset the simulator to the |00..0⟩ state."""

    self.state = self._init_statevector()
    self.metrics.fidelity = 1.0
    self.metrics.gate_count = 0

  def state_vector(self) -> np.ndarray:
    """Return a copy of the current statevector."""

    return self.state.copy()

  # ---------------------------------------------------------------------------
  # Gate applications
  # ---------------------------------------------------------------------------
  def hadamard(self, qubit: int) -> None:
    gate = (1 / math.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    self._apply_single_qubit_gate(gate, qubit)

  def rx(self, qubit: int, theta: float) -> None:
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    gate = np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
    self._apply_single_qubit_gate(gate, qubit)

  def ry(self, qubit: int, theta: float) -> None:
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    gate = np.array([[c, -s], [s, c]], dtype=np.complex128)
    self._apply_single_qubit_gate(gate, qubit)

  def rz(self, qubit: int, theta: float) -> None:
    gate = np.array(
      [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
      dtype=np.complex128,
    )
    self._apply_single_qubit_gate(gate, qubit)

  def phase(self, qubit: int, phi: float) -> None:
    gate = np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=np.complex128)
    self._apply_single_qubit_gate(gate, qubit)

  def cnot(self, control: int, target: int) -> None:
    self._ensure_indices(control, target)
    self._cnot_statevector(control, target)

  def apply_custom_gate(self, matrix: Sequence[Sequence[complex]], qubit: int) -> None:
    gate = np.asarray(matrix, dtype=np.complex128)
    if gate.shape != (2, 2):
      raise ValueError("Custom gate must be 2x2 for single-qubit operations.")
    self._apply_single_qubit_gate(gate, qubit)

  def apply_sequence(self, operations: Iterable[Tuple[str, Tuple]]) -> None:
    for name, args in operations:
      getattr(self, name)(*args)

  # ---------------------------------------------------------------------------
  # Measurement
  # ---------------------------------------------------------------------------
  def measure(self, qubit: int) -> int:
    self._ensure_indices(qubit)
    prob_zero = 0.0
    mask = 1 << qubit
    for index, amplitude in enumerate(self.state):
      if index & mask == 0:
        prob_zero += (amplitude.conjugate() * amplitude).real
    outcome = 0 if self.rng.random() < prob_zero else 1
    self._collapse(qubit, outcome, prob_zero)
    return outcome

  def measure_all(self) -> List[int]:
    outcomes = []
    for qubit in range(self.num_qubits):
      outcomes.append(self.measure(qubit))
    return outcomes

  # ---------------------------------------------------------------------------
  # Internal helpers
  # ---------------------------------------------------------------------------
  def _ensure_indices(self, *qubits: int) -> None:
    for qubit in qubits:
      if qubit < 0 or qubit >= self.num_qubits:
        raise IndexError(f"Qubit index {qubit} out of range (0-{self.num_qubits - 1}).")

  def _apply_single_qubit_gate(self, gate: np.ndarray, qubit: int) -> None:
    self._ensure_indices(qubit)
    stride = 1 << qubit
    period = stride << 1
    vector = self.state
    for start in range(0, len(vector), period):
      for offset in range(stride):
        idx0 = start + offset
        idx1 = idx0 + stride
        amp0 = vector[idx0]
        amp1 = vector[idx1]
        vector[idx0] = gate[0, 0] * amp0 + gate[0, 1] * amp1
        vector[idx1] = gate[1, 0] * amp0 + gate[1, 1] * amp1
    self.metrics.gate_count += 1

  def _cnot_statevector(self, control: int, target: int) -> None:
    control_mask = 1 << control
    target_mask = 1 << target
    vector = self.state
    for index in range(len(vector)):
      if index & control_mask:
        flipped = index ^ target_mask
        if flipped > index:
          vector[index], vector[flipped] = vector[flipped], vector[index]
    self.metrics.gate_count += 1

  def _collapse(self, qubit: int, outcome: int, prob_zero: float) -> None:
    mask = 1 << qubit
    vector = self.state
    norm = math.sqrt(prob_zero if outcome == 0 else (1 - prob_zero))
    if norm == 0:
      return
    for index in range(len(vector)):
      bit = 1 if index & mask else 0
      if bit != outcome:
        vector[index] = 0.0
      else:
        vector[index] /= norm


__all__ = [
  "SimulationBackend",
  "SimulationMetrics",
  "QuantumStateEngine",
]
