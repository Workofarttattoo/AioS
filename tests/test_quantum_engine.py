import math
import unittest

from aios.probabilistic_core import agentaos_load
from aios.quantum import QuantumStateEngine


class QuantumStateEngineTests(unittest.TestCase):
  def test_hadamard_creates_superposition(self) -> None:
    engine = QuantumStateEngine(1, rng=np_random())
    engine.hadamard(0)
    vector = engine.state_vector()
    self.assertAlmostEqual(abs(vector[0]), 1 / math.sqrt(2), places=6)
    self.assertAlmostEqual(abs(vector[1]), 1 / math.sqrt(2), places=6)

  def test_cnot_entangles_qubits(self) -> None:
    engine = QuantumStateEngine(2, rng=np_random())
    engine.hadamard(0)
    engine.cnot(0, 1)
    state = engine.state_vector()
    self.assertAlmostEqual(abs(state[0]), 1 / math.sqrt(2), places=6)
    self.assertAlmostEqual(abs(state[3]), 1 / math.sqrt(2), places=6)
    self.assertAlmostEqual(abs(state[1]), 0.0, places=6)
    self.assertAlmostEqual(abs(state[2]), 0.0, places=6)

  def test_registry_adapter_executes_sequence(self) -> None:
    adapter = agentaos_load("quantum.engine", num_qubits=1, rng=np_random())
    adapter_forward = adapter.forward
    state = adapter_forward([
      ("hadamard", (0,)),
      ("rz", (0, math.pi)),
    ])
    self.assertEqual(len(state), 2)
    outcomes = adapter.sample()
    self.assertEqual(len(outcomes), 1)


# Helper ---------------------------------------------------------------------


def np_random(seed: int = 0):
  import numpy as _np

  return _np.random.default_rng(seed)


if __name__ == "__main__":
  unittest.main()
