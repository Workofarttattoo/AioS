import math
import unittest

from aios.probabilistic_core import agentaos_load, device


class ProbabilisticCoreTests(unittest.TestCase):
  def test_device_prefers_named_device(self) -> None:
    current = str(device())
    self.assertTrue("cuda" in current or "cpu" in current or current in {"cuda", "cpu"})

  def test_ssm_adapter_smoothed_output(self) -> None:
    operator = agentaos_load("ssm.mamba", hidden_size=2)
    series = [1.0, 2.0, 3.0]
    smoothed = operator.forward(series)
    self.assertEqual(len(smoothed), len(series))
    self.assertAlmostEqual(smoothed[-1], 2.5)

  def test_flowmatch_loss_and_sample(self) -> None:
    operator = agentaos_load("gen.flowmatch", net=lambda x: x * 2, sigma=0.0)
    loss = operator.forward([1.0, 2.0], [2.0, 4.0])
    self.assertLess(loss, 1e-6)
    sampled = operator.sample([1.0], steps=1)
    self.assertEqual(sampled, [2.0])

  def test_nuts_adapter_sampling(self) -> None:
    operator = agentaos_load("mcmc.nuts", log_prob=lambda x: -0.5 * x * x)
    samples = operator.sample(0.0, samples=5)
    self.assertEqual(len(samples), 5)
    self.assertTrue(all(math.isfinite(value) for value in samples))

  def test_mcts_direct_access(self) -> None:
    controller = agentaos_load("rl.mcts")
    result = controller.run({"state": 1})
    self.assertIn("visits", result)
    self.assertIn("value", result)


if __name__ == "__main__":
  unittest.main()
