"""
Probabilistic operator registry for AgentaOS.

This module exposes a lightweight abstraction layer so meta-agents can request
probabilistic algorithms by name without binding to specific implementations.
Production deployments can swap the underlying suite while preserving the
public interface offered by ``agentaos_load``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

try:  # pragma: no cover - torch may be absent in minimal environments
  import torch
except Exception:  # pragma: no cover - fallback stub
  class _TorchStub:  # pylint: disable=too-few-public-methods
    class cuda:  # type: ignore[attr-defined]
      @staticmethod
      def is_available() -> bool:
        return False

    @staticmethod
    def device(name: str) -> str:
      return name

  torch = _TorchStub()  # type: ignore[assignment]

from .probabilistic_suite import (  # type: ignore
  AdaptiveParticleFilter,
  AdaptiveStateSpace,
  AmortizedPosteriorNetwork,
  ArchitectureSearchController,
  BayesianLayer,
  NeuralGuidedMCTS,
  NoUTurnSampler,
  OptimalTransportFlowMatcher,
  SparseGaussianProcess,
  StructuredStateDuality,
)
from .quantum import QuantumStateEngine


class ProbabilisticOperator:
  """Unified interface for probabilistic components."""

  def forward(self, *args, **kwargs):  # pylint: disable=unused-argument
    raise NotImplementedError

  def sample(self, *args, **kwargs):  # pylint: disable=unused-argument
    raise NotImplementedError

  def posterior(self, *args, **kwargs):  # pylint: disable=unused-argument
    return None


class AlgorithmRegistry:
  """Registry mapping human-readable names to operator constructors."""

  def __init__(self) -> None:
    self._mods: Dict[str, Any] = {}

  def register(self, name: str, cls: Any) -> None:
    self._mods[name] = cls

  def create(self, name: str, *args, **kwargs):
    if name not in self._mods:
      raise KeyError(f"Probabilistic operator '{name}' not registered.")
    return self._mods[name](*args, **kwargs)


REGISTRY = AlgorithmRegistry()


def device():
  """Return the preferred compute device."""

  has_cuda = getattr(torch.cuda, "is_available", lambda: False)()  # type: ignore[arg-type]
  return torch.device("cuda" if has_cuda else "cpu")


class SSMAdapter(ProbabilisticOperator):
  def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
    self.mod = AdaptiveStateSpace(*args, **kwargs)

  def forward(self, series):
    return self.mod.selective_scan(series)


class FlowMatchAdapter(ProbabilisticOperator):
  def __init__(self, net, sigma: float = 1e-3):
    self.mod = OptimalTransportFlowMatcher(net, sigma)

  def forward(self, x0, x1):
    return self.mod.conditional_flow_matching_loss(x0, x1)

  def sample(self, x0, steps: int = 50):
    return self.mod.sample(x0, steps)


class SSDAdapter(ProbabilisticOperator):
  def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
    self.mod = StructuredStateDuality(*args, **kwargs)

  def forward(self, series):
    return self.mod.structured_scan(series)


class AmortizedVIAdapter(ProbabilisticOperator):
  def __init__(self, encoder, num_flows: int = 4):
    self.mod = AmortizedPosteriorNetwork(encoder, num_flows)

  def forward(self, observations, likelihood_fn):
    return self.mod.amortized_elbo(observations, likelihood_fn)


class NUTSAdapter(ProbabilisticOperator):
  def __init__(self, log_prob, step_size: float = 0.1, max_tree_depth: int = 10):
    self.mod = NoUTurnSampler(log_prob, step_size, max_tree_depth)

  def sample(self, initial, samples: int = 1000):
    return self.mod.sample(initial, samples)


class QuantumEngineAdapter(ProbabilisticOperator):
  def __init__(self, *args, **kwargs):
    self.engine = QuantumStateEngine(*args, **kwargs)

  def forward(self, operations: Optional[Iterable[Tuple[str, Tuple]]] = None):
    if operations:
      for name, args in operations:
        getattr(self.engine, name)(*args)
    return self.engine.state_vector()

  def sample(self, *_, **__):
    return self.engine.measure_all()


REGISTRY.register("ssm.mamba", SSMAdapter)
REGISTRY.register("gen.flowmatch", FlowMatchAdapter)
REGISTRY.register("ssm.ssd", SSDAdapter)
REGISTRY.register("vi.amortized", AmortizedVIAdapter)
REGISTRY.register("rl.mcts", NeuralGuidedMCTS)
REGISTRY.register("bayes.layer", BayesianLayer)
REGISTRY.register("smc.pfilter", AdaptiveParticleFilter)
REGISTRY.register("mcmc.nuts", NUTSAdapter)
REGISTRY.register("gp.sparse", SparseGaussianProcess)
REGISTRY.register("nas.controller", ArchitectureSearchController)
REGISTRY.register("quantum.engine", QuantumEngineAdapter)


def agentaos_load(name: str, *args, **kwargs):
  """Instantiate a probabilistic operator by registry name."""

  return REGISTRY.create(name, *args, **kwargs)
