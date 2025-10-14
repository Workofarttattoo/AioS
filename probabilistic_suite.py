"""
Lightweight probabilistic algorithm stubs for AgentaOS.

These implementations provide deterministic, dependency-light stand-ins so the
probabilistic registry can be exercised without requiring heavyweight ML
frameworks at runtime.  Real deployments can replace these classes with
production implementations while retaining the shared interfaces.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Sequence, Tuple


@dataclass
class AdaptiveStateSpace:
  hidden_size: int = 8

  def selective_scan(self, series: Sequence[float]) -> List[float]:
    """Return a moving-average style smoothing over the input series."""

    if not series:
      return []
    window = max(1, min(len(series), self.hidden_size))
    smoothed: List[float] = []
    for index in range(len(series)):
      start = max(0, index - window + 1)
      window_slice = series[start : index + 1]
      smoothed.append(sum(window_slice) / len(window_slice))
    return smoothed


class OptimalTransportFlowMatcher:
  def __init__(self, net: Callable[[float], float], sigma: float = 1e-3):
    self.net = net
    self.sigma = float(sigma)

  def conditional_flow_matching_loss(self, source: Iterable[float], target: Iterable[float]) -> float:
    diff = 0.0
    count = 0
    for x0, x1 in zip(source, target):
      pred = self.net(x0)
      diff += (pred - x1) ** 2
      count += 1
    if count == 0:
      return 0.0
    return diff / count + self.sigma

  def sample(self, start: Iterable[float], steps: int = 50) -> List[float]:
    values = list(start)
    for _ in range(max(1, steps)):
      values = [self.net(x) for x in values]
    return values


@dataclass
class StructuredStateDuality:
  order: int = 3

  def structured_scan(self, series: Sequence[float]) -> List[Tuple[float, float]]:
    """Return cumulative mean/variance pairs across the series."""

    results: List[Tuple[float, float]] = []
    total = 0.0
    squares = 0.0
    for index, value in enumerate(series, start=1):
      total += value
      squares += value * value
      mean = total / index
      variance = max(0.0, (squares / index) - mean * mean)
      results.append((mean, variance))
    return results


class AmortizedPosteriorNetwork:
  def __init__(self, encoder: Callable[[Any], float], num_flows: int = 4):
    self.encoder = encoder
    self.num_flows = num_flows

  def amortized_elbo(self, observations: Iterable[Any], likelihood_fn: Callable[[Any], float]) -> float:
    kl_term = 0.0
    recon = 0.0
    count = 0
    for item in observations:
      latent = self.encoder(item)
      kl_term += latent * latent * 0.5
      recon += likelihood_fn(item)
      count += 1
    if count == 0:
      return 0.0
    return recon / count - kl_term / max(1, self.num_flows)


class NeuralGuidedMCTS:
  def __init__(self, policy: Callable[[Any], List[float]] | None = None, depth: int = 5):
    self.policy = policy or (lambda _: [1.0])
    self.depth = max(1, depth)

  def run(self, state: Any) -> dict:
    priors = self.policy(state)
    visits = len(priors) * self.depth
    value_estimate = sum(priors) / max(1, len(priors))
    return {
      "visits": visits,
      "value": value_estimate,
      "policy": priors,
    }


class BayesianLayer:
  def __init__(self, weight_mean: float = 0.0, weight_std: float = 0.1):
    self.weight_mean = weight_mean
    self.weight_std = max(1e-6, weight_std)

  def forward(self, inputs: Sequence[float]) -> List[float]:
    return [self.weight_mean * x for x in inputs]

  def sample(self, inputs: Sequence[float]) -> List[float]:
    return [
      random.gauss(self.weight_mean, self.weight_std) * x
      for x in inputs
    ]


class AdaptiveParticleFilter:
  def __init__(self, particles: int = 32):
    self.particles = max(1, particles)

  def step(self, observation: float) -> dict:
    weights = [1.0 / self.particles] * self.particles
    estimate = observation
    return {
      "weights": weights,
      "estimate": estimate,
    }


class NoUTurnSampler:
  def __init__(self, log_prob: Callable[[float], float], step_size: float = 0.1, max_tree_depth: int = 10):
    self.log_prob = log_prob
    self.step_size = step_size
    self.max_tree_depth = max_tree_depth

  def sample(self, initial: float, num_samples: int = 100) -> List[float]:
    samples = [initial]
    current = initial
    for _ in range(max(1, num_samples - 1)):
      gradient = self.log_prob(current + self.step_size) - self.log_prob(current - self.step_size)
      current = current + 0.5 * self.step_size * gradient
      samples.append(current)
    return samples


class SparseGaussianProcess:
  def __init__(self, inducing_points: Sequence[float] | None = None):
    self.inducing_points = list(inducing_points or [0.0])

  def predict(self, x: Sequence[float]) -> List[Tuple[float, float]]:
    predictions: List[Tuple[float, float]] = []
    for value in x:
      mean = sum(self.inducing_points) / len(self.inducing_points)
      variance = abs(value - mean)
      predictions.append((mean, variance))
    return predictions


class ArchitectureSearchController:
  def __init__(self, search_space: Sequence[str] | None = None):
    self.search_space = list(search_space or ["conv", "transformer", "mamba"])
    self.index = 0

  def propose(self) -> str:
    choice = self.search_space[self.index % len(self.search_space)]
    self.index += 1
    return choice
