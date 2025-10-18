"""
Beta-Bernoulli evidence ledger for tracking teleportation fidelity.

Implements a Bayesian evidence accumulation system using Beta-Bernoulli
conjugate priors for tracking teleportation fidelity over time.

References:
- Gelman, A., et al. (2013). Bayesian data analysis.
- Murphy, K. P. (2012). Machine learning: a probabilistic perspective.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy import stats
from scipy.special import beta, betaln, digamma
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


class EvidenceEntry(BaseModel):
    """Single entry in the evidence ledger."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestamp: datetime = Field(..., description="Timestamp of evidence collection")
    fidelity: float = Field(..., ge=0.0, le=1.0, description="Observed fidelity")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in measurement")
    shots: int = Field(..., gt=0, description="Number of measurement shots")
    success_count: int = Field(..., ge=0, description="Number of successful teleportations")
    experiment_id: str = Field(..., description="Unique experiment identifier")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Additional metadata")
    weight: float = Field(1.0, ge=0.0, description="Additional weighting applied to this evidence")
    effective_confidence: float = Field(1.0, ge=0.0, description="Confidence after weighting/decay adjustments")
    effective_successes: float = Field(0.0, ge=0.0, description="Weighted successes contributed to posterior")
    effective_failures: float = Field(0.0, ge=0.0, description="Weighted failures contributed to posterior")


class EvidenceLedger:
    """
    Beta-Bernoulli evidence ledger for teleportation fidelity tracking.
    
    Uses Beta-Bernoulli conjugate priors to maintain a Bayesian estimate
    of teleportation fidelity. The Beta distribution is the conjugate prior
    for the Bernoulli distribution, making it ideal for tracking binary
    success/failure outcomes.
    
    The Beta distribution is parameterized by α (successes) and β (failures):
    Beta(α, β) with mean μ = α/(α + β) and variance σ² = αβ/((α+β)²(α+β+1))
    """
    
    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        """
        Initialize evidence ledger with Beta prior.
        
        Args:
            alpha_prior: Prior number of successes (default: 1.0 for uniform prior)
            beta_prior: Prior number of failures (default: 1.0 for uniform prior)
        """
        if alpha_prior <= 0 or beta_prior <= 0:
            raise ValueError("Prior parameters must be positive")
        
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.alpha = alpha_prior
        self.beta = beta_prior
        self.evidence_entries: List[EvidenceEntry] = []
        self.total_shots = 0.0
        self.total_successes = 0.0
        self.total_weighted_shots = 0.0
        self.total_weighted_successes = 0.0
        self.total_weighted_failures = 0.0

    def update_evidence(self, fidelity: float, confidence: float, shots: int,
                        experiment_id: str, metadata: Optional[Dict[str, str]] = None,
                        timestamp: Optional[datetime] = None, weight: float = 1.0) -> None:
        """
        Update evidence ledger with new teleportation results.
        
        Args:
            fidelity: Observed fidelity (0 ≤ fidelity ≤ 1)
            confidence: Confidence in measurement (0 ≤ confidence ≤ 1)
            shots: Number of measurement shots
            experiment_id: Unique experiment identifier
            metadata: Additional metadata
            timestamp: Optional explicit timestamp for the evidence entry
            weight: Optional multiplicative weight applied to confidence (e.g., manual adjustments)
        """
        if not 0 <= fidelity <= 1:
            raise ValueError("Fidelity must be between 0 and 1")
        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if shots <= 0:
            raise ValueError("Number of shots must be positive")
        
        # Calculate effective successes and failures
        # Weight by confidence to account for measurement uncertainty
        effective_confidence = confidence * weight
        effective_successes = fidelity * shots * effective_confidence
        effective_failures = (1 - fidelity) * shots * effective_confidence
        
        # Update Beta parameters
        self.alpha += effective_successes
        self.beta += effective_failures
        
        # Update totals
        self.total_shots += float(shots)
        self.total_successes += float(fidelity * shots)
        self.total_weighted_shots += shots * effective_confidence
        self.total_weighted_successes += effective_successes
        self.total_weighted_failures += effective_failures

        # Create evidence entry
        entry = EvidenceEntry(
            timestamp=timestamp or datetime.now(timezone.utc),
            fidelity=fidelity,
            confidence=confidence,
            shots=shots,
            success_count=int(fidelity * shots),
            experiment_id=experiment_id,
            metadata=metadata or {},
            weight=weight,
            effective_confidence=effective_confidence,
            effective_successes=effective_successes,
            effective_failures=effective_failures
        )
        
        self.evidence_entries.append(entry)

        logger.info(f"Updated evidence: α={self.alpha:.2f}, β={self.beta:.2f}, "
                   f"mean={self.get_mean():.4f}, std={self.get_std():.4f}")

    def get_posterior_parameters(
        self,
        decay_half_life_days: Optional[float] = None,
        reference_time: Optional[datetime] = None,
    ) -> Tuple[float, float]:
        """Return posterior α, β with optional exponential decay on historical evidence."""

        alpha = float(self.alpha_prior)
        beta = float(self.beta_prior)

        if not self.evidence_entries:
            return alpha, beta

        if reference_time is None:
            reference_time = max(entry.timestamp for entry in self.evidence_entries)

        for entry in self.evidence_entries:
            weight_factor = 1.0
            if decay_half_life_days and decay_half_life_days > 0:
                age_seconds = max((reference_time - entry.timestamp).total_seconds(), 0.0)
                age_days = age_seconds / 86400.0
                weight_factor *= 0.5 ** (age_days / decay_half_life_days)

            alpha += entry.effective_successes * weight_factor
            beta += entry.effective_failures * weight_factor

        return alpha, beta

    def get_mean(self) -> float:
        """
        Get posterior mean fidelity.

        Returns:
            Mean of Beta(α, β) distribution
        """
        return self.alpha / (self.alpha + self.beta)
    
    def get_std(self) -> float:
        """
        Get posterior standard deviation.
        
        Returns:
            Standard deviation of Beta(α, β) distribution
        """
        n = self.alpha + self.beta
        variance = (self.alpha * self.beta) / (n**2 * (n + 1))
        return np.sqrt(variance)
    
    def get_credible_interval(self, confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Get credible interval for fidelity.
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        alpha_level = (1 - confidence_level) / 2
        lower = stats.beta.ppf(alpha_level, self.alpha, self.beta)
        upper = stats.beta.ppf(1 - alpha_level, self.alpha, self.beta)
        return (lower, upper)
    
    def get_probability_above_threshold(self, threshold: float) -> float:
        """
        Get probability that fidelity is above threshold.
        
        Args:
            threshold: Fidelity threshold
            
        Returns:
            Probability P(fidelity > threshold)
        """
        return 1 - stats.beta.cdf(threshold, self.alpha, self.beta)
    
    def get_probability_below_threshold(self, threshold: float) -> float:
        """
        Get probability that fidelity is below threshold.
        
        Args:
            threshold: Fidelity threshold
            
        Returns:
            Probability P(fidelity < threshold)
        """
        return stats.beta.cdf(threshold, self.alpha, self.beta)
    
    def get_evidence_ratio(self, threshold: float) -> float:
        """
        Get evidence ratio for threshold hypothesis.
        
        Args:
            threshold: Fidelity threshold to test
            
        Returns:
            Evidence ratio (Bayes factor)
        """
        # Prior probability above threshold
        prior_alpha = self.alpha_prior
        prior_beta = self.beta_prior
        prior_prob_above = 1 - stats.beta.cdf(threshold, prior_alpha, prior_beta)
        
        # Posterior probability above threshold
        posterior_prob_above = self.get_probability_above_threshold(threshold)
        
        # Evidence ratio
        if prior_prob_above > 0 and posterior_prob_above > 0:
            evidence_ratio = (posterior_prob_above / (1 - posterior_prob_above)) / \
                           (prior_prob_above / (1 - prior_prob_above))
        else:
            evidence_ratio = float('inf') if posterior_prob_above > prior_prob_above else 0.0
        
        return evidence_ratio
    
    def get_effective_sample_size(self) -> float:
        """
        Get effective sample size of the posterior.
        
        Returns:
            Effective sample size (α + β)
        """
        return self.alpha + self.beta
    
    def get_entropy(self) -> float:
        """
        Get entropy of the posterior Beta distribution.
        
        Returns:
            Differential entropy of Beta(α, β)
        """
        n = self.alpha + self.beta
        return (
            betaln(self.alpha, self.beta)
            - (self.alpha - 1) * (digamma(self.alpha) - digamma(n))
            - (self.beta - 1) * (digamma(self.beta) - digamma(n))
        )
    
    def get_kl_divergence_from_prior(self) -> float:
        """
        Get KL divergence from prior to posterior.
        
        Returns:
            KL(Beta(α,β) || Beta(α₀,β₀))
        """
        # KL divergence between Beta distributions
        kl = (betaln(self.alpha_prior, self.beta_prior) - betaln(self.alpha, self.beta) +
              (self.alpha - self.alpha_prior) * (stats.digamma(self.alpha) - stats.digamma(self.alpha + self.beta)) +
              (self.beta - self.beta_prior) * (stats.digamma(self.beta) - stats.digamma(self.alpha + self.beta)))
        
        return kl
    
    def sample_posterior(
        self,
        n_samples: int = 1000,
        decay_half_life_days: Optional[float] = None,
    ) -> np.ndarray:
        """
        Sample from posterior Beta distribution.

        Args:
            n_samples: Number of samples to draw
            decay_half_life_days: Optional half-life (in days) for exponential decay weighting
            
        Returns:
            Array of samples from Beta(α, β)
        """
        alpha, beta = self.get_posterior_parameters(decay_half_life_days=decay_half_life_days)
        return stats.beta.rvs(alpha, beta, size=n_samples)
    
    def get_summary_statistics(self) -> Dict[str, float]:
        """
        Get comprehensive summary statistics.
        
        Returns:
            Dictionary of summary statistics
        """
        ci_95 = self.get_credible_interval(0.95)
        ci_99 = self.get_credible_interval(0.99)
        
        return {
            "mean": self.get_mean(),
            "std": self.get_std(),
            "ci_95_lower": ci_95[0],
            "ci_95_upper": ci_95[1],
            "ci_99_lower": ci_99[0],
            "ci_99_upper": ci_99[1],
            "effective_sample_size": self.get_effective_sample_size(),
            "entropy": self.get_entropy(),
            "kl_divergence_from_prior": self.get_kl_divergence_from_prior(),
            "total_evidence_entries": len(self.evidence_entries),
            "total_shots": self.total_shots,
            "total_successes": self.total_successes,
        }
    
    def reset(self) -> None:
        """Reset ledger to initial state."""
        self.alpha = self.alpha_prior
        self.beta = self.beta_prior
        self.evidence_entries.clear()
        self.total_shots = 0.0
        self.total_successes = 0.0
        self.total_weighted_shots = 0.0
        self.total_weighted_successes = 0.0
        self.total_weighted_failures = 0.0

    def export_evidence(self) -> List[Dict]:
        """
        Export evidence entries as list of dictionaries.

        Returns:
            List of evidence entry dictionaries
        """
        return [entry.model_dump(mode="python") for entry in self.evidence_entries]
    
    def import_evidence(self, evidence_data: List[Dict]) -> None:
        """
        Import evidence entries from list of dictionaries.
        
        Args:
            evidence_data: List of evidence entry dictionaries
        """
        self.reset()
        
        for entry_data in evidence_data:
            entry = EvidenceEntry(**entry_data)
            self.update_evidence(
                fidelity=entry.fidelity,
                confidence=entry.confidence,
                shots=entry.shots,
                experiment_id=entry.experiment_id,
                metadata=entry.metadata,
                timestamp=entry.timestamp,
                weight=entry.weight,
            )

    def get_evidence_timeline(
        self,
        decay_half_life_days: Optional[float] = None,
        reference_time: Optional[datetime] = None,
    ) -> List[Tuple[datetime, float, float]]:
        """
        Get timeline of evidence accumulation.

        Returns:
            List of (timestamp, mean, std) tuples
        """
        timeline = []
        temp_alpha = float(self.alpha_prior)
        temp_beta = float(self.beta_prior)

        if reference_time is None and self.evidence_entries:
            reference_time = max(entry.timestamp for entry in self.evidence_entries)

        for entry in self.evidence_entries:
            weight_factor = 1.0
            if decay_half_life_days and decay_half_life_days > 0:
                if reference_time is None:
                    reference_time = entry.timestamp
                age_seconds = max((reference_time - entry.timestamp).total_seconds(), 0.0)
                age_days = age_seconds / 86400.0
                weight_factor *= 0.5 ** (age_days / decay_half_life_days)

            temp_alpha += entry.effective_successes * weight_factor
            temp_beta += entry.effective_failures * weight_factor

            # Calculate mean and std
            mean = temp_alpha / (temp_alpha + temp_beta)
            n = temp_alpha + temp_beta
            std = np.sqrt((temp_alpha * temp_beta) / (n**2 * (n + 1)))

            timeline.append((entry.timestamp, mean, std))

        return timeline
