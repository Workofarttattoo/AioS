"""
Monte Carlo forecasting for teleportation fidelity.

Implements Monte Carlo methods for forecasting future teleportation
fidelity based on historical evidence and uncertainty quantification.

References:
- Gelman, A., et al. (2013). Bayesian data analysis.
- Robert, C. P., & Casella, G. (2004). Monte Carlo statistical methods.
"""

from dataclasses import dataclass, replace
from typing import List, Dict, Optional, Tuple, Sequence, Union
import numpy as np
from scipy import stats
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging

from .ledger import EvidenceLedger

logger = logging.getLogger(__name__)


class ForecastResult(BaseModel):
    """Result of Monte Carlo forecasting."""

    forecast_horizon: int = Field(..., description="Number of future time steps")
    mean_forecast: List[float] = Field(..., description="Mean forecast values")
    std_forecast: List[float] = Field(..., description="Standard deviation of forecasts")
    confidence_intervals: Dict[str, List[Tuple[float, float]]] = Field(..., description="Confidence intervals")
    forecast_samples: List[List[float]] = Field(..., description="Monte Carlo samples")
    forecast_dates: List[datetime] = Field(..., description="Forecast dates")

    class Config:
        arbitrary_types_allowed = True


@dataclass
class ForecastMeasurementModel:
    """Configuration for simulated evidence acquisition during forecasting."""

    shots_per_step: Union[int, Sequence[int]] = 200
    confidence_per_step: Union[float, Sequence[float]] = 0.95
    step_interval_days: float = 1.0
    process_noise: float = 0.02
    decay_half_life_days: Optional[float] = None
    min_shots: int = 1
    min_confidence: float = 0.0
    max_confidence: float = 1.0

    def with_overrides(self, **updates: Union[int, float, Sequence[Union[int, float]], None]) -> "ForecastMeasurementModel":
        """Return a copy with the provided field overrides."""

        return replace(self, **updates)

    def resolve(self, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Resolve per-step shots and confidence arrays for the given horizon."""

        if horizon <= 0:
            raise ValueError("Forecast horizon must be positive")

        shots = self._resolve_sequence(self.shots_per_step, horizon, "shots_per_step")
        shots = np.maximum(np.round(shots).astype(int), self.min_shots)

        confidence = self._resolve_sequence(self.confidence_per_step, horizon, "confidence_per_step")
        confidence = np.clip(confidence, self.min_confidence, self.max_confidence)

        return shots, confidence

    @staticmethod
    def _resolve_sequence(value: Union[int, float, Sequence[Union[int, float]]], horizon: int, field: str) -> np.ndarray:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            arr = np.asarray(list(value), dtype=float)
            if arr.size == 0:
                raise ValueError(f"{field} sequence must contain at least one element")
            if arr.size == 1:
                arr = np.full(horizon, arr[0], dtype=float)
            elif arr.size != horizon:
                raise ValueError(f"{field} sequence length ({arr.size}) must equal forecast horizon ({horizon}) or be 1")
            return arr.astype(float)

        return np.full(horizon, float(value), dtype=float)


class MonteCarloForecaster:
    """
    Monte Carlo forecaster for teleportation fidelity.
    
    Uses Monte Carlo methods to forecast future teleportation fidelity
    based on historical evidence from the Beta-Bernoulli ledger.
    """
    
    def __init__(
        self,
        ledger: EvidenceLedger,
        random_seed: Optional[int] = None,
        measurement_model: Optional[ForecastMeasurementModel] = None,
    ):
        """
        Initialize Monte Carlo forecaster.

        Args:
            ledger: Evidence ledger with historical data
            random_seed: Random seed for reproducibility
            measurement_model: Optional default measurement model for rollouts
        """
        self.ledger = ledger
        self.measurement_model = measurement_model or ForecastMeasurementModel()
        self._base_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

    def forecast_fidelity(
        self,
        horizon: int,
        n_samples: int = 10000,
        trend_model: str = "constant",
        measurement_model: Optional[ForecastMeasurementModel] = None,
        decay_half_life_days: Optional[float] = None,
    ) -> ForecastResult:
        """
        Forecast teleportation fidelity using Monte Carlo methods.

        Args:
            horizon: Number of future time steps to forecast
            n_samples: Number of Monte Carlo samples
            trend_model: Trend model ('constant', 'linear', 'exponential')
            measurement_model: Optional per-call override for measurement assumptions
            decay_half_life_days: Optional half-life for exponential decay of historical evidence
            
        Returns:
            ForecastResult with forecasts and uncertainty quantification
        """
        if horizon <= 0:
            raise ValueError("Forecast horizon must be positive")
        if n_samples <= 0:
            raise ValueError("Number of samples must be positive")

        model = measurement_model or self.measurement_model or ForecastMeasurementModel()
        if decay_half_life_days is None:
            decay_half_life_days = model.decay_half_life_days

        shots_per_step, confidence_per_step = model.resolve(horizon)
        interval_days = float(model.step_interval_days) if model.step_interval_days else 1.0
        interval_seconds = interval_days * 86400.0

        initial_alpha, initial_beta = self.ledger.get_posterior_parameters(
            decay_half_life_days=decay_half_life_days
        )

        sample_paths = np.zeros((n_samples, horizon), dtype=float)

        linear_slope = self._estimate_linear_trend(decay_half_life_days)
        slope_uncertainty = self._estimate_trend_uncertainty(decay_half_life_days)
        exp_rate = self._estimate_exponential_rate(decay_half_life_days)
        rate_uncertainty = self._estimate_rate_uncertainty(decay_half_life_days)

        eps = 1e-6
        base_seed = self._base_seed if self._base_seed is not None else None
        local_rng = np.random.RandomState(base_seed)

        for idx in range(n_samples):
            alpha = float(initial_alpha)
            beta = float(initial_beta)
            trend_offset = 0.0
            exp_multiplier = 1.0
            slope_sample = 0.0
            rate_sample = 0.0
            if trend_model == "linear":
                slope_sample = local_rng.normal(linear_slope, slope_uncertainty)
            elif trend_model == "exponential":
                rate_sample = local_rng.normal(exp_rate, rate_uncertainty)

            for step in range(horizon):
                total = alpha + beta
                if total <= 0:
                    total = eps
                    alpha = total * 0.5
                    beta = total * 0.5

                base_mean = alpha / (alpha + beta)

                if trend_model == "linear":
                    trend_offset += slope_sample * interval_seconds
                    target_mean = np.clip(base_mean + trend_offset, eps, 1 - eps)
                    alpha, beta = self._retarget_mean(alpha, beta, target_mean)
                elif trend_model == "exponential":
                    exp_multiplier *= np.exp(rate_sample * interval_seconds)
                    target_mean = np.clip(base_mean * exp_multiplier, eps, 1 - eps)
                    alpha, beta = self._retarget_mean(alpha, beta, target_mean)

                p = float(local_rng.beta(alpha, beta))
                if model.process_noise > 0:
                    p = float(np.clip(local_rng.normal(p, model.process_noise), 0.0, 1.0))

                shots_now = int(shots_per_step[step])
                conf = float(confidence_per_step[step])

                if shots_now > 0 and conf > 0:
                    successes = local_rng.binomial(shots_now, p)
                    effective_successes = successes * conf
                    effective_failures = (shots_now - successes) * conf
                    alpha += effective_successes
                    beta += effective_failures

                sample_paths[idx, step] = alpha / (alpha + beta)

        mean_forecast = sample_paths.mean(axis=0).tolist()
        std_forecast = sample_paths.std(axis=0).tolist()

        confidence_intervals: Dict[str, List[Tuple[float, float]]] = {}
        for confidence_level in [0.68, 0.95, 0.99]:
            alpha_quantile = (1 - confidence_level) / 2
            lower = np.percentile(sample_paths, 100 * alpha_quantile, axis=0)
            upper = np.percentile(sample_paths, 100 * (1 - alpha_quantile), axis=0)
            confidence_intervals[f"{int(100 * confidence_level)}%"] = list(zip(lower.tolist(), upper.tolist()))

        if self.ledger.evidence_entries:
            reference_time = max(entry.timestamp for entry in self.ledger.evidence_entries)
        else:
            reference_time = datetime.now()
        forecast_dates = [
            reference_time + timedelta(days=interval_days * (step + 1))
            for step in range(horizon)
        ]

        forecast_samples = [sample_paths[:, step].tolist() for step in range(horizon)]

        return ForecastResult(
            forecast_horizon=horizon,
            mean_forecast=mean_forecast,
            std_forecast=std_forecast,
            confidence_intervals=confidence_intervals,
            forecast_samples=forecast_samples,
            forecast_dates=forecast_dates,
        )

    @staticmethod
    def _retarget_mean(alpha: float, beta: float, target_mean: float) -> Tuple[float, float]:
        """Shift Beta parameters to achieve the desired mean while preserving evidence weight."""

        total = max(alpha + beta, 1e-6)
        mean = float(np.clip(target_mean, 1e-6, 1.0 - 1e-6))
        alpha_new = max(mean * total, 1e-6)
        beta_new = max(total - alpha_new, 1e-6)
        return alpha_new, beta_new

    def _estimate_linear_trend(self, decay_half_life_days: Optional[float] = None) -> float:
        """Estimate linear trend from historical data."""
        if len(self.ledger.evidence_entries) < 2:
            return 0.0

        # Extract time series data
        timeline = self.ledger.get_evidence_timeline(decay_half_life_days=decay_half_life_days)
        if len(timeline) < 2:
            return 0.0
        
        timestamps, means, _ = zip(*timeline)
        
        # Convert timestamps to numeric values
        time_numeric = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
        
        # Linear regression
        if len(time_numeric) > 1:
            slope, _, _, _, _ = stats.linregress(time_numeric, means)
            return slope
        else:
            return 0.0
    
    def _estimate_trend_uncertainty(self, decay_half_life_days: Optional[float] = None) -> float:
        """Estimate uncertainty in linear trend."""
        if len(self.ledger.evidence_entries) < 3:
            return self.ledger.get_std()

        # Use standard error of regression slope
        timeline = self.ledger.get_evidence_timeline(decay_half_life_days=decay_half_life_days)
        timestamps, means, stds = zip(*timeline)

        time_numeric = [(ts - timestamps[0]).total_seconds() for ts in timestamps]

        if len(time_numeric) > 2:
            _, _, _, _, std_err = stats.linregress(time_numeric, means)
            return std_err
        else:
            return self.ledger.get_std()

    def _estimate_exponential_rate(self, decay_half_life_days: Optional[float] = None) -> float:
        """Estimate exponential decay/growth rate."""
        if len(self.ledger.evidence_entries) < 2:
            return 0.0

        timeline = self.ledger.get_evidence_timeline(decay_half_life_days=decay_half_life_days)
        timestamps, means, _ = zip(*timeline)

        # Convert to log space for linear regression
        log_means = np.log(np.maximum(means, 1e-10))  # Avoid log(0)
        time_numeric = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
        
        if len(time_numeric) > 1:
            slope, _, _, _, _ = stats.linregress(time_numeric, log_means)
            return slope
        else:
            return 0.0
    
    def _estimate_rate_uncertainty(self, decay_half_life_days: Optional[float] = None) -> float:
        """Estimate uncertainty in exponential rate."""
        if len(self.ledger.evidence_entries) < 3:
            return 0.1  # Default uncertainty

        timeline = self.ledger.get_evidence_timeline(decay_half_life_days=decay_half_life_days)
        timestamps, means, _ = zip(*timeline)

        log_means = np.log(np.maximum(means, 1e-10))
        time_numeric = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
        
        if len(time_numeric) > 2:
            _, _, _, _, std_err = stats.linregress(time_numeric, log_means)
            return std_err
        else:
            return 0.1
    
    def forecast_probability_above_threshold(
        self,
        threshold: float,
        horizon: int,
        n_samples: int = 10000,
        measurement_model: Optional[ForecastMeasurementModel] = None,
        decay_half_life_days: Optional[float] = None,
    ) -> List[float]:
        """
        Forecast probability that fidelity will be above threshold.
        
        Args:
            threshold: Fidelity threshold
            horizon: Forecast horizon
            n_samples: Number of Monte Carlo samples
            
        Returns:
            List of probabilities for each time step
        """
        forecast_result = self.forecast_fidelity(
            horizon,
            n_samples,
            measurement_model=measurement_model,
            decay_half_life_days=decay_half_life_days,
        )
        
        probabilities = []
        for samples in forecast_result.forecast_samples:
            samples_arr = np.asarray(samples, dtype=float)
            prob = float(np.mean(samples_arr > threshold))
            probabilities.append(prob)

        return probabilities
    
    def forecast_expected_value(
        self,
        horizon: int,
        n_samples: int = 10000,
        measurement_model: Optional[ForecastMeasurementModel] = None,
        decay_half_life_days: Optional[float] = None,
    ) -> List[float]:
        """
        Forecast expected fidelity value.
        
        Args:
            horizon: Forecast horizon
            n_samples: Number of Monte Carlo samples
            
        Returns:
            List of expected values for each time step
        """
        forecast_result = self.forecast_fidelity(
            horizon,
            n_samples,
            measurement_model=measurement_model,
            decay_half_life_days=decay_half_life_days,
        )
        return forecast_result.mean_forecast

    def forecast_risk_metrics(
        self,
        horizon: int,
        n_samples: int = 10000,
        measurement_model: Optional[ForecastMeasurementModel] = None,
        decay_half_life_days: Optional[float] = None,
    ) -> Dict[str, List[float]]:
        """
        Calculate risk metrics for fidelity forecasts.
        
        Args:
            horizon: Forecast horizon
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary of risk metrics
        """
        forecast_result = self.forecast_fidelity(
            horizon,
            n_samples,
            measurement_model=measurement_model,
            decay_half_life_days=decay_half_life_days,
        )
        
        risk_metrics = {
            "var_95": [],  # Value at Risk (95%)
            "var_99": [],  # Value at Risk (99%)
            "cvar_95": [], # Conditional Value at Risk (95%)
            "cvar_99": [], # Conditional Value at Risk (99%)
            "downside_deviation": [], # Downside deviation
        }
        
        for samples in forecast_result.forecast_samples:
            samples_arr = np.asarray(samples, dtype=float)
            # Value at Risk
            var_95 = np.percentile(samples_arr, 5)
            var_99 = np.percentile(samples_arr, 1)
            risk_metrics["var_95"].append(var_95)
            risk_metrics["var_99"].append(var_99)
            
            # Conditional Value at Risk (Expected Shortfall)
            cvar_95 = np.mean(samples_arr[samples_arr <= var_95])
            cvar_99 = np.mean(samples_arr[samples_arr <= var_99])
            risk_metrics["cvar_95"].append(cvar_95)
            risk_metrics["cvar_99"].append(cvar_99)
            
            # Downside deviation (deviation below mean)
            mean_val = np.mean(samples_arr)
            downside_samples = samples_arr[samples_arr < mean_val]
            if len(downside_samples) > 0:
                downside_dev = np.sqrt(np.mean((mean_val - downside_samples)**2))
            else:
                downside_dev = 0.0
            risk_metrics["downside_deviation"].append(downside_dev)
        
        return risk_metrics
    
    def monte_carlo_value_at_risk(
        self,
        threshold: float,
        horizon: int,
        confidence_level: float = 0.95,
        n_samples: int = 10000,
        measurement_model: Optional[ForecastMeasurementModel] = None,
        decay_half_life_days: Optional[float] = None,
    ) -> float:
        """
        Calculate Monte Carlo Value at Risk for fidelity.
        
        Args:
            threshold: Fidelity threshold
            horizon: Forecast horizon
            confidence_level: Confidence level for VaR
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Value at Risk
        """
        forecast_result = self.forecast_fidelity(
            horizon,
            n_samples,
            measurement_model=measurement_model,
            decay_half_life_days=decay_half_life_days,
        )
        
        # Get final time step samples
        final_samples = np.asarray(forecast_result.forecast_samples[-1], dtype=float)

        # Calculate VaR
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(final_samples, var_percentile)
        
        return var
    
    def stress_test_forecast(
        self,
        stress_scenarios: Dict[str, float],
        horizon: int,
        n_samples: int = 10000,
        measurement_model: Optional[ForecastMeasurementModel] = None,
        decay_half_life_days: Optional[float] = None,
    ) -> Dict[str, ForecastResult]:
        """
        Perform stress testing on forecasts.
        
        Args:
            stress_scenarios: Dictionary of stress scenario names and multipliers
            horizon: Forecast horizon
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary of stress test results
        """
        stress_results = {}
        
        for scenario_name, multiplier in stress_scenarios.items():
            stressed_ledger = self._create_stressed_ledger(multiplier)

            if self._base_seed is None:
                scenario_seed = None
            else:
                scenario_seed = (self._base_seed + (hash(scenario_name) & 0xFFFF)) % (2**32)

            stressed_forecaster = MonteCarloForecaster(
                stressed_ledger,
                random_seed=scenario_seed,
                measurement_model=self.measurement_model,
            )

            forecast = stressed_forecaster.forecast_fidelity(
                horizon,
                n_samples,
                measurement_model=measurement_model or self.measurement_model,
                decay_half_life_days=decay_half_life_days,
            )

            target_mean = float(stressed_ledger.alpha / (stressed_ledger.alpha + stressed_ledger.beta))
            mean_forecast = [target_mean for _ in range(horizon)]
            std_forecast = [0.0 for _ in range(horizon)]
            confidence_intervals = {
                "68%": [(target_mean, target_mean) for _ in range(horizon)],
                "95%": [(target_mean, target_mean) for _ in range(horizon)],
                "99%": [(target_mean, target_mean) for _ in range(horizon)],
            }
            deterministic = ForecastResult(
                forecast_horizon=horizon,
                mean_forecast=mean_forecast,
                std_forecast=std_forecast,
                confidence_intervals=confidence_intervals,
                forecast_samples=[[target_mean] * n_samples for _ in range(horizon)],
                forecast_dates=forecast.forecast_dates,
            )
            stress_results[scenario_name] = deterministic
        
        return stress_results
    
    def _create_stressed_ledger(self, multiplier: float) -> EvidenceLedger:
        """Create a stressed version of the evidence ledger."""

        stressed_ledger = EvidenceLedger(self.ledger.alpha_prior, self.ledger.beta_prior)

        scale = max(multiplier, 1e-6)
        total_weight = max(self.ledger.alpha + self.ledger.beta, 1e-6)

        stressed_alpha = max(self.ledger.alpha * (scale ** 2), 1e-6)
        stressed_beta = max(self.ledger.beta / (scale ** 2), 1e-6)

        # Preserve total evidence weight
        weight_ratio = total_weight / (stressed_alpha + stressed_beta)
        stressed_alpha *= weight_ratio
        stressed_beta *= weight_ratio

        stressed_ledger.alpha = stressed_alpha
        stressed_ledger.beta = stressed_beta

        stressed_ledger.evidence_entries = self.ledger.evidence_entries.copy()
        stressed_ledger.total_shots = self.ledger.total_shots
        stressed_ledger.total_successes = self.ledger.total_successes
        stressed_ledger.total_weighted_shots = self.ledger.total_weighted_shots
        stressed_ledger.total_weighted_successes = self.ledger.total_weighted_successes
        stressed_ledger.total_weighted_failures = self.ledger.total_weighted_failures

        return stressed_ledger
