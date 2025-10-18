"""
Unit tests for governance system.

Tests evidence ledger, Monte Carlo forecasting, and cadence planning
with statistical validation and edge cases.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta, timezone

from qulab.governance.ledger import EvidenceLedger, EvidenceEntry
from qulab.governance.forecasting import (
    MonteCarloForecaster,
    ForecastResult,
    ForecastMeasurementModel,
)
from qulab.governance.cadence import CadencePlanner, CadencePlan


class TestEvidenceLedger:
    """Test cases for EvidenceLedger."""
    
    def test_initialization(self):
        """Test ledger initialization."""
        ledger = EvidenceLedger()
        assert ledger.alpha_prior == 1.0
        assert ledger.beta_prior == 1.0
        assert ledger.alpha == 1.0
        assert ledger.beta == 1.0
        assert len(ledger.evidence_entries) == 0
        assert ledger.total_shots == 0
        assert ledger.total_successes == 0
    
    def test_initialization_custom_priors(self):
        """Test ledger initialization with custom priors."""
        ledger = EvidenceLedger(alpha_prior=2.0, beta_prior=3.0)
        assert ledger.alpha_prior == 2.0
        assert ledger.beta_prior == 3.0
        assert ledger.alpha == 2.0
        assert ledger.beta == 3.0
    
    def test_update_evidence(self):
        """Test evidence update."""
        ledger = EvidenceLedger()
        
        # Update with perfect fidelity
        ledger.update_evidence(
            fidelity=1.0,
            confidence=0.95,
            shots=1000,
            experiment_id="test_1"
        )
        
        assert len(ledger.evidence_entries) == 1
        assert ledger.total_shots == 1000
        assert ledger.total_successes == 1000
        assert ledger.alpha > ledger.alpha_prior
        assert ledger.beta == ledger.beta_prior  # No failures

    def test_weighted_totals(self):
        """Confidence weighting should reflect in weighted totals."""
        ledger = EvidenceLedger()

        ledger.update_evidence(
            fidelity=0.75,
            confidence=0.5,
            shots=200,
            experiment_id="weighted"
        )

        assert np.isclose(ledger.total_weighted_shots, 100.0)
        assert np.isclose(ledger.total_weighted_successes, 0.75 * 200 * 0.5)
        assert np.isclose(ledger.total_weighted_failures, (1 - 0.75) * 200 * 0.5)

    def test_posterior_decay_half_life(self):
        """Older evidence should be down-weighted when applying decay."""
        ledger = EvidenceLedger()
        old_timestamp = datetime.now(timezone.utc) - timedelta(days=30)
        recent_timestamp = datetime.now(timezone.utc)

        ledger.update_evidence(
            fidelity=0.2,
            confidence=1.0,
            shots=500,
            experiment_id="old",
            timestamp=old_timestamp
        )
        ledger.update_evidence(
            fidelity=0.9,
            confidence=1.0,
            shots=500,
            experiment_id="recent",
            timestamp=recent_timestamp
        )

        alpha_full, beta_full = ledger.get_posterior_parameters()
        alpha_decay, beta_decay = ledger.get_posterior_parameters(decay_half_life_days=7)

        mean_full = alpha_full / (alpha_full + beta_full)
        mean_decay = alpha_decay / (alpha_decay + beta_decay)

        assert alpha_decay < alpha_full  # older successes contribute less
        assert beta_decay < beta_full    # older failures down-weighted more strongly
        assert mean_decay > mean_full    # posterior mean shifts toward recent high fidelity

    def test_update_evidence_imperfect_fidelity(self):
        """Test evidence update with imperfect fidelity."""
        ledger = EvidenceLedger()
        
        # Update with 80% fidelity
        ledger.update_evidence(
            fidelity=0.8,
            confidence=0.95,
            shots=1000,
            experiment_id="test_2"
        )
        
        assert len(ledger.evidence_entries) == 1
        assert ledger.total_shots == 1000
        assert ledger.total_successes == 800
        assert ledger.alpha > ledger.alpha_prior
        assert ledger.beta > ledger.beta_prior  # Some failures
    
    def test_update_evidence_multiple(self):
        """Test multiple evidence updates."""
        ledger = EvidenceLedger()
        
        # Add multiple evidence entries
        for i in range(5):
            ledger.update_evidence(
                fidelity=0.9,
                confidence=0.95,
                shots=100,
                experiment_id=f"test_{i}"
            )
        
        assert len(ledger.evidence_entries) == 5
        assert ledger.total_shots == 500
        assert ledger.total_successes == 450  # 90% of 500
    
    def test_get_mean(self):
        """Test mean calculation."""
        ledger = EvidenceLedger()
        
        # With uniform prior (1, 1), mean should be 0.5
        assert np.isclose(ledger.get_mean(), 0.5)
        
        # Add evidence
        ledger.update_evidence(1.0, 0.95, 100, "test")
        assert ledger.get_mean() > 0.5  # Should increase with success
    
    def test_get_std(self):
        """Test standard deviation calculation."""
        ledger = EvidenceLedger()
        
        # With uniform prior, std should be reasonable
        std = ledger.get_std()
        assert 0.0 <= std <= 1.0
        
        # Add evidence
        ledger.update_evidence(1.0, 0.95, 100, "test")
        new_std = ledger.get_std()
        assert new_std < std  # Should decrease with more evidence
    
    def test_get_credible_interval(self):
        """Test credible interval calculation."""
        ledger = EvidenceLedger()
        
        # Test 95% credible interval
        ci_95 = ledger.get_credible_interval(0.95)
        assert len(ci_95) == 2
        assert ci_95[0] <= ci_95[1]
        assert 0.0 <= ci_95[0] <= 1.0
        assert 0.0 <= ci_95[1] <= 1.0
        
        # Test 99% credible interval
        ci_99 = ledger.get_credible_interval(0.99)
        assert ci_99[0] <= ci_95[0]  # 99% CI should be wider
        assert ci_95[1] <= ci_99[1]
    
    def test_get_probability_above_threshold(self):
        """Test probability above threshold calculation."""
        ledger = EvidenceLedger()
        
        # With uniform prior, P(fidelity > 0.5) should be 0.5
        prob = ledger.get_probability_above_threshold(0.5)
        assert np.isclose(prob, 0.5, atol=0.1)
        
        # Add evidence of high fidelity
        ledger.update_evidence(0.9, 0.95, 100, "test")
        prob_high = ledger.get_probability_above_threshold(0.5)
        assert prob_high > prob  # Should increase
    
    def test_get_effective_sample_size(self):
        """Test effective sample size calculation."""
        ledger = EvidenceLedger()
        
        # Initial sample size should be 2 (alpha + beta = 1 + 1)
        assert ledger.get_effective_sample_size() == 2.0
        
        # Add evidence
        ledger.update_evidence(0.8, 0.95, 100, "test")
        new_size = ledger.get_effective_sample_size()
        assert new_size > 2.0  # Should increase
    
    def test_sample_posterior(self):
        """Test posterior sampling."""
        ledger = EvidenceLedger()
        
        # Sample from posterior
        samples = ledger.sample_posterior(1000)
        
        assert len(samples) == 1000
        assert all(0.0 <= s <= 1.0 for s in samples)
        
        # Mean of samples should be close to posterior mean
        sample_mean = np.mean(samples)
        posterior_mean = ledger.get_mean()
        assert np.isclose(sample_mean, posterior_mean, atol=0.1)
    
    def test_reset(self):
        """Test ledger reset."""
        ledger = EvidenceLedger()
        
        # Add some evidence
        ledger.update_evidence(0.8, 0.95, 100, "test")
        assert len(ledger.evidence_entries) > 0
        
        # Reset
        ledger.reset()
        assert ledger.alpha == ledger.alpha_prior
        assert ledger.beta == ledger.beta_prior
        assert len(ledger.evidence_entries) == 0
        assert ledger.total_shots == 0
        assert ledger.total_successes == 0
    
    def test_export_import_evidence(self):
        """Test evidence export and import."""
        ledger1 = EvidenceLedger()
        
        # Add evidence
        ledger1.update_evidence(0.8, 0.95, 100, "test1")
        ledger1.update_evidence(0.9, 0.95, 100, "test2")
        
        # Export
        evidence_data = ledger1.export_evidence()
        assert len(evidence_data) == 2
        assert "effective_successes" in evidence_data[0]
        assert "effective_failures" in evidence_data[0]

        # Create new ledger and import
        ledger2 = EvidenceLedger()
        ledger2.import_evidence(evidence_data)

        # Should have same statistics
        assert np.isclose(ledger1.get_mean(), ledger2.get_mean())
        assert np.isclose(ledger1.get_std(), ledger2.get_std())
        assert ledger1.total_shots == ledger2.total_shots


class TestMonteCarloForecaster:
    """Test cases for MonteCarloForecaster."""
    
    def test_initialization(self):
        """Test forecaster initialization."""
        ledger = EvidenceLedger()
        forecaster = MonteCarloForecaster(ledger)
        
        assert forecaster.ledger is ledger
        assert forecaster.rng is not None
    
    def test_forecast_fidelity_constant_trend(self):
        """Test fidelity forecasting with constant trend."""
        ledger = EvidenceLedger()
        
        # Add some evidence
        ledger.update_evidence(0.8, 0.95, 100, "test1")
        ledger.update_evidence(0.9, 0.95, 100, "test2")
        
        forecaster = MonteCarloForecaster(ledger)
        forecast = forecaster.forecast_fidelity(horizon=10, n_samples=1000, trend_model='constant')
        
        assert isinstance(forecast, ForecastResult)
        assert forecast.forecast_horizon == 10
        assert len(forecast.mean_forecast) == 10
        assert len(forecast.std_forecast) == 10
        assert len(forecast.forecast_dates) == 10
        assert "95%" in forecast.confidence_intervals
        assert "99%" in forecast.confidence_intervals
    
    def test_forecast_fidelity_linear_trend(self):
        """Test fidelity forecasting with linear trend."""
        ledger = EvidenceLedger()
        
        # Add evidence with trend
        for i in range(5):
            fidelity = 0.7 + 0.05 * i  # Increasing trend
            ledger.update_evidence(fidelity, 0.95, 100, f"test_{i}")
        
        forecaster = MonteCarloForecaster(ledger)
        forecast = forecaster.forecast_fidelity(horizon=5, n_samples=1000, trend_model='linear')
        
        assert isinstance(forecast, ForecastResult)
        assert forecast.forecast_horizon == 5
        assert len(forecast.mean_forecast) == 5
    
    def test_forecast_probability_above_threshold(self):
        """Test probability above threshold forecasting."""
        ledger = EvidenceLedger()
        
        # Add evidence
        ledger.update_evidence(0.8, 0.95, 100, "test")
        
        forecaster = MonteCarloForecaster(ledger)
        probabilities = forecaster.forecast_probability_above_threshold(0.7, horizon=5)
        
        assert len(probabilities) == 5
        assert all(0.0 <= p <= 1.0 for p in probabilities)
    
    def test_forecast_risk_metrics(self):
        """Test risk metrics forecasting."""
        ledger = EvidenceLedger()
        
        # Add evidence
        ledger.update_evidence(0.8, 0.95, 100, "test")
        
        forecaster = MonteCarloForecaster(ledger)
        risk_metrics = forecaster.forecast_risk_metrics(horizon=5)
        
        assert "var_95" in risk_metrics
        assert "var_99" in risk_metrics
        assert "cvar_95" in risk_metrics
        assert "cvar_99" in risk_metrics
        assert "downside_deviation" in risk_metrics
        
        # Check that all metrics have correct length
        for metric_name, values in risk_metrics.items():
            assert len(values) == 5
    
    def test_stress_test_forecast(self):
        """Test stress testing forecasts."""
        ledger = EvidenceLedger()
        
        # Add evidence
        ledger.update_evidence(0.8, 0.95, 100, "test")
        
        forecaster = MonteCarloForecaster(ledger)
        stress_scenarios = {
            "optimistic": 1.2,
            "pessimistic": 0.8,
            "extreme": 0.5
        }
        
        stress_results = forecaster.stress_test_forecast(stress_scenarios, horizon=5)
        
        assert len(stress_results) == 3
        assert "optimistic" in stress_results
        assert "pessimistic" in stress_results
        assert "extreme" in stress_results
        
        # Check that optimistic scenario has higher mean than pessimistic
        opt_mean = stress_results["optimistic"].mean_forecast[-1]
        pess_mean = stress_results["pessimistic"].mean_forecast[-1]
        assert opt_mean > pess_mean

    def test_forecast_beta_binomial_measurement_model(self):
        """Forecast should respect measurement model assumptions."""
        ledger = EvidenceLedger()
        ledger.update_evidence(0.85, 1.0, 400, "base")

        model = ForecastMeasurementModel(
            shots_per_step=50,
            confidence_per_step=1.0,
            process_noise=0.0,
            step_interval_days=1.0,
        )

        forecaster = MonteCarloForecaster(ledger, random_seed=42, measurement_model=model)
        forecast = forecaster.forecast_fidelity(horizon=3, n_samples=500, measurement_model=model)

        assert len(forecast.mean_forecast) == 3
        assert all(0.0 <= m <= 1.0 for m in forecast.mean_forecast)

        probs = forecaster.forecast_probability_above_threshold(
            threshold=0.7,
            horizon=3,
            n_samples=500,
            measurement_model=model,
        )

        assert len(probs) == 3
        assert probs[0] > 0.5
        assert probs[-1] >= probs[0] - 0.1

    def test_stress_scenarios_shift_mean(self):
        """Stress scenarios should shift posterior mean as expected."""
        ledger = EvidenceLedger()
        ledger.update_evidence(0.75, 1.0, 300, "baseline")

        model = ForecastMeasurementModel(shots_per_step=40, confidence_per_step=1.0)
        forecaster = MonteCarloForecaster(ledger, random_seed=7, measurement_model=model)

        stress_results = forecaster.stress_test_forecast(
            {"down": 0.5, "up": 2.0},
            horizon=1,
            n_samples=400,
            measurement_model=model,
        )

        down_mean = stress_results["down"].mean_forecast[0]
        up_mean = stress_results["up"].mean_forecast[0]

        assert up_mean > down_mean


class TestCadencePlanner:
    """Test cases for CadencePlanner."""
    
    def test_initialization(self):
        """Test planner initialization."""
        ledger = EvidenceLedger()
        planner = CadencePlanner(ledger)
        
        assert planner.ledger is ledger
        assert planner.measurement_cost == 1.0
        assert planner.time_horizon_days == 30
    
    def test_plan_optimal_cadence_uncertainty_reduction(self):
        """Test optimal cadence planning for uncertainty reduction."""
        ledger = EvidenceLedger()
        
        # Add some evidence
        ledger.update_evidence(0.8, 0.95, 100, "test")
        
        planner = CadencePlanner(ledger)
        plan = planner.plan_optimal_cadence(
            n_measurements=5,
            objective='uncertainty_reduction'
        )
        
        assert isinstance(plan, CadencePlan)
        assert len(plan.measurement_times) == 5
        assert len(plan.expected_uncertainty) == 5
        assert len(plan.information_gain) == 5
        assert plan.optimization_objective == 'uncertainty_reduction'
        assert plan.total_cost == 5.0  # 5 measurements * 1.0 cost
    
    def test_plan_optimal_cadence_information_gain(self):
        """Test optimal cadence planning for information gain."""
        ledger = EvidenceLedger()
        
        # Add some evidence
        ledger.update_evidence(0.8, 0.95, 100, "test")
        
        planner = CadencePlanner(ledger)
        plan = planner.plan_optimal_cadence(
            n_measurements=3,
            objective='information_gain'
        )
        
        assert isinstance(plan, CadencePlan)
        assert len(plan.measurement_times) == 3
        assert plan.optimization_objective == 'information_gain'
    
    def test_plan_optimal_cadence_cost_effective(self):
        """Test optimal cadence planning for cost effectiveness."""
        ledger = EvidenceLedger()
        
        # Add some evidence
        ledger.update_evidence(0.8, 0.95, 100, "test")
        
        planner = CadencePlanner(ledger)
        plan = planner.plan_optimal_cadence(
            n_measurements=10,
            objective='cost_effective'
        )
        
        assert isinstance(plan, CadencePlan)
        assert plan.optimization_objective == 'cost_effective'
    
    def test_adaptive_cadence_plan(self):
        """Test adaptive cadence planning."""
        ledger = EvidenceLedger()
        
        # Add some evidence
        ledger.update_evidence(0.8, 0.95, 100, "test")
        
        planner = CadencePlanner(ledger)
        plan = planner.adaptive_cadence_plan(
            target_uncertainty=0.01,
            max_measurements=10
        )
        
        assert isinstance(plan, CadencePlan)
        assert plan.optimization_objective == 'adaptive_uncertainty_target'
        assert len(plan.measurement_times) <= 10
    
    def test_cadence_plan_validation(self):
        """Test cadence plan validation."""
        ledger = EvidenceLedger()
        planner = CadencePlanner(ledger)
        
        # Test with invalid objective
        with pytest.raises(ValueError):
            planner.plan_optimal_cadence(
                n_measurements=5,
                objective='invalid_objective'
            )
        
        # Test with zero measurements
        with pytest.raises(ValueError):
            planner.plan_optimal_cadence(
                n_measurements=0,
                objective='uncertainty_reduction'
            )


class TestEvidenceEntry:
    """Test cases for EvidenceEntry."""
    
    def test_evidence_entry_creation(self):
        """Test EvidenceEntry creation."""
        entry = EvidenceEntry(
            timestamp=datetime.now(),
            fidelity=0.8,
            confidence=0.95,
            shots=1000,
            success_count=800,
            experiment_id="test_1"
        )
        
        assert entry.fidelity == 0.8
        assert entry.confidence == 0.95
        assert entry.shots == 1000
        assert entry.success_count == 800
        assert entry.experiment_id == "test_1"
    
    def test_evidence_entry_validation(self):
        """Test EvidenceEntry validation."""
        # Test valid entry
        entry = EvidenceEntry(
            timestamp=datetime.now(),
            fidelity=0.8,
            confidence=0.95,
            shots=1000,
            success_count=800,
            experiment_id="test_1"
        )
        assert entry.fidelity == 0.8
        
        # Test invalid fidelity
        with pytest.raises(ValueError):
            EvidenceEntry(
                timestamp=datetime.now(),
                fidelity=1.5,  # Invalid: > 1.0
                confidence=0.95,
                shots=1000,
                success_count=800,
                experiment_id="test_1"
            )
        
        # Test invalid confidence
        with pytest.raises(ValueError):
            EvidenceEntry(
                timestamp=datetime.now(),
                fidelity=0.8,
                confidence=1.5,  # Invalid: > 1.0
                shots=1000,
                success_count=800,
                experiment_id="test_1"
            )


# Property-based tests
class TestGovernanceProperties:
    """Property-based tests for governance system."""
    
    def test_ledger_monotonicity(self):
        """Test that adding evidence monotonically changes statistics."""
        ledger = EvidenceLedger()
        
        initial_mean = ledger.get_mean()
        initial_std = ledger.get_std()
        
        # Add evidence
        ledger.update_evidence(0.8, 0.95, 100, "test")
        
        new_mean = ledger.get_mean()
        new_std = ledger.get_std()
        
        # Mean should change (not necessarily monotonically, but should be different)
        assert new_mean != initial_mean
        
        # Std should generally decrease with more evidence
        assert new_std <= initial_std
    
    def test_forecast_consistency(self):
        """Test that forecasts are consistent across multiple runs."""
        ledger = EvidenceLedger()
        
        # Add evidence
        ledger.update_evidence(0.8, 0.95, 100, "test")
        
        forecaster = MonteCarloForecaster(ledger, random_seed=42)
        
        # Run forecast multiple times with same seed
        forecast1 = forecaster.forecast_fidelity(horizon=5, n_samples=1000)
        forecast2 = forecaster.forecast_fidelity(horizon=5, n_samples=1000)
        
        # Should be identical with same seed
        assert np.allclose(forecast1.mean_forecast, forecast2.mean_forecast)
        assert np.allclose(forecast1.std_forecast, forecast2.std_forecast)
    
    def test_cadence_plan_feasibility(self):
        """Test that cadence plans are feasible."""
        ledger = EvidenceLedger()
        planner = CadencePlanner(ledger)
        
        plan = planner.plan_optimal_cadence(n_measurements=5)
        
        # Check that measurement times are ordered
        for i in range(len(plan.measurement_times) - 1):
            assert plan.measurement_times[i] <= plan.measurement_times[i + 1]
        
        # Check that uncertainties are reasonable
        for uncertainty in plan.expected_uncertainty:
            assert 0.0 <= uncertainty <= 1.0
        
        # Check that information gains are non-negative
        for gain in plan.information_gain:
            assert gain >= 0.0
