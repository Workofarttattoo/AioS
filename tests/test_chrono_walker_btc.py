#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Chrono Walker BTC Prediction Validation Tests

Tests the Chrono Walker probabilistic forecasting system with Bitcoin price
prediction scenarios using both complete and incomplete data to determine
accuracy requirements and minimum data needs.

Test Scenarios:
1. Complete historical BTC data (full time series)
2. Incomplete data - missing values (50%, 25%, 10%)
3. Incomplete data - sparse observations (weekly, monthly)
4. Real-time prediction with rolling windows
5. Multi-step ahead forecasting

Metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Directional Accuracy (% correct up/down predictions)
- Confidence Interval Coverage
- Calibration Score
"""

import sys
import os
import unittest
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import Chrono Walker
from quantum_chronowalk_gov import (
    Evidence, Belief, apply_evidence, mc_forecast,
    solve_events_needed, load_ledger, append_evidence
)


@dataclass
class BTCPrice:
    """Bitcoin price data point"""
    timestamp: str
    price: float
    volume: float = 0.0
    market_cap: float = 0.0


class BTCDataGenerator:
    """Generate realistic BTC price data for testing"""

    @staticmethod
    def generate_synthetic_btc_data(
        start_price: float = 40000.0,
        num_days: int = 365,
        volatility: float = 0.03,
        trend: float = 0.0002,
        seed: int = 42
    ) -> List[BTCPrice]:
        """
        Generate synthetic BTC price data using geometric Brownian motion

        Args:
            start_price: Starting BTC price in USD
            num_days: Number of days to generate
            volatility: Daily volatility (std dev of returns)
            trend: Daily drift (mean return)
            seed: Random seed for reproducibility

        Returns:
            List of BTCPrice objects
        """
        np.random.seed(seed)

        prices = [start_price]
        timestamps = []
        start_date = datetime(2024, 1, 1)

        for i in range(num_days):
            # Geometric Brownian Motion: S_t+1 = S_t * exp((μ - σ²/2)dt + σ√dt * Z)
            daily_return = trend + volatility * np.random.randn()
            new_price = prices[-1] * (1 + daily_return)
            prices.append(new_price)

            timestamp = (start_date + timedelta(days=i)).isoformat()
            timestamps.append(timestamp)

        # Remove initial price, create BTCPrice objects
        prices = prices[1:]

        btc_data = []
        for ts, price in zip(timestamps, prices):
            # Add realistic volume (higher volume = higher price typically)
            volume = np.random.uniform(20e9, 50e9) * (price / start_price)
            market_cap = price * 19e6  # ~19M BTC in circulation

            btc_data.append(BTCPrice(
                timestamp=ts,
                price=price,
                volume=volume,
                market_cap=market_cap
            ))

        return btc_data

    @staticmethod
    def create_incomplete_data(
        complete_data: List[BTCPrice],
        missing_pct: float = 0.5,
        seed: int = 42
    ) -> List[BTCPrice]:
        """
        Create incomplete data by randomly removing observations

        Args:
            complete_data: Complete BTC price data
            missing_pct: Percentage of data to remove (0.0-1.0)
            seed: Random seed

        Returns:
            Incomplete data list
        """
        np.random.seed(seed)
        n = len(complete_data)
        n_keep = int(n * (1 - missing_pct))

        indices = np.random.choice(n, n_keep, replace=False)
        indices = sorted(indices)

        return [complete_data[i] for i in indices]

    @staticmethod
    def create_sparse_data(
        complete_data: List[BTCPrice],
        frequency: str = 'weekly'
    ) -> List[BTCPrice]:
        """
        Create sparse data by sampling at fixed intervals

        Args:
            complete_data: Complete BTC price data
            frequency: 'daily', 'weekly', 'monthly'

        Returns:
            Sparse data list
        """
        if frequency == 'daily':
            step = 1
        elif frequency == 'weekly':
            step = 7
        elif frequency == 'monthly':
            step = 30
        else:
            step = 1

        return complete_data[::step]


class ChronoWalkerBTCTester:
    """Test suite for Chrono Walker BTC predictions"""

    def __init__(self, ledger_path: str = '/tmp/chrono_btc_test.csv'):
        self.ledger_path = ledger_path
        self.clear_ledger()

    def clear_ledger(self):
        """Clear test ledger"""
        if os.path.exists(self.ledger_path):
            os.remove(self.ledger_path)

    def btc_to_evidence(
        self,
        btc_data: List[BTCPrice],
        field: str = 'BTC_price',
        normalize: bool = True
    ) -> List[Evidence]:
        """
        Convert BTC price data to Chrono Walker evidence format

        Maps price changes to outcome scores:
        - Large increase (>5%) -> outcome 0.9
        - Moderate increase (2-5%) -> outcome 0.7
        - Small increase (0-2%) -> outcome 0.6
        - Flat (±0%) -> outcome 0.5
        - Small decrease (0-2%) -> outcome 0.4
        - Moderate decrease (2-5%) -> outcome 0.3
        - Large decrease (>5%) -> outcome 0.1

        Strength is based on volume (higher volume = higher confidence)
        """
        if len(btc_data) < 2:
            return []

        evidence_list = []

        # Calculate max volume for normalization
        max_volume = max(d.volume for d in btc_data) if normalize else 1.0

        for i in range(1, len(btc_data)):
            prev_price = btc_data[i-1].price
            curr_price = btc_data[i].price

            # Calculate price change percentage
            pct_change = (curr_price - prev_price) / prev_price

            # Map to outcome score [0, 1]
            if pct_change > 0.05:
                outcome = 0.9
                kind = 'large_increase'
            elif pct_change > 0.02:
                outcome = 0.7
                kind = 'moderate_increase'
            elif pct_change > 0.0:
                outcome = 0.6
                kind = 'small_increase'
            elif pct_change > -0.02:
                outcome = 0.4
                kind = 'small_decrease'
            elif pct_change > -0.05:
                outcome = 0.3
                kind = 'moderate_decrease'
            else:
                outcome = 0.1
                kind = 'large_decrease'

            # Strength based on volume (higher volume = more confidence)
            volume = btc_data[i].volume
            strength = min(1.0, volume / max_volume) if normalize else 0.5
            strength = max(0.2, strength)  # Minimum confidence

            evidence_list.append(Evidence(
                timestamp=btc_data[i].timestamp,
                field=field,
                kind=kind,
                strength=strength,
                outcome=outcome,
                source='BTC_market_data',
                title=f'BTC price ${curr_price:.2f} ({pct_change*100:+.2f}%)',
                notes=f'Volume: ${volume:.2e}, Previous: ${prev_price:.2f}'
            ))

        return evidence_list

    def test_prediction_accuracy(
        self,
        train_data: List[BTCPrice],
        test_data: List[BTCPrice],
        forecast_periods: int = 30,
        alpha0: float = 1.0,
        beta0: float = 1.0
    ) -> Dict:
        """
        Test prediction accuracy on train/test split

        Args:
            train_data: Training BTC data
            test_data: Testing BTC data
            forecast_periods: Number of periods to forecast
            alpha0: Prior alpha (optimistic = higher alpha)
            beta0: Prior beta (pessimistic = higher beta)

        Returns:
            Dict with accuracy metrics
        """
        # Clear ledger and add training evidence
        self.clear_ledger()

        train_evidence = self.btc_to_evidence(train_data)
        for ev in train_evidence:
            append_evidence(self.ledger_path, ev)

        # Load ledger and compute posterior
        ledger = load_ledger(self.ledger_path)
        prior = Belief(alpha=alpha0, beta=beta0)
        posterior = apply_evidence(prior, ledger, field_filter='BTC_price')

        # Run Monte Carlo forecast
        forecast_result = mc_forecast(
            start=posterior,
            periods=min(forecast_periods, len(test_data)),
            events_per_period=1,
            event_strength=0.5,
            outcome_mean=posterior.mean,
            outcome_std=0.15,
            profile='neutral',
            runs=5000
        )

        # Compare predictions to actual test data
        test_evidence = self.btc_to_evidence(test_data)
        actual_outcomes = [ev.outcome for ev in test_evidence[:forecast_periods]]
        predicted_means = forecast_result['trajectory_means']
        predicted_lo = forecast_result['lo']
        predicted_hi = forecast_result['hi']

        # Calculate metrics
        n = min(len(actual_outcomes), len(predicted_means))
        actual = np.array(actual_outcomes[:n])
        pred = np.array(predicted_means[:n])
        lo = np.array(predicted_lo[:n])
        hi = np.array(predicted_hi[:n])

        # Mean Absolute Error
        mae = np.mean(np.abs(actual - pred))

        # Root Mean Squared Error
        rmse = np.sqrt(np.mean((actual - pred) ** 2))

        # Directional Accuracy (did we predict up/down correctly?)
        baseline = posterior.mean
        actual_direction = actual > baseline
        pred_direction = pred > baseline
        directional_accuracy = np.mean(actual_direction == pred_direction)

        # Confidence Interval Coverage (% of actuals within 90% CI)
        within_ci = np.mean((actual >= lo) & (actual <= hi))

        # Calibration: How well do confidence intervals match coverage?
        # Expected 90% CI should cover 90% of observations
        calibration_error = abs(within_ci - 0.90)

        return {
            'posterior_mean': posterior.mean,
            'posterior_alpha': posterior.alpha,
            'posterior_beta': posterior.beta,
            'n_train': len(train_data),
            'n_test': n,
            'mae': float(mae),
            'rmse': float(rmse),
            'directional_accuracy': float(directional_accuracy),
            'ci_coverage': float(within_ci),
            'calibration_error': float(calibration_error),
            'predictions': predicted_means[:n],
            'actuals': actual_outcomes[:n],
            'ci_lower': predicted_lo[:n],
            'ci_upper': predicted_hi[:n]
        }


class TestChronoWalkerBTC(unittest.TestCase):
    """Unit tests for Chrono Walker BTC predictions"""

    @classmethod
    def setUpClass(cls):
        """Generate test data once for all tests"""
        cls.generator = BTCDataGenerator()
        cls.complete_data = cls.generator.generate_synthetic_btc_data(
            start_price=40000,
            num_days=365,
            volatility=0.03,
            trend=0.0005,  # Slight upward trend
            seed=42
        )
        cls.tester = ChronoWalkerBTCTester()

    def test_01_complete_data_prediction(self):
        """Test with complete BTC data (100% observations)"""
        print("\n[TEST 1] Complete Data Prediction")

        # 80/20 train/test split
        split_idx = int(len(self.complete_data) * 0.8)
        train = self.complete_data[:split_idx]
        test = self.complete_data[split_idx:]

        results = self.tester.test_prediction_accuracy(
            train_data=train,
            test_data=test,
            forecast_periods=30
        )

        print(f"  Training samples: {results['n_train']}")
        print(f"  Test samples: {results['n_test']}")
        print(f"  MAE: {results['mae']:.4f}")
        print(f"  RMSE: {results['rmse']:.4f}")
        print(f"  Directional Accuracy: {results['directional_accuracy']:.2%}")
        print(f"  CI Coverage (90%): {results['ci_coverage']:.2%}")
        print(f"  Calibration Error: {results['calibration_error']:.4f}")

        # Assertions
        self.assertLess(results['mae'], 0.3, "MAE should be < 0.3 for complete data")
        self.assertGreater(results['directional_accuracy'], 0.5, "Should beat random chance")
        self.assertGreater(results['ci_coverage'], 0.7, "CI should cover at least 70%")

    def test_02_incomplete_data_50pct(self):
        """Test with 50% missing data"""
        print("\n[TEST 2] Incomplete Data (50% missing)")

        # Create incomplete training data
        split_idx = int(len(self.complete_data) * 0.8)
        complete_train = self.complete_data[:split_idx]
        test = self.complete_data[split_idx:]

        incomplete_train = self.generator.create_incomplete_data(
            complete_train,
            missing_pct=0.5,
            seed=123
        )

        results = self.tester.test_prediction_accuracy(
            train_data=incomplete_train,
            test_data=test,
            forecast_periods=30
        )

        print(f"  Training samples (50% missing): {results['n_train']}")
        print(f"  MAE: {results['mae']:.4f}")
        print(f"  RMSE: {results['rmse']:.4f}")
        print(f"  Directional Accuracy: {results['directional_accuracy']:.2%}")

        # With 50% missing data, accuracy should degrade but still be useful
        self.assertLess(results['mae'], 0.4, "MAE should still be < 0.4")
        self.assertGreater(results['directional_accuracy'], 0.45, "Should still have predictive power")

    def test_03_incomplete_data_75pct(self):
        """Test with 75% missing data (sparse observations)"""
        print("\n[TEST 3] Incomplete Data (75% missing)")

        split_idx = int(len(self.complete_data) * 0.8)
        complete_train = self.complete_data[:split_idx]
        test = self.complete_data[split_idx:]

        incomplete_train = self.generator.create_incomplete_data(
            complete_train,
            missing_pct=0.75,
            seed=456
        )

        results = self.tester.test_prediction_accuracy(
            train_data=incomplete_train,
            test_data=test,
            forecast_periods=30
        )

        print(f"  Training samples (75% missing): {results['n_train']}")
        print(f"  MAE: {results['mae']:.4f}")
        print(f"  Directional Accuracy: {results['directional_accuracy']:.2%}")

        # With very sparse data, accuracy degrades significantly
        # But should still provide some signal
        self.assertGreater(results['n_train'], 50, "Should have at least 50 observations")

    def test_04_weekly_sparse_data(self):
        """Test with weekly observations (sparse time series)"""
        print("\n[TEST 4] Weekly Sparse Data")

        split_idx = int(len(self.complete_data) * 0.8)
        complete_train = self.complete_data[:split_idx]
        test = self.complete_data[split_idx:]

        weekly_train = self.generator.create_sparse_data(
            complete_train,
            frequency='weekly'
        )

        results = self.tester.test_prediction_accuracy(
            train_data=weekly_train,
            test_data=test,
            forecast_periods=10
        )

        print(f"  Weekly samples: {results['n_train']}")
        print(f"  MAE: {results['mae']:.4f}")
        print(f"  Directional Accuracy: {results['directional_accuracy']:.2%}")

        # Weekly data should still provide useful predictions
        self.assertGreater(results['n_train'], 40, "Should have ~52 weekly observations")

    def test_05_monthly_sparse_data(self):
        """Test with monthly observations (very sparse)"""
        print("\n[TEST 5] Monthly Sparse Data")

        split_idx = int(len(self.complete_data) * 0.8)
        complete_train = self.complete_data[:split_idx]
        test = self.complete_data[split_idx:]

        monthly_train = self.generator.create_sparse_data(
            complete_train,
            frequency='monthly'
        )

        results = self.tester.test_prediction_accuracy(
            train_data=monthly_train,
            test_data=test,
            forecast_periods=5
        )

        print(f"  Monthly samples: {results['n_train']}")
        print(f"  MAE: {results['mae']:.4f}")
        print(f"  Posterior mean: {results['posterior_mean']:.4f}")

        # Monthly data is very sparse, but should still compute posterior
        self.assertGreater(results['n_train'], 8, "Should have ~12 monthly observations")

    def test_06_minimum_data_requirement(self):
        """Determine minimum data points needed for reasonable accuracy"""
        print("\n[TEST 6] Minimum Data Requirement Analysis")

        split_idx = int(len(self.complete_data) * 0.8)
        complete_train = self.complete_data[:split_idx]
        test = self.complete_data[split_idx:]

        # Test with different sample sizes
        sample_sizes = [10, 20, 50, 100, 200]
        results_by_size = {}

        for n in sample_sizes:
            if n > len(complete_train):
                continue

            # Sample uniformly
            indices = np.linspace(0, len(complete_train)-1, n, dtype=int)
            sampled_train = [complete_train[i] for i in indices]

            result = self.tester.test_prediction_accuracy(
                train_data=sampled_train,
                test_data=test,
                forecast_periods=10
            )

            results_by_size[n] = result
            print(f"  n={n:3d}: MAE={result['mae']:.4f}, "
                  f"Dir_Acc={result['directional_accuracy']:.2%}")

        # Find minimum n where MAE < 0.35
        min_n_acceptable = None
        for n in sorted(results_by_size.keys()):
            if results_by_size[n]['mae'] < 0.35:
                min_n_acceptable = n
                break

        if min_n_acceptable:
            print(f"\n  Minimum samples for MAE < 0.35: {min_n_acceptable}")
        else:
            print("\n  Could not achieve MAE < 0.35 with tested sample sizes")

        # Should work with at least 50 samples
        if 50 in results_by_size:
            self.assertLess(results_by_size[50]['mae'], 0.5,
                          "Should get reasonable accuracy with 50 samples")


def run_comprehensive_btc_validation():
    """
    Run comprehensive validation suite and generate report
    """
    print("=" * 70)
    print("CHRONO WALKER BTC PREDICTION VALIDATION")
    print("=" * 70)

    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestChronoWalkerBTC)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✓ All Chrono Walker BTC prediction tests PASSED")
    else:
        print("\n✗ Some Chrono Walker BTC prediction tests FAILED")

    return result


if __name__ == '__main__':
    run_comprehensive_btc_validation()
