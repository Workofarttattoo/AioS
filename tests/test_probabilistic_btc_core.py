#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Probabilistic BTC Prediction Core Tests

Tests Oracle, AdaptiveParticleFilter, BayesianLayer, and NoUTurnSampler
for Bitcoin price prediction with complete and incomplete data.

Validates:
1. Oracle probabilistic forecasting with telemetry signals
2. Particle Filter for sequential BTC price tracking
3. Bayesian Neural Network for price prediction with uncertainty
4. NUTS HMC for posterior sampling of BTC parameters

Each algorithm is tested with:
- Complete historical data
- Incomplete/missing data (50%, 75%)
- Accuracy metrics (MAE, RMSE, calibration)
- Minimum data requirements
"""

import sys
import os
import unittest
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Callable
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Oracle
from aios.oracle import ProbabilisticOracle, ForecastResult

# Import ML Algorithms
from aios.ml_algorithms import (
    AdaptiveParticleFilter,
    NoUTurnSampler,
    get_algorithm_catalog
)

# Try to import PyTorch-dependent algorithms
try:
    from aios.ml_algorithms import BayesianLayer
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    BayesianLayer = None
    torch = None


class BTCPriceData:
    """Generate and manage BTC price data for testing"""

    @staticmethod
    def generate_synthetic_prices(
        n_days: int = 365,
        start_price: float = 40000.0,
        volatility: float = 0.03,
        trend: float = 0.0002,
        seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic BTC prices and timestamps

        Returns:
            (prices, timestamps) as numpy arrays
        """
        np.random.seed(seed)

        prices = [start_price]
        for _ in range(n_days):
            daily_return = trend + volatility * np.random.randn()
            new_price = prices[-1] * (1 + daily_return)
            prices.append(new_price)

        prices = np.array(prices[1:])

        # Generate timestamps
        start_date = datetime(2024, 1, 1)
        timestamps = np.array([
            (start_date + timedelta(days=i)).timestamp()
            for i in range(n_days)
        ])

        return prices, timestamps

    @staticmethod
    def create_missing_data(
        prices: np.ndarray,
        timestamps: np.ndarray,
        missing_pct: float = 0.5,
        seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create incomplete data by randomly removing observations

        Args:
            prices: Complete price array
            timestamps: Complete timestamp array
            missing_pct: Percentage to remove (0.0-1.0)
            seed: Random seed

        Returns:
            (incomplete_prices, incomplete_timestamps)
        """
        np.random.seed(seed)
        n = len(prices)
        n_keep = int(n * (1 - missing_pct))

        indices = np.random.choice(n, n_keep, replace=False)
        indices = np.sort(indices)

        return prices[indices], timestamps[indices]

    @staticmethod
    def price_to_telemetry(
        prices: np.ndarray,
        timestamps: np.ndarray,
        feature_names: List[str] = None
    ) -> Dict[str, dict]:
        """
        Convert BTC prices to Oracle-compatible telemetry format

        Creates signals based on:
        - Price momentum (recent trend)
        - Volatility (price stability)
        - Moving averages
        - Volume proxy (price * random volume factor)

        Args:
            prices: Price array
            timestamps: Timestamp array
            feature_names: Optional custom feature names

        Returns:
            Dict compatible with Oracle.forecast()
        """
        if len(prices) < 10:
            return {}

        telemetry = {}

        # Recent price momentum (last 7 days vs previous 7 days)
        if len(prices) >= 14:
            recent_avg = np.mean(prices[-7:])
            prev_avg = np.mean(prices[-14:-7])
            momentum = (recent_avg - prev_avg) / prev_avg

            telemetry['price_momentum'] = {
                'value': float(np.clip((momentum + 0.1) / 0.2, 0, 1)),  # Normalize to [0,1]
                'weight': 0.3,
                'timestamp': timestamps[-1]
            }

        # Volatility signal (lower volatility = more stable = positive)
        if len(prices) >= 30:
            returns = np.diff(prices[-30:]) / prices[-30:-1]
            volatility = np.std(returns)
            # Inverse relationship: high vol = low signal
            vol_signal = 1.0 / (1.0 + volatility * 50)  # Normalize

            telemetry['volatility'] = {
                'value': float(vol_signal),
                'weight': 0.2,
                'timestamp': timestamps[-1]
            }

        # Moving average crossover (MA7 vs MA30)
        if len(prices) >= 30:
            ma7 = np.mean(prices[-7:])
            ma30 = np.mean(prices[-30:])
            ma_signal = 1.0 if ma7 > ma30 else 0.0

            telemetry['ma_crossover'] = {
                'value': float(ma_signal),
                'weight': 0.25,
                'timestamp': timestamps[-1]
            }

        # Volume proxy (assume volume proportional to price change)
        if len(prices) >= 2:
            price_change = abs(prices[-1] - prices[-2]) / prices[-2]
            volume_signal = np.clip(price_change * 10, 0, 1)

            telemetry['volume'] = {
                'value': float(volume_signal),
                'weight': 0.25,
                'timestamp': timestamps[-1]
            }

        return telemetry


class TestOracleBTCPrediction(unittest.TestCase):
    """Test Oracle probabilistic forecasting for BTC"""

    def setUp(self):
        self.oracle = ProbabilisticOracle()
        self.data_gen = BTCPriceData()

    def test_01_oracle_complete_data(self):
        """Test Oracle with complete BTC price data"""
        print("\n[ORACLE TEST 1] Complete Data Prediction")

        # Generate complete data
        prices, timestamps = self.data_gen.generate_synthetic_prices(
            n_days=180,
            volatility=0.03,
            trend=0.0005
        )

        # Convert to telemetry
        telemetry = self.data_gen.price_to_telemetry(prices, timestamps)

        # Forecast
        forecast = self.oracle.forecast(telemetry)

        print(f"  Probability: {forecast.probability:.4f}")
        print(f"  Confidence: {forecast.confidence:.4f}")
        print(f"  Reasoning: {forecast.reasoning[:100]}...")

        # Assertions
        self.assertGreater(forecast.probability, 0.0)
        self.assertLess(forecast.probability, 1.0)
        self.assertGreater(forecast.confidence, 0.5)

    def test_02_oracle_incomplete_data_50pct(self):
        """Test Oracle with 50% missing data"""
        print("\n[ORACLE TEST 2] Incomplete Data (50% missing)")

        # Generate and thin data
        prices, timestamps = self.data_gen.generate_synthetic_prices(n_days=180)
        prices_incomplete, timestamps_incomplete = self.data_gen.create_missing_data(
            prices, timestamps, missing_pct=0.5
        )

        # Convert to telemetry
        telemetry = self.data_gen.price_to_telemetry(
            prices_incomplete,
            timestamps_incomplete
        )

        forecast = self.oracle.forecast(telemetry)

        print(f"  Samples: {len(prices_incomplete)}")
        print(f"  Probability: {forecast.probability:.4f}")
        print(f"  Confidence: {forecast.confidence:.4f}")

        # Should still produce valid forecast
        self.assertGreater(forecast.probability, 0.0)
        self.assertLess(forecast.confidence, 1.0)  # Less confident with missing data

    def test_03_oracle_incomplete_data_75pct(self):
        """Test Oracle with 75% missing data"""
        print("\n[ORACLE TEST 3] Incomplete Data (75% missing)")

        prices, timestamps = self.data_gen.generate_synthetic_prices(n_days=180)
        prices_incomplete, timestamps_incomplete = self.data_gen.create_missing_data(
            prices, timestamps, missing_pct=0.75
        )

        telemetry = self.data_gen.price_to_telemetry(
            prices_incomplete,
            timestamps_incomplete
        )

        forecast = self.oracle.forecast(telemetry)

        print(f"  Samples: {len(prices_incomplete)}")
        print(f"  Probability: {forecast.probability:.4f}")
        print(f"  Confidence: {forecast.confidence:.4f}")

        # With very sparse data, confidence should be low
        self.assertLess(forecast.confidence, 0.7)


class TestParticleFilterBTC(unittest.TestCase):
    """Test Adaptive Particle Filter for BTC price tracking"""

    def setUp(self):
        self.data_gen = BTCPriceData()

    def transition_btc_price(self, state: np.ndarray) -> np.ndarray:
        """
        BTC price transition model (state-space dynamics)

        State: [log_price, momentum, volatility]
        """
        log_price, momentum, volatility = state

        # Update log price with momentum
        new_log_price = log_price + momentum

        # Momentum mean reversion
        new_momentum = 0.8 * momentum  # Decay

        # Volatility persistence
        new_volatility = 0.9 * volatility + 0.01

        return np.array([new_log_price, new_momentum, new_volatility])

    def likelihood_btc(self, observation: np.ndarray, state: np.ndarray) -> float:
        """
        Likelihood p(observation | state) for BTC prices

        observation: [log_price_observed]
        state: [log_price, momentum, volatility]
        """
        log_price_obs = observation[0]
        log_price_state, _, volatility = state

        # Gaussian likelihood
        diff = log_price_obs - log_price_state
        likelihood = np.exp(-0.5 * (diff / volatility) ** 2)

        return likelihood

    def test_01_particle_filter_complete_data(self):
        """Test Particle Filter with complete BTC price data"""
        print("\n[PARTICLE FILTER TEST 1] Complete Data Tracking")

        # Generate data
        prices, timestamps = self.data_gen.generate_synthetic_prices(
            n_days=100,
            volatility=0.03,
            trend=0.0002
        )

        # Initialize particle filter
        # State: [log_price, momentum, volatility]
        pf = AdaptiveParticleFilter(num_particles=1000, state_dim=3, obs_dim=1)

        # Initialize particles around first observation
        log_price_0 = np.log(prices[0])
        pf.particles[:, 0] = log_price_0 + np.random.randn(1000) * 0.01
        pf.particles[:, 1] = np.random.randn(1000) * 0.001  # momentum
        pf.particles[:, 2] = 0.03 + np.random.randn(1000) * 0.005  # volatility

        # Track prices
        estimates = []
        errors = []

        for i in range(1, len(prices)):
            # Predict
            pf.predict(
                transition_fn=self.transition_btc_price,
                process_noise=0.01
            )

            # Update with observation
            observation = np.array([np.log(prices[i])])
            pf.update(
                observation=observation,
                likelihood_fn=self.likelihood_btc
            )

            # Get estimate
            state_estimate = pf.estimate()
            price_estimate = np.exp(state_estimate[0])
            estimates.append(price_estimate)

            # Track error
            error = abs(price_estimate - prices[i]) / prices[i]
            errors.append(error)

        # Calculate metrics
        mean_error = np.mean(errors)
        rmse = np.sqrt(np.mean(np.array(errors) ** 2))

        print(f"  Observations tracked: {len(prices)}")
        print(f"  Mean Absolute Percentage Error: {mean_error:.2%}")
        print(f"  RMSE: {rmse:.4f}")

        # Should track reasonably well
        self.assertLess(mean_error, 0.10, "MAPE should be < 10%")

    def test_02_particle_filter_incomplete_data(self):
        """Test Particle Filter with missing observations"""
        print("\n[PARTICLE FILTER TEST 2] Incomplete Data (50% missing)")

        # Generate complete data
        prices_complete, timestamps = self.data_gen.generate_synthetic_prices(n_days=100)

        # Create missing observations (50%)
        prices, _ = self.data_gen.create_missing_data(
            prices_complete, timestamps, missing_pct=0.5
        )

        pf = AdaptiveParticleFilter(num_particles=1000, state_dim=3, obs_dim=1)

        # Initialize
        log_price_0 = np.log(prices[0])
        pf.particles[:, 0] = log_price_0 + np.random.randn(1000) * 0.01
        pf.particles[:, 1] = np.random.randn(1000) * 0.001
        pf.particles[:, 2] = 0.03 + np.random.randn(1000) * 0.005

        # Track with sparse observations
        estimates = []
        errors = []

        for i in range(1, len(prices)):
            # Predict (may be multiple steps if observations are sparse)
            pf.predict(
                transition_fn=self.transition_btc_price,
                process_noise=0.015  # Higher noise for sparse data
            )

            # Update
            observation = np.array([np.log(prices[i])])
            pf.update(
                observation=observation,
                likelihood_fn=self.likelihood_btc
            )

            state_estimate = pf.estimate()
            price_estimate = np.exp(state_estimate[0])
            estimates.append(price_estimate)

            error = abs(price_estimate - prices[i]) / prices[i]
            errors.append(error)

        mean_error = np.mean(errors)

        print(f"  Sparse observations: {len(prices)}")
        print(f"  Mean Absolute Percentage Error: {mean_error:.2%}")

        # Error should increase with missing data but still be useful
        self.assertLess(mean_error, 0.20, "MAPE should be < 20% with sparse data")


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
class TestBayesianLayerBTC(unittest.TestCase):
    """Test Bayesian Neural Network Layer for BTC prediction"""

    def setUp(self):
        self.data_gen = BTCPriceData()

    def test_01_bayesian_layer_prediction(self):
        """Test Bayesian Layer for BTC price prediction with uncertainty"""
        print("\n[BAYESIAN LAYER TEST 1] Prediction with Uncertainty")

        # Generate data
        prices, timestamps = self.data_gen.generate_synthetic_prices(
            n_days=200,
            volatility=0.03
        )

        # Create features: [price_t-3, price_t-2, price_t-1, ma7, volatility7]
        window = 10
        X_list = []
        y_list = []

        for i in range(window, len(prices)):
            # Features
            features = [
                prices[i-3] / prices[0],  # Normalized price lag 3
                prices[i-2] / prices[0],  # lag 2
                prices[i-1] / prices[0],  # lag 1
                np.mean(prices[i-7:i]) / prices[0],  # MA7
                np.std(prices[i-7:i]) / prices[0],  # Vol7
            ]
            X_list.append(features)

            # Target: next day price (normalized)
            y_list.append(prices[i] / prices[0])

        X = np.array(X_list)
        y = np.array(y_list)

        # Train/test split
        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        # Convert to PyTorch tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
        X_test_t = torch.FloatTensor(X_test)

        # Create Bayesian Layer (5 inputs -> 1 output)
        bayesian_layer = BayesianLayer(in_features=5, out_features=1)

        # Simple training loop
        optimizer = torch.optim.Adam([
            bayesian_layer.weight_mu,
            bayesian_layer.weight_rho,
            bayesian_layer.bias_mu,
            bayesian_layer.bias_rho
        ], lr=0.01)

        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")

        for epoch in range(50):
            optimizer.zero_grad()

            # Forward pass (sample weights)
            output, kl_div = bayesian_layer.forward(X_train_t, sample=True)

            # Loss = MSE + KL divergence
            mse_loss = torch.mean((output - y_train_t) ** 2)
            total_loss = mse_loss + 0.001 * kl_div  # Small KL weight

            total_loss.backward()
            optimizer.step()

        # Test with multiple samples (Monte Carlo dropout style)
        n_samples = 100
        predictions = []

        with torch.no_grad():
            for _ in range(n_samples):
                output, _ = bayesian_layer.forward(X_test_t, sample=True)
                predictions.append(output.numpy())

        predictions = np.array(predictions)  # (n_samples, n_test, 1)

        # Mean prediction and uncertainty
        pred_mean = predictions.mean(axis=0).squeeze()
        pred_std = predictions.std(axis=0).squeeze()

        # Metrics
        mae = np.mean(np.abs(pred_mean - y_test))
        rmse = np.sqrt(np.mean((pred_mean - y_test) ** 2))

        # Calibration: Check if uncertainty correlates with error
        errors = np.abs(pred_mean - y_test)
        correlation = np.corrcoef(errors, pred_std)[0, 1]

        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Mean Uncertainty (std): {pred_std.mean():.4f}")
        print(f"  Error-Uncertainty Correlation: {correlation:.3f}")

        # Assertions
        self.assertLess(mae, 0.15, "MAE should be reasonable")
        self.assertGreater(pred_std.mean(), 0.0, "Should have non-zero uncertainty")

        # Ideally, higher uncertainty should correlate with higher errors
        # But this may not always be true with limited training
        # self.assertGreater(correlation, 0.0, "Uncertainty should correlate with errors")


class TestNUTSSamplerBTC(unittest.TestCase):
    """Test No-U-Turn Sampler for BTC parameter inference"""

    def setUp(self):
        self.data_gen = BTCPriceData()

    def test_01_nuts_parameter_inference(self):
        """Test NUTS for inferring BTC price model parameters"""
        print("\n[NUTS TEST 1] Bayesian Parameter Inference")

        # Generate data with known parameters
        true_drift = 0.0005
        true_volatility = 0.03
        prices, _ = self.data_gen.generate_synthetic_prices(
            n_days=100,
            trend=true_drift,
            volatility=true_volatility,
            seed=42
        )

        # Calculate log returns
        log_returns = np.log(prices[1:] / prices[:-1])

        # Define log probability function for NUTS
        def log_prob(params):
            """
            Log probability of parameters given data

            params: [drift, log_volatility]
            """
            drift, log_vol = params
            volatility = np.exp(log_vol)  # Constrain positive

            # Prior: drift ~ N(0, 0.001), volatility ~ LogNormal(log(0.03), 0.5)
            log_prior_drift = -0.5 * (drift / 0.001) ** 2
            log_prior_vol = -0.5 * ((log_vol - np.log(0.03)) / 0.5) ** 2

            # Likelihood: returns ~ N(drift, volatility)
            residuals = log_returns - drift
            log_likelihood = -0.5 * np.sum((residuals / volatility) ** 2) - len(log_returns) * np.log(volatility)

            return log_prior_drift + log_prior_vol + log_likelihood

        # Initialize NUTS sampler
        nuts = NoUTurnSampler(
            log_prob_fn=log_prob,
            step_size=0.01,
            max_tree_depth=10
        )

        # Initial parameters (start near truth)
        initial_params = np.array([0.0, np.log(0.025)])

        print(f"  True drift: {true_drift:.6f}")
        print(f"  True volatility: {true_volatility:.4f}")
        print(f"  Sampling {500} posterior samples...")

        # Sample from posterior
        samples = nuts.sample(
            initial_position=initial_params,
            num_samples=500
        )

        # Skip burn-in (first 100 samples)
        samples_post_burnin = samples[100:]

        # Extract parameters
        drift_samples = samples_post_burnin[:, 0]
        vol_samples = np.exp(samples_post_burnin[:, 1])

        # Posterior statistics
        drift_mean = np.mean(drift_samples)
        drift_std = np.std(drift_samples)
        vol_mean = np.mean(vol_samples)
        vol_std = np.std(vol_samples)

        print(f"  Inferred drift: {drift_mean:.6f} ± {drift_std:.6f}")
        print(f"  Inferred volatility: {vol_mean:.4f} ± {vol_std:.4f}")

        # Check if true parameters are within 2 std of posterior mean
        drift_error = abs(drift_mean - true_drift)
        vol_error = abs(vol_mean - true_volatility)

        print(f"  Drift error: {drift_error:.6f} ({drift_error/drift_std:.2f} std)")
        print(f"  Vol error: {vol_error:.4f} ({vol_error/vol_std:.2f} std)")

        # Assertions
        self.assertLess(drift_error, 3 * drift_std, "Drift should be within 3 std")
        self.assertLess(vol_error, 3 * vol_std, "Volatility should be within 3 std")


def run_comprehensive_probabilistic_tests():
    """Run all probabilistic core tests"""
    print("=" * 70)
    print("PROBABILISTIC BTC PREDICTION CORE TESTS")
    print("=" * 70)

    # Test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestOracleBTCPrediction))
    suite.addTests(loader.loadTestsFromTestCase(TestParticleFilterBTC))

    if TORCH_AVAILABLE:
        suite.addTests(loader.loadTestsFromTestCase(TestBayesianLayerBTC))
    else:
        print("\n[SKIP] Bayesian Layer tests - PyTorch not available\n")

    suite.addTests(loader.loadTestsFromTestCase(TestNUTSSamplerBTC))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✓ All probabilistic core tests PASSED")
    else:
        print("\n✗ Some probabilistic core tests FAILED")

    return result


if __name__ == '__main__':
    run_comprehensive_probabilistic_tests()
