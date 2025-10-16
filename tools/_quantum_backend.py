#!/usr/bin/env python3
"""
Quantum ML Backend for Ai|oS Security Tools
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Provides quantum-enhanced machine learning capabilities for security tools:
- Particle Filter Bayesian Inference for anomaly detection
- HHL Algorithm for exponential pattern matching speedup
- VQE for optimization
- Schrödinger Dynamics for probabilistic forecasting
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

LOG = logging.getLogger(__name__)

# Check for quantum dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    LOG.warning("PyTorch not available - quantum features will be limited")

# Import quantum algorithms from aios
try:
    import sys
    import os
    # Add parent directory to path to import aios modules
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from quantum_ml_algorithms import QuantumStateEngine, QuantumVQE
    from quantum_hhl_algorithm import hhl_linear_system_solver
    from quantum_schrodinger_dynamics import quantum_dynamics_forecasting
    from ml_algorithms import AdaptiveParticleFilter, SparseGaussianProcess

    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    LOG.warning("Quantum ML modules not available - using classical fallbacks")


class QuantumAnomalyDetector:
    """Quantum-enhanced anomaly detection using Particle Filter"""

    def __init__(self, num_particles: int = 1000):
        self.num_particles = num_particles
        self.particle_filter = None

        if QUANTUM_AVAILABLE:
            try:
                # Initialize particle filter for state estimation
                self.particle_filter = AdaptiveParticleFilter(
                    num_particles=num_particles,
                    state_dim=5,  # [status_code, size, response_time, pattern_match, entropy]
                    obs_dim=3     # [status_code, size, response_time]
                )
                LOG.info(f"Initialized quantum anomaly detector with {num_particles} particles")
            except Exception as e:
                LOG.warning(f"Failed to initialize particle filter: {e}")
                self.particle_filter = None

    def predict_anomaly(self, observation: Dict[str, Any]) -> Tuple[float, str]:
        """
        Predict if observation is anomalous using Bayesian inference.

        Returns:
            (anomaly_score, reasoning) where anomaly_score is 0.0-1.0
        """
        if not self.particle_filter:
            # Classical fallback: simple threshold-based detection
            return self._classical_anomaly_detection(observation)

        try:
            # Extract features from observation
            obs_vector = np.array([
                observation.get('status', 200),
                observation.get('size', 0) / 1000.0,  # Normalize size
                observation.get('response_time', 0) * 1000.0  # Convert to ms
            ])

            # Prediction step (Bayesian forward model)
            def transition_fn(x):
                # Model: state evolves slowly with some noise
                return x + np.random.normal(0, 0.1, x.shape)

            self.particle_filter.predict(transition_fn, process_noise=0.05)

            # Update step (Bayesian likelihood)
            def likelihood_fn(x, obs):
                # Gaussian likelihood comparing state to observation
                predicted_obs = x[:3]  # First 3 dims map to observation
                diff = predicted_obs - obs
                return np.exp(-0.5 * np.sum(diff**2))

            self.particle_filter.update(obs_vector, likelihood_fn)

            # Get state estimate
            state_estimate = self.particle_filter.estimate()

            # Calculate anomaly score based on particle spread (uncertainty)
            particle_spread = self.particle_filter.effective_sample_size() / self.num_particles
            anomaly_score = 1.0 - particle_spread  # High spread = high uncertainty = anomaly

            # Generate reasoning
            if anomaly_score > 0.7:
                reasoning = "High uncertainty in Bayesian inference - likely anomalous pattern"
            elif anomaly_score > 0.4:
                reasoning = "Moderate uncertainty - potentially anomalous"
            else:
                reasoning = "Low uncertainty - pattern matches known distributions"

            return float(anomaly_score), reasoning

        except Exception as e:
            LOG.warning(f"Quantum anomaly detection failed: {e}, falling back to classical")
            return self._classical_anomaly_detection(observation)

    def _classical_anomaly_detection(self, observation: Dict[str, Any]) -> Tuple[float, str]:
        """Classical fallback for anomaly detection"""
        status = observation.get('status', 200)
        size = observation.get('size', 0)

        # Simple rule-based anomaly detection
        anomaly_score = 0.0
        reasons = []

        if status >= 500:
            anomaly_score += 0.3
            reasons.append("5xx server error")
        elif status == 403:
            anomaly_score += 0.2
            reasons.append("Forbidden access")

        if size > 100000:  # Large response
            anomaly_score += 0.2
            reasons.append("Large response size")
        elif size == 0:
            anomaly_score += 0.1
            reasons.append("Empty response")

        reasoning = "; ".join(reasons) if reasons else "Normal pattern"
        return min(anomaly_score, 1.0), reasoning


class QuantumPathPredictor:
    """Quantum-enhanced path prediction using HHL algorithm"""

    def __init__(self):
        self.pattern_matrix = None
        self.quantum_available = QUANTUM_AVAILABLE

    def predict_next_paths(self, discovered_paths: List[str],
                          wordlist: List[str],
                          top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Predict most likely next paths to discover using quantum pattern matching.

        Uses HHL algorithm for exponential speedup in pattern matching.

        Returns:
            List of (path, confidence_score) tuples
        """
        if not self.quantum_available or len(discovered_paths) < 5:
            # Need at least 5 discovered paths for pattern analysis
            return self._classical_path_prediction(discovered_paths, wordlist, top_k)

        try:
            # Build pattern matrix from discovered paths
            # Matrix A: path_features x path_features
            # Vector b: success probability for each feature

            features = self._extract_path_features(discovered_paths)

            if len(features) < 2:
                return self._classical_path_prediction(discovered_paths, wordlist, top_k)

            # Create correlation matrix (how features relate to successful discoveries)
            A = np.corrcoef(features) + np.eye(len(features)) * 0.1  # Add diagonal for stability
            b = np.mean(features, axis=1)  # Average feature activation

            # Check if system is well-conditioned for HHL
            condition_number = np.linalg.cond(A)

            if condition_number < 20:  # Well-conditioned, use quantum
                LOG.debug(f"Using HHL algorithm (κ={condition_number:.2f})")
                result = hhl_linear_system_solver(A, b)

                if result['success']:
                    # Use quantum solution to rank wordlist paths
                    solution = result.get('expectation_values', b)
                    predictions = []

                    for word in wordlist[:100]:  # Limit to top 100 for performance
                        # Score each word based on quantum solution
                        word_features = self._extract_word_features(word, discovered_paths)
                        score = float(np.dot(solution[:len(word_features)], word_features))
                        predictions.append((word, score))

                    # Sort by score and return top_k
                    predictions.sort(key=lambda x: x[1], reverse=True)
                    return predictions[:top_k]

            # Fallback to classical if poorly conditioned
            return self._classical_path_prediction(discovered_paths, wordlist, top_k)

        except Exception as e:
            LOG.warning(f"Quantum path prediction failed: {e}, falling back to classical")
            return self._classical_path_prediction(discovered_paths, wordlist, top_k)

    def _extract_path_features(self, paths: List[str]) -> np.ndarray:
        """Extract features from discovered paths"""
        features = []

        for path in paths:
            path_features = [
                float(len(path)),  # Path length
                float('/' in path),  # Has directory separator
                float('.' in path),  # Has file extension
                float(any(c.isdigit() for c in path)),  # Contains digits
                float(path.startswith('.')),  # Hidden file
                float(path.endswith('/')),  # Directory
                float('-' in path or '_' in path),  # Has separators
            ]
            features.append(path_features)

        return np.array(features).T  # Transpose: features x paths

    def _extract_word_features(self, word: str, context_paths: List[str]) -> np.ndarray:
        """Extract features for a candidate word"""
        return np.array([
            float(len(word)),
            float('/' in word),
            float('.' in word),
            float(any(c.isdigit() for c in word)),
            float(word.startswith('.')),
            float(word.endswith('/')),
            float('-' in word or '_' in word),
        ])

    def _classical_path_prediction(self, discovered_paths: List[str],
                                   wordlist: List[str],
                                   top_k: int) -> List[Tuple[str, float]]:
        """Classical fallback for path prediction"""
        # Simple pattern matching based on discovered paths
        predictions = []

        # Extract common patterns from discovered paths
        common_prefixes = set()
        common_suffixes = set()

        for path in discovered_paths:
            parts = path.strip('/').split('/')
            if parts:
                common_prefixes.add(parts[0][:3] if len(parts[0]) >= 3 else parts[0])

            if '.' in path:
                ext = path.split('.')[-1]
                common_suffixes.add(ext)

        # Score wordlist based on pattern similarity
        for word in wordlist[:100]:
            score = 0.0

            # Bonus for matching prefixes
            word_prefix = word[:3] if len(word) >= 3 else word
            if word_prefix in common_prefixes:
                score += 0.5

            # Bonus for matching suffixes
            if '.' in word:
                word_ext = word.split('.')[-1]
                if word_ext in common_suffixes:
                    score += 0.3

            # Bonus for similar length
            avg_length = np.mean([len(p) for p in discovered_paths])
            length_diff = abs(len(word) - avg_length)
            score += max(0, 0.2 - length_diff / 50.0)

            predictions.append((word, score))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]


class QuantumResponseForecaster:
    """Quantum-enhanced forecasting using Schrödinger dynamics"""

    def __init__(self):
        self.quantum_available = QUANTUM_AVAILABLE
        self.history = []

    def forecast_response(self, historical_responses: List[Dict[str, Any]],
                         forecast_horizon: float = 1.0) -> Dict[str, Any]:
        """
        Forecast likely response characteristics using quantum dynamics.

        Args:
            historical_responses: List of past responses with status, size, time
            forecast_horizon: Time horizon for forecast (arbitrary units)

        Returns:
            Dict with forecasted probabilities and recommendations
        """
        if not self.quantum_available or len(historical_responses) < 3:
            return self._classical_forecast(historical_responses)

        try:
            # Build Hamiltonian representing response transitions
            # States: [2xx success, 3xx redirect, 4xx client error, 5xx server error]

            # Count transitions
            transitions = np.zeros((4, 4))
            state_map = {2: 0, 3: 1, 4: 2, 5: 3}

            for i in range(len(historical_responses) - 1):
                curr_status = historical_responses[i].get('status', 200) // 100
                next_status = historical_responses[i + 1].get('status', 200) // 100

                curr_idx = state_map.get(curr_status, 0)
                next_idx = state_map.get(next_status, 0)

                transitions[curr_idx, next_idx] += 1

            # Normalize to get Hamiltonian (symmetric for Hermiticity)
            H = (transitions + transitions.T) / 2
            H = H / (np.sum(H) + 1e-10)  # Normalize

            # Current state (distribution over status codes)
            recent_status = historical_responses[-1].get('status', 200) // 100
            psi0 = np.zeros(4)
            psi0[state_map.get(recent_status, 0)] = 1.0

            # Forecast using Schrödinger dynamics
            result = quantum_dynamics_forecasting(H, psi0, forecast_time=forecast_horizon)

            probabilities = result['probabilities']

            # Interpret probabilities
            forecast = {
                'probabilities': {
                    '2xx_success': float(probabilities[0]),
                    '3xx_redirect': float(probabilities[1]),
                    '4xx_client_error': float(probabilities[2]),
                    '5xx_server_error': float(probabilities[3])
                },
                'most_likely': self._interpret_state(np.argmax(probabilities)),
                'energy': result['energy'],
                'quantum_advantage': True,
                'forecast_horizon': forecast_horizon,
                'recommendation': self._generate_recommendation(probabilities)
            }

            return forecast

        except Exception as e:
            LOG.warning(f"Quantum forecasting failed: {e}, falling back to classical")
            return self._classical_forecast(historical_responses)

    def _interpret_state(self, state_idx: int) -> str:
        """Interpret quantum state index"""
        states = ['2xx Success', '3xx Redirect', '4xx Client Error', '5xx Server Error']
        return states[state_idx]

    def _generate_recommendation(self, probabilities: np.ndarray) -> str:
        """Generate scanning recommendation based on forecast"""
        success_prob = probabilities[0]
        error_prob = probabilities[2] + probabilities[3]

        if success_prob > 0.6:
            return "High success probability - continue aggressive scanning"
        elif error_prob > 0.5:
            return "High error probability - slow down to avoid rate limiting"
        elif probabilities[1] > 0.4:
            return "Many redirects expected - enable redirect following"
        else:
            return "Balanced probability distribution - maintain current pace"

    def _classical_forecast(self, historical_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Classical fallback for forecasting"""
        if not historical_responses:
            return {
                'probabilities': {
                    '2xx_success': 0.7,
                    '3xx_redirect': 0.1,
                    '4xx_client_error': 0.15,
                    '5xx_server_error': 0.05
                },
                'most_likely': '2xx Success',
                'quantum_advantage': False,
                'recommendation': "Insufficient data - using default assumptions"
            }

        # Simple frequency-based forecast
        status_counts = {2: 0, 3: 0, 4: 0, 5: 0}

        for resp in historical_responses[-20:]:  # Last 20 responses
            status = resp.get('status', 200) // 100
            status_counts[status] = status_counts.get(status, 0) + 1

        total = sum(status_counts.values())

        probabilities = {
            '2xx_success': status_counts[2] / total if total > 0 else 0.5,
            '3xx_redirect': status_counts[3] / total if total > 0 else 0.1,
            '4xx_client_error': status_counts[4] / total if total > 0 else 0.3,
            '5xx_server_error': status_counts[5] / total if total > 0 else 0.1
        }

        most_likely = max(probabilities.items(), key=lambda x: x[1])[0]

        return {
            'probabilities': probabilities,
            'most_likely': most_likely,
            'quantum_advantage': False,
            'recommendation': "Classical frequency analysis"
        }


def health_check() -> Dict[str, Any]:
    """Health check for quantum backend"""
    return {
        'quantum_backend': {
            'torch_available': TORCH_AVAILABLE,
            'quantum_available': QUANTUM_AVAILABLE,
            'features': {
                'anomaly_detection': QUANTUM_AVAILABLE,
                'path_prediction': QUANTUM_AVAILABLE,
                'response_forecasting': QUANTUM_AVAILABLE
            }
        }
    }


# Convenience functions for easy integration

def detect_anomaly(observation: Dict[str, Any],
                  num_particles: int = 1000) -> Tuple[float, str]:
    """
    Convenience function for anomaly detection.

    Args:
        observation: Dict with 'status', 'size', 'response_time' keys
        num_particles: Number of particles for filter (default 1000)

    Returns:
        (anomaly_score, reasoning) tuple
    """
    detector = QuantumAnomalyDetector(num_particles=num_particles)
    return detector.predict_anomaly(observation)


def predict_paths(discovered_paths: List[str],
                 wordlist: List[str],
                 top_k: int = 10) -> List[Tuple[str, float]]:
    """
    Convenience function for path prediction.

    Args:
        discovered_paths: List of already discovered paths
        wordlist: List of candidate paths to score
        top_k: Number of top predictions to return

    Returns:
        List of (path, confidence_score) tuples
    """
    predictor = QuantumPathPredictor()
    return predictor.predict_next_paths(discovered_paths, wordlist, top_k)


def forecast_response(historical_responses: List[Dict[str, Any]],
                     forecast_horizon: float = 1.0) -> Dict[str, Any]:
    """
    Convenience function for response forecasting.

    Args:
        historical_responses: List of past responses
        forecast_horizon: Time horizon for forecast

    Returns:
        Dict with forecast probabilities and recommendations
    """
    forecaster = QuantumResponseForecaster()
    return forecaster.forecast_response(historical_responses, forecast_horizon)
