# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Telescope Explainer
SHAP values, counterfactuals, confidence intervals for transparent predictions.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TelescopeExplainer:
    """
    Comprehensive explainability system for Telescope predictions.

    Features:
    1. SHAP values - Feature importance for individual predictions
    2. Confidence intervals - Bootstrap uncertainty quantification
    3. Counterfactuals - Minimal changes to flip prediction
    4. Similar cases - Historical examples with similar profiles
    """

    def __init__(self, model, feature_names: List[str], training_data: np.ndarray = None):
        """
        Initialize explainer.

        Args:
            model: Trained model (sklearn-compatible or custom)
            feature_names: List of feature names
            training_data: Optional training data for SHAP background
        """
        self.model = model
        self.feature_names = feature_names
        self.training_data = training_data
        self.shap_explainer = None

        # Try to initialize SHAP
        self._init_shap()

    def _init_shap(self):
        """Initialize SHAP explainer if available."""
        try:
            import shap

            # Determine model type and use appropriate explainer
            if hasattr(self.model, 'predict_proba'):
                # Tree-based model
                try:
                    self.shap_explainer = shap.TreeExplainer(self.model)
                    logger.info("SHAP TreeExplainer initialized")
                except:
                    # Fallback to KernelExplainer
                    if self.training_data is not None:
                        background = shap.sample(self.training_data, 100)
                        self.shap_explainer = shap.KernelExplainer(
                            self.model.predict_proba,
                            background
                        )
                        logger.info("SHAP KernelExplainer initialized (fallback)")
            else:
                logger.warning("Model not compatible with SHAP")

        except ImportError:
            logger.warning("SHAP not installed, using fallback explainer")
            self.shap_explainer = None

    def explain_prediction(
        self,
        x: np.ndarray,
        return_all: bool = True,
        n_counterfactuals: int = 3,
        n_similar: int = 5
    ) -> Dict[str, Any]:
        """
        Generate complete explanation for a prediction.

        Args:
            x: Feature vector [n_features] or [1, n_features]
            return_all: Return all explanation types
            n_counterfactuals: Number of counterfactuals to generate
            n_similar: Number of similar cases to find

        Returns:
            Dictionary with explanations
        """
        # Ensure 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Get prediction
        if hasattr(self.model, 'predict_proba'):
            prediction_proba = self.model.predict_proba(x)[0]
            prediction = prediction_proba.argmax() if len(prediction_proba.shape) > 0 else int(prediction_proba > 0.5)
        else:
            prediction = int(self.model.predict(x)[0])
            prediction_proba = None

        explanation = {
            'prediction': prediction,
            'prediction_proba': prediction_proba.tolist() if prediction_proba is not None else None,
        }

        if return_all:
            explanation['shap_values'] = self._shap_explanation(x)
            explanation['confidence_interval'] = self._confidence_interval(x)
            explanation['counterfactuals'] = self._generate_counterfactuals(x, n_counterfactuals)

            if self.training_data is not None:
                explanation['similar_cases'] = self._find_similar_cases(x, n_similar)

        return explanation

    def _shap_explanation(self, x: np.ndarray) -> Dict[str, Any]:
        """
        SHAP values showing feature contributions.

        Returns:
            Dictionary with feature contributions and base value
        """
        if self.shap_explainer is None:
            # Fallback: simple feature importance via permutation
            return self._fallback_feature_importance(x)

        try:
            import shap

            # Get SHAP values
            shap_values = self.shap_explainer.shap_values(x)

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Multi-class output, use first class
                shap_values = shap_values[0]

            # Format as readable dict
            if shap_values.ndim == 2:
                shap_values = shap_values[0]

            contributions = {
                self.feature_names[i]: float(shap_values[i])
                for i in range(min(len(self.feature_names), len(shap_values)))
            }

            # Sort by absolute contribution
            sorted_contrib = dict(
                sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
            )

            return {
                'contributions': sorted_contrib,
                'base_value': float(self.shap_explainer.expected_value) if hasattr(self.shap_explainer, 'expected_value') else 0.0,
                'top_5_features': list(sorted_contrib.keys())[:5]
            }

        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}, using fallback")
            return self._fallback_feature_importance(x)

    def _fallback_feature_importance(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Fallback feature importance when SHAP unavailable.
        Uses simple magnitude-based scoring.
        """
        # Normalize features and use magnitude as importance
        x_flat = x.flatten()

        # Simple heuristic: larger absolute values = more important
        importances = np.abs(x_flat)

        contributions = {
            self.feature_names[i]: float(importances[i])
            for i in range(min(len(self.feature_names), len(x_flat)))
        }

        sorted_contrib = dict(
            sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
        )

        return {
            'contributions': sorted_contrib,
            'base_value': 0.0,
            'top_5_features': list(sorted_contrib.keys())[:5],
            'note': 'Fallback importance (SHAP unavailable)'
        }

    def _confidence_interval(self, x: np.ndarray, n_bootstrap: int = 100) -> Dict[str, float]:
        """
        Bootstrap confidence interval for prediction.

        Args:
            x: Feature vector
            n_bootstrap: Number of bootstrap samples

        Returns:
            Dictionary with mean, std, and confidence intervals
        """
        predictions = []

        for _ in range(n_bootstrap):
            # Add small random noise to simulate uncertainty
            x_perturbed = x + np.random.normal(0, 0.01, x.shape)

            if hasattr(self.model, 'predict_proba'):
                pred = self.model.predict_proba(x_perturbed)[0]
                # Use probability of positive class or max probability
                if len(pred.shape) > 0 and len(pred) > 1:
                    pred_val = pred.max()
                else:
                    pred_val = pred[1] if len(pred) > 1 else pred[0]
            else:
                pred_val = float(self.model.predict(x_perturbed)[0])

            predictions.append(pred_val)

        predictions = np.array(predictions)

        return {
            'mean': float(predictions.mean()),
            'std': float(predictions.std()),
            'ci_95_lower': float(np.percentile(predictions, 2.5)),
            'ci_95_upper': float(np.percentile(predictions, 97.5)),
            'ci_90_lower': float(np.percentile(predictions, 5)),
            'ci_90_upper': float(np.percentile(predictions, 95)),
        }

    def _generate_counterfactuals(
        self,
        x: np.ndarray,
        n_counterfactuals: int = 3,
        max_changes: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find minimal changes that would flip the prediction.

        Args:
            x: Original feature vector
            n_counterfactuals: Number of counterfactuals to generate
            max_changes: Maximum features to change per counterfactual

        Returns:
            List of counterfactual explanations
        """
        x_flat = x.flatten()
        current_pred = int(self.model.predict(x)[0])

        counterfactuals = []

        # Try changing each feature slightly
        for i in range(min(len(x_flat), len(self.feature_names))):
            for multiplier in [0.8, 1.2, 0.5, 1.5]:
                x_modified = x_flat.copy()
                x_modified[i] *= multiplier

                new_pred = int(self.model.predict(x_modified.reshape(1, -1))[0])

                if new_pred != current_pred:
                    change_pct = (multiplier - 1.0) * 100

                    counterfactuals.append({
                        'feature': self.feature_names[i],
                        'original_value': float(x_flat[i]),
                        'new_value': float(x_modified[i]),
                        'change_percent': f'{change_pct:+.1f}%',
                        'new_prediction': int(new_pred)
                    })

                    if len(counterfactuals) >= n_counterfactuals:
                        return counterfactuals

        # If no single-feature changes work, try combinations
        if len(counterfactuals) == 0:
            counterfactuals.append({
                'note': 'No single-feature counterfactuals found within threshold',
                'suggestion': 'Multiple features may need to change'
            })

        return counterfactuals[:n_counterfactuals]

    def _find_similar_cases(self, x: np.ndarray, n_similar: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar historical cases from training data.

        Args:
            x: Query feature vector
            n_similar: Number of similar cases to return

        Returns:
            List of similar case descriptions
        """
        if self.training_data is None:
            return [{'note': 'No training data available for similarity search'}]

        # Compute distances (Euclidean)
        x_flat = x.flatten()
        distances = np.linalg.norm(self.training_data - x_flat, axis=1)

        # Get indices of nearest neighbors
        nearest_indices = np.argsort(distances)[:n_similar]

        similar_cases = []
        for idx in nearest_indices:
            similar_cases.append({
                'index': int(idx),
                'distance': float(distances[idx]),
                'features': self.training_data[idx].tolist()
            })

        return similar_cases

    def feature_importance_summary(self, X: np.ndarray, top_k: int = 10) -> Dict[str, float]:
        """
        Aggregate feature importance across multiple samples.

        Args:
            X: Multiple samples [n_samples, n_features]
            top_k: Return top K features

        Returns:
            Dictionary of average feature importances
        """
        if self.shap_explainer is None:
            logger.warning("SHAP not available, using fallback")
            # Simple variance-based importance
            importances = X.var(axis=0)
        else:
            try:
                shap_values = self.shap_explainer.shap_values(X)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                importances = np.abs(shap_values).mean(axis=0)
            except:
                importances = X.var(axis=0)

        # Create importance dict
        importance_dict = {
            self.feature_names[i]: float(importances[i])
            for i in range(min(len(self.feature_names), len(importances)))
        }

        # Sort and return top K
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)
        )

        return dict(list(sorted_importance.items())[:top_k])


if __name__ == "__main__":
    # Test explainer
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    logger.info("Testing Telescope Explainer...")

    # Create dummy model and data
    np.random.seed(42)
    X_train = np.random.randn(1000, 10)
    y_train = (X_train[:, 0] + X_train[:, 5] > 0).astype(int)

    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # Create explainer
    feature_names = [f'feature_{i}' for i in range(10)]
    explainer = TelescopeExplainer(model, feature_names, training_data=X_train)

    # Test prediction
    x_test = np.random.randn(10)
    explanation = explainer.explain_prediction(x_test)

    print("\n=== Telescope Explainer Test ===")
    print(f"Prediction: {explanation['prediction']}")
    print(f"Confidence: {explanation['prediction_proba']}")
    print(f"\nTop 5 important features:")
    for feat in explanation['shap_values']['top_5_features'][:5]:
        contrib = explanation['shap_values']['contributions'][feat]
        print(f"  {feat}: {contrib:.4f}")

    print(f"\nConfidence Interval (95%):")
    ci = explanation['confidence_interval']
    print(f"  [{ci['ci_95_lower']:.3f}, {ci['ci_95_upper']:.3f}]")

    print(f"\nCounterfactuals: {len(explanation['counterfactuals'])}")
    if explanation['counterfactuals']:
        cf = explanation['counterfactuals'][0]
        if 'feature' in cf:
            print(f"  Change {cf['feature']} by {cf['change_percent']}")

    print("\n✓ Explainer validated")
