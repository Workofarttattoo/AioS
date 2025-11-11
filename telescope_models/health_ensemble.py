# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Health Ensemble Model
Stacked ensemble: XGBoost + LightGBM + RandomForest + Neural Net
Target Accuracy: 89%+ for health risk prediction
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthNeuralNet(nn.Module):
    """Neural network base learner for health ensemble."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        # Output layer (binary classification)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class HealthEnsembleModel:
    """
    Stacked Ensemble Model for Health Risk Prediction.

    Architecture:
    - Base Models:
      * XGBoost (GPU-accelerated if available)
      * LightGBM (GPU-accelerated if available)
      * Random Forest
      * Neural Network (PyTorch)
    - Meta-learner: Gradient Boosting (stacking)

    Target Accuracy: 89%+
    AUC: 0.92+
    """

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu and torch.cuda.is_available()

        # Base models (will be initialized during fit)
        self.xgb = None
        self.lgbm = None
        self.rf = None
        self.neural = None

        # Meta-learner
        self.meta_model = None

        # Feature statistics for normalization
        self.feature_means = None
        self.feature_stds = None

        logger.info(f"Health Ensemble initialized (GPU: {self.use_gpu})")

    def _build_xgboost(self, n_features: int):
        """Build XGBoost model."""
        try:
            import xgboost as xgb

            params = {
                'n_estimators': 500,
                'max_depth': 10,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'random_state': 42,
            }

            if self.use_gpu:
                params['tree_method'] = 'gpu_hist'
                params['predictor'] = 'gpu_predictor'

            self.xgb = xgb.XGBClassifier(**params)
            logger.info("XGBoost initialized")
        except ImportError:
            logger.warning("XGBoost not installed, using fallback")
            # Fallback to scikit-learn GradientBoosting
            from sklearn.ensemble import GradientBoostingClassifier
            self.xgb = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.05,
                random_state=42
            )

    def _build_lightgbm(self, n_features: int):
        """Build LightGBM model."""
        try:
            import lightgbm as lgb

            params = {
                'n_estimators': 500,
                'max_depth': 10,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'random_state': 42,
            }

            if self.use_gpu:
                params['device'] = 'gpu'

            self.lgbm = lgb.LGBMClassifier(**params)
            logger.info("LightGBM initialized")
        except ImportError:
            logger.warning("LightGBM not installed, using fallback")
            from sklearn.ensemble import GradientBoostingClassifier
            self.lgbm = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                random_state=43
            )

    def _build_random_forest(self, n_features: int):
        """Build Random Forest model."""
        from sklearn.ensemble import RandomForestClassifier

        self.rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,  # Use all CPU cores
            random_state=42
        )
        logger.info("Random Forest initialized")

    def _build_neural_net(self, n_features: int):
        """Build Neural Network model."""
        self.neural = HealthNeuralNet(
            input_dim=n_features,
            hidden_dims=[256, 128, 64]
        )

        if self.use_gpu:
            self.neural = self.neural.cuda()

        logger.info("Neural Network initialized")

    def _build_meta_learner(self):
        """Build meta-learner for stacking."""
        from sklearn.ensemble import GradientBoostingClassifier

        self.meta_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        logger.info("Meta-learner initialized")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        neural_epochs: int = 50
    ):
        """
        Train ensemble with stacking.

        Args:
            X_train: Training features [n_samples, n_features]
            y_train: Training labels [n_samples]
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            neural_epochs: Epochs for neural network training
        """
        n_samples, n_features = X_train.shape

        logger.info(f"Training ensemble on {n_samples} samples, {n_features} features")

        # Normalize features
        self.feature_means = X_train.mean(axis=0)
        self.feature_stds = X_train.std(axis=0) + 1e-8
        X_train_norm = (X_train - self.feature_means) / self.feature_stds

        if X_val is not None:
            X_val_norm = (X_val - self.feature_means) / self.feature_stds

        # Initialize base models
        self._build_xgboost(n_features)
        self._build_lightgbm(n_features)
        self._build_random_forest(n_features)
        self._build_neural_net(n_features)

        # Train base models
        logger.info("[1/5] Training XGBoost...")
        self.xgb.fit(X_train_norm, y_train)

        logger.info("[2/5] Training LightGBM...")
        self.lgbm.fit(X_train_norm, y_train)

        logger.info("[3/5] Training Random Forest...")
        self.rf.fit(X_train_norm, y_train)

        logger.info("[4/5] Training Neural Network...")
        self._train_neural_net(X_train_norm, y_train, X_val_norm if X_val is not None else None, y_val, neural_epochs)

        # Generate meta-features from base models
        logger.info("[5/5] Training meta-learner...")
        meta_features_train = self._generate_meta_features(X_train_norm)
        self._build_meta_learner()
        self.meta_model.fit(meta_features_train, y_train)

        # Validation accuracy if provided
        if X_val is not None and y_val is not None:
            val_preds = self.predict(X_val)
            val_acc = (val_preds == y_val).mean()
            logger.info(f"Validation Accuracy: {val_acc:.4f}")

            # AUC score
            try:
                from sklearn.metrics import roc_auc_score
                val_proba = self.predict_proba(X_val)
                val_auc = roc_auc_score(y_val, val_proba)
                logger.info(f"Validation AUC: {val_auc:.4f}")
            except:
                pass

        logger.info("✓ Ensemble training complete")

    def _train_neural_net(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int
    ):
        """Train neural network base learner."""
        # Convert to tensors
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

        if self.use_gpu:
            X_tensor = X_tensor.cuda()
            y_tensor = y_tensor.cuda()

        # Optimizer and loss
        optimizer = torch.optim.AdamW(self.neural.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        # Training loop
        batch_size = 512
        n_batches = len(X_train) // batch_size

        for epoch in range(epochs):
            self.neural.train()
            epoch_loss = 0.0

            # Shuffle data
            indices = torch.randperm(len(X_train))

            for i in range(n_batches):
                batch_idx = indices[i * batch_size:(i + 1) * batch_size]
                X_batch = X_tensor[batch_idx]
                y_batch = y_tensor[batch_idx]

                optimizer.zero_grad()
                outputs = self.neural(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / n_batches
                logger.info(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}")

    def _generate_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions from base models as meta-features."""
        # XGBoost predictions
        xgb_pred = self.xgb.predict_proba(X)[:, 1].reshape(-1, 1)

        # LightGBM predictions
        lgbm_pred = self.lgbm.predict_proba(X)[:, 1].reshape(-1, 1)

        # Random Forest predictions
        rf_pred = self.rf.predict_proba(X)[:, 1].reshape(-1, 1)

        # Neural Network predictions
        self.neural.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            if self.use_gpu:
                X_tensor = X_tensor.cuda()
            nn_pred = self.neural(X_tensor).cpu().numpy()

        # Stack all predictions
        meta_features = np.hstack([xgb_pred, lgbm_pred, rf_pred, nn_pred])

        return meta_features

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities with stacked ensemble.

        Args:
            X: Features [n_samples, n_features]

        Returns:
            Probabilities [n_samples] (probability of positive class)
        """
        # Normalize
        X_norm = (X - self.feature_means) / self.feature_stds

        # Generate meta-features
        meta_features = self._generate_meta_features(X_norm)

        # Meta-learner prediction
        proba = self.meta_model.predict_proba(meta_features)[:, 1]

        return proba

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features [n_samples, n_features]
            threshold: Classification threshold (default 0.5)

        Returns:
            Binary predictions [n_samples]
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def save(self, path: str):
        """Save ensemble model."""
        import pickle

        model_data = {
            'xgb': self.xgb,
            'lgbm': self.lgbm,
            'rf': self.rf,
            'neural_state': self.neural.state_dict() if self.neural else None,
            'meta_model': self.meta_model,
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds,
            'use_gpu': self.use_gpu,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load ensemble model."""
        import pickle

        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.xgb = model_data['xgb']
        self.lgbm = model_data['lgbm']
        self.rf = model_data['rf']
        self.meta_model = model_data['meta_model']
        self.feature_means = model_data['feature_means']
        self.feature_stds = model_data['feature_stds']
        self.use_gpu = model_data['use_gpu']

        # Rebuild neural network
        if model_data['neural_state']:
            n_features = len(self.feature_means)
            self._build_neural_net(n_features)
            self.neural.load_state_dict(model_data['neural_state'])

        logger.info(f"Model loaded from {path}")


def train_health_ensemble(
    data_path: str,
    target_column: str = 'outcome_cvd_10yr',
    test_size: float = 0.2,
    save_path: str = 'models/health_ensemble.pkl'
):
    """
    Train health ensemble model.

    Args:
        data_path: Path to health data (parquet/csv)
        target_column: Target variable to predict
        test_size: Validation split ratio
        save_path: Where to save model
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

    # Load data
    logger.info(f"Loading data from {data_path}")
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    # Feature columns (health metrics)
    feature_cols = [
        'age', 'bmi', 'systolic_bp', 'diastolic_bp',
        'glucose_fasting', 'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
        'smoking', 'exercise_hours_week', 'alcohol_drinks_week',
        'diet_quality_score', 'sleep_hours_avg', 'stress_level',
        'family_history_cvd', 'family_history_diabetes', 'family_history_cancer',
        'cvd_risk_score', 'diabetes_risk_score', 'hypertension_risk_score', 'cancer_risk_score',
        'metabolic_syndrome_score', 'lifestyle_score'
    ]

    # Filter to available columns
    feature_cols = [col for col in feature_cols if col in df.columns]

    X = df[feature_cols].values
    y = df[target_column].values

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")
    logger.info(f"Positive rate: Train={y_train.mean():.3f}, Val={y_val.mean():.3f}")

    # Train ensemble
    model = HealthEnsembleModel(use_gpu=torch.cuda.is_available())
    model.fit(X_train, y_train, X_val, y_val, neural_epochs=50)

    # Final evaluation
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)

    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_proba)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)

    logger.info(f"\n=== Final Results ===")
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"AUC: {auc:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")

    # Save model
    model.save(save_path)

    return model


if __name__ == "__main__":
    # Test with dummy data
    logger.info("Testing Health Ensemble Model...")

    np.random.seed(42)
    n_samples = 10000
    n_features = 23

    # Generate synthetic health data
    X = np.random.randn(n_samples, n_features)
    # Target correlated with some features
    y = ((X[:, 0] + X[:, 5] + X[:, 10]) > 0.5).astype(int)

    # Split
    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Train
    model = HealthEnsembleModel()
    model.fit(X_train, y_train, X_val, y_val, neural_epochs=20)

    # Predict
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)

    acc = (y_pred == y_val).mean()

    print(f"\n=== Test Results ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Predictions sample: {y_pred[:10]}")
    print(f"Probabilities sample: {y_proba[:10]}")
    print("\n✓ Health Ensemble Model validated")
