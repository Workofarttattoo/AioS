#!/usr/bin/env python3
"""
Telescope Suite & Oracle of Light: Comprehensive Metrics & Evaluation Framework
Tracks accuracy, validates predictions, and drives continuous improvement

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# ============================================================================
# Metrics Data Structures
# ============================================================================

@dataclass
class PredictionMetrics:
    """Metrics for a single prediction"""
    tool: str
    prediction_id: str
    algorithm: str
    predicted_value: float
    actual_value: Optional[float] = None
    confidence: float = 0.5
    error: Optional[float] = None
    abs_error: Optional[float] = None
    abs_percent_error: Optional[float] = None
    timestamp: str = ""
    validated: bool = False
    horizon: int = 1  # 1-step, multi-step forecast

@dataclass
class ToolAccuracy:
    """Accuracy summary for a tool"""
    tool: str
    algorithm: str
    mape: float  # Mean Absolute Percent Error
    mae: float   # Mean Absolute Error
    rmse: float  # Root Mean Squared Error
    r_squared: float  # R² coefficient
    directional_accuracy: float  # For classification
    count: int  # Number of predictions
    timestamp: str = ""

@dataclass
class EnsemblePerformance:
    """Performance metrics for ensemble"""
    ensemble_accuracy: float
    component_accuracies: Dict[str, float]
    weights: Dict[str, float]
    improvement: float  # vs best individual
    timestamp: str = ""

# ============================================================================
# Metrics Calculation Engine
# ============================================================================

class MetricsCalculator:
    """Calculates various accuracy metrics"""

    @staticmethod
    def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percent Error"""
        mask = y_true != 0
        if not mask.any():
            return 0.0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) * 100

    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def calculate_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R² coefficient"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1 - (ss_res / ss_tot))

    @staticmethod
    def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Percentage of correct direction predictions"""
        if len(y_true) < 2:
            return 0.0

        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        return float(np.mean(true_direction == pred_direction)) * 100

    @staticmethod
    def calculate_precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Tuple[float, float, float]:
        """Precision, Recall, F1 for binary classification"""
        y_pred_binary = (y_pred > threshold).astype(int)
        y_true_binary = (y_true > threshold).astype(int)

        tp = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
        fp = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
        fn = np.sum((y_pred_binary == 0) & (y_true_binary == 1))

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

        return float(precision), float(recall), float(f1)

# ============================================================================
# Prediction Validator
# ============================================================================

class PredictionValidator:
    """Validates predictions against actual outcomes"""

    def __init__(self, db_path: Path = Path("/tmp/telescope_predictions.db")):
        self.db_path = db_path
        self.metrics = []

    def validate_prediction(
        self,
        tool: str,
        prediction_id: str,
        algorithm: str,
        predicted_value: float,
        actual_value: float,
        confidence: float,
        horizon: int = 1
    ) -> PredictionMetrics:
        """Validate a single prediction"""

        metric = PredictionMetrics(
            tool=tool,
            prediction_id=prediction_id,
            algorithm=algorithm,
            predicted_value=predicted_value,
            actual_value=actual_value,
            confidence=confidence,
            error=actual_value - predicted_value,
            abs_error=abs(actual_value - predicted_value),
            timestamp=datetime.now().isoformat(),
            validated=True,
            horizon=horizon,
        )

        # Calculate percent error
        if actual_value != 0:
            metric.abs_percent_error = abs(metric.error / actual_value) * 100
        else:
            metric.abs_percent_error = 0.0

        self.metrics.append(metric)
        self._save_metric(metric)

        return metric

    def batch_validate(self, predictions_df: pd.DataFrame) -> List[PredictionMetrics]:
        """Validate multiple predictions"""
        results = []

        for _, row in predictions_df.iterrows():
            metric = self.validate_prediction(
                tool=row.get('tool'),
                prediction_id=row.get('prediction_id'),
                algorithm=row.get('algorithm'),
                predicted_value=row.get('predicted'),
                actual_value=row.get('actual'),
                confidence=row.get('confidence', 0.5),
                horizon=row.get('horizon', 1)
            )
            results.append(metric)

        LOG.info(f"[info] Validated {len(results)} predictions")
        return results

    def _save_metric(self, metric: PredictionMetrics):
        """Save metric to persistent storage"""
        try:
            import sqlite3
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Create table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions_history (
                    id TEXT PRIMARY KEY,
                    tool TEXT,
                    algorithm TEXT,
                    predicted REAL,
                    actual REAL,
                    abs_error REAL,
                    abs_percent_error REAL,
                    confidence REAL,
                    horizon INTEGER,
                    timestamp TEXT,
                    validated INTEGER
                )
            """)

            cursor.execute("""
                INSERT OR REPLACE INTO predictions_history
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.prediction_id,
                metric.tool,
                metric.algorithm,
                metric.predicted_value,
                metric.actual_value,
                metric.abs_error,
                metric.abs_percent_error,
                metric.confidence,
                metric.horizon,
                metric.timestamp,
                int(metric.validated)
            ))

            conn.commit()
            conn.close()
        except Exception as e:
            LOG.warn(f"[warn] Failed to save metric: {e}")

# ============================================================================
# Accuracy Analytics Engine
# ============================================================================

class AccuracyAnalytics:
    """Analyzes accuracy trends and generates reports"""

    def __init__(self, db_path: Path = Path("/tmp/telescope_predictions.db")):
        self.db_path = db_path

    def get_tool_accuracy(self, tool: str, lookback_days: int = 7) -> Optional[ToolAccuracy]:
        """Calculate accuracy metrics for a tool"""
        try:
            df = self._load_predictions(tool, lookback_days)

            if df.empty or 'actual' not in df.columns:
                return None

            y_true = df['actual'].values
            y_pred = df['predicted'].values

            return ToolAccuracy(
                tool=tool,
                algorithm='ensemble',
                mape=MetricsCalculator.calculate_mape(y_true, y_pred),
                mae=MetricsCalculator.calculate_mae(y_true, y_pred),
                rmse=MetricsCalculator.calculate_rmse(y_true, y_pred),
                r_squared=MetricsCalculator.calculate_r_squared(y_true, y_pred),
                directional_accuracy=MetricsCalculator.calculate_directional_accuracy(y_true, y_pred),
                count=len(df),
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            LOG.warn(f"[warn] Failed to calculate accuracy for {tool}: {e}")
            return None

    def get_algorithm_comparison(self, tool: str, lookback_days: int = 7) -> Dict[str, ToolAccuracy]:
        """Compare accuracy across algorithms"""
        try:
            df = self._load_predictions(tool, lookback_days)

            if df.empty:
                return {}

            comparison = {}

            for algo in df['algorithm'].unique():
                algo_df = df[df['algorithm'] == algo]

                if len(algo_df) < 10:
                    continue

                y_true = algo_df['actual'].values
                y_pred = algo_df['predicted'].values

                comparison[algo] = ToolAccuracy(
                    tool=tool,
                    algorithm=algo,
                    mape=MetricsCalculator.calculate_mape(y_true, y_pred),
                    mae=MetricsCalculator.calculate_mae(y_true, y_pred),
                    rmse=MetricsCalculator.calculate_rmse(y_true, y_pred),
                    r_squared=MetricsCalculator.calculate_r_squared(y_true, y_pred),
                    directional_accuracy=MetricsCalculator.calculate_directional_accuracy(y_true, y_pred),
                    count=len(algo_df),
                    timestamp=datetime.now().isoformat()
                )

            return comparison

        except Exception as e:
            LOG.warn(f"[warn] Failed to compare algorithms: {e}")
            return {}

    def get_accuracy_trend(self, tool: str, days: int = 30) -> List[Tuple[str, float]]:
        """Get accuracy trend over time"""
        try:
            df = self._load_predictions(tool, days)

            if df.empty:
                return []

            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            daily_accuracy = []

            for date, group in df.groupby('date'):
                y_true = group['actual'].values
                y_pred = group['predicted'].values

                mape = MetricsCalculator.calculate_mape(y_true, y_pred)
                accuracy = 100 - mape  # Approximate

                daily_accuracy.append((str(date), accuracy))

            return daily_accuracy

        except Exception as e:
            LOG.warn(f"[warn] Failed to calculate trend: {e}")
            return []

    def detect_drift(self, tool: str, threshold: float = 0.05) -> bool:
        """Detect data distribution drift"""
        try:
            df = self._load_predictions(tool, lookback_days=14)

            if len(df) < 100:
                return False

            # Compare first half to second half
            mid = len(df) // 2
            first_half = df.iloc[:mid]['predicted'].values
            second_half = df.iloc[mid:]['predicted'].values

            # KL divergence approximation
            mean_diff = abs(np.mean(first_half) - np.mean(second_half)) / (np.mean(first_half) + 1e-10)

            return mean_diff > threshold

        except Exception as e:
            LOG.warn(f"[warn] Drift detection failed: {e}")
            return False

    def generate_daily_report(self, tools: List[str]) -> Dict[str, Any]:
        """Generate daily accuracy report"""
        LOG.info("[info] Generating daily accuracy report...")

        report = {
            'timestamp': datetime.now().isoformat(),
            'tools': {},
            'alerts': [],
            'recommendations': [],
        }

        for tool in tools:
            accuracy = self.get_tool_accuracy(tool)

            if accuracy:
                report['tools'][tool] = asdict(accuracy)

                # Check thresholds
                if accuracy.mape > 20:
                    report['alerts'].append(f"⚠️  {tool}: MAPE {accuracy.mape:.1f}% > 20% (retrain recommended)")

                if self.detect_drift(tool):
                    report['alerts'].append(f"⚠️  {tool}: Data drift detected")

        return report

    def _load_predictions(self, tool: str, lookback_days: int = 7) -> pd.DataFrame:
        """Load predictions from database"""
        try:
            import sqlite3
            conn = sqlite3.connect(str(self.db_path))
            cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()

            df = pd.read_sql(f"""
                SELECT * FROM predictions_history
                WHERE tool = '{tool}' AND timestamp > '{cutoff_date}' AND validated = 1
            """, conn)

            conn.close()
            return df

        except Exception as e:
            LOG.warn(f"[warn] Failed to load predictions: {e}")
            return pd.DataFrame()

# ============================================================================
# Ensemble Performance Tracker
# ============================================================================

class EnsemblePerformanceTracker:
    """Tracks and optimizes ensemble performance"""

    def __init__(self):
        self.history: List[EnsemblePerformance] = []

    def record_performance(
        self,
        ensemble_accuracy: float,
        component_accuracies: Dict[str, float],
        weights: Dict[str, float],
        improvement: Optional[float] = None
    ) -> EnsemblePerformance:
        """Record ensemble performance"""

        if improvement is None:
            best_component = max(component_accuracies.values())
            improvement = ensemble_accuracy - best_component

        perf = EnsemblePerformance(
            ensemble_accuracy=ensemble_accuracy,
            component_accuracies=component_accuracies,
            weights=weights,
            improvement=improvement,
            timestamp=datetime.now().isoformat()
        )

        self.history.append(perf)
        LOG.info(f"[info] Ensemble accuracy: {ensemble_accuracy:.1%}, Improvement: {improvement:+.1%}")

        return perf

    def get_latest_performance(self) -> Optional[EnsemblePerformance]:
        """Get most recent ensemble performance"""
        if self.history:
            return self.history[-1]
        return None

    def get_performance_trend(self, lookback: int = 30) -> List[float]:
        """Get ensemble accuracy trend"""
        return [perf.ensemble_accuracy for perf in self.history[-lookback:]]

    def is_improving(self, window: int = 7) -> bool:
        """Check if ensemble is improving"""
        if len(self.history) < window:
            return True

        recent = self.get_performance_trend(window)
        if len(recent) < 2:
            return True

        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        return trend > 0

# ============================================================================
# Cross-Tool Learning Analyzer
# ============================================================================

class CrossToolLearningAnalyzer:
    """Analyzes patterns across different prediction tools"""

    def __init__(self):
        self.insights: List[Dict[str, Any]] = []

    def analyze_correlation(self, predictions_df: pd.DataFrame) -> Dict[str, float]:
        """Analyze correlation between tools"""
        correlations = {}

        tools = predictions_df['tool'].unique()

        for i, tool1 in enumerate(tools):
            for tool2 in tools[i+1:]:
                df1 = predictions_df[predictions_df['tool'] == tool1]
                df2 = predictions_df[predictions_df['tool'] == tool2]

                # Merge on date
                if 'timestamp' in df1.columns and 'timestamp' in df2.columns:
                    merged = df1.merge(df2, on='timestamp', suffixes=('_1', '_2'))

                    if len(merged) > 10:
                        corr = merged[['predicted_1', 'predicted_2']].corr().iloc[0, 1]
                        correlations[f"{tool1}↔{tool2}"] = float(corr)

        return correlations

    def identify_transferable_patterns(self, predictions_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify patterns that can transfer between tools"""
        insights = []

        # Example: If career and relationships predictions agree, they reinforce
        career = predictions_df[predictions_df['tool'] == 'telescope_career']
        relations = predictions_df[predictions_df['tool'] == 'telescope_relationships']

        if len(career) > 50 and len(relations) > 50:
            career_accuracy = MetricsCalculator.calculate_mape(
                career['actual'].values,
                career['predicted'].values
            )
            relations_accuracy = MetricsCalculator.calculate_mape(
                relations['actual'].values,
                relations['predicted'].values
            )

            if career_accuracy < relations_accuracy:
                insights.append({
                    'pattern': 'Career predictions outperform relationships',
                    'source_tool': 'telescope_career',
                    'target_tool': 'telescope_relationships',
                    'recommendation': 'Apply career prediction patterns to relationships',
                    'expected_improvement': '1-2%'
                })

        return insights

# ============================================================================
# Reporting & Visualization
# ============================================================================

def generate_metrics_report(tools: List[str]) -> Dict[str, Any]:
    """Generate comprehensive metrics report"""
    LOG.info("[info] Generating comprehensive metrics report...")

    analytics = AccuracyAnalytics()
    tracker = EnsemblePerformanceTracker()

    report = {
        'generated_at': datetime.now().isoformat(),
        'accuracy_by_tool': {},
        'algorithm_comparison': {},
        'accuracy_trends': {},
        'ensemble_status': {},
        'alerts': [],
        'recommendations': [],
    }

    # Tool accuracies
    for tool in tools:
        accuracy = analytics.get_tool_accuracy(tool)
        if accuracy:
            report['accuracy_by_tool'][tool] = asdict(accuracy)

        # Algorithm comparison
        comparison = analytics.get_algorithm_comparison(tool)
        if comparison:
            report['algorithm_comparison'][tool] = {
                algo: asdict(acc) for algo, acc in comparison.items()
            }

        # Trends
        trend = analytics.get_accuracy_trend(tool)
        if trend:
            report['accuracy_trends'][tool] = trend

    # Ensemble status
    latest = tracker.get_latest_performance()
    if latest:
        report['ensemble_status'] = asdict(latest)

    # Format report
    LOG.info("[info] ====== METRICS REPORT ======")
    for tool, metrics in report['accuracy_by_tool'].items():
        LOG.info(f"[info] {tool}:")
        LOG.info(f"[info]   MAPE: {metrics.get('mape', 0):.1f}%")
        LOG.info(f"[info]   MAE: {metrics.get('mae', 0):.2f}")
        LOG.info(f"[info]   Directional Accuracy: {metrics.get('directional_accuracy', 0):.1f}%")
        LOG.info(f"[info]   Samples: {metrics.get('count', 0)}")

    return report

if __name__ == "__main__":
    # Example usage
    tools = ['telescope_career', 'telescope_health', 'bear_tamer']
    report = generate_metrics_report(tools)

    print(json.dumps(report, indent=2, default=str))
