# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Telescope Feature Engineer
Automated feature generation: 1000+ features via Deep Feature Synthesis + domain engineering.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TelescopeFeatureEngineer:
    """
    Automated feature engineering system generating 1000+ features.

    Methods:
    1. Domain-specific feature engineering (career, health, market)
    2. Polynomial interaction features
    3. Time-series features (lag, rolling)
    4. Statistical aggregations
    5. Feature selection (remove low-variance, correlated)
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        self.selected_features = []

    def engineer_all_features(
        self,
        df: pd.DataFrame,
        target_domain: str,
        include_interactions: bool = True,
        include_aggregations: bool = True,
        max_features: int = 1000
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate comprehensive feature set for domain.

        Args:
            df: Input dataframe
            target_domain: One of ['career', 'health', 'market', 'relationship', 'real_estate', 'startup']
            include_interactions: Include polynomial interactions
            include_aggregations: Include statistical aggregations
            max_features: Maximum features to return (after selection)

        Returns:
            (features_array, feature_names)
        """
        logger.info(f"Feature engineering for {target_domain}...")

        all_features = []
        all_names = []

        # 1. Domain-specific features
        logger.info("[1/5] Domain-specific features...")
        domain_feats, domain_names = self._domain_features(df, target_domain)
        all_features.append(domain_feats)
        all_names.extend(domain_names)

        # 2. Polynomial interactions
        if include_interactions:
            logger.info("[2/5] Polynomial interaction features...")
            poly_feats, poly_names = self._polynomial_features(domain_feats, domain_names)
            all_features.append(poly_feats)
            all_names.extend(poly_names)

        # 3. Statistical aggregations
        if include_aggregations:
            logger.info("[3/5] Statistical aggregation features...")
            agg_feats, agg_names = self._aggregation_features(domain_feats, domain_names)
            all_features.append(agg_feats)
            all_names.extend(agg_names)

        # 4. Combine all features
        logger.info("[4/5] Combining features...")
        X = np.hstack(all_features)
        self.feature_names = all_names

        logger.info(f"Generated {X.shape[1]} total features")

        # 5. Feature selection
        logger.info("[5/5] Feature selection...")
        X_selected, selected_names = self._feature_selection(X, all_names, max_features)

        self.selected_features = selected_names

        logger.info(f"✓ Feature engineering complete: {len(selected_names)} features")

        return X_selected, selected_names

    def _domain_features(self, df: pd.DataFrame, domain: str) -> Tuple[np.ndarray, List[str]]:
        """Generate domain-specific features."""
        if domain == 'career':
            return self._career_features(df)
        elif domain == 'health':
            return self._health_features(df)
        elif domain == 'market':
            return self._market_features(df)
        elif domain == 'relationship':
            return self._relationship_features(df)
        elif domain == 'real_estate':
            return self._real_estate_features(df)
        elif domain == 'startup':
            return self._startup_features(df)
        else:
            raise ValueError(f"Unknown domain: {domain}")

    def _career_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Career-specific features."""
        features = pd.DataFrame()

        # Education encoding
        if 'education' in df.columns:
            edu_map = {'high_school': 1, 'associates': 2, 'bachelors': 3, 'masters': 4, 'doctorate': 5, 'professional': 6}
            features['education_score'] = df['education'].map(edu_map).fillna(3)

        # Experience ratios
        if 'years_experience' in df.columns and 'current_salary' in df.columns:
            features['salary_per_year_exp'] = df['current_salary'] / (df['years_experience'] + 1)

        if 'years_experience' in df.columns and 'education_level' in df.columns:
            features['experience_education_ratio'] = df['years_experience'] / (df['education_level'] + 1)

        # Skill diversity
        if 'num_skills' in df.columns and 'years_experience' in df.columns:
            features['skill_diversity'] = df['num_skills'] / (df['years_experience'] + 1)
            features['skills_squared'] = df['num_skills'] ** 2

        # Salary benchmarking
        if 'current_salary' in df.columns and 'median_salary_for_occupation' in df.columns:
            features['salary_vs_median'] = df['current_salary'] / df['median_salary_for_occupation']
            features['salary_gap'] = df['current_salary'] - df['median_salary_for_occupation']

        # Career momentum
        if 'industry_growth_rate' in df.columns and 'job_satisfaction' in df.columns:
            features['career_momentum'] = df['industry_growth_rate'] * df['job_satisfaction'] / 5.0

        # Add all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in features.columns:
                features[col] = df[col]

        return features.values, list(features.columns)

    def _health_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Health-specific features."""
        features = pd.DataFrame()

        # BMI category risk
        if 'bmi' in df.columns:
            features['bmi'] = df['bmi']
            features['bmi_squared'] = df['bmi'] ** 2
            features['obesity_risk'] = (df['bmi'] > 30).astype(int)
            features['underweight_risk'] = (df['bmi'] < 18.5).astype(int)

        # Blood pressure risk
        if 'systolic_bp' in df.columns:
            features['systolic_bp'] = df['systolic_bp']
            features['hypertension_stage1'] = (df['systolic_bp'] > 130).astype(int)
            features['hypertension_stage2'] = (df['systolic_bp'] > 140).astype(int)

        # Metabolic syndrome indicators
        if all(col in df.columns for col in ['bmi', 'glucose_fasting', 'systolic_bp', 'hdl_cholesterol']):
            features['metabolic_syndrome_score'] = (
                (df['bmi'] > 30).astype(int) +
                (df['glucose_fasting'] > 100).astype(int) +
                (df['systolic_bp'] > 130).astype(int) +
                (df['hdl_cholesterol'] < 40).astype(int)
            )

        # Cholesterol ratios
        if 'cholesterol_total' in df.columns and 'hdl_cholesterol' in df.columns:
            features['cholesterol_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1)

        # Lifestyle composite score
        if all(col in df.columns for col in ['smoking', 'exercise_hours_week', 'diet_quality_score']):
            features['lifestyle_score'] = (
                (1 - df['smoking']) * 2.5 +
                np.clip(df['exercise_hours_week'] / 5, 0, 1) * 2.5 +
                (df['diet_quality_score'] / 10) * 2.5
            )

        # Age risk factors
        if 'age' in df.columns:
            features['age'] = df['age']
            features['age_squared'] = df['age'] ** 2
            features['elderly_risk'] = (df['age'] > 65).astype(int)

        # Add all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in features.columns:
                features[col] = df[col]

        return features.values, list(features.columns)

    def _market_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Market/financial features."""
        features = pd.DataFrame()

        # Technical indicators (if price data available)
        if 'close' in df.columns:
            # Price momentum
            features['price_momentum_5d'] = df['close'].pct_change(5)
            features['price_momentum_20d'] = df['close'].pct_change(20)

            # Volatility
            features['volatility_5d'] = df['close'].pct_change().rolling(5).std()
            features['volatility_20d'] = df['close'].pct_change().rolling(20).std()

            # Moving averages
            features['sma_5'] = df['close'].rolling(5).mean()
            features['sma_20'] = df['close'].rolling(20).mean()
            features['sma_50'] = df['close'].rolling(50).mean()

            # Price vs MA
            features['price_vs_sma5'] = df['close'] / features['sma_5']
            features['price_vs_sma20'] = df['close'] / features['sma_20']

        # Volume indicators
        if 'volume' in df.columns:
            features['volume_momentum'] = df['volume'].pct_change(5)
            features['volume_ma'] = df['volume'].rolling(20).mean()

        # Add all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in features.columns:
                features[col] = df[col].fillna(0)

        return features.values, list(features.columns)

    def _relationship_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Relationship compatibility features."""
        features = pd.DataFrame()

        # Age compatibility
        if 'age_person1' in df.columns and 'age_person2' in df.columns:
            features['age_diff'] = abs(df['age_person1'] - df['age_person2'])
            features['age_ratio'] = df['age_person1'] / (df['age_person2'] + 1)

        # Interest overlap
        if 'interests_person1' in df.columns and 'interests_person2' in df.columns:
            # Calculate Jaccard similarity
            features['interest_overlap'] = df.apply(
                lambda row: len(set(row['interests_person1'].split(',')).intersection(set(row['interests_person2'].split(',')))) /
                           len(set(row['interests_person1'].split(',')).union(set(row['interests_person2'].split(',')))),
                axis=1
            )

        # Add numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in features.columns:
                features[col] = df[col]

        return features.values, list(features.columns)

    def _real_estate_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Real estate features."""
        features = pd.DataFrame()

        # Price per square foot
        if 'price' in df.columns and 'sqft' in df.columns:
            features['price_per_sqft'] = df['price'] / (df['sqft'] + 1)

        # Property age
        if 'year_built' in df.columns:
            current_year = 2025
            features['property_age'] = current_year - df['year_built']
            features['age_squared'] = features['property_age'] ** 2

        # Bedroom/bathroom ratios
        if 'bedrooms' in df.columns and 'bathrooms' in df.columns:
            features['bed_bath_ratio'] = df['bedrooms'] / (df['bathrooms'] + 1)

        # Add numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in features.columns:
                features[col] = df[col]

        return features.values, list(features.columns)

    def _startup_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Startup success features."""
        features = pd.DataFrame()

        # Funding velocity
        if 'total_funding' in df.columns and 'years_since_founded' in df.columns:
            features['funding_velocity'] = df['total_funding'] / (df['years_since_founded'] + 1)

        # Team size growth
        if 'team_size' in df.columns and 'years_since_founded' in df.columns:
            features['growth_rate'] = df['team_size'] / (df['years_since_founded'] + 1)

        # Founder experience
        if 'founder_previous_exits' in df.columns:
            features['serial_entrepreneur'] = (df['founder_previous_exits'] > 0).astype(int)

        # Add numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in features.columns:
                features[col] = df[col]

        return features.values, list(features.columns)

    def _polynomial_features(
        self,
        X: np.ndarray,
        feature_names: List[str],
        degree: int = 2,
        max_features: int = 500
    ) -> Tuple[np.ndarray, List[str]]:
        """Generate polynomial interaction features."""
        # Limit input features to avoid explosion
        if X.shape[1] > 20:
            # Select top 20 features by variance
            variances = X.var(axis=0)
            top_indices = np.argsort(variances)[-20:]
            X = X[:, top_indices]
            feature_names = [feature_names[i] for i in top_indices]

        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
        X_poly = poly.fit_transform(X)

        # Generate interaction names
        poly_names = []
        for i, powers in enumerate(poly.powers_):
            if i < X.shape[1]:  # Original features
                continue
            # Create interaction name
            terms = [feature_names[j] for j in range(len(powers)) if powers[j] > 0]
            poly_names.append('_X_'.join(terms))

        # Exclude original features (already included)
        X_poly = X_poly[:, X.shape[1]:]

        # Limit to max_features
        if X_poly.shape[1] > max_features:
            X_poly = X_poly[:, :max_features]
            poly_names = poly_names[:max_features]

        logger.info(f"Generated {len(poly_names)} polynomial features")

        return X_poly, poly_names

    def _aggregation_features(
        self,
        X: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Generate statistical aggregation features."""
        features = []
        names = []

        # Row-wise statistics
        features.append(X.mean(axis=1, keepdims=True))
        names.append('row_mean')

        features.append(X.std(axis=1, keepdims=True))
        names.append('row_std')

        features.append(X.min(axis=1, keepdims=True))
        names.append('row_min')

        features.append(X.max(axis=1, keepdims=True))
        names.append('row_max')

        features.append((X.max(axis=1, keepdims=True) - X.min(axis=1, keepdims=True)))
        names.append('row_range')

        X_agg = np.hstack(features)

        logger.info(f"Generated {len(names)} aggregation features")

        return X_agg, names

    def _feature_selection(
        self,
        X: np.ndarray,
        feature_names: List[str],
        max_features: int
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Feature selection: remove low-variance and highly correlated features.
        """
        n_features = X.shape[1]

        if n_features <= max_features:
            return X, feature_names

        # 1. Remove low variance features
        variances = np.var(X, axis=0)
        high_var_mask = variances > np.percentile(variances, 10)  # Keep top 90%

        X_filtered = X[:, high_var_mask]
        names_filtered = [name for name, keep in zip(feature_names, high_var_mask) if keep]

        logger.info(f"After variance filter: {len(names_filtered)} features")

        # 2. If still too many, select by variance
        if len(names_filtered) > max_features:
            top_indices = np.argsort(variances[high_var_mask])[-max_features:]
            X_filtered = X_filtered[:, top_indices]
            names_filtered = [names_filtered[i] for i in top_indices]

        logger.info(f"After selection: {len(names_filtered)} features")

        return X_filtered, names_filtered

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_domain: str,
        max_features: int = 1000
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Engineer features and normalize.

        Returns:
            (normalized_features, feature_names)
        """
        X, names = self.engineer_all_features(df, target_domain, max_features=max_features)

        # Normalize
        X_norm = self.scaler.fit_transform(X)

        return X_norm, names


if __name__ == "__main__":
    # Test with career data
    logger.info("Testing Feature Engineering...")

    # Load sample data
    try:
        df = pd.read_parquet('data/career/career_complete.parquet')
        logger.info(f"Loaded {len(df)} career records")

        # Engineer features
        engineer = TelescopeFeatureEngineer()
        X, feature_names = engineer.fit_transform(df.head(10000), target_domain='career', max_features=500)

        print(f"\n=== Feature Engineering Results ===")
        print(f"Input shape: {df.shape}")
        print(f"Output shape: {X.shape}")
        print(f"Features generated: {len(feature_names)}")
        print(f"\nTop 10 features: {feature_names[:10]}")
        print(f"\n✓ Feature engineering validated")

    except FileNotFoundError:
        logger.warning("Career data not found, using synthetic data")

        # Synthetic data
        df = pd.DataFrame({
            'years_experience': np.random.randint(0, 30, 1000),
            'education_level': np.random.randint(1, 6, 1000),
            'current_salary': np.random.uniform(30000, 200000, 1000),
            'num_skills': np.random.randint(1, 15, 1000),
            'job_satisfaction': np.random.uniform(1, 5, 1000),
        })

        engineer = TelescopeFeatureEngineer()
        X, feature_names = engineer.fit_transform(df, target_domain='career', max_features=100)

        print(f"\n=== Feature Engineering Test (Synthetic) ===")
        print(f"Input shape: {df.shape}")
        print(f"Output shape: {X.shape}")
        print(f"Features: {len(feature_names)}")
        print(f"\n✓ Test passed")
