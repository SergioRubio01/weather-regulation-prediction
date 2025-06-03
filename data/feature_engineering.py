"""
Feature Engineering Module for Weather Regulation Prediction

This module provides advanced feature engineering capabilities including:
- Weather-specific feature creation
- Time-based feature engineering
- Statistical feature generation
- Domain-specific transformations
- Feature selection utilities
- Automated feature generation
- Feature importance analysis
"""

import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    SelectKBest, SelectFromModel, RFE, RFECV,
    chi2, f_classif, mutual_info_classif,
    VarianceThreshold
)
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import logging
from tqdm import tqdm
from scipy import stats
from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf

from config import DataConfig


@dataclass
class FeatureSet:
    """Container for a set of engineered features"""
    name: str
    features: pd.DataFrame
    feature_names: List[str]
    feature_types: Dict[str, str]  # feature_name -> type (numerical, categorical, etc.)
    creation_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_numerical_features(self) -> List[str]:
        """Get list of numerical features"""
        return [f for f, t in self.feature_types.items() if t == 'numerical']
        
    def get_categorical_features(self) -> List[str]:
        """Get list of categorical features"""
        return [f for f, t in self.feature_types.items() if t == 'categorical']


class WeatherFeatureEngineer(BaseEstimator, TransformerMixin):
    """Advanced weather-specific feature engineering"""
    
    def __init__(self, create_derivatives: bool = True,
                 create_interactions: bool = True,
                 create_domain_features: bool = True):
        self.create_derivatives = create_derivatives
        self.create_interactions = create_interactions
        self.create_domain_features = create_domain_features
        self.feature_names_ = []
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the feature engineer"""
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data with weather-specific features"""
        X_feat = X.copy()
        
        # 1. Derivative features
        if self.create_derivatives:
            X_feat = self._add_derivative_features(X_feat)
            
        # 2. Interaction features
        if self.create_interactions:
            X_feat = self._add_interaction_features(X_feat)
            
        # 3. Domain-specific features
        if self.create_domain_features:
            X_feat = self._add_domain_features(X_feat)
            
        self.feature_names_ = list(X_feat.columns)
        return X_feat
        
    def _add_derivative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derivative features (rate of change)"""
        derivative_cols = ['temperature', 'pressure', 'wind_speed', 'visibility']
        
        for col in derivative_cols:
            if col in df.columns:
                # First derivative (rate of change)
                df[f'{col}_diff'] = df[col].diff()
                
                # Second derivative (acceleration)
                df[f'{col}_diff2'] = df[f'{col}_diff'].diff()
                
                # Moving average of derivatives
                df[f'{col}_diff_ma3'] = df[f'{col}_diff'].rolling(3).mean()
                
        return df
        
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between weather variables"""
        # Temperature-based interactions
        if 'temperature' in df.columns and 'dewpoint' in df.columns:
            # Relative humidity approximation
            df['relative_humidity'] = 100 * np.exp(
                17.625 * df['dewpoint'] / (243.04 + df['dewpoint'])
            ) / np.exp(
                17.625 * df['temperature'] / (243.04 + df['temperature'])
            )
            
            # Temperature-dewpoint spread
            df['temp_dewpoint_spread'] = df['temperature'] - df['dewpoint']
            
        # Wind chill and heat index
        if 'temperature' in df.columns and 'wind_speed' in df.columns:
            # Wind chill (when temp < 10°C and wind > 4.8 km/h)
            df['wind_chill'] = np.where(
                (df['temperature'] < 10) & (df['wind_speed'] > 4.8),
                13.12 + 0.6215 * df['temperature'] - 
                11.37 * (df['wind_speed'] ** 0.16) + 
                0.3965 * df['temperature'] * (df['wind_speed'] ** 0.16),
                df['temperature']
            )
            
        # Pressure tendency features
        if 'pressure' in df.columns:
            df['pressure_tendency'] = df['pressure'].diff().rolling(3).mean()
            df['pressure_dropping'] = (df['pressure_tendency'] < -0.5).astype(int)
            df['pressure_rising'] = (df['pressure_tendency'] > 0.5).astype(int)
            
        # Visibility-based features
        if 'visibility' in df.columns:
            # Visibility categories
            df['visibility_cat'] = pd.cut(
                df['visibility'],
                bins=[0, 1000, 3000, 5000, 10000, np.inf],
                labels=['very_poor', 'poor', 'moderate', 'good', 'excellent']
            )
            
        # Wind features
        if 'wind_speed' in df.columns and 'wind_gust' in df.columns:
            df['gust_factor'] = df['wind_gust'] / (df['wind_speed'] + 1)
            df['turbulence'] = (df['gust_factor'] > 1.5).astype(int)
            
        return df
        
    def _add_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add aviation domain-specific features"""
        # Ceiling and visibility product (used in aviation)
        if 'ceiling' in df.columns and 'visibility' in df.columns:
            df['ceiling_visibility_product'] = df['ceiling'] * df['visibility']
            
        # Flight categories based on ceiling and visibility
        if 'ceiling' in df.columns and 'visibility' in df.columns:
            def categorize_flight_conditions(row):
                ceiling = row['ceiling']
                vis = row['visibility']
                
                if pd.isna(ceiling) or pd.isna(vis):
                    return 'unknown'
                elif ceiling < 200 or vis < 550:  # ~0.5 SM
                    return 'LIFR'  # Low IFR
                elif ceiling < 500 or vis < 1600:  # ~1 SM
                    return 'IFR'   # Instrument Flight Rules
                elif ceiling < 1000 or vis < 5000:  # ~3 SM
                    return 'MVFR'  # Marginal VFR
                else:
                    return 'VFR'   # Visual Flight Rules
                    
            df['flight_category'] = df.apply(categorize_flight_conditions, axis=1)
            
        # Crosswind component (if runway direction known)
        if 'wind_direction' in df.columns and 'wind_speed' in df.columns:
            # Assuming main runway directions (would be airport-specific)
            runway_directions = [90, 270]  # East-West runways
            
            for rwy_dir in runway_directions:
                angle_diff = np.abs(df['wind_direction'] - rwy_dir)
                angle_diff = np.minimum(angle_diff, 360 - angle_diff)
                df[f'crosswind_rwy_{rwy_dir}'] = df['wind_speed'] * np.sin(np.radians(angle_diff))
                df[f'headwind_rwy_{rwy_dir}'] = df['wind_speed'] * np.cos(np.radians(angle_diff))
                
        # Weather severity index
        df['weather_severity'] = 0
        
        if 'visibility' in df.columns:
            df['weather_severity'] += (df['visibility'] < 1000).astype(int) * 3
            df['weather_severity'] += (df['visibility'] < 3000).astype(int) * 2
            
        if 'ceiling' in df.columns:
            df['weather_severity'] += (df['ceiling'] < 200).astype(int) * 3
            df['weather_severity'] += (df['ceiling'] < 500).astype(int) * 2
            
        if 'wind_speed' in df.columns:
            df['weather_severity'] += (df['wind_speed'] > 25).astype(int) * 2
            df['weather_severity'] += (df['wind_speed'] > 35).astype(int) * 3
            
        return df


class TimeSeriesFeatureEngineer(BaseEstimator, TransformerMixin):
    """Time series specific feature engineering"""
    
    def __init__(self, lags: List[int] = [1, 2, 3, 6, 12, 24],
                 rolling_windows: List[int] = [3, 6, 12, 24],
                 create_seasonal: bool = True):
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.create_seasonal = create_seasonal
        self.feature_names_ = []
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the feature engineer"""
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data with time series features"""
        X_feat = X.copy()
        
        # Ensure datetime index
        if not isinstance(X_feat.index, pd.DatetimeIndex):
            if 'timestamp' in X_feat.columns:
                X_feat.set_index('timestamp', inplace=True)
                
        # 1. Lag features
        X_feat = self._add_lag_features(X_feat)
        
        # 2. Rolling statistics
        X_feat = self._add_rolling_features(X_feat)
        
        # 3. Seasonal features
        if self.create_seasonal:
            X_feat = self._add_seasonal_features(X_feat)
            
        # 4. Time-based features
        X_feat = self._add_time_features(X_feat)
        
        self.feature_names_ = list(X_feat.columns)
        return X_feat
        
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            for lag in self.lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                
                # Lag differences
                if lag > 1:
                    df[f'{col}_lag_diff_{lag}'] = df[col] - df[col].shift(lag)
                    
        return df
        
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window statistics"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            for window in self.rolling_windows:
                # Basic statistics
                df[f'{col}_roll_mean_{window}'] = df[col].rolling(window).mean()
                df[f'{col}_roll_std_{window}'] = df[col].rolling(window).std()
                df[f'{col}_roll_min_{window}'] = df[col].rolling(window).min()
                df[f'{col}_roll_max_{window}'] = df[col].rolling(window).max()
                
                # More advanced statistics
                df[f'{col}_roll_skew_{window}'] = df[col].rolling(window).skew()
                df[f'{col}_roll_kurt_{window}'] = df[col].rolling(window).kurt()
                
                # Trend features
                df[f'{col}_roll_trend_{window}'] = (
                    df[col] - df[f'{col}_roll_mean_{window}']
                )
                
                # Normalized by rolling std
                df[f'{col}_roll_zscore_{window}'] = (
                    df[f'{col}_roll_trend_{window}'] / 
                    (df[f'{col}_roll_std_{window}'] + 1e-8)
                )
                
        return df
        
    def _add_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add seasonal decomposition features"""
        numerical_cols = ['temperature', 'pressure', 'wind_speed']
        
        for col in numerical_cols:
            if col in df.columns and len(df) > 2 * 24:  # Need enough data
                try:
                    # Seasonal decomposition
                    decomposition = seasonal_decompose(
                        df[col].fillna(method='ffill').fillna(method='bfill'),
                        model='additive',
                        period=24  # Daily seasonality
                    )
                    
                    df[f'{col}_trend'] = decomposition.trend
                    df[f'{col}_seasonal'] = decomposition.seasonal
                    df[f'{col}_residual'] = decomposition.resid
                    
                except Exception as e:
                    logging.warning(f"Could not decompose {col}: {str(e)}")
                    
        return df
        
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        # Temporal features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['day_of_year'] = df.index.dayofyear
        df['week_of_year'] = df.index.isocalendar().week
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Time-based indicators
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        df['is_rush_hour'] = (
            ((df['hour'] >= 7) & (df['hour'] <= 9)) |
            ((df['hour'] >= 17) & (df['hour'] <= 19))
        ).astype(int)
        
        # Season
        df['season'] = pd.cut(
            df['month'],
            bins=[0, 3, 6, 9, 12],
            labels=['winter', 'spring', 'summer', 'fall']
        )
        
        return df


class StatisticalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Statistical feature engineering"""
    
    def __init__(self, create_polynomial: bool = True,
                 polynomial_degree: int = 2,
                 create_bins: bool = True,
                 n_quantiles: int = 10):
        self.create_polynomial = create_polynomial
        self.polynomial_degree = polynomial_degree
        self.create_bins = create_bins
        self.n_quantiles = n_quantiles
        self.bin_edges_ = {}
        self.feature_names_ = []
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the feature engineer"""
        if self.create_bins:
            # Calculate quantile bin edges
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            
            for col in numerical_cols:
                if X[col].nunique() > self.n_quantiles:
                    _, edges = pd.qcut(
                        X[col].dropna(),
                        q=self.n_quantiles,
                        retbins=True,
                        duplicates='drop'
                    )
                    self.bin_edges_[col] = edges
                    
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data with statistical features"""
        X_feat = X.copy()
        
        # 1. Polynomial features
        if self.create_polynomial:
            X_feat = self._add_polynomial_features(X_feat)
            
        # 2. Binned features
        if self.create_bins:
            X_feat = self._add_binned_features(X_feat)
            
        # 3. Statistical aggregates
        X_feat = self._add_statistical_features(X_feat)
        
        self.feature_names_ = list(X_feat.columns)
        return X_feat
        
    def _add_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add polynomial features for important variables"""
        poly_cols = ['temperature', 'pressure', 'wind_speed']
        
        for col in poly_cols:
            if col in df.columns:
                for degree in range(2, self.polynomial_degree + 1):
                    df[f'{col}_pow_{degree}'] = df[col] ** degree
                    
                # Square root and log transforms
                df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
                df[f'{col}_log'] = np.log1p(np.abs(df[col]))
                
        return df
        
    def _add_binned_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add binned/discretized features"""
        for col, edges in self.bin_edges_.items():
            if col in df.columns:
                # Create bins
                df[f'{col}_bin'] = pd.cut(
                    df[col],
                    bins=edges,
                    labels=range(len(edges) - 1),
                    include_lowest=True
                )
                
                # One-hot encode bins
                bin_dummies = pd.get_dummies(
                    df[f'{col}_bin'],
                    prefix=f'{col}_bin'
                )
                df = pd.concat([df, bin_dummies], axis=1)
                
        return df
        
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical aggregate features"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        # Row-wise statistics
        if len(numerical_cols) > 3:
            df['row_mean'] = df[numerical_cols].mean(axis=1)
            df['row_std'] = df[numerical_cols].std(axis=1)
            df['row_max'] = df[numerical_cols].max(axis=1)
            df['row_min'] = df[numerical_cols].min(axis=1)
            df['row_range'] = df['row_max'] - df['row_min']
            
        # Entropy-like features
        for col in ['wind_direction', 'cloud_coverage']:
            if col in df.columns:
                # Shannon entropy of recent values
                window = 24
                df[f'{col}_entropy'] = df[col].rolling(window).apply(
                    lambda x: stats.entropy(x.value_counts() + 1e-8)
                )
                
        return df


class AutomatedFeatureEngineer:
    """Automated feature engineering with feature selection"""
    
    def __init__(self, config: DataConfig,
                 max_features: Optional[int] = None,
                 selection_method: str = 'mutual_info'):
        self.config = config
        self.max_features = max_features
        self.selection_method = selection_method
        self.engineers = []
        self.selector = None
        self.selected_features = None
        self.feature_importance = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform with automated feature engineering"""
        self.logger.info("Starting automated feature engineering...")
        
        # 1. Apply all feature engineers
        X_feat = self._apply_engineers(X)
        
        # 2. Handle missing values created by feature engineering
        X_feat = self._handle_missing(X_feat)
        
        # 3. Select best features
        X_selected = self._select_features(X_feat, y)
        
        self.logger.info(f"Created {len(X_feat.columns)} features, selected {len(X_selected.columns)}")
        
        return X_selected
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted engineers and selector"""
        # Apply engineers
        X_feat = X.copy()
        for engineer in self.engineers:
            X_feat = engineer.transform(X_feat)
            
        # Handle missing values
        X_feat = self._handle_missing(X_feat)
        
        # Select features
        if self.selected_features is not None:
            X_feat = X_feat[self.selected_features]
            
        return X_feat
        
    def _apply_engineers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering transformations"""
        X_feat = X.copy()
        
        # Initialize engineers
        self.engineers = [
            WeatherFeatureEngineer(
                create_derivatives=True,
                create_interactions=True,
                create_domain_features=True
            ),
            TimeSeriesFeatureEngineer(
                lags=[1, 3, 6, 12, 24],
                rolling_windows=[6, 12, 24],
                create_seasonal=True
            ),
            StatisticalFeatureEngineer(
                create_polynomial=True,
                polynomial_degree=2,
                create_bins=True,
                n_quantiles=5
            )
        ]
        
        # Apply each engineer
        for engineer in self.engineers:
            self.logger.info(f"Applying {engineer.__class__.__name__}...")
            engineer.fit(X_feat)
            X_feat = engineer.transform(X_feat)
            
        return X_feat
        
    def _handle_missing(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values created by feature engineering"""
        # Forward fill for time series features
        time_cols = [col for col in X.columns if any(
            pattern in col for pattern in ['lag_', 'roll_', 'diff']
        )]
        X[time_cols] = X[time_cols].fillna(method='ffill').fillna(method='bfill')
        
        # Fill remaining with median
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())
        
        # Drop columns with too many missing values
        missing_threshold = 0.5
        missing_ratio = X.isnull().sum() / len(X)
        cols_to_drop = missing_ratio[missing_ratio > missing_threshold].index
        if len(cols_to_drop) > 0:
            self.logger.info(f"Dropping {len(cols_to_drop)} columns with >50% missing values")
            X = X.drop(columns=cols_to_drop)
            
        return X
        
    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Select best features using specified method"""
        self.logger.info(f"Selecting features using {self.selection_method}...")
        
        # Remove non-numeric columns for selection
        X_numeric = X.select_dtypes(include=[np.number])
        
        if self.selection_method == 'mutual_info':
            selector = SelectKBest(
                score_func=mutual_info_classif,
                k=min(self.max_features or len(X_numeric.columns), len(X_numeric.columns))
            )
        elif self.selection_method == 'f_classif':
            selector = SelectKBest(
                score_func=f_classif,
                k=min(self.max_features or len(X_numeric.columns), len(X_numeric.columns))
            )
        elif self.selection_method == 'random_forest':
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            selector = SelectFromModel(
                rf,
                max_features=self.max_features,
                threshold='median'
            )
        elif self.selection_method == 'rfe':
            rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            selector = RFE(
                rf,
                n_features_to_select=self.max_features or len(X_numeric.columns) // 2,
                step=0.1
            )
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
            
        # Fit selector
        X_selected = selector.fit_transform(X_numeric, y)
        
        # Get selected features
        if hasattr(selector, 'get_support'):
            feature_mask = selector.get_support()
            self.selected_features = X_numeric.columns[feature_mask].tolist()
        else:
            self.selected_features = X_numeric.columns.tolist()
            
        # Calculate feature importance
        if self.selection_method == 'random_forest':
            self.feature_importance = pd.DataFrame({
                'feature': X_numeric.columns,
                'importance': selector.estimator_.feature_importances_
            }).sort_values('importance', ascending=False)
        elif hasattr(selector, 'scores_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_numeric.columns,
                'score': selector.scores_
            }).sort_values('score', ascending=False)
            
        self.selector = selector
        
        return X[self.selected_features]
        
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores"""
        return self.feature_importance
        
    def save_feature_pipeline(self, path: str) -> None:
        """Save the feature engineering pipeline"""
        import joblib
        
        pipeline_data = {
            'engineers': self.engineers,
            'selector': self.selector,
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(pipeline_data, path)
        self.logger.info(f"Saved feature pipeline to {path}")
        
    def load_feature_pipeline(self, path: str) -> None:
        """Load a saved feature engineering pipeline"""
        import joblib
        
        pipeline_data = joblib.load(path)
        self.engineers = pipeline_data['engineers']
        self.selector = pipeline_data['selector']
        self.selected_features = pipeline_data['selected_features']
        self.feature_importance = pipeline_data['feature_importance']
        
        self.logger.info(f"Loaded feature pipeline from {path}")


class FeatureInteractionAnalyzer:
    """Analyze feature interactions and relationships"""
    
    def __init__(self):
        self.interaction_scores = None
        self.correlation_matrix = None
        self.vif_scores = None
        
    def analyze_interactions(self, X: pd.DataFrame, y: pd.Series,
                           top_k: int = 20) -> pd.DataFrame:
        """Analyze feature interactions"""
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.feature_selection import f_classif
        
        # Select numerical features
        X_numeric = X.select_dtypes(include=[np.number])
        
        # Create pairwise interactions
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_interactions = poly.fit_transform(X_numeric)
        
        # Get feature names
        feature_names = poly.get_feature_names_out(X_numeric.columns)
        
        # Calculate interaction scores
        scores, _ = f_classif(X_interactions, y)
        
        # Create results dataframe
        interaction_df = pd.DataFrame({
            'feature': feature_names,
            'score': scores
        }).sort_values('score', ascending=False)
        
        # Filter to only interaction terms
        interaction_df = interaction_df[
            interaction_df['feature'].str.contains(' ')
        ].head(top_k)
        
        self.interaction_scores = interaction_df
        return interaction_df
        
    def analyze_correlations(self, X: pd.DataFrame,
                           threshold: float = 0.8) -> pd.DataFrame:
        """Analyze feature correlations"""
        # Calculate correlation matrix
        X_numeric = X.select_dtypes(include=[np.number])
        self.correlation_matrix = X_numeric.corr()
        
        # Find highly correlated features
        upper_tri = np.triu(np.ones_like(self.correlation_matrix), k=1).astype(bool)
        upper_tri_corr = self.correlation_matrix.where(upper_tri)
        
        # Get pairs with high correlation
        high_corr_pairs = []
        for col in upper_tri_corr.columns:
            for row in upper_tri_corr.index:
                corr_value = upper_tri_corr.loc[row, col]
                if abs(corr_value) > threshold:
                    high_corr_pairs.append({
                        'feature_1': row,
                        'feature_2': col,
                        'correlation': corr_value
                    })
                    
        return pd.DataFrame(high_corr_pairs).sort_values(
            'correlation', key=abs, ascending=False
        )
        
    def calculate_vif(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate Variance Inflation Factor for multicollinearity"""
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        X_numeric = X.select_dtypes(include=[np.number]).dropna()
        
        vif_data = []
        for i in range(X_numeric.shape[1]):
            vif_data.append({
                'feature': X_numeric.columns[i],
                'VIF': variance_inflation_factor(X_numeric.values, i)
            })
            
        self.vif_scores = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
        return self.vif_scores


# Utility functions for feature engineering
def create_feature_report(X: pd.DataFrame, y: pd.Series,
                         output_path: Optional[str] = None) -> Dict[str, Any]:
    """Create comprehensive feature engineering report"""
    report = {
        'n_samples': len(X),
        'n_features': len(X.columns),
        'feature_types': {},
        'missing_values': {},
        'unique_values': {},
        'correlations': {},
        'importance_scores': {}
    }
    
    # Analyze each feature
    for col in X.columns:
        # Type
        if pd.api.types.is_numeric_dtype(X[col]):
            report['feature_types'][col] = 'numerical'
        else:
            report['feature_types'][col] = 'categorical'
            
        # Missing values
        report['missing_values'][col] = X[col].isnull().sum()
        
        # Unique values
        report['unique_values'][col] = X[col].nunique()
        
    # Feature importance using mutual information
    X_numeric = X.select_dtypes(include=[np.number])
    if len(X_numeric.columns) > 0:
        mi_scores = mutual_info_classif(X_numeric.fillna(0), y)
        for col, score in zip(X_numeric.columns, mi_scores):
            report['importance_scores'][col] = score
            
    # Save report if path provided
    if output_path:
        import json
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
    return report


def engineer_features_for_model(X: pd.DataFrame, y: pd.Series,
                              model_type: str,
                              config: DataConfig) -> pd.DataFrame:
    """Engineer features optimized for specific model type"""
    if model_type in ['lstm', 'gru', 'rnn', 'cnn']:
        # Deep learning models - focus on sequential features
        engineer = TimeSeriesFeatureEngineer(
            lags=[1, 2, 3, 6, 12],
            rolling_windows=[3, 6, 12],
            create_seasonal=False  # Let the model learn seasonality
        )
    elif model_type in ['random_forest', 'xgboost']:
        # Tree-based models - can handle many features
        engineer = AutomatedFeatureEngineer(
            config=config,
            max_features=None,  # Let model handle feature selection
            selection_method='random_forest'
        )
        return engineer.fit_transform(X, y)
    else:
        # Linear models - need good features with less multicollinearity
        engineer = StatisticalFeatureEngineer(
            create_polynomial=True,
            polynomial_degree=2,
            create_bins=False
        )
        
    return engineer.fit_transform(X)


def create_synthetic_features(X: pd.DataFrame,
                            n_synthetic: int = 10,
                            method: str = 'random_projection') -> pd.DataFrame:
    """Create synthetic features using dimensionality expansion"""
    from sklearn.random_projection import GaussianRandomProjection
    from sklearn.kernel_approximation import RBFSampler
    
    X_numeric = X.select_dtypes(include=[np.number])
    
    if method == 'random_projection':
        transformer = GaussianRandomProjection(n_components=n_synthetic, random_state=42)
    elif method == 'rbf':
        transformer = RBFSampler(n_components=n_synthetic, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
        
    X_synthetic = transformer.fit_transform(X_numeric)
    
    # Create dataframe with synthetic features
    synthetic_df = pd.DataFrame(
        X_synthetic,
        columns=[f'synthetic_{i}' for i in range(n_synthetic)],
        index=X.index
    )
    
    return pd.concat([X, synthetic_df], axis=1)