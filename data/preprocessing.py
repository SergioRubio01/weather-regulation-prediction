"""
Advanced Preprocessing Module for Weather Regulation Prediction

This module provides modular preprocessing pipelines including:
- Feature scaling and normalization
- Encoding categorical variables
- Time series specific preprocessing
- Feature selection and dimensionality reduction
- Data quality improvements
- Pipeline composition and management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder,
    PolynomialFeatures, KBinsDiscretizer
)
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, NMF
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, SelectFromModel,
    mutual_info_classif, mutual_info_regression,
    f_classif, chi2, RFE, RFECV
)
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from pathlib import Path
import logging
import warnings
from functools import partial
from scipy import signal
from scipy.stats import boxcox, yeojohnson
import category_encoders as ce

from config import ExperimentConfig, DataConfig


class TimeSeriesScaler(BaseEstimator, TransformerMixin):
    """Custom scaler for time series data that preserves temporal relationships"""
    
    def __init__(self, method: str = 'standard', window_size: Optional[int] = None):
        self.method = method
        self.window_size = window_size
        self.scalers_ = {}
        
    def fit(self, X: np.ndarray, y=None) -> 'TimeSeriesScaler':
        """Fit the scaler"""
        if self.window_size:
            # Fit separate scalers for different time windows
            n_windows = len(X) // self.window_size + 1
            
            for i in range(n_windows):
                start_idx = i * self.window_size
                end_idx = min((i + 1) * self.window_size, len(X))
                
                if self.method == 'standard':
                    scaler = StandardScaler()
                elif self.method == 'minmax':
                    scaler = MinMaxScaler()
                elif self.method == 'robust':
                    scaler = RobustScaler()
                else:
                    raise ValueError(f"Unknown method: {self.method}")
                    
                scaler.fit(X[start_idx:end_idx])
                self.scalers_[i] = scaler
        else:
            # Fit single scaler
            if self.method == 'standard':
                self.scalers_[0] = StandardScaler().fit(X)
            elif self.method == 'minmax':
                self.scalers_[0] = MinMaxScaler().fit(X)
            elif self.method == 'robust':
                self.scalers_[0] = RobustScaler().fit(X)
                
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data"""
        if self.window_size:
            # Transform using window-specific scalers
            X_scaled = np.zeros_like(X)
            n_windows = len(X) // self.window_size + 1
            
            for i in range(n_windows):
                start_idx = i * self.window_size
                end_idx = min((i + 1) * self.window_size, len(X))
                
                if i in self.scalers_:
                    X_scaled[start_idx:end_idx] = self.scalers_[i].transform(X[start_idx:end_idx])
                else:
                    # Use last available scaler
                    last_scaler = self.scalers_[max(self.scalers_.keys())]
                    X_scaled[start_idx:end_idx] = last_scaler.transform(X[start_idx:end_idx])
                    
            return X_scaled
        else:
            return self.scalers_[0].transform(X)
            
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform the data"""
        if self.window_size:
            X_original = np.zeros_like(X)
            n_windows = len(X) // self.window_size + 1
            
            for i in range(n_windows):
                start_idx = i * self.window_size
                end_idx = min((i + 1) * self.window_size, len(X))
                
                if i in self.scalers_:
                    X_original[start_idx:end_idx] = self.scalers_[i].inverse_transform(X[start_idx:end_idx])
                else:
                    last_scaler = self.scalers_[max(self.scalers_.keys())]
                    X_original[start_idx:end_idx] = last_scaler.inverse_transform(X[start_idx:end_idx])
                    
            return X_original
        else:
            return self.scalers_[0].inverse_transform(X)


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """Encode cyclical features using sin/cos transformation"""
    
    def __init__(self, features_periods: Dict[str, int]):
        """
        Args:
            features_periods: Dictionary mapping feature names to their periods
                             e.g., {'hour': 24, 'month': 12, 'day_of_week': 7}
        """
        self.features_periods = features_periods
        
    def fit(self, X: pd.DataFrame, y=None) -> 'CyclicalEncoder':
        """Fit the encoder"""
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform cyclical features"""
        X_transformed = X.copy()
        
        for feature, period in self.features_periods.items():
            if feature in X.columns:
                X_transformed[f'{feature}_sin'] = np.sin(2 * np.pi * X[feature] / period)
                X_transformed[f'{feature}_cos'] = np.cos(2 * np.pi * X[feature] / period)
                X_transformed.drop(feature, axis=1, inplace=True)
                
        return X_transformed


class LagFeatureCreator(BaseEstimator, TransformerMixin):
    """Create lag features for time series data"""
    
    def __init__(self, features: List[str], lags: List[int], 
                 rolling_windows: Optional[List[int]] = None,
                 rolling_funcs: Optional[List[str]] = None):
        self.features = features
        self.lags = lags
        self.rolling_windows = rolling_windows or []
        self.rolling_funcs = rolling_funcs or ['mean', 'std', 'min', 'max']
        
    def fit(self, X: pd.DataFrame, y=None) -> 'LagFeatureCreator':
        """Fit the transformer"""
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create lag features"""
        X_transformed = X.copy()
        
        for feature in self.features:
            if feature in X.columns:
                # Create lag features
                for lag in self.lags:
                    X_transformed[f'{feature}_lag_{lag}'] = X[feature].shift(lag)
                    
                # Create rolling features
                for window in self.rolling_windows:
                    for func in self.rolling_funcs:
                        if func == 'mean':
                            X_transformed[f'{feature}_rolling_{func}_{window}'] = X[feature].rolling(window).mean()
                        elif func == 'std':
                            X_transformed[f'{feature}_rolling_{func}_{window}'] = X[feature].rolling(window).std()
                        elif func == 'min':
                            X_transformed[f'{feature}_rolling_{func}_{window}'] = X[feature].rolling(window).min()
                        elif func == 'max':
                            X_transformed[f'{feature}_rolling_{func}_{window}'] = X[feature].rolling(window).max()
                        elif func == 'median':
                            X_transformed[f'{feature}_rolling_{func}_{window}'] = X[feature].rolling(window).median()
                            
        return X_transformed


class OutlierDetector(BaseEstimator, TransformerMixin):
    """Detect and handle outliers in the data"""
    
    def __init__(self, method: str = 'iqr', threshold: float = 1.5,
                 contamination: float = 0.1, handling: str = 'clip'):
        self.method = method
        self.threshold = threshold
        self.contamination = contamination
        self.handling = handling
        self.bounds_ = {}
        
    def fit(self, X: pd.DataFrame, y=None) -> 'OutlierDetector':
        """Fit the outlier detector"""
        if self.method == 'iqr':
            # IQR method
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            
            self.bounds_['lower'] = Q1 - self.threshold * IQR
            self.bounds_['upper'] = Q3 + self.threshold * IQR
            
        elif self.method == 'zscore':
            # Z-score method
            mean = X.mean()
            std = X.std()
            
            self.bounds_['lower'] = mean - self.threshold * std
            self.bounds_['upper'] = mean + self.threshold * std
            
        elif self.method == 'isolation_forest':
            # Use Isolation Forest
            from sklearn.ensemble import IsolationForest
            
            self.detector_ = IsolationForest(
                contamination=self.contamination,
                random_state=42
            )
            self.detector_.fit(X)
            
        elif self.method == 'local_outlier_factor':
            # Use Local Outlier Factor
            from sklearn.neighbors import LocalOutlierFactor
            
            self.detector_ = LocalOutlierFactor(
                contamination=self.contamination,
                novelty=True
            )
            self.detector_.fit(X)
            
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by handling outliers"""
        X_transformed = X.copy()
        
        if self.method in ['iqr', 'zscore']:
            # Handle outliers based on bounds
            if self.handling == 'clip':
                # Clip values to bounds
                for col in X.columns:
                    if col in self.bounds_['lower'].index:
                        X_transformed[col] = X_transformed[col].clip(
                            lower=self.bounds_['lower'][col],
                            upper=self.bounds_['upper'][col]
                        )
            elif self.handling == 'remove':
                # Mark outliers as NaN
                for col in X.columns:
                    if col in self.bounds_['lower'].index:
                        mask = (X_transformed[col] < self.bounds_['lower'][col]) | \
                               (X_transformed[col] > self.bounds_['upper'][col])
                        X_transformed.loc[mask, col] = np.nan
                        
        elif self.method in ['isolation_forest', 'local_outlier_factor']:
            # Use fitted detector
            outliers = self.detector_.predict(X) == -1
            
            if self.handling == 'remove':
                X_transformed[outliers] = np.nan
            elif self.handling == 'clip':
                # For tree-based methods, clip to percentiles
                for col in X.columns:
                    X_transformed.loc[outliers, col] = X_transformed[col].clip(
                        lower=X_transformed[col].quantile(0.01),
                        upper=X_transformed[col].quantile(0.99)
                    )
                    
        return X_transformed


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Advanced feature selection with multiple methods"""
    
    def __init__(self, method: str = 'mutual_info', n_features: Union[int, float] = 0.8,
                 task: str = 'classification', cv: int = 5):
        self.method = method
        self.n_features = n_features
        self.task = task
        self.cv = cv
        self.selector_ = None
        self.selected_features_ = None
        
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'FeatureSelector':
        """Fit the feature selector"""
        if self.method == 'mutual_info':
            # Mutual information
            if self.task == 'classification':
                score_func = mutual_info_classif
            else:
                score_func = mutual_info_regression
                
            if isinstance(self.n_features, int):
                self.selector_ = SelectKBest(score_func=score_func, k=self.n_features)
            else:
                self.selector_ = SelectPercentile(score_func=score_func, 
                                                percentile=int(self.n_features * 100))
                                                
        elif self.method == 'f_score':
            # F-score
            if self.task == 'classification':
                score_func = f_classif
            else:
                from sklearn.feature_selection import f_regression
                score_func = f_regression
                
            if isinstance(self.n_features, int):
                self.selector_ = SelectKBest(score_func=score_func, k=self.n_features)
            else:
                self.selector_ = SelectPercentile(score_func=score_func,
                                                percentile=int(self.n_features * 100))
                                                
        elif self.method == 'chi2':
            # Chi-squared (only for non-negative features)
            if isinstance(self.n_features, int):
                self.selector_ = SelectKBest(score_func=chi2, k=self.n_features)
            else:
                self.selector_ = SelectPercentile(score_func=chi2,
                                                percentile=int(self.n_features * 100))
                                                
        elif self.method == 'rfe':
            # Recursive Feature Elimination
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            
            if self.task == 'classification':
                estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=100, random_state=42)
                
            n_features_to_select = self.n_features if isinstance(self.n_features, int) \
                                  else int(self.n_features * X.shape[1])
                                  
            self.selector_ = RFE(estimator=estimator, 
                               n_features_to_select=n_features_to_select)
                               
        elif self.method == 'rfecv':
            # RFE with cross-validation
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            
            if self.task == 'classification':
                estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=100, random_state=42)
                
            self.selector_ = RFECV(estimator=estimator, cv=self.cv)
            
        elif self.method == 'lasso':
            # Lasso-based selection
            from sklearn.linear_model import LassoCV, LogisticRegressionCV
            
            if self.task == 'classification':
                estimator = LogisticRegressionCV(cv=self.cv, penalty='l1', 
                                               solver='liblinear', random_state=42)
            else:
                estimator = LassoCV(cv=self.cv, random_state=42)
                
            self.selector_ = SelectFromModel(estimator=estimator)
            
        elif self.method == 'tree_based':
            # Tree-based feature importance
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            
            if self.task == 'classification':
                estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=100, random_state=42)
                
            self.selector_ = SelectFromModel(estimator=estimator, 
                                           threshold='median')
        
        # Fit the selector
        self.selector_.fit(X, y)
        
        # Get selected features
        feature_mask = self.selector_.get_support()
        self.selected_features_ = X.columns[feature_mask].tolist()
        
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by selecting features"""
        if self.selector_ is None:
            raise ValueError("FeatureSelector must be fitted before transform")
            
        X_selected = self.selector_.transform(X)
        
        # Convert back to DataFrame with selected feature names
        return pd.DataFrame(X_selected, 
                          columns=self.selected_features_,
                          index=X.index)


class PreprocessingPipeline:
    """Main preprocessing pipeline manager"""
    
    def __init__(self, config: Union[DataConfig, ExperimentConfig]):
        self.config = config.data if isinstance(config, ExperimentConfig) else config
        self.logger = self._setup_logger()
        self.pipeline = None
        self.feature_names_ = None
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def create_pipeline(self, 
                       numerical_features: List[str],
                       categorical_features: List[str],
                       cyclical_features: Optional[Dict[str, int]] = None,
                       lag_features: Optional[Dict[str, List[int]]] = None,
                       feature_selection: bool = True,
                       task: str = 'classification') -> Pipeline:
        """
        Create preprocessing pipeline
        
        Args:
            numerical_features: List of numerical feature names
            categorical_features: List of categorical feature names
            cyclical_features: Dict of cyclical features and their periods
            lag_features: Dict of features and their lag values
            feature_selection: Whether to perform feature selection
            task: 'classification' or 'regression'
            
        Returns:
            Sklearn Pipeline object
        """
        self.logger.info("Creating preprocessing pipeline")
        
        steps = []
        
        # 1. Handle missing values
        if self.config.handle_missing:
            self.logger.info("Adding missing value imputation")
            
            # Numerical imputation
            num_imputer = SimpleImputer(strategy='median')
            
            # Categorical imputation
            cat_imputer = SimpleImputer(strategy='most_frequent')
            
            # Combine imputers
            from sklearn.compose import ColumnTransformer
            
            imputation = ColumnTransformer([
                ('num_imputer', num_imputer, numerical_features),
                ('cat_imputer', cat_imputer, categorical_features)
            ], remainder='passthrough')
            
            steps.append(('imputation', imputation))
            
        # 2. Handle outliers
        if self.config.outlier_method:
            self.logger.info(f"Adding outlier detection: {self.config.outlier_method}")
            outlier_detector = OutlierDetector(
                method=self.config.outlier_method,
                threshold=self.config.outlier_threshold
            )
            steps.append(('outlier_detection', outlier_detector))
            
        # 3. Create lag features
        if lag_features:
            self.logger.info("Adding lag feature creation")
            for feature, lags in lag_features.items():
                lag_creator = LagFeatureCreator(
                    features=[feature],
                    lags=lags,
                    rolling_windows=[3, 6, 12, 24]
                )
                steps.append((f'lag_{feature}', lag_creator))
                
        # 4. Encode cyclical features
        if cyclical_features:
            self.logger.info("Adding cyclical encoding")
            cyclical_encoder = CyclicalEncoder(cyclical_features)
            steps.append(('cyclical_encoding', cyclical_encoder))
            
        # 5. Encode categorical variables
        if categorical_features:
            self.logger.info("Encoding categorical variables")
            
            # Use different encoders based on cardinality
            from sklearn.compose import ColumnTransformer
            
            # Get cardinality of categorical features
            high_cardinality = []
            low_cardinality = []
            
            for feat in categorical_features:
                # This is a simplification - in practice, you'd check actual cardinality
                if feat in ['airport', 'city', 'country']:
                    high_cardinality.append(feat)
                else:
                    low_cardinality.append(feat)
                    
            encoders = []
            
            if low_cardinality:
                encoders.append(('onehot', OneHotEncoder(drop='first', sparse=False), 
                               low_cardinality))
                               
            if high_cardinality:
                # Use target encoding for high cardinality
                if task == 'classification':
                    target_encoder = ce.TargetEncoder()
                else:
                    target_encoder = ce.TargetEncoder()
                encoders.append(('target', target_encoder, high_cardinality))
                
            if encoders:
                encoding = ColumnTransformer(encoders, remainder='passthrough')
                steps.append(('encoding', encoding))
                
        # 6. Scale numerical features
        if self.config.scaling_method:
            self.logger.info(f"Adding feature scaling: {self.config.scaling_method}")
            
            if self.config.scaling_method == 'standard':
                scaler = StandardScaler()
            elif self.config.scaling_method == 'minmax':
                scaler = MinMaxScaler()
            elif self.config.scaling_method == 'robust':
                scaler = RobustScaler()
            elif self.config.scaling_method == 'maxabs':
                scaler = MaxAbsScaler()
            elif self.config.scaling_method == 'time_series':
                scaler = TimeSeriesScaler(method='standard', window_size=24*7)  # Weekly windows
            else:
                scaler = StandardScaler()
                
            steps.append(('scaling', scaler))
            
        # 7. Feature transformation
        if self.config.feature_transform:
            self.logger.info("Adding feature transformations")
            
            if self.config.feature_transform == 'polynomial':
                poly_features = PolynomialFeatures(degree=2, include_bias=False)
                steps.append(('polynomial', poly_features))
                
            elif self.config.feature_transform == 'log':
                # Custom log transformer
                from sklearn.preprocessing import FunctionTransformer
                log_transformer = FunctionTransformer(
                    func=lambda X: np.log1p(np.abs(X)),
                    validate=False
                )
                steps.append(('log_transform', log_transformer))
                
        # 8. Dimensionality reduction
        if self.config.dim_reduction:
            self.logger.info(f"Adding dimensionality reduction: {self.config.dim_reduction}")
            
            if self.config.dim_reduction == 'pca':
                reducer = PCA(n_components=0.95)  # Keep 95% variance
            elif self.config.dim_reduction == 'truncated_svd':
                reducer = TruncatedSVD(n_components=50)
            elif self.config.dim_reduction == 'ica':
                reducer = FastICA(n_components=30, random_state=42)
            elif self.config.dim_reduction == 'nmf':
                reducer = NMF(n_components=30, random_state=42)
            else:
                reducer = PCA(n_components=0.95)
                
            steps.append(('dim_reduction', reducer))
            
        # 9. Feature selection
        if feature_selection and self.config.feature_selection:
            self.logger.info(f"Adding feature selection: {self.config.feature_selection}")
            
            selector = FeatureSelector(
                method=self.config.feature_selection,
                n_features=0.8,  # Keep 80% of features
                task=task
            )
            steps.append(('feature_selection', selector))
            
        # Create pipeline
        self.pipeline = Pipeline(steps)
        
        return self.pipeline
        
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'PreprocessingPipeline':
        """Fit the preprocessing pipeline"""
        if self.pipeline is None:
            raise ValueError("Pipeline not created. Call create_pipeline() first")
            
        self.logger.info(f"Fitting preprocessing pipeline on {X.shape} data")
        self.pipeline.fit(X, y)
        
        # Store feature names after transformation
        self._extract_feature_names(X)
        
        return self
        
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform the data"""
        if self.pipeline is None:
            raise ValueError("Pipeline not created. Call create_pipeline() first")
            
        self.logger.info(f"Transforming {X.shape} data")
        return self.pipeline.transform(X)
        
    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Fit and transform the data"""
        self.fit(X, y)
        return self.transform(X)
        
    def _extract_feature_names(self, X: pd.DataFrame) -> None:
        """Extract feature names after transformation"""
        # This is complex due to various transformers
        # For now, we'll store original names
        self.feature_names_ = list(X.columns)
        
    def save(self, filepath: str) -> None:
        """Save the fitted pipeline"""
        if self.pipeline is None:
            raise ValueError("No pipeline to save")
            
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'pipeline': self.pipeline,
            'feature_names': self.feature_names_,
            'config': self.config
        }, filepath)
        
        self.logger.info(f"Pipeline saved to {filepath}")
        
    def load(self, filepath: str) -> None:
        """Load a fitted pipeline"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Pipeline file not found: {filepath}")
            
        data = joblib.load(filepath)
        self.pipeline = data['pipeline']
        self.feature_names_ = data['feature_names']
        self.config = data['config']
        
        self.logger.info(f"Pipeline loaded from {filepath}")


def create_preprocessing_pipeline(config: Union[str, ExperimentConfig],
                                feature_info: Dict[str, List[str]],
                                task: str = 'classification') -> PreprocessingPipeline:
    """
    Quick function to create preprocessing pipeline
    
    Args:
        config: Configuration file path or ExperimentConfig object
        feature_info: Dictionary with feature type information
                     {'numerical': [...], 'categorical': [...], 'cyclical': {...}}
        task: 'classification' or 'regression'
        
    Returns:
        PreprocessingPipeline object
    """
    if isinstance(config, str):
        from config_parser import ConfigParser
        parser = ConfigParser()
        config = parser.load_config(config)
        
    preprocessor = PreprocessingPipeline(config)
    
    pipeline = preprocessor.create_pipeline(
        numerical_features=feature_info.get('numerical', []),
        categorical_features=feature_info.get('categorical', []),
        cyclical_features=feature_info.get('cyclical', None),
        lag_features=feature_info.get('lag_features', None),
        feature_selection=True,
        task=task
    )
    
    return preprocessor


def preprocess_weather_data(data: pd.DataFrame, 
                          config: Union[str, ExperimentConfig],
                          target_column: str = 'has_regulation') -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess weather data with automatic feature detection
    
    Args:
        data: Raw feature DataFrame
        config: Configuration
        target_column: Name of target column
        
    Returns:
        Tuple of (X_processed, y)
    """
    # Separate features and target
    y = data[target_column].values
    X = data.drop(columns=[target_column])
    
    # Detect feature types
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Define cyclical features
    cyclical_features = {
        'hour': 24,
        'day_of_week': 7,
        'month': 12
    }
    
    # Define lag features for key weather variables
    lag_features = {
        'temperature_c': [1, 3, 6, 12, 24],
        'wind_speed_kt': [1, 3, 6, 12],
        'visibility_m': [1, 3, 6],
        'pressure_hpa': [1, 3, 6, 12, 24]
    }
    
    # Create feature info
    feature_info = {
        'numerical': numerical_features,
        'categorical': categorical_features,
        'cyclical': cyclical_features,
        'lag_features': lag_features
    }
    
    # Create and fit pipeline
    preprocessor = create_preprocessing_pipeline(config, feature_info)
    X_processed = preprocessor.fit_transform(X, y)
    
    return X_processed, y