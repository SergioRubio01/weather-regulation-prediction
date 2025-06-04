"""
Data Validation Module for Weather Regulation Prediction

This module provides comprehensive data validation and quality checks including:
- Schema validation
- Data type validation
- Range and constraint validation
- Consistency checks
- Anomaly detection
- Data drift detection
- Missing value analysis
- Automated data quality reporting
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from config import DataConfig


@dataclass
class ValidationResult:
    """Container for validation results"""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "statistics": self.statistics,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat(),
        }

    def save(self, path: str) -> None:
        """Save validation results"""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


@dataclass
class DataSchema:
    """Schema definition for data validation"""

    columns: dict[str, str]  # column_name -> data_type
    required_columns: list[str]
    constraints: dict[str, dict[str, Any]]  # column_name -> {min, max, values, etc.}
    datetime_columns: list[str]
    categorical_columns: list[str]
    numerical_columns: list[str]

    @classmethod
    def from_yaml(cls, path: str) -> "DataSchema":
        """Load schema from YAML file"""
        with open(path) as f:
            schema_dict = yaml.safe_load(f)
        return cls(**schema_dict)

    def to_yaml(self, path: str) -> None:
        """Save schema to YAML file"""
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)


class BaseValidator(ABC):
    """Abstract base class for validators"""

    def __init__(self, config: DataConfig | None = None):
        self.config = config
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        return logger

    @abstractmethod
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate data"""
        pass


class SchemaValidator(BaseValidator):
    """Validate data against a defined schema"""

    def __init__(self, schema: DataSchema, **kwargs):
        super().__init__(**kwargs)
        self.schema = schema

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate data against schema"""
        result = ValidationResult(is_valid=True)

        # Check required columns
        missing_columns = set(self.schema.required_columns) - set(data.columns)
        if missing_columns:
            result.is_valid = False
            result.errors.append(f"Missing required columns: {missing_columns}")

        # Check data types
        for col, expected_type in self.schema.columns.items():
            if col in data.columns:
                actual_type = str(data[col].dtype)
                if not self._check_dtype_compatibility(actual_type, expected_type):
                    result.warnings.append(
                        f"Column '{col}' has type '{actual_type}', expected '{expected_type}'"
                    )

        # Check constraints
        for col, constraints in self.schema.constraints.items():
            if col in data.columns:
                self._validate_constraints(data[col], constraints, col, result)

        # Additional checks
        result.statistics["n_rows"] = len(data)
        result.statistics["n_columns"] = len(data.columns)
        result.statistics["memory_usage_mb"] = data.memory_usage(deep=True).sum() / 1e6

        return result

    def _check_dtype_compatibility(self, actual: str, expected: str) -> bool:
        """Check if data types are compatible"""
        compatible_types = {
            "float": ["float16", "float32", "float64", "int"],
            "int": ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"],
            "object": ["object", "string", "category"],
            "datetime": ["datetime64", "datetime64[ns]", "datetime64[ns, UTC]"],
        }

        for expected_base, compatible in compatible_types.items():
            if expected_base in expected.lower():
                return any(c in actual.lower() for c in compatible)

        return actual == expected

    def _validate_constraints(
        self,
        series: pd.Series,
        constraints: dict[str, Any],
        col_name: str,
        result: ValidationResult,
    ) -> None:
        """Validate column constraints"""
        # Min/max constraints
        if "min" in constraints:
            violations = series < constraints["min"]
            if violations.any():
                result.warnings.append(
                    f"Column '{col_name}' has {violations.sum()} values below minimum {constraints['min']}"
                )

        if "max" in constraints:
            violations = series > constraints["max"]
            if violations.any():
                result.warnings.append(
                    f"Column '{col_name}' has {violations.sum()} values above maximum {constraints['max']}"
                )

        # Allowed values constraint
        if "values" in constraints:
            invalid_values = ~series.isin(constraints["values"])
            if invalid_values.any():
                unique_invalid = series[invalid_values].unique()[:5]  # Show first 5
                result.warnings.append(f"Column '{col_name}' has invalid values: {unique_invalid}")

        # Not null constraint
        if constraints.get("not_null", False):
            null_count = series.isnull().sum()
            if null_count > 0:
                result.errors.append(
                    f"Column '{col_name}' has {null_count} null values but should not be null"
                )
                result.is_valid = False


class DataQualityValidator(BaseValidator):
    """Comprehensive data quality validation"""

    def __init__(
        self, missing_threshold: float = 0.5, duplication_threshold: float = 0.1, **kwargs
    ):
        super().__init__(**kwargs)
        self.missing_threshold = missing_threshold
        self.duplication_threshold = duplication_threshold

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Perform comprehensive data quality checks"""
        result = ValidationResult(is_valid=True)

        # 1. Missing value analysis
        self._check_missing_values(data, result)

        # 2. Duplicate analysis
        self._check_duplicates(data, result)

        # 3. Data consistency
        self._check_consistency(data, result)

        # 4. Statistical anomalies
        self._check_statistical_anomalies(data, result)

        # 5. Data completeness
        self._check_completeness(data, result)

        # Generate recommendations
        self._generate_recommendations(result)

        return result

    def _check_missing_values(self, data: pd.DataFrame, result: ValidationResult) -> None:
        """Check for missing values"""
        missing_counts = data.isnull().sum()
        missing_ratios = missing_counts / len(data)

        # Overall missing ratio
        total_missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        result.statistics["total_missing_ratio"] = total_missing_ratio

        # Columns with high missing ratio
        high_missing_cols = missing_ratios[missing_ratios > self.missing_threshold]
        if len(high_missing_cols) > 0:
            result.warnings.append(
                f"Columns with >{self.missing_threshold*100}% missing: {list(high_missing_cols.index)}"
            )

        # Missing value patterns
        if "timestamp" in data.columns:
            # Check for systematic missing patterns
            missing_by_hour = (
                data.set_index("timestamp").resample("H").apply(lambda x: x.isnull().sum().sum())
            )
            if missing_by_hour.std() > missing_by_hour.mean():
                result.warnings.append("Detected non-uniform missing value patterns over time")

        result.statistics["missing_by_column"] = missing_counts.to_dict()

    def _check_duplicates(self, data: pd.DataFrame, result: ValidationResult) -> None:
        """Check for duplicate records"""
        # Exact duplicates
        n_duplicates = data.duplicated().sum()
        duplicate_ratio = n_duplicates / len(data)

        result.statistics["n_duplicates"] = n_duplicates
        result.statistics["duplicate_ratio"] = duplicate_ratio

        if duplicate_ratio > self.duplication_threshold:
            result.warnings.append(
                f"High duplicate ratio: {duplicate_ratio:.2%} ({n_duplicates} records)"
            )

        # Check for duplicates on key columns
        if "timestamp" in data.columns and "airport" in data.columns:
            key_duplicates = data.duplicated(subset=["timestamp", "airport"]).sum()
            if key_duplicates > 0:
                result.errors.append(
                    f"Found {key_duplicates} duplicate timestamp-airport combinations"
                )
                result.is_valid = False

    def _check_consistency(self, data: pd.DataFrame, result: ValidationResult) -> None:
        """Check data consistency"""
        # Temperature consistency
        if "temperature" in data.columns and "dewpoint" in data.columns:
            invalid_temp = data["temperature"] < data["dewpoint"]
            if invalid_temp.any():
                result.warnings.append(
                    f"Found {invalid_temp.sum()} records where temperature < dewpoint"
                )

        # Wind consistency
        if "wind_speed" in data.columns and "wind_gust" in data.columns:
            invalid_wind = data["wind_gust"] < data["wind_speed"]
            if invalid_wind.any():
                result.warnings.append(
                    f"Found {invalid_wind.sum()} records where wind gust < wind speed"
                )

        # Time consistency
        if "timestamp" in data.columns:
            data_sorted = data.sort_values("timestamp")
            time_diffs = data_sorted["timestamp"].diff()

            # Check for backwards time jumps
            backwards_jumps = (time_diffs < pd.Timedelta(0)).sum()
            if backwards_jumps > 0:
                result.errors.append(f"Found {backwards_jumps} backwards time jumps")
                result.is_valid = False

            # Check for large gaps
            large_gaps = time_diffs > pd.Timedelta(hours=24)
            if large_gaps.any():
                result.warnings.append(f"Found {large_gaps.sum()} time gaps larger than 24 hours")

    def _check_statistical_anomalies(self, data: pd.DataFrame, result: ValidationResult) -> None:
        """Check for statistical anomalies"""
        numerical_cols = data.select_dtypes(include=[np.number]).columns

        anomaly_counts = {}
        for col in numerical_cols:
            if col in data.columns:
                # Z-score based anomalies
                z_scores = np.abs(stats.zscore(data[col].dropna()))
                n_anomalies = (z_scores > 3).sum()

                if n_anomalies > 0:
                    anomaly_counts[col] = n_anomalies

        if anomaly_counts:
            result.statistics["anomaly_counts"] = anomaly_counts
            total_anomalies = sum(anomaly_counts.values())
            result.warnings.append(
                f"Found {total_anomalies} statistical anomalies across {len(anomaly_counts)} columns"
            )

    def _check_completeness(self, data: pd.DataFrame, result: ValidationResult) -> None:
        """Check data completeness"""
        if "timestamp" in data.columns:
            # Check for expected time range coverage
            time_range = data["timestamp"].max() - data["timestamp"].min()
            expected_records = time_range / pd.Timedelta(minutes=30)  # Assuming 30-min intervals
            actual_records = len(data)
            completeness_ratio = actual_records / expected_records

            result.statistics["temporal_completeness"] = completeness_ratio

            if completeness_ratio < 0.8:
                result.warnings.append(f"Low temporal completeness: {completeness_ratio:.2%}")

    def _generate_recommendations(self, result: ValidationResult) -> None:
        """Generate recommendations based on validation results"""
        if result.statistics.get("total_missing_ratio", 0) > 0.1:
            result.recommendations.append(
                "Consider advanced imputation techniques for missing values"
            )

        if result.statistics.get("n_duplicates", 0) > 0:
            result.recommendations.append("Remove duplicate records before training")

        if "anomaly_counts" in result.statistics:
            result.recommendations.append("Apply outlier detection and treatment before modeling")


class AnomalyDetector:
    """Advanced anomaly detection for weather data"""

    def __init__(
        self,
        contamination: float = 0.01,
        methods: list[str] | None = None,
    ):
        self.contamination = contamination
        self.methods = methods or ["isolation_forest", "elliptic_envelope"]
        self.detectors = {}
        self.anomaly_scores = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def fit(self, X: pd.DataFrame) -> None:
        """Fit anomaly detectors"""
        X_numeric = X.select_dtypes(include=[np.number]).fillna(X.median())

        for method in self.methods:
            self.logger.info(f"Fitting {method} detector...")

            if method == "isolation_forest":
                detector = IsolationForest(
                    contamination=self.contamination, random_state=42, n_jobs=-1
                )
            elif method == "elliptic_envelope":
                detector = EllipticEnvelope(contamination=self.contamination, random_state=42)
            elif method == "local_outlier_factor":
                detector = LocalOutlierFactor(
                    contamination=self.contamination, novelty=True, n_jobs=-1
                )
            else:
                continue

            detector.fit(X_numeric)
            self.detectors[method] = detector

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict anomalies"""
        X_numeric = X.select_dtypes(include=[np.number]).fillna(X.median())

        predictions = pd.DataFrame(index=X.index)
        scores = pd.DataFrame(index=X.index)

        for method, detector in self.detectors.items():
            # Get predictions (-1 for anomaly, 1 for normal)
            pred = detector.predict(X_numeric)
            predictions[f"{method}_anomaly"] = (pred == -1).astype(int)

            # Get anomaly scores if available
            if hasattr(detector, "score_samples"):
                scores[f"{method}_score"] = detector.score_samples(X_numeric)
            elif hasattr(detector, "decision_function"):
                scores[f"{method}_score"] = detector.decision_function(X_numeric)

        # Ensemble prediction
        predictions["ensemble_anomaly"] = (predictions.mean(axis=1) > 0.5).astype(int)

        self.anomaly_scores = scores
        return predictions

    def explain_anomalies(self, X: pd.DataFrame, anomalies: pd.DataFrame) -> pd.DataFrame:
        """Explain why records were flagged as anomalies"""
        explanations = []

        X_numeric = X.select_dtypes(include=[np.number])

        # For each anomaly, find which features are most unusual
        anomaly_indices = anomalies[anomalies["ensemble_anomaly"] == 1].index

        for idx in anomaly_indices:
            record = X_numeric.loc[idx]
            explanation = {"index": idx}

            # Calculate z-scores for this record
            z_scores = {}
            for col in X_numeric.columns:
                z_score = abs((record[col] - X_numeric[col].mean()) / X_numeric[col].std())
                if z_score > 2:
                    z_scores[col] = z_score

            # Sort by z-score
            top_features = sorted(z_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            explanation["unusual_features"] = top_features
            explanation["anomaly_type"] = self._classify_anomaly_type(record, X_numeric)

            explanations.append(explanation)

        return pd.DataFrame(explanations)

    def _classify_anomaly_type(self, record: pd.Series, data: pd.DataFrame) -> str:
        """Classify the type of anomaly"""
        # Simple classification based on feature patterns
        extreme_features = []

        for col in record.index:
            if record[col] < data[col].quantile(0.01):
                extreme_features.append(f"{col}_very_low")
            elif record[col] > data[col].quantile(0.99):
                extreme_features.append(f"{col}_very_high")

        if len(extreme_features) >= 3:
            return "multivariate_extreme"
        elif len(extreme_features) > 0:
            return f"extreme_{extreme_features[0]}"
        else:
            return "subtle_anomaly"


class DataDriftDetector:
    """Detect data drift between datasets"""

    def __init__(self, statistical_tests: list[str] | None = None, drift_threshold: float = 0.05):
        self.statistical_tests = statistical_tests or ["ks", "chi2"]
        self.drift_threshold = drift_threshold
        self.reference_stats = None
        self.drift_scores = None

    def fit_reference(self, X_reference: pd.DataFrame) -> None:
        """Fit on reference dataset"""
        self.reference_stats = {
            "means": X_reference.select_dtypes(include=[np.number]).mean(),
            "stds": X_reference.select_dtypes(include=[np.number]).std(),
            "distributions": {},
        }

        # Store distributions for numerical columns
        for col in X_reference.select_dtypes(include=[np.number]).columns:
            self.reference_stats["distributions"][col] = X_reference[col].dropna().values

    def detect_drift(self, X_current: pd.DataFrame) -> dict[str, Any]:
        """Detect drift in current dataset"""
        if self.reference_stats is None:
            raise ValueError("Must fit on reference dataset first")

        drift_results = {
            "overall_drift": False,
            "drifted_features": [],
            "drift_scores": {},
            "statistical_tests": {},
        }

        numerical_cols = X_current.select_dtypes(include=[np.number]).columns

        for col in numerical_cols:
            if col in self.reference_stats["distributions"]:
                # Perform statistical tests
                test_results = {}

                if "ks" in self.statistical_tests:
                    # Kolmogorov-Smirnov test
                    statistic, p_value = stats.ks_2samp(
                        self.reference_stats["distributions"][col], X_current[col].dropna().values
                    )
                    test_results["ks"] = {"statistic": statistic, "p_value": p_value}

                if "chi2" in self.statistical_tests:
                    # Chi-square test (for binned data)
                    # Create bins
                    bins = np.histogram_bin_edges(
                        np.concatenate(
                            [
                                self.reference_stats["distributions"][col],
                                X_current[col].dropna().values,
                            ]
                        ),
                        bins=10,
                    )

                    ref_hist, _ = np.histogram(
                        self.reference_stats["distributions"][col], bins=bins
                    )
                    curr_hist, _ = np.histogram(X_current[col].dropna().values, bins=bins)

                    # Normalize
                    ref_hist = ref_hist / ref_hist.sum()
                    curr_hist = curr_hist / curr_hist.sum()

                    # Chi-square test
                    statistic, p_value = stats.chisquare(
                        curr_hist + 1e-10,
                        ref_hist + 1e-10,  # Avoid division by zero
                    )
                    test_results["chi2"] = {"statistic": statistic, "p_value": p_value}

                drift_results["statistical_tests"][col] = test_results

                # Check for drift
                is_drifted = any(
                    result["p_value"] < self.drift_threshold for result in test_results.values()
                )

                if is_drifted:
                    drift_results["drifted_features"].append(col)

                # Calculate drift score
                drift_score = 1 - min(result["p_value"] for result in test_results.values())
                drift_results["drift_scores"][col] = drift_score

        # Overall drift detection
        if len(drift_results["drifted_features"]) > len(numerical_cols) * 0.3:
            drift_results["overall_drift"] = True

        self.drift_scores = drift_results["drift_scores"]

        return drift_results

    def visualize_drift(
        self,
        X_reference: pd.DataFrame,
        X_current: pd.DataFrame,
        features: list[str] | None = None,
        save_path: str | None = None,
    ) -> None:
        """Visualize data drift"""
        if features is None:
            features = list(self.drift_scores.keys())[:6]  # Top 6 features

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        for idx, feature in enumerate(features[:6]):
            ax = axes[idx]

            # Plot distributions
            ax.hist(
                X_reference[feature].dropna(), bins=30, alpha=0.5, label="Reference", density=True
            )
            ax.hist(X_current[feature].dropna(), bins=30, alpha=0.5, label="Current", density=True)

            # Add drift score
            drift_score = self.drift_scores.get(feature, 0)
            ax.set_title(f"{feature}\nDrift Score: {drift_score:.3f}")
            ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


class DataValidator:
    """Main data validation orchestrator"""

    def __init__(self, config: DataConfig):
        self.config = config
        self.schema_validator = None
        self.quality_validator = DataQualityValidator()
        self.anomaly_detector = AnomalyDetector()
        self.drift_detector = DataDriftDetector()
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_schema_from_data(self, data: pd.DataFrame) -> DataSchema:
        """Create schema from sample data"""
        schema = DataSchema(
            columns={col: str(dtype) for col, dtype in data.dtypes.items()},
            required_columns=list(data.columns),
            constraints={},
            datetime_columns=[
                col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col])
            ],
            categorical_columns=list(data.select_dtypes(include=["object", "category"]).columns),
            numerical_columns=list(data.select_dtypes(include=[np.number]).columns),
        )

        # Add basic constraints for numerical columns
        for col in schema.numerical_columns:
            schema.constraints[col] = {
                "min": data[col].min(),
                "max": data[col].max(),
                "not_null": data[col].isnull().sum() == 0,
            }

        return schema

    def validate_dataset(
        self,
        data: pd.DataFrame,
        schema: DataSchema | None = None,
        check_anomalies: bool = True,
        reference_data: pd.DataFrame | None = None,
    ) -> dict[str, ValidationResult]:
        """Perform comprehensive dataset validation"""
        results = {}

        # Schema validation
        if schema:
            self.logger.info("Performing schema validation...")
            self.schema_validator = SchemaValidator(schema)
            results["schema"] = self.schema_validator.validate(data)

        # Quality validation
        self.logger.info("Performing quality validation...")
        results["quality"] = self.quality_validator.validate(data)

        # Anomaly detection
        if check_anomalies:
            self.logger.info("Performing anomaly detection...")
            self.anomaly_detector.fit(data)
            anomalies = self.anomaly_detector.predict(data)

            anomaly_result = ValidationResult(
                is_valid=True,
                statistics={
                    "n_anomalies": anomalies["ensemble_anomaly"].sum(),
                    "anomaly_rate": anomalies["ensemble_anomaly"].mean(),
                },
            )

            if anomaly_result.statistics["anomaly_rate"] > 0.05:
                anomaly_result.warnings.append(
                    f"High anomaly rate: {anomaly_result.statistics['anomaly_rate']:.2%}"
                )

            results["anomalies"] = anomaly_result

        # Drift detection
        if reference_data is not None:
            self.logger.info("Performing drift detection...")
            self.drift_detector.fit_reference(reference_data)
            drift_results = self.drift_detector.detect_drift(data)

            drift_result = ValidationResult(
                is_valid=not drift_results["overall_drift"], statistics=drift_results
            )

            if drift_results["overall_drift"]:
                drift_result.errors.append(
                    f"Significant data drift detected in {len(drift_results['drifted_features'])} features"
                )

            results["drift"] = drift_result

        return results

    def generate_validation_report(
        self, results: dict[str, ValidationResult], output_path: str
    ) -> None:
        """Generate comprehensive validation report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "all_valid": all(r.is_valid for r in results.values()),
                "n_errors": sum(len(r.errors) for r in results.values()),
                "n_warnings": sum(len(r.warnings) for r in results.values()),
            },
            "results": {name: result.to_dict() for name, result in results.items()},
        }

        # Save JSON report
        json_path = Path(output_path).with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Generate HTML report
        html_content = self._generate_html_report(report)
        html_path = Path(output_path).with_suffix(".html")
        with open(html_path, "w") as f:
            f.write(html_content)

        self.logger.info(f"Validation report saved to {output_path}")

    def _generate_html_report(self, report: dict[str, Any]) -> str:
        """Generate HTML validation report"""
        from jinja2 import Template

        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Validation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .summary { background-color: #f0f0f0; padding: 15px; border-radius: 5px; }
                .valid { color: green; }
                .invalid { color: red; }
                .warning { color: orange; }
                .section { margin: 20px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Data Validation Report</h1>
            <p>Generated: {{ timestamp }}</p>

            <div class="summary">
                <h2>Summary</h2>
                <p>Overall Status:
                    <span class="{{ 'valid' if summary.all_valid else 'invalid' }}">
                        {{ 'VALID' if summary.all_valid else 'INVALID' }}
                    </span>
                </p>
                <p>Total Errors: {{ summary.n_errors }}</p>
                <p>Total Warnings: {{ summary.n_warnings }}</p>
            </div>

            {% for name, result in results.items() %}
            <div class="section">
                <h2>{{ name|title }} Validation</h2>
                <p>Status:
                    <span class="{{ 'valid' if result.is_valid else 'invalid' }}">
                        {{ 'VALID' if result.is_valid else 'INVALID' }}
                    </span>
                </p>

                {% if result.errors %}
                <h3>Errors</h3>
                <ul>
                    {% for error in result.errors %}
                    <li class="invalid">{{ error }}</li>
                    {% endfor %}
                </ul>
                {% endif %}

                {% if result.warnings %}
                <h3>Warnings</h3>
                <ul>
                    {% for warning in result.warnings %}
                    <li class="warning">{{ warning }}</li>
                    {% endfor %}
                </ul>
                {% endif %}

                {% if result.recommendations %}
                <h3>Recommendations</h3>
                <ul>
                    {% for rec in result.recommendations %}
                    <li>{{ rec }}</li>
                    {% endfor %}
                </ul>
                {% endif %}

                {% if result.statistics %}
                <h3>Statistics</h3>
                <table>
                    {% for key, value in result.statistics.items() %}
                    <tr>
                        <td>{{ key }}</td>
                        <td>{{ value }}</td>
                    </tr>
                    {% endfor %}
                </table>
                {% endif %}
            </div>
            {% endfor %}
        </body>
        </html>
        """

        template_obj = Template(template)
        return template_obj.render(**report)


# Utility functions
def validate_weather_data(data: pd.DataFrame) -> ValidationResult:
    """Quick validation function for weather data"""
    # Create weather-specific schema
    schema = DataSchema(
        columns={
            "timestamp": "datetime64[ns]",
            "airport": "object",
            "temperature": "float",
            "dewpoint": "float",
            "pressure": "float",
            "wind_speed": "float",
            "wind_direction": "float",
            "visibility": "float",
            "ceiling": "float",
            "cloud_coverage": "object",
        },
        required_columns=["timestamp", "airport", "temperature"],
        constraints={
            "temperature": {"min": -60, "max": 60},
            "dewpoint": {"min": -60, "max": 50},
            "pressure": {"min": 850, "max": 1100},
            "wind_speed": {"min": 0, "max": 200},
            "wind_direction": {"min": 0, "max": 360},
            "visibility": {"min": 0, "max": 100000},
            "ceiling": {"min": 0, "max": 60000},
        },
        datetime_columns=["timestamp"],
        categorical_columns=["airport", "cloud_coverage"],
        numerical_columns=[
            "temperature",
            "dewpoint",
            "pressure",
            "wind_speed",
            "wind_direction",
            "visibility",
            "ceiling",
        ],
    )

    validator = SchemaValidator(schema)
    return validator.validate(data)


def create_validation_pipeline(config: DataConfig) -> DataValidator:
    """Create configured validation pipeline"""
    return DataValidator(config)
