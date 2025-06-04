"""
Comprehensive Unit Tests for Data Pipeline

This module tests all data processing components:
- Data loading and caching
- Data validation and quality checks
- Feature engineering
- Preprocessing pipelines
- Error handling and edge cases
"""

import shutil
import tempfile
import warnings
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

# Import data pipeline modules
from data.data_loader import DataCache, DataLoader
from data.data_validation import (
    AnomalyDetector,
    DataDriftDetector,
    DataQualityValidator,
    DataValidator,
    SchemaValidator,
)
from data.feature_engineering import (
    AutomatedFeatureEngineer,
    StatisticalFeatureEngineer,
    TimeSeriesFeatureEngineer,
    WeatherFeatureEngineer,
)
from data.preprocessing import (
    CyclicalEncoder,
    FeatureSelector,
    LagFeatureCreator,
    OutlierDetector,
    PreprocessingPipeline,
    TimeSeriesScaler,
)

warnings.filterwarnings("ignore")


class TestDataLoader:
    """Test DataLoader class functionality"""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with sample data files"""
        temp_dir = tempfile.mkdtemp()

        # Create sample METAR data
        metar_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="30min"),
                "airport": ["EGLL"] * 100,
                "temperature": np.random.randn(100) * 10 + 15,
                "pressure": np.random.randn(100) * 20 + 1013,
                "wind_speed": np.random.randn(100) * 5 + 10,
                "visibility": np.random.randn(100) * 2000 + 8000,
                "weather_code": np.random.choice(["RA", "SN", "FG", "CLR"], 100),
            }
        )
        metar_path = Path(temp_dir) / "metar_test.csv"
        metar_data.to_csv(metar_path, index=False)

        # Create sample regulation data
        regulation_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=50, freq="1H"),
                "airport": ["EGLL"] * 50,
                "regulation_type": np.random.choice(["WX", "ATC", "EQ"], 50),
                "duration": np.random.randint(10, 120, 50),
                "has_regulation": np.random.choice([0, 1], 50, p=[0.7, 0.3]),
            }
        )
        reg_path = Path(temp_dir) / "regulations_test.csv"
        regulation_data.to_csv(reg_path, index=False)

        # Create sample TAF data
        taf_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=30, freq="6H"),
                "airport": ["EGLL"] * 30,
                "forecast_temp": np.random.randn(30) * 8 + 15,
                "forecast_wind": np.random.randn(30) * 4 + 12,
                "forecast_vis": np.random.randn(30) * 1500 + 9000,
            }
        )
        taf_path = Path(temp_dir) / "taf_test.csv"
        taf_data.to_csv(taf_path, index=False)

        yield temp_dir, metar_path, reg_path, taf_path

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_initialization(self, temp_data_dir):
        """Test DataLoader initialization"""
        temp_dir, _, _, _ = temp_data_dir
        loader = DataLoader(data_path=temp_dir)

        assert loader.data_path == Path(temp_dir)
        assert isinstance(loader.cache, DataCache)

    def test_load_metar_data(self, temp_data_dir):
        """Test METAR data loading"""
        temp_dir, metar_path, _, _ = temp_data_dir
        loader = DataLoader(data_path=temp_dir)

        # Test loading with file path
        metar_df = loader.load_metar_data(str(metar_path))
        assert isinstance(metar_df, pd.DataFrame)
        assert len(metar_df) > 0
        assert "timestamp" in metar_df.columns
        assert "temperature" in metar_df.columns

        # Test timestamp parsing
        assert pd.api.types.is_datetime64_any_dtype(metar_df["timestamp"])

    def test_load_regulation_data(self, temp_data_dir):
        """Test regulation data loading"""
        temp_dir, _, reg_path, _ = temp_data_dir
        loader = DataLoader(data_path=temp_dir)

        reg_df = loader.load_regulation_data(str(reg_path))
        assert isinstance(reg_df, pd.DataFrame)
        assert len(reg_df) > 0
        assert "has_regulation" in reg_df.columns
        assert all(reg_df["has_regulation"].isin([0, 1]))

    def test_load_taf_data(self, temp_data_dir):
        """Test TAF data loading"""
        temp_dir, _, _, taf_path = temp_data_dir
        loader = DataLoader(data_path=temp_dir)

        taf_df = loader.load_taf_data(str(taf_path))
        assert isinstance(taf_df, pd.DataFrame)
        assert len(taf_df) > 0
        assert "forecast_temp" in taf_df.columns

    def test_create_features(self, temp_data_dir):
        """Test feature matrix creation"""
        temp_dir, metar_path, reg_path, taf_path = temp_data_dir
        loader = DataLoader(data_path=temp_dir)

        # Load all data
        metar_df = loader.load_metar_data(str(metar_path))
        reg_df = loader.load_regulation_data(str(reg_path))
        taf_df = loader.load_taf_data(str(taf_path))

        # Create feature matrix
        features_df = loader.create_features(
            metar_data=metar_df,
            regulation_data=reg_df,
            taf_data=taf_df,
            target_column="has_regulation",
        )

        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) > 0
        assert "target" in features_df.columns

    def test_caching_functionality(self, temp_data_dir):
        """Test data caching"""
        temp_dir, metar_path, _, _ = temp_data_dir
        loader = DataLoader(data_path=temp_dir, enable_cache=True)

        # First load (should cache)
        df1 = loader.load_metar_data(str(metar_path))

        # Second load (should use cache)
        df2 = loader.load_metar_data(str(metar_path))

        pd.testing.assert_frame_equal(df1, df2)

    def test_parallel_loading(self, temp_data_dir):
        """Test parallel data loading"""
        temp_dir, metar_path, _, _ = temp_data_dir
        loader = DataLoader(data_path=temp_dir, n_jobs=2)

        # Create multiple files for parallel loading
        file_paths = [str(metar_path) for _ in range(3)]
        results = loader.load_multiple_files(file_paths, file_type="metar")

        assert len(results) == 3
        assert all(isinstance(df, pd.DataFrame) for df in results)

    def test_error_handling(self, temp_data_dir):
        """Test error handling for invalid data"""
        temp_dir, _, _, _ = temp_data_dir
        loader = DataLoader(data_path=temp_dir)

        # Test loading non-existent file
        with pytest.raises(FileNotFoundError):
            loader.load_metar_data("non_existent_file.csv")

        # Test invalid file format
        invalid_path = Path(temp_dir) / "invalid.txt"
        invalid_path.write_text("invalid data")

        with pytest.raises((ValueError, pd.errors.EmptyDataError)):
            loader.load_metar_data(str(invalid_path))


class TestDataCache:
    """Test DataCache functionality"""

    def test_cache_operations(self):
        """Test basic cache operations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DataCache(cache_dir=temp_dir, max_size_gb=0.1)

            # Test caching
            test_data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            cache_key = "test_data"

            # Store in cache
            cache.set(cache_key, test_data)
            assert cache.exists(cache_key)

            # Retrieve from cache
            cached_data = cache.get(cache_key)
            pd.testing.assert_frame_equal(test_data, cached_data)

            # Clear cache
            cache.clear()
            assert not cache.exists(cache_key)

    def test_cache_expiration(self):
        """Test cache expiration"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DataCache(cache_dir=temp_dir, ttl_hours=0.001)  # Very short TTL

            test_data = pd.DataFrame({"a": [1, 2, 3]})
            cache.set("test_key", test_data)

            # Wait for expiration
            import time

            time.sleep(0.1)

            # Should be expired
            assert not cache.exists("test_key")


class TestDataValidator:
    """Test DataValidator functionality"""

    @pytest.fixture
    def sample_weather_data(self):
        """Create sample weather data for testing"""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="1H"),
                "temperature": np.random.randn(100) * 10 + 15,
                "pressure": np.random.randn(100) * 20 + 1013,
                "wind_speed": np.random.randn(100) * 5 + 10,
                "visibility": np.random.randn(100) * 2000 + 8000,
                "humidity": np.random.randint(0, 101, 100),
            }
        )

    def test_schema_validation(self, sample_weather_data):
        """Test schema validation"""
        validator = SchemaValidator()

        # Define expected schema
        expected_schema = {
            "timestamp": "datetime64[ns]",
            "temperature": "float64",
            "pressure": "float64",
            "wind_speed": "float64",
            "visibility": "float64",
            "humidity": "int64",
        }

        # Test valid schema
        is_valid, errors = validator.validate_schema(sample_weather_data, expected_schema)
        assert is_valid
        assert len(errors) == 0

        # Test invalid schema (missing column)
        invalid_data = sample_weather_data.drop("temperature", axis=1)
        is_valid, errors = validator.validate_schema(invalid_data, expected_schema)
        assert not is_valid
        assert len(errors) > 0

    def test_data_quality_validation(self, sample_weather_data):
        """Test data quality validation"""
        validator = DataQualityValidator()

        # Test with clean data
        quality_report = validator.validate_quality(sample_weather_data)
        assert "missing_values" in quality_report
        assert "duplicate_rows" in quality_report
        assert "outliers" in quality_report

        # Add some quality issues
        dirty_data = sample_weather_data.copy()
        dirty_data.loc[0, "temperature"] = np.nan  # Missing value
        dirty_data.loc[10, "pressure"] = 2000  # Outlier
        dirty_data = pd.concat([dirty_data, dirty_data.iloc[:1]])  # Duplicate

        quality_report = validator.validate_quality(dirty_data)
        assert quality_report["missing_values"]["temperature"] > 0
        assert quality_report["duplicate_rows"] > 0

    def test_anomaly_detection(self, sample_weather_data):
        """Test anomaly detection"""
        detector = AnomalyDetector()

        # Add some anomalies
        anomalous_data = sample_weather_data.copy()
        anomalous_data.loc[0, "temperature"] = 100  # Extreme temperature
        anomalous_data.loc[1, "pressure"] = 500  # Extreme pressure

        anomalies = detector.detect_anomalies(
            anomalous_data, columns=["temperature", "pressure"], method="isolation_forest"
        )

        assert len(anomalies) > 0
        assert 0 in anomalies or 1 in anomalies  # Should detect our inserted anomalies

    def test_data_drift_detection(self, sample_weather_data):
        """Test data drift detection"""
        detector = DataDriftDetector()

        # Create reference and current datasets
        reference_data = sample_weather_data[:50]

        # Create drifted data (shifted distribution)
        current_data = sample_weather_data[50:].copy()
        current_data["temperature"] += 10  # Temperature drift

        drift_report = detector.detect_drift(
            reference_data=reference_data,
            current_data=current_data,
            columns=["temperature", "pressure"],
        )

        assert "temperature" in drift_report
        assert drift_report["temperature"]["has_drift"] == True  # Should detect temperature drift
        assert drift_report["pressure"]["has_drift"] == False  # No pressure drift

    def test_weather_specific_validation(self, sample_weather_data):
        """Test weather-specific validation rules"""
        validator = DataValidator()

        # Add invalid weather values
        invalid_data = sample_weather_data.copy()
        invalid_data.loc[0, "temperature"] = -100  # Invalid temperature
        invalid_data.loc[1, "wind_speed"] = -5  # Negative wind speed
        invalid_data.loc[2, "visibility"] = -1000  # Negative visibility

        validation_report = validator.validate_weather_data(invalid_data)

        assert len(validation_report["errors"]) > 0
        assert any("temperature" in error for error in validation_report["errors"])
        assert any("wind_speed" in error for error in validation_report["errors"])


class TestFeatureEngineering:
    """Test feature engineering modules"""

    @pytest.fixture
    def sample_weather_data(self):
        """Create sample weather data for feature engineering"""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="1H"),
                "temperature": np.random.randn(100) * 10 + 15,
                "pressure": np.random.randn(100) * 20 + 1013,
                "wind_speed": np.random.randn(100) * 5 + 10,
                "wind_direction": np.random.randint(0, 360, 100),
                "visibility": np.random.randn(100) * 2000 + 8000,
                "humidity": np.random.randint(0, 101, 100),
                "weather_code": np.random.choice(["RA", "SN", "FG", "CLR"], 100),
            }
        )

    def test_weather_feature_engineer(self, sample_weather_data):
        """Test weather-specific feature engineering"""
        engineer = WeatherFeatureEngineer()

        enhanced_data = engineer.create_features(sample_weather_data)

        # Check that new features were created
        assert "flight_category" in enhanced_data.columns
        assert "weather_severity" in enhanced_data.columns
        assert "wind_components_u" in enhanced_data.columns
        assert "wind_components_v" in enhanced_data.columns
        assert "dewpoint" in enhanced_data.columns

        # Check flight category is valid
        valid_categories = ["VFR", "MVFR", "IFR", "LIFR"]
        assert all(cat in valid_categories for cat in enhanced_data["flight_category"].unique())

    def test_time_series_feature_engineer(self, sample_weather_data):
        """Test time series feature engineering"""
        engineer = TimeSeriesFeatureEngineer()

        enhanced_data = engineer.create_features(
            sample_weather_data,
            timestamp_col="timestamp",
            value_cols=["temperature", "pressure"],
            lags=[1, 2, 3],
            rolling_windows=[6, 12],
        )

        # Check lag features
        assert "temperature_lag_1" in enhanced_data.columns
        assert "pressure_lag_2" in enhanced_data.columns

        # Check rolling features
        assert "temperature_rolling_mean_6" in enhanced_data.columns
        assert "pressure_rolling_std_12" in enhanced_data.columns

        # Check temporal features
        assert "hour" in enhanced_data.columns
        assert "day_of_week" in enhanced_data.columns
        assert "month" in enhanced_data.columns

    def test_statistical_feature_engineer(self, sample_weather_data):
        """Test statistical feature engineering"""
        engineer = StatisticalFeatureEngineer()

        enhanced_data = engineer.create_features(
            sample_weather_data, numeric_cols=["temperature", "pressure", "wind_speed"]
        )

        # Check statistical features
        assert "temperature_squared" in enhanced_data.columns
        assert "temperature_log" in enhanced_data.columns
        assert "temperature_pressure_interaction" in enhanced_data.columns

        # Check polynomial features were created
        poly_cols = [col for col in enhanced_data.columns if "poly_" in col]
        assert len(poly_cols) > 0

    def test_automated_feature_engineer(self, sample_weather_data):
        """Test automated feature engineering"""
        engineer = AutomatedFeatureEngineer()

        # Add a target variable
        sample_weather_data["target"] = (sample_weather_data["temperature"] > 15).astype(int)

        enhanced_data = engineer.create_features(
            sample_weather_data, target_col="target", max_features=50
        )

        # Should have created new features
        assert enhanced_data.shape[1] > sample_weather_data.shape[1]

        # Should have selected best features
        selected_features = engineer.get_selected_features()
        assert len(selected_features) <= 50
        assert len(selected_features) > 0


class TestPreprocessing:
    """Test preprocessing pipeline components"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for preprocessing"""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "numeric_1": np.random.randn(100) * 10 + 50,
                "numeric_2": np.random.randn(100) * 5 + 20,
                "categorical": np.random.choice(["A", "B", "C"], 100),
                "cyclical": np.random.randint(0, 24, 100),  # Hour of day
                "target": np.random.choice([0, 1], 100),
            }
        )

    def test_time_series_scaler(self, sample_data):
        """Test time series scaler"""
        scaler = TimeSeriesScaler(method="minmax")

        # Fit and transform
        scaled_data = scaler.fit_transform(sample_data[["numeric_1", "numeric_2"]])

        assert scaled_data.shape == (100, 2)
        assert scaled_data.min().min() >= 0
        assert scaled_data.max().max() <= 1

        # Test inverse transform
        original_data = scaler.inverse_transform(scaled_data)
        np.testing.assert_allclose(
            original_data, sample_data[["numeric_1", "numeric_2"]].values, rtol=1e-10
        )

    def test_cyclical_encoder(self, sample_data):
        """Test cyclical encoding"""
        encoder = CyclicalEncoder()

        encoded_data = encoder.fit_transform(sample_data[["cyclical"]], periods={"cyclical": 24})

        assert "cyclical_sin" in encoded_data.columns
        assert "cyclical_cos" in encoded_data.columns

        # Check that values are in valid range
        assert encoded_data["cyclical_sin"].min() >= -1
        assert encoded_data["cyclical_sin"].max() <= 1
        assert encoded_data["cyclical_cos"].min() >= -1
        assert encoded_data["cyclical_cos"].max() <= 1

    def test_lag_feature_creator(self):
        """Test lag feature creation"""
        creator = LagFeatureCreator(lags=[1, 2, 3])

        # Create time series data
        data = pd.DataFrame(
            {
                "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="1H"),
            }
        )

        lagged_data = creator.fit_transform(data, timestamp_col="timestamp")

        assert "value_lag_1" in lagged_data.columns
        assert "value_lag_2" in lagged_data.columns
        assert "value_lag_3" in lagged_data.columns

        # Check lag values are correct
        assert lagged_data["value_lag_1"].iloc[3] == 3  # Previous value
        assert lagged_data["value_lag_2"].iloc[3] == 2  # Two steps back

    def test_outlier_detector(self, sample_data):
        """Test outlier detection"""
        detector = OutlierDetector(method="iqr")

        # Add some outliers
        outlier_data = sample_data.copy()
        outlier_data.loc[0, "numeric_1"] = 1000  # Extreme outlier

        cleaned_data = detector.fit_transform(outlier_data[["numeric_1", "numeric_2"]])

        # Should have removed or modified the outlier
        assert cleaned_data["numeric_1"].max() < 1000

    def test_feature_selector(self, sample_data):
        """Test feature selection"""
        selector = FeatureSelector(method="correlation", max_features=2)

        X = sample_data[["numeric_1", "numeric_2"]]
        y = sample_data["target"]

        selected_data = selector.fit_transform(X, y)

        # Should select at most 2 features
        assert selected_data.shape[1] <= 2

        # Check which features were selected
        selected_features = selector.get_selected_features()
        assert len(selected_features) <= 2

    def test_preprocessing_pipeline(self, sample_data):
        """Test complete preprocessing pipeline"""
        pipeline = PreprocessingPipeline()

        # Add preprocessing steps
        pipeline.add_step("scaler", TimeSeriesScaler(method="standard"))
        pipeline.add_step("cyclical", CyclicalEncoder())
        pipeline.add_step("outlier_detector", OutlierDetector(method="iqr"))

        # Fit and transform
        X = sample_data[["numeric_1", "numeric_2", "cyclical"]]
        processed_data = pipeline.fit_transform(X, cyclical_periods={"cyclical": 24})

        assert processed_data.shape[0] == sample_data.shape[0]
        assert processed_data.shape[1] > X.shape[1]  # Should have added cyclical features

        # Test pipeline persistence
        with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp_file:
            pipeline.save(tmp_file.name)

            new_pipeline = PreprocessingPipeline.load(tmp_file.name)

            # Should produce same results
            new_processed_data = new_pipeline.transform(X, cyclical_periods={"cyclical": 24})
            pd.testing.assert_frame_equal(processed_data, new_processed_data)


class TestErrorHandling:
    """Test error handling across data pipeline"""

    def test_missing_file_handling(self):
        """Test handling of missing files"""
        loader = DataLoader(data_path="/non/existent/path")

        with pytest.raises(FileNotFoundError):
            loader.load_metar_data("missing_file.csv")

    def test_corrupt_data_handling(self):
        """Test handling of corrupt data"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_file:
            # Write corrupt CSV
            tmp_file.write("header1,header2\nvalue1\nvalue2,value3,extra_value\n")
            tmp_file.flush()

            loader = DataLoader()

            # Should handle gracefully
            try:
                df = loader.load_metar_data(tmp_file.name)
                # If it loads, check it has reasonable shape
                assert df.shape[0] > 0
            except (pd.errors.ParserError, ValueError):
                # Expected for corrupt data
                pass

            # Cleanup
            Path(tmp_file.name).unlink()

    def test_empty_data_handling(self):
        """Test handling of empty datasets"""
        empty_df = pd.DataFrame()

        validator = DataValidator()

        # Should handle empty data gracefully
        validation_report = validator.validate_data(empty_df)
        assert "errors" in validation_report
        assert len(validation_report["errors"]) > 0  # Should report empty data as error

    def test_invalid_configuration_handling(self):
        """Test handling of invalid configurations"""
        # Test invalid feature engineering config
        with pytest.raises((ValueError, TypeError)):
            WeatherFeatureEngineer(invalid_param="invalid")

        # Test invalid preprocessing config
        with pytest.raises((ValueError, TypeError)):
            TimeSeriesScaler(method="invalid_method")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
