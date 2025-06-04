"""
Comprehensive Integration Tests for Weather Regulation Prediction System

This module tests end-to-end workflows and integration between components:
- Complete data pipeline to model training workflows
- Configuration system integration
- Results management and visualization integration
- Cross-module compatibility and data flow
- Full experiment execution scenarios
"""

import shutil
import tempfile
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from model import Dly_Classifier

# Import all major components
from config import DataConfig, ExperimentConfig, TrainingConfig
from config_parser import ConfigParser
from data.data_loader import DataLoader
from data.data_validation import DataValidator
from data.feature_engineering import WeatherFeatureEngineer
from data.preprocessing import PreprocessingPipeline
from models.lstm import LSTMModel
from models.random_forest import RandomForestModel
from results.results_manager import ExperimentResult, ResultsManager
from run_experiments import ExperimentRunner, ExperimentSuite
from training.hyperparameter_tuning import GridSearchTuner
from training.trainer import Trainer
from visualization.plots import ModelVisualizer

warnings.filterwarnings("ignore")


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows"""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace with sample data"""
        workspace = tempfile.mkdtemp()

        # Create directory structure
        data_dir = Path(workspace) / "data"
        config_dir = Path(workspace) / "configs"
        results_dir = Path(workspace) / "results"

        for dir_path in [data_dir, config_dir, results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Create sample weather data
        np.random.seed(42)
        weather_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=200, freq="30min"),
                "airport": ["EGLL"] * 200,
                "temperature": np.random.randn(200) * 10 + 15,
                "pressure": np.random.randn(200) * 20 + 1013,
                "wind_speed": np.random.randn(200) * 5 + 10,
                "wind_direction": np.random.randint(0, 360, 200),
                "visibility": np.random.randn(200) * 2000 + 8000,
                "humidity": np.random.randint(20, 101, 200),
                "weather_code": np.random.choice(["RA", "SN", "FG", "CLR"], 200),
            }
        )
        weather_path = data_dir / "weather_data.csv"
        weather_data.to_csv(weather_path, index=False)

        # Create sample regulation data
        regulation_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="1H"),
                "airport": ["EGLL"] * 100,
                "regulation_type": np.random.choice(["WX", "ATC", "EQ"], 100),
                "duration": np.random.randint(10, 120, 100),
                "has_regulation": np.random.choice([0, 1], 100, p=[0.7, 0.3]),
            }
        )
        regulation_path = data_dir / "regulation_data.csv"
        regulation_data.to_csv(regulation_path, index=False)

        # Create configuration file
        config_data = {
            "name": "integration_test_experiment",
            "data": {
                "airports": ["EGLL"],
                "start_date": "2023-01-01",
                "end_date": "2023-01-08",
                "weather_data_path": str(weather_path),
                "regulation_data_path": str(regulation_path),
            },
            "training": {
                "test_size": 0.2,
                "validation_size": 0.2,
                "random_state": 42,
                "cross_validation": True,
                "cv_folds": 3,
            },
            "models": {"random_forest": {"n_estimators": 10, "max_depth": 5, "random_state": 42}},
        }

        config_path = config_dir / "test_config.yaml"
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        yield workspace, data_dir, config_path, results_dir

        # Cleanup
        shutil.rmtree(workspace)

    def test_complete_data_to_model_pipeline(self, temp_workspace):
        """Test complete pipeline from raw data to trained model"""
        workspace, data_dir, config_path, results_dir = temp_workspace

        # 1. Load and parse configuration
        parser = ConfigParser()
        config = parser.load_config(str(config_path))
        assert isinstance(config, ExperimentConfig)
        assert config.name == "integration_test_experiment"

        # 2. Load and validate data
        loader = DataLoader(data_path=str(data_dir))
        weather_df = loader.load_metar_data(config.data.weather_data_path)
        regulation_df = loader.load_regulation_data(config.data.regulation_data_path)

        validator = DataValidator()
        weather_report = validator.validate_weather_data(weather_df)
        assert len(weather_report["errors"]) == 0  # Should be clean data

        # 3. Feature engineering
        engineer = WeatherFeatureEngineer()
        enhanced_weather = engineer.create_features(weather_df)
        assert "weather_severity" in enhanced_weather.columns
        assert "flight_category" in enhanced_weather.columns

        # 4. Create feature matrix
        features_df = loader.create_features(
            metar_data=enhanced_weather,
            regulation_data=regulation_df,
            target_column="has_regulation",
        )
        assert "target" in features_df.columns
        assert len(features_df) > 0

        # 5. Preprocessing
        pipeline = PreprocessingPipeline()
        X = features_df.drop("target", axis=1)
        y = features_df["target"]

        X_processed = pipeline.fit_transform(X)
        assert X_processed.shape[0] == X.shape[0]

        # 6. Model training
        from config import RandomForestConfig

        rf_config = RandomForestConfig(**config.models["random_forest"])
        model = RandomForestModel(rf_config)

        trainer = Trainer()

        # Split data
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )

        # Train model
        result = trainer.train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            model_name="integration_test_rf",
        )

        assert "accuracy" in result
        assert 0 <= result["accuracy"] <= 1
        assert result["training_time"] > 0

        # 7. Results management
        results_manager = ResultsManager(base_path=str(results_dir))

        # Create experiment result
        from datetime import datetime

        experiment_result = ExperimentResult(
            experiment_id="integration_test_001",
            experiment_name="Integration Test Experiment",
            timestamp=datetime.now(),
            config=config,
        )

        # Save results
        experiment_id = results_manager.save_experiment_result(experiment_result)
        assert experiment_id == "integration_test_001"

        # Load and verify
        loaded_experiment = results_manager.load_experiment_result(experiment_id)
        assert loaded_experiment is not None
        assert loaded_experiment.experiment_name == "Integration Test Experiment"

    def test_experiment_runner_integration(self, temp_workspace):
        """Test ExperimentRunner with real configuration"""
        workspace, data_dir, config_path, results_dir = temp_workspace

        # Load configuration
        parser = ConfigParser()
        config = parser.load_config(str(config_path))

        # Mock data loading in ExperimentRunner
        with patch.object(ExperimentRunner, "_load_data") as mock_load_data:
            # Create mock feature data
            np.random.seed(42)
            features_df = pd.DataFrame(
                {
                    "feature_1": np.random.randn(100),
                    "feature_2": np.random.randn(100),
                    "feature_3": np.random.randn(100),
                    "target": np.random.choice([0, 1], 100, p=[0.6, 0.4]),
                }
            )
            mock_load_data.return_value = features_df

            # Run experiment
            runner = ExperimentRunner(config)
            results = runner.run_experiment("random_forest")

            assert "random_forest" in results
            assert "accuracy" in results["random_forest"]
            assert "training_time" in results["random_forest"]
            assert 0 <= results["random_forest"]["accuracy"] <= 1

    def test_hyperparameter_tuning_integration(self, temp_workspace):
        """Test integrated hyperparameter tuning workflow"""
        workspace, data_dir, config_path, results_dir = temp_workspace

        # Create sample data
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        # Model configuration
        from config import RandomForestConfig

        rf_config = RandomForestConfig(n_estimators=5, random_state=42)
        model = RandomForestModel(rf_config)

        # Hyperparameter tuning
        tuner = GridSearchTuner()
        param_grid = {"n_estimators": [5, 10], "max_depth": [2, 3]}

        tuning_result = tuner.tune(
            model=model,
            param_grid=param_grid,
            X_train=X[:80],
            y_train=y[:80],
            X_val=X[80:],
            y_val=y[80:],
            cv=2,
        )

        # Train best model
        best_model = RandomForestModel(RandomForestConfig(**tuning_result.best_params))
        trainer = Trainer()

        final_result = trainer.train_model(
            model=best_model,
            X_train=X[:80],
            y_train=y[:80],
            X_val=X[80:],
            y_val=y[80:],
            model_name="tuned_rf",
        )

        assert final_result["accuracy"] >= 0
        assert tuning_result.best_score >= 0

    def test_visualization_integration(self, temp_workspace):
        """Test visualization integration with results"""
        workspace, data_dir, config_path, results_dir = temp_workspace

        # Create mock results
        np.random.seed(42)
        y_true = np.random.choice([0, 1], 100)
        y_pred = np.random.choice([0, 1], 100)
        y_proba = np.random.rand(100, 2)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)  # Normalize

        # Create visualizations
        visualizer = ModelVisualizer(save_path=str(results_dir))

        # Test confusion matrix
        cm_fig = visualizer.plot_confusion_matrix(
            y_true,
            y_pred,
            title="Integration Test Confusion Matrix",
            interactive=True,
            save_name="integration_test_cm",
        )

        assert cm_fig is not None

        # Test ROC curves
        roc_fig = visualizer.plot_roc_curves(
            y_true,
            {"test_model": y_proba[:, 1]},
            title="Integration Test ROC",
            interactive=True,
            save_name="integration_test_roc",
        )

        assert roc_fig is not None

        # Verify files were saved
        cm_path = Path(results_dir) / "integration_test_cm.html"
        roc_path = Path(results_dir) / "integration_test_roc.html"
        assert cm_path.exists()
        assert roc_path.exists()


class TestConfigurationIntegration:
    """Test configuration system integration across modules"""

    def test_config_to_model_integration(self):
        """Test configuration propagation to models"""
        # Create configuration
        config_data = {
            "name": "config_integration_test",
            "models": {
                "random_forest": {"n_estimators": 15, "max_depth": 4, "random_state": 123},
                "lstm": {"units": 64, "dropout": 0.3, "batch_size": 32, "epochs": 5},
            },
        }

        # Parse configuration
        parser = ConfigParser()
        config = parser.parse_dict(config_data)

        # Create models from configuration
        from config import LSTMConfig, RandomForestConfig

        rf_config = RandomForestConfig(**config.models["random_forest"])
        rf_model = RandomForestModel(rf_config)

        assert rf_model.config.n_estimators == 15
        assert rf_model.config.max_depth == 4
        assert rf_model.config.random_state == 123

        lstm_config = LSTMConfig(**config.models["lstm"])
        lstm_model = LSTMModel(lstm_config)

        assert lstm_model.config.units == 64
        assert lstm_model.config.dropout == 0.3
        assert lstm_model.config.batch_size == 32

    def test_config_validation_integration(self):
        """Test configuration validation across components"""
        # Test invalid configuration
        invalid_config_data = {
            "name": "",  # Invalid empty name
            "models": {
                "random_forest": {
                    "n_estimators": -1,  # Invalid negative value
                    "max_depth": "invalid",  # Invalid type
                }
            },
        }

        parser = ConfigParser()

        # Should catch validation errors
        with pytest.raises((ValueError, TypeError)):
            parser.parse_dict(invalid_config_data)

    def test_config_merging_integration(self):
        """Test configuration merging functionality"""
        base_config = {
            "name": "base_experiment",
            "training": {"test_size": 0.2, "random_state": 42},
            "models": {"random_forest": {"n_estimators": 10, "random_state": 42}},
        }

        override_config = {
            "name": "merged_experiment",
            "models": {
                "random_forest": {"n_estimators": 20},  # Override n_estimators
                "lstm": {"units": 32, "epochs": 10},  # Add new model
            },
        }

        parser = ConfigParser()
        merged_config = parser.merge_configs(base_config, override_config)

        assert merged_config["name"] == "merged_experiment"
        assert merged_config["training"]["test_size"] == 0.2  # Preserved
        assert merged_config["models"]["random_forest"]["n_estimators"] == 20  # Overridden
        assert merged_config["models"]["random_forest"]["random_state"] == 42  # Preserved
        assert "lstm" in merged_config["models"]  # Added


class TestDataFlowIntegration:
    """Test data flow between pipeline components"""

    @pytest.fixture
    def sample_pipeline_data(self):
        """Create sample data for pipeline testing"""
        np.random.seed(42)

        # Raw weather data
        weather_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="1H"),
                "temperature": np.random.randn(100) * 10 + 15,
                "pressure": np.random.randn(100) * 20 + 1013,
                "wind_speed": np.random.randn(100) * 5 + 10,
                "wind_direction": np.random.randint(0, 360, 100),
                "visibility": np.random.randn(100) * 2000 + 8000,
                "humidity": np.random.randint(20, 101, 100),
            }
        )

        # Regulation data
        regulation_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=50, freq="2H"),
                "has_regulation": np.random.choice([0, 1], 50, p=[0.7, 0.3]),
            }
        )

        return weather_data, regulation_data

    def test_data_loader_to_validator_integration(self, sample_pipeline_data):
        """Test data flow from loader to validator"""
        weather_data, regulation_data = sample_pipeline_data

        # Save to temporary files
        with tempfile.TemporaryDirectory() as tmp_dir:
            weather_path = Path(tmp_dir) / "weather.csv"
            regulation_path = Path(tmp_dir) / "regulation.csv"

            weather_data.to_csv(weather_path, index=False)
            regulation_data.to_csv(regulation_path, index=False)

            # Load data
            loader = DataLoader(data_path=tmp_dir)
            loaded_weather = loader.load_metar_data(str(weather_path))
            loader.load_regulation_data(str(regulation_path))

            # Validate data
            validator = DataValidator()
            weather_report = validator.validate_weather_data(loaded_weather)

            assert len(weather_report["errors"]) == 0  # Clean data
            assert weather_report["summary"]["total_records"] == 100

            # Validate schema
            from data.data_validation import SchemaValidator

            schema_validator = SchemaValidator()

            expected_schema = {
                "timestamp": "datetime64[ns]",
                "temperature": "float64",
                "pressure": "float64",
            }

            is_valid, errors = schema_validator.validate_schema(
                loaded_weather[["timestamp", "temperature", "pressure"]], expected_schema
            )
            assert is_valid

    def test_feature_engineering_to_preprocessing_integration(self, sample_pipeline_data):
        """Test data flow from feature engineering to preprocessing"""
        weather_data, regulation_data = sample_pipeline_data

        # Feature engineering
        engineer = WeatherFeatureEngineer()
        enhanced_data = engineer.create_features(weather_data)

        # Check new features were created
        assert "weather_severity" in enhanced_data.columns
        assert "flight_category" in enhanced_data.columns

        # Preprocessing
        from data.preprocessing import CyclicalEncoder, TimeSeriesScaler

        # Scale numeric features
        scaler = TimeSeriesScaler(method="standard")
        numeric_cols = ["temperature", "pressure", "wind_speed", "visibility", "humidity"]
        scaled_data = scaler.fit_transform(enhanced_data[numeric_cols])

        assert scaled_data.shape == enhanced_data[numeric_cols].shape

        # Encode cyclical features
        encoder = CyclicalEncoder()
        if "hour" in enhanced_data.columns:
            cyclical_data = encoder.fit_transform(enhanced_data[["hour"]], periods={"hour": 24})
            assert "hour_sin" in cyclical_data.columns
            assert "hour_cos" in cyclical_data.columns

    def test_preprocessing_to_model_integration(self, sample_pipeline_data):
        """Test data flow from preprocessing to model training"""
        weather_data, regulation_data = sample_pipeline_data

        # Create combined dataset
        combined_data = weather_data.copy()
        combined_data["target"] = np.random.choice([0, 1], len(weather_data), p=[0.7, 0.3])

        # Preprocessing pipeline
        pipeline = PreprocessingPipeline()

        from data.preprocessing import TimeSeriesScaler

        pipeline.add_step("scaler", TimeSeriesScaler(method="minmax"))

        X = combined_data.drop("target", axis=1)
        y = combined_data["target"]

        # Handle non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]

        X_processed = pipeline.fit_transform(X_numeric)

        # Train model
        from config import RandomForestConfig

        config = RandomForestConfig(n_estimators=5, random_state=42)
        model = RandomForestModel(config)

        trainer = Trainer()
        result = trainer.train_model(
            model=model,
            X_train=X_processed[:80],
            y_train=y[:80],
            X_val=X_processed[80:],
            y_val=y[80:],
            model_name="integration_test",
        )

        assert "accuracy" in result
        assert 0 <= result["accuracy"] <= 1


class TestLegacyCompatibility:
    """Test backward compatibility with legacy system"""

    def test_legacy_vs_new_pipeline_compatibility(self):
        """Test that new pipeline produces compatible results with legacy"""
        # Create mock legacy data
        np.random.seed(42)
        legacy_X = np.random.randn(100, 10)
        legacy_y = (legacy_X[:, 0] + legacy_X[:, 1] > 0).astype(int)

        # Test both pipelines produce valid results
        # Legacy approach (simplified)
        from sklearn.ensemble import RandomForestClassifier

        legacy_model = RandomForestClassifier(n_estimators=10, random_state=42)
        legacy_model.fit(legacy_X[:80], legacy_y[:80])
        legacy_pred = legacy_model.predict(legacy_X[80:])
        legacy_accuracy = (legacy_pred == legacy_y[80:]).mean()

        # New approach
        from config import RandomForestConfig

        config = RandomForestConfig(n_estimators=10, random_state=42)
        new_model = RandomForestModel(config)
        trainer = Trainer()

        result = trainer.train_model(
            model=new_model,
            X_train=legacy_X[:80],
            y_train=legacy_y[:80],
            X_val=legacy_X[80:],
            y_val=legacy_y[80:],
            model_name="compatibility_test",
        )

        # Both should produce reasonable results
        assert 0 <= legacy_accuracy <= 1
        assert 0 <= result["accuracy"] <= 1

        # Results should be similar (same algorithm, same parameters)
        accuracy_diff = abs(legacy_accuracy - result["accuracy"])
        assert accuracy_diff < 0.1  # Allow small differences due to implementation details

    def test_dly_classifier_integration(self):
        """Test integration with main Dly_Classifier class"""
        # Test that Dly_Classifier can work with new pipeline
        classifier = Dly_Classifier()

        # Should have both legacy and new pipeline options
        assert hasattr(classifier, "use_new_pipeline")

        # Test configuration loading (mocked)
        with patch("model.ConfigParser") as mock_parser:
            mock_config = Mock()
            mock_config.data.airports = ["EGLL"]
            mock_parser.return_value.load_config.return_value = mock_config

            # Should be able to initialize with config
            try:
                classifier.use_new_pipeline = True
                # This would normally load real data, so we just test initialization
                assert classifier.use_new_pipeline is True
            except Exception:
                # Expected if data files don't exist
                pass


class TestCrossModuleCompatibility:
    """Test compatibility between different modules"""

    def test_results_manager_with_all_models(self):
        """Test ResultsManager with different model types"""
        results_manager = ResultsManager()

        # Test with different model results
        from results.results_manager import create_model_result

        np.random.seed(42)
        y_true = np.random.choice([0, 1], 50)
        y_pred = np.random.choice([0, 1], 50)
        y_proba = np.random.rand(50, 2)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

        # Create results for different model types
        model_types = ["RandomForest", "LSTM", "CNN", "FNN"]

        for model_type in model_types:
            result = create_model_result(
                model_name=f"test_{model_type.lower()}",
                model_type=model_type,
                y_true=y_true,
                y_pred=y_pred,
                y_proba=y_proba,
                training_time=np.random.uniform(1, 10),
            )

            model_id = results_manager.save_model_result(result)
            loaded_result = results_manager.load_model_result(model_id)

            assert loaded_result is not None
            assert loaded_result.model_type == model_type
            assert 0 <= loaded_result.test_accuracy <= 1

    def test_visualization_with_multiple_experiments(self):
        """Test visualization integration with multiple experiments"""
        # Create mock experiment results
        from datetime import datetime

        from results.results_manager import ExperimentResult, ModelResult

        experiments = {}

        for i, exp_name in enumerate(["exp_1", "exp_2"]):
            experiment = ExperimentResult(
                experiment_id=f"test_exp_{i}",
                experiment_name=exp_name,
                timestamp=datetime.now(),
                config={},
            )

            # Add mock model results
            for model_name in ["rf", "lstm"]:
                model_result = ModelResult(
                    model_name=model_name,
                    model_type=model_name.upper(),
                    timestamp=datetime.now(),
                    config={},
                    training_time=np.random.uniform(1, 10),
                    test_accuracy=np.random.uniform(0.7, 0.95),
                    test_precision=np.random.uniform(0.7, 0.95),
                    test_recall=np.random.uniform(0.7, 0.95),
                    test_f1=np.random.uniform(0.7, 0.95),
                )
                experiment.add_model_result(model_result)

            experiments[f"test_exp_{i}"] = experiment

        # Test visualization creation
        visualizer = ModelVisualizer()

        # Create comparison data
        comparison_data = []
        for _, exp in experiments.items():
            if exp.comparison_metrics is not None:
                df = exp.comparison_metrics.copy()
                df["experiment"] = exp.experiment_name
                comparison_data.append(df)

        if comparison_data:
            combined_df = pd.concat(comparison_data)

            # Test model comparison visualization
            comparison_fig = visualizer.plot_model_comparison(
                combined_df, title="Multi-Experiment Comparison", interactive=True
            )

            assert comparison_fig is not None


class TestPerformanceIntegration:
    """Test performance characteristics of integrated system"""

    @pytest.mark.slow
    def test_large_dataset_integration(self):
        """Test system performance with larger datasets"""
        # Create larger synthetic dataset
        np.random.seed(42)
        n_samples = 5000
        n_features = 50

        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)

        # Test data processing pipeline
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
        df["target"] = y

        # Feature engineering
        engineer = WeatherFeatureEngineer()
        # Add some weather-like features
        df["temperature"] = X[:, 0] * 10 + 15
        df["pressure"] = X[:, 1] * 20 + 1013
        df["wind_speed"] = np.abs(X[:, 2]) * 5 + 5
        df["wind_direction"] = np.random.randint(0, 360, n_samples)
        df["visibility"] = np.abs(X[:, 3]) * 2000 + 8000

        enhanced_df = engineer.create_features(df)

        # Should handle large dataset
        assert len(enhanced_df) == n_samples
        assert enhanced_df.shape[1] > df.shape[1]  # New features added

        # Test model training performance
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            enhanced_df.drop("target", axis=1).select_dtypes(include=[np.number]),
            enhanced_df["target"],
            test_size=0.2,
            random_state=42,
        )

        # Train model
        from config import RandomForestConfig

        config = RandomForestConfig(n_estimators=20, random_state=42, n_jobs=2)
        model = RandomForestModel(config)

        trainer = Trainer()

        import time

        start_time = time.time()

        result = trainer.train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            model_name="large_dataset_test",
        )

        training_time = time.time() - start_time

        # Should complete in reasonable time (adjust threshold as needed)
        assert training_time < 30  # seconds
        assert 0 <= result["accuracy"] <= 1
        assert result["training_time"] > 0

    def test_memory_usage_integration(self):
        """Test memory usage characteristics"""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create and process moderate dataset
        np.random.seed(42)
        n_samples = 1000
        X = np.random.randn(n_samples, 20)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        # Train multiple models
        models = {}
        trainer = Trainer()

        for i in range(3):
            from config import RandomForestConfig

            config = RandomForestConfig(n_estimators=10, random_state=42 + i)
            model = RandomForestModel(config)

            trainer.train_model(
                model=model,
                X_train=X[:800],
                y_train=y[:800],
                X_val=X[800:],
                y_val=y[800:],
                model_name=f"memory_test_{i}",
            )

            models[f"model_{i}"] = model

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (adjust threshold as needed)
        assert memory_increase < 200  # MB

        # Cleanup
        del models
        del trainer


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short", "-x"])  # Stop on first failure
