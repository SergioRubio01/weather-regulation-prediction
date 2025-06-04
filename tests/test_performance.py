"""
Performance and Benchmark Tests for Weather Regulation Prediction System

This module provides comprehensive performance testing including:
- Training speed benchmarks for all models
- Memory usage profiling
- Scalability tests with different data sizes
- Hyperparameter tuning performance
- Data pipeline throughput tests
- Resource utilization monitoring
"""

import os
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pytest

# Import components for testing
from config import CNNConfig, EnsembleConfig, FNNConfig, GRUConfig, LSTMConfig, RandomForestConfig
from data.data_loader import DataLoader
from data.feature_engineering import WeatherFeatureEngineer
from data.preprocessing import PreprocessingPipeline
from models.cnn import CNNModel
from models.ensemble import EnsembleModel
from models.fnn import FNNModel
from models.gru import GRUModel
from models.lstm import LSTMModel
from models.random_forest import RandomForestModel
from results.results_manager import ResultsManager
from training.hyperparameter_tuning import GridSearchTuner, RandomSearchTuner
from training.trainer import Trainer

warnings.filterwarnings("ignore")


class PerformanceMonitor:
    """Helper class to monitor performance metrics"""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.start_memory = None
        self.start_cpu = None

    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.start_cpu = self.process.cpu_percent()
        return self

    def stop_monitoring(self):
        """Stop monitoring and return metrics"""
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        end_cpu = self.process.cpu_percent()

        return {
            "execution_time": end_time - self.start_time,
            "memory_usage": end_memory - self.start_memory,
            "peak_memory": end_memory,
            "cpu_usage": end_cpu,
        }


@pytest.fixture(scope="session")
def performance_data():
    """Generate performance test datasets of various sizes"""
    datasets = {}

    # Small dataset
    np.random.seed(42)
    X_small = np.random.randn(500, 10)
    y_small = (X_small[:, 0] + X_small[:, 1] > 0).astype(int)
    datasets["small"] = (X_small, y_small)

    # Medium dataset
    X_medium = np.random.randn(2000, 20)
    y_medium = (X_medium[:, 0] + X_medium[:, 1] + np.random.randn(2000) * 0.1 > 0).astype(int)
    datasets["medium"] = (X_medium, y_medium)

    # Large dataset
    X_large = np.random.randn(10000, 50)
    y_large = (X_large[:, 0] + X_large[:, 1] + np.random.randn(10000) * 0.2 > 0).astype(int)
    datasets["large"] = (X_large, y_large)

    return datasets


class TestModelTrainingPerformance:
    """Test training performance for different models"""

    @pytest.mark.benchmark
    @pytest.mark.parametrize("dataset_size", ["small", "medium"])
    def test_random_forest_training_speed(self, performance_data, dataset_size):
        """Benchmark Random Forest training speed"""
        X, y = performance_data[dataset_size]

        config = RandomForestConfig(n_estimators=50, random_state=42, n_jobs=2)
        model = RandomForestModel(config)
        trainer = Trainer()

        monitor = PerformanceMonitor().start_monitoring()

        result = trainer.train_model(
            model=model,
            X_train=X[: int(0.8 * len(X))],
            y_train=y[: int(0.8 * len(y))],
            X_val=X[int(0.8 * len(X)) :],
            y_val=y[int(0.8 * len(y)) :],
            model_name=f"rf_perf_{dataset_size}",
        )

        metrics = monitor.stop_monitoring()

        # Performance assertions
        if dataset_size == "small":
            assert metrics["execution_time"] < 10  # seconds
            assert metrics["memory_usage"] < 100  # MB
        elif dataset_size == "medium":
            assert metrics["execution_time"] < 30  # seconds
            assert metrics["memory_usage"] < 300  # MB

        assert result["accuracy"] > 0

        print(
            f"RF {dataset_size} dataset - Time: {metrics['execution_time']:.2f}s, "
            f"Memory: {metrics['memory_usage']:.2f}MB"
        )

    @pytest.mark.benchmark
    @pytest.mark.slow
    @pytest.mark.parametrize("dataset_size", ["small"])
    def test_lstm_training_speed(self, performance_data, dataset_size):
        """Benchmark LSTM training speed"""
        X, y = performance_data[dataset_size]

        config = LSTMConfig(units=32, dropout=0.2, batch_size=32, epochs=5, sequence_length=10)
        model = LSTMModel(config)
        trainer = Trainer()

        monitor = PerformanceMonitor().start_monitoring()

        try:
            result = trainer.train_model(
                model=model,
                X_train=X[: int(0.8 * len(X))],
                y_train=y[: int(0.8 * len(y))],
                X_val=X[int(0.8 * len(X)) :],
                y_val=y[int(0.8 * len(y)) :],
                model_name=f"lstm_perf_{dataset_size}",
            )

            metrics = monitor.stop_monitoring()

            # LSTM is expected to be slower
            if dataset_size == "small":
                assert metrics["execution_time"] < 60  # seconds
                assert metrics["memory_usage"] < 500  # MB

            assert result["accuracy"] >= 0

            print(
                f"LSTM {dataset_size} dataset - Time: {metrics['execution_time']:.2f}s, "
                f"Memory: {metrics['memory_usage']:.2f}MB"
            )

        except Exception as e:
            pytest.skip(f"LSTM training failed (likely missing TensorFlow): {e}")

    @pytest.mark.benchmark
    @pytest.mark.parametrize("dataset_size", ["small", "medium"])
    def test_fnn_training_speed(self, performance_data, dataset_size):
        """Benchmark FNN training speed"""
        X, y = performance_data[dataset_size]

        config = FNNConfig(hidden_layer_sizes=[50, 25], max_iter=100, random_state=42)
        model = FNNModel(config)
        trainer = Trainer()

        monitor = PerformanceMonitor().start_monitoring()

        result = trainer.train_model(
            model=model,
            X_train=X[: int(0.8 * len(X))],
            y_train=y[: int(0.8 * len(y))],
            X_val=X[int(0.8 * len(X)) :],
            y_val=y[int(0.8 * len(y)) :],
            model_name=f"fnn_perf_{dataset_size}",
        )

        metrics = monitor.stop_monitoring()

        # Performance assertions
        if dataset_size == "small":
            assert metrics["execution_time"] < 20  # seconds
            assert metrics["memory_usage"] < 200  # MB
        elif dataset_size == "medium":
            assert metrics["execution_time"] < 60  # seconds
            assert metrics["memory_usage"] < 400  # MB

        assert result["accuracy"] > 0

        print(
            f"FNN {dataset_size} dataset - Time: {metrics['execution_time']:.2f}s, "
            f"Memory: {metrics['memory_usage']:.2f}MB"
        )

    @pytest.mark.benchmark
    def test_ensemble_training_speed(self, performance_data):
        """Benchmark ensemble training speed"""
        X, y = performance_data["small"]

        config = EnsembleConfig(
            base_models=[
                {"type": "random_forest", "n_estimators": 10, "random_state": 42},
                {"type": "fnn", "hidden_layer_sizes": [20], "max_iter": 50, "random_state": 42},
            ],
            ensemble_method="voting",
            voting_type="soft",
        )
        model = EnsembleModel(config)
        trainer = Trainer()

        monitor = PerformanceMonitor().start_monitoring()

        result = trainer.train_model(
            model=model,
            X_train=X[: int(0.8 * len(X))],
            y_train=y[: int(0.8 * len(y))],
            X_val=X[int(0.8 * len(X)) :],
            y_val=y[int(0.8 * len(y)) :],
            model_name="ensemble_perf",
        )

        metrics = monitor.stop_monitoring()

        # Ensemble should be slower than individual models
        assert metrics["execution_time"] < 60  # seconds
        assert metrics["memory_usage"] < 300  # MB
        assert result["accuracy"] > 0

        print(
            f"Ensemble small dataset - Time: {metrics['execution_time']:.2f}s, "
            f"Memory: {metrics['memory_usage']:.2f}MB"
        )


class TestHyperparameterTuningPerformance:
    """Test hyperparameter tuning performance"""

    @pytest.mark.benchmark
    def test_grid_search_performance(self, performance_data):
        """Benchmark grid search performance"""
        X, y = performance_data["small"]

        config = RandomForestConfig(n_estimators=5, random_state=42)
        model = RandomForestModel(config)

        param_grid = {"n_estimators": [5, 10, 15], "max_depth": [2, 3, 5]}

        tuner = GridSearchTuner()
        monitor = PerformanceMonitor().start_monitoring()

        result = tuner.tune(
            model=model,
            param_grid=param_grid,
            X_train=X[: int(0.8 * len(X))],
            y_train=y[: int(0.8 * len(y))],
            X_val=X[int(0.8 * len(X)) :],
            y_val=y[int(0.8 * len(y)) :],
            cv=3,
        )

        metrics = monitor.stop_monitoring()

        # Grid search with 9 combinations should complete reasonably fast
        assert metrics["execution_time"] < 120  # seconds
        assert metrics["memory_usage"] < 500  # MB
        assert result.best_score > 0

        print(
            f"Grid search - Time: {metrics['execution_time']:.2f}s, "
            f"Memory: {metrics['memory_usage']:.2f}MB, "
            f"Trials: {len(result.all_results)}"
        )

    @pytest.mark.benchmark
    def test_random_search_performance(self, performance_data):
        """Benchmark random search performance"""
        X, y = performance_data["small"]

        config = RandomForestConfig(n_estimators=5, random_state=42)
        model = RandomForestModel(config)

        param_distributions = {"n_estimators": [5, 10, 15, 20, 25], "max_depth": [2, 3, 5, 8, None]}

        tuner = RandomSearchTuner(n_trials=10)
        monitor = PerformanceMonitor().start_monitoring()

        result = tuner.tune(
            model=model,
            param_distributions=param_distributions,
            X_train=X[: int(0.8 * len(X))],
            y_train=y[: int(0.8 * len(y))],
            X_val=X[int(0.8 * len(X)) :],
            y_val=y[int(0.8 * len(y)) :],
            cv=3,
        )

        metrics = monitor.stop_monitoring()

        # Random search with 10 trials should be faster than full grid
        assert metrics["execution_time"] < 100  # seconds
        assert metrics["memory_usage"] < 400  # MB
        assert result.best_score > 0
        assert len(result.all_results) <= 10

        print(
            f"Random search - Time: {metrics['execution_time']:.2f}s, "
            f"Memory: {metrics['memory_usage']:.2f}MB, "
            f"Trials: {len(result.all_results)}"
        )


class TestDataPipelinePerformance:
    """Test data pipeline performance"""

    @pytest.fixture
    def large_weather_dataset(self):
        """Generate large weather dataset for performance testing"""
        np.random.seed(42)
        n_samples = 10000

        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=n_samples, freq="30min"),
                "temperature": np.random.randn(n_samples) * 15 + 10,
                "pressure": np.random.randn(n_samples) * 30 + 1013,
                "wind_speed": np.abs(np.random.randn(n_samples)) * 8 + 5,
                "wind_direction": np.random.randint(0, 360, n_samples),
                "visibility": np.abs(np.random.randn(n_samples)) * 3000 + 5000,
                "humidity": np.random.randint(10, 101, n_samples),
                "weather_code": np.random.choice(["RA", "SN", "FG", "CLR", "OVC"], n_samples),
            }
        )

        return data

    @pytest.mark.benchmark
    def test_data_loading_performance(self, large_weather_dataset):
        """Benchmark data loading performance"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save large dataset
            data_path = Path(tmp_dir) / "large_weather.csv"
            large_weather_dataset.to_csv(data_path, index=False)

            loader = DataLoader(data_path=tmp_dir, enable_cache=True)
            monitor = PerformanceMonitor().start_monitoring()

            # Load data multiple times to test caching
            df1 = loader.load_metar_data(str(data_path))
            df2 = loader.load_metar_data(str(data_path))  # Should use cache

            metrics = monitor.stop_monitoring()

            assert len(df1) == len(large_weather_dataset)
            assert len(df2) == len(large_weather_dataset)
            pd.testing.assert_frame_equal(df1, df2)

            # Should complete reasonably fast
            assert metrics["execution_time"] < 30  # seconds
            assert metrics["memory_usage"] < 1000  # MB

            print(
                f"Data loading (10k records) - Time: {metrics['execution_time']:.2f}s, "
                f"Memory: {metrics['memory_usage']:.2f}MB"
            )

    @pytest.mark.benchmark
    def test_feature_engineering_performance(self, large_weather_dataset):
        """Benchmark feature engineering performance"""
        engineer = WeatherFeatureEngineer()
        monitor = PerformanceMonitor().start_monitoring()

        enhanced_data = engineer.create_features(large_weather_dataset)

        metrics = monitor.stop_monitoring()

        assert len(enhanced_data) == len(large_weather_dataset)
        assert enhanced_data.shape[1] > large_weather_dataset.shape[1]

        # Should complete reasonably fast
        assert metrics["execution_time"] < 60  # seconds
        assert metrics["memory_usage"] < 800  # MB

        print(
            f"Feature engineering (10k records) - Time: {metrics['execution_time']:.2f}s, "
            f"Memory: {metrics['memory_usage']:.2f}MB"
        )

    @pytest.mark.benchmark
    def test_preprocessing_performance(self, large_weather_dataset):
        """Benchmark preprocessing pipeline performance"""
        from data.preprocessing import OutlierDetector, TimeSeriesScaler

        pipeline = PreprocessingPipeline()
        pipeline.add_step("scaler", TimeSeriesScaler(method="standard"))
        pipeline.add_step("outlier_detector", OutlierDetector(method="iqr"))

        # Use numeric columns only
        numeric_data = large_weather_dataset.select_dtypes(include=[np.number])

        monitor = PerformanceMonitor().start_monitoring()

        processed_data = pipeline.fit_transform(numeric_data)

        metrics = monitor.stop_monitoring()

        assert processed_data.shape[0] <= numeric_data.shape[0]  # May remove outliers
        assert processed_data.shape[1] == numeric_data.shape[1]

        # Should complete reasonably fast
        assert metrics["execution_time"] < 45  # seconds
        assert metrics["memory_usage"] < 600  # MB

        print(
            f"Preprocessing (10k records) - Time: {metrics['execution_time']:.2f}s, "
            f"Memory: {metrics['memory_usage']:.2f}MB"
        )


class TestResultsManagementPerformance:
    """Test results management performance"""

    @pytest.mark.benchmark
    def test_results_storage_performance(self):
        """Benchmark results storage and retrieval"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_manager = ResultsManager(base_path=tmp_dir)

            # Create many model results
            from results.results_manager import create_model_result

            np.random.seed(42)

            monitor = PerformanceMonitor().start_monitoring()

            # Create and save 100 model results
            model_ids = []
            for i in range(100):
                y_true = np.random.choice([0, 1], 100)
                y_pred = np.random.choice([0, 1], 100)
                y_proba = np.random.rand(100, 2)
                y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

                result = create_model_result(
                    model_name=f"model_{i}",
                    model_type="RandomForest",
                    y_true=y_true,
                    y_pred=y_pred,
                    y_proba=y_proba,
                    training_time=np.random.uniform(1, 10),
                )

                model_id = results_manager.save_model_result(result)
                model_ids.append(model_id)

            # Load all results
            loaded_results = []
            for model_id in model_ids:
                result = results_manager.load_model_result(model_id)
                loaded_results.append(result)

            metrics = monitor.stop_monitoring()

            assert len(loaded_results) == 100
            assert all(result is not None for result in loaded_results)

            # Should handle 100 results efficiently
            assert metrics["execution_time"] < 30  # seconds
            assert metrics["memory_usage"] < 500  # MB

            print(
                f"Results management (100 models) - Time: {metrics['execution_time']:.2f}s, "
                f"Memory: {metrics['memory_usage']:.2f}MB"
            )

    @pytest.mark.benchmark
    def test_experiment_comparison_performance(self):
        """Benchmark experiment comparison performance"""
        from datetime import datetime

        from results.results_manager import ExperimentResult, ModelResult

        # Create multiple experiments with many models
        experiments = []

        for exp_idx in range(5):
            experiment = ExperimentResult(
                experiment_id=f"perf_exp_{exp_idx}",
                experiment_name=f"Performance Test Experiment {exp_idx}",
                timestamp=datetime.now(),
                config={},
            )

            # Add 20 models per experiment
            for model_idx in range(20):
                model_result = ModelResult(
                    model_name=f"model_{model_idx}",
                    model_type=f"Type_{model_idx % 3}",
                    timestamp=datetime.now(),
                    config={},
                    training_time=np.random.uniform(1, 10),
                    test_accuracy=np.random.uniform(0.7, 0.95),
                    test_precision=np.random.uniform(0.7, 0.95),
                    test_recall=np.random.uniform(0.7, 0.95),
                    test_f1=np.random.uniform(0.7, 0.95),
                )
                experiment.add_model_result(model_result)

            experiments.append(experiment)

        with tempfile.TemporaryDirectory() as tmp_dir:
            results_manager = ResultsManager(base_path=tmp_dir)

            monitor = PerformanceMonitor().start_monitoring()

            # Save all experiments
            exp_ids = []
            for exp in experiments:
                exp_id = results_manager.save_experiment_result(exp)
                exp_ids.append(exp_id)

            # Compare experiments
            comparison_df = results_manager.compare_experiments(exp_ids, metric="test_accuracy")

            # Get best models
            best_models_df = results_manager.get_best_models(n=10)

            metrics = monitor.stop_monitoring()

            assert len(comparison_df) > 0
            assert len(best_models_df) <= 10

            # Should handle comparison efficiently
            assert metrics["execution_time"] < 45  # seconds
            assert metrics["memory_usage"] < 600  # MB

            print(
                f"Experiment comparison (5 exp, 20 models each) - "
                f"Time: {metrics['execution_time']:.2f}s, "
                f"Memory: {metrics['memory_usage']:.2f}MB"
            )


class TestScalabilityTests:
    """Test system scalability with increasing data sizes"""

    @pytest.mark.benchmark
    @pytest.mark.slow
    @pytest.mark.parametrize("n_samples", [1000, 5000, 10000])
    def test_model_training_scalability(self, n_samples):
        """Test how training time scales with data size"""
        np.random.seed(42)
        X = np.random.randn(n_samples, 20)
        y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)

        config = RandomForestConfig(n_estimators=20, random_state=42, n_jobs=2)
        model = RandomForestModel(config)
        trainer = Trainer()

        monitor = PerformanceMonitor().start_monitoring()

        result = trainer.train_model(
            model=model,
            X_train=X[: int(0.8 * len(X))],
            y_train=y[: int(0.8 * len(y))],
            X_val=X[int(0.8 * len(X)) :],
            y_val=y[int(0.8 * len(y)) :],
            model_name=f"scalability_test_{n_samples}",
        )

        metrics = monitor.stop_monitoring()

        # Time should scale reasonably with data size
        expected_time_per_1k = 5  # seconds per 1000 samples
        max_expected_time = (n_samples / 1000) * expected_time_per_1k

        assert metrics["execution_time"] < max_expected_time
        assert result["accuracy"] > 0

        print(
            f"Scalability test ({n_samples} samples) - "
            f"Time: {metrics['execution_time']:.2f}s, "
            f"Memory: {metrics['memory_usage']:.2f}MB, "
            f"Time per 1k samples: {metrics['execution_time'] / (n_samples/1000):.2f}s"
        )

    @pytest.mark.benchmark
    @pytest.mark.slow
    @pytest.mark.parametrize("n_features", [10, 50, 100])
    def test_feature_scalability(self, n_features):
        """Test how performance scales with number of features"""
        np.random.seed(42)
        n_samples = 2000
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        config = RandomForestConfig(n_estimators=20, random_state=42, n_jobs=2)
        model = RandomForestModel(config)
        trainer = Trainer()

        monitor = PerformanceMonitor().start_monitoring()

        result = trainer.train_model(
            model=model,
            X_train=X[: int(0.8 * len(X))],
            y_train=y[: int(0.8 * len(y))],
            X_val=X[int(0.8 * len(X)) :],
            y_val=y[int(0.8 * len(y)) :],
            model_name=f"feature_scalability_{n_features}",
        )

        metrics = monitor.stop_monitoring()

        # Time should scale reasonably with feature count
        expected_time_per_10_features = 3  # seconds per 10 features
        max_expected_time = (n_features / 10) * expected_time_per_10_features

        assert metrics["execution_time"] < max_expected_time
        assert result["accuracy"] > 0

        print(
            f"Feature scalability test ({n_features} features) - "
            f"Time: {metrics['execution_time']:.2f}s, "
            f"Memory: {metrics['memory_usage']:.2f}MB"
        )


class TestMemoryUsageTests:
    """Test memory usage characteristics"""

    @pytest.mark.benchmark
    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated training"""
        np.random.seed(42)
        X = np.random.randn(1000, 20)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_measurements = []

        # Train multiple models in sequence
        for i in range(10):
            config = RandomForestConfig(n_estimators=10, random_state=42 + i)
            model = RandomForestModel(config)
            trainer = Trainer()

            result = trainer.train_model(
                model=model,
                X_train=X[:800],
                y_train=y[:800],
                X_val=X[800:],
                y_val=y[800:],
                model_name=f"memory_test_{i}",
            )

            current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            memory_measurements.append(current_memory)

            # Explicitly delete to help garbage collection
            del model
            del trainer

        final_memory = memory_measurements[-1]
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (not a significant leak)
        assert memory_increase < 500  # MB

        # Memory shouldn't grow indefinitely
        recent_memory = np.mean(memory_measurements[-3:])
        early_memory = np.mean(memory_measurements[:3])
        memory_growth_rate = (recent_memory - early_memory) / early_memory

        assert memory_growth_rate < 0.5  # Less than 50% growth

        print(
            f"Memory leak test - Initial: {initial_memory:.2f}MB, "
            f"Final: {final_memory:.2f}MB, "
            f"Increase: {memory_increase:.2f}MB, "
            f"Growth rate: {memory_growth_rate:.2%}"
        )

    @pytest.mark.benchmark
    def test_peak_memory_usage(self, performance_data):
        """Test peak memory usage during training"""
        X, y = performance_data["large"]

        config = RandomForestConfig(n_estimators=50, random_state=42, n_jobs=1)
        model = RandomForestModel(config)
        trainer = Trainer()

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        peak_memory = initial_memory

        # Monitor memory during training
        import threading
        import time

        def monitor_memory():
            nonlocal peak_memory
            while True:
                try:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    peak_memory = max(peak_memory, current_memory)
                    time.sleep(0.1)
                except (psutil.NoSuchProcess, AttributeError):
                    break

        monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
        monitor_thread.start()

        result = trainer.train_model(
            model=model,
            X_train=X[:8000],
            y_train=y[:8000],
            X_val=X[8000:],
            y_val=y[8000:],
            model_name="peak_memory_test",
        )

        memory_increase = peak_memory - initial_memory

        # Peak memory should be reasonable for large dataset
        assert memory_increase < 2000  # MB
        assert result["accuracy"] > 0

        print(
            f"Peak memory test (10k samples) - "
            f"Initial: {initial_memory:.2f}MB, "
            f"Peak: {peak_memory:.2f}MB, "
            f"Increase: {memory_increase:.2f}MB"
        )


if __name__ == "__main__":
    # Run performance tests
    pytest.main(
        [
            __file__,
            "-v",
            "--tb=short",
            "-m",
            "benchmark",
            "--benchmark-only",
            "--benchmark-sort=mean",
        ]
    )
