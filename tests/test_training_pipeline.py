"""
Comprehensive Unit Tests for Training Pipeline

This module tests all training components:
- Trainer classes and training workflows
- Hyperparameter tuning methods
- Experiment management and tracking
- Model checkpointing and persistence
- Distributed training capabilities
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import warnings
warnings.filterwarnings('ignore')

# Import training pipeline modules
from training.trainer import (
    Trainer, DistributedTrainer, ExperimentTracker, ModelCheckpointer
)
from training.hyperparameter_tuning import (
    GridSearchTuner, RandomSearchTuner, BayesianOptimizationTuner,
    TuningResult, create_tuner
)
from run_experiments import ExperimentRunner, ExperimentSuite

# Import models and configs for testing
from models.random_forest import RandomForestModel
from models.lstm import LSTMModel
from models.fnn import FNNModel
from config import RandomForestConfig, LSTMConfig, FNNConfig, ExperimentConfig


class TestTrainer:
    """Test Trainer class functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y
    
    @pytest.fixture
    def rf_model_config(self):
        """Create Random Forest model and config"""
        config = RandomForestConfig(n_estimators=10, random_state=42)
        model = RandomForestModel(config)
        return model, config
    
    def test_trainer_initialization(self):
        """Test Trainer initialization"""
        trainer = Trainer()
        assert trainer.models == {}
        assert trainer.results == {}
        assert trainer.current_experiment is None
    
    def test_train_sklearn_model(self, sample_data, rf_model_config):
        """Test training sklearn-based model"""
        X, y = sample_data
        model, config = rf_model_config
        trainer = Trainer()
        
        # Train model
        result = trainer.train_model(
            model=model,
            X_train=X[:80],
            y_train=y[:80],
            X_val=X[80:],
            y_val=y[80:],
            model_name="test_rf"
        )
        
        assert result is not None
        assert 'test_rf' in trainer.models
        assert 'test_rf' in trainer.results
        assert 0 <= result['accuracy'] <= 1
        assert result['training_time'] > 0
    
    def test_cross_validation(self, sample_data, rf_model_config):
        """Test cross-validation functionality"""
        X, y = sample_data
        model, config = rf_model_config
        trainer = Trainer()
        
        cv_results = trainer.cross_validate(
            model=model,
            X=X,
            y=y,
            cv_folds=3,
            model_name="test_rf_cv"
        )
        
        assert 'mean_accuracy' in cv_results
        assert 'std_accuracy' in cv_results
        assert 'fold_results' in cv_results
        assert len(cv_results['fold_results']) == 3
        assert 0 <= cv_results['mean_accuracy'] <= 1
    
    def test_ensemble_training(self, sample_data):
        """Test ensemble model training"""
        X, y = sample_data
        trainer = Trainer()
        
        # Create multiple models
        models = {
            'rf': RandomForestModel(RandomForestConfig(n_estimators=5, random_state=42)),
            'fnn': FNNModel(FNNConfig(hidden_layer_sizes=[10], max_iter=100, random_state=42))
        }
        
        ensemble_results = trainer.train_ensemble(
            models=models,
            X_train=X[:80],
            y_train=y[:80],
            X_val=X[80:],
            y_val=y[80:],
            ensemble_method='voting'
        )
        
        assert 'ensemble' in ensemble_results
        assert 'individual_results' in ensemble_results
        assert len(ensemble_results['individual_results']) == 2
        assert 0 <= ensemble_results['ensemble']['accuracy'] <= 1
    
    def test_model_persistence(self, sample_data, rf_model_config):
        """Test model saving and loading"""
        X, y = sample_data
        model, config = rf_model_config
        trainer = Trainer()
        
        # Train and save model
        trainer.train_model(
            model=model,
            X_train=X[:80],
            y_train=y[:80],
            X_val=X[80:],
            y_val=y[80:],
            model_name="test_persistence"
        )
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "test_model.pkl"
            trainer.save_model("test_persistence", str(save_path))
            assert save_path.exists()
            
            # Load model in new trainer
            new_trainer = Trainer()
            new_trainer.load_model("test_persistence", str(save_path), config)
            
            # Compare predictions
            original_pred = trainer.predict("test_persistence", X[:5])
            loaded_pred = new_trainer.predict("test_persistence", X[:5])
            np.testing.assert_array_equal(original_pred, loaded_pred)


class TestModelCheckpointer:
    """Test ModelCheckpointer functionality"""
    
    def test_checkpointer_initialization(self):
        """Test checkpointer initialization"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpointer = ModelCheckpointer(checkpoint_dir=tmp_dir)
            assert checkpointer.checkpoint_dir == Path(tmp_dir)
            assert checkpointer.checkpoint_dir.exists()
    
    def test_checkpoint_saving_loading(self):
        """Test checkpoint save/load operations"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpointer = ModelCheckpointer(checkpoint_dir=tmp_dir)
            
            # Create mock model state
            model_state = {
                'epoch': 10,
                'accuracy': 0.85,
                'loss': 0.3,
                'model_params': {'param1': 'value1'}
            }
            
            # Save checkpoint
            checkpoint_path = checkpointer.save_checkpoint(
                model_state=model_state,
                experiment_id="test_exp",
                model_name="test_model",
                epoch=10
            )
            
            assert checkpoint_path.exists()
            
            # Load checkpoint
            loaded_state = checkpointer.load_checkpoint(checkpoint_path)
            assert loaded_state['epoch'] == 10
            assert loaded_state['accuracy'] == 0.85
            assert loaded_state['model_params'] == {'param1': 'value1'}
    
    def test_best_checkpoint_tracking(self):
        """Test tracking of best checkpoints"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpointer = ModelCheckpointer(checkpoint_dir=tmp_dir)
            
            # Save multiple checkpoints with different accuracies
            for epoch, accuracy in [(1, 0.7), (2, 0.8), (3, 0.75), (4, 0.85)]:
                model_state = {'epoch': epoch, 'accuracy': accuracy}
                checkpointer.save_checkpoint(
                    model_state=model_state,
                    experiment_id="test_exp",
                    model_name="test_model",
                    epoch=epoch,
                    metric_value=accuracy,
                    metric_name='accuracy'
                )
            
            # Get best checkpoint
            best_checkpoint = checkpointer.get_best_checkpoint(
                experiment_id="test_exp",
                model_name="test_model",
                metric_name='accuracy'
            )
            
            assert best_checkpoint is not None
            loaded_best = checkpointer.load_checkpoint(best_checkpoint)
            assert loaded_best['accuracy'] == 0.85  # Best accuracy
            assert loaded_best['epoch'] == 4


class TestExperimentTracker:
    """Test MLflow experiment tracking"""
    
    @pytest.fixture
    def mock_mlflow(self):
        """Mock MLflow for testing"""
        with patch('training.trainer.mlflow') as mock_mlflow:
            mock_mlflow.start_run.return_value.__enter__ = Mock()
            mock_mlflow.start_run.return_value.__exit__ = Mock()
            yield mock_mlflow
    
    def test_tracker_initialization(self, mock_mlflow):
        """Test experiment tracker initialization"""
        tracker = ExperimentTracker(experiment_name="test_experiment")
        assert tracker.experiment_name == "test_experiment"
        mock_mlflow.set_experiment.assert_called_once_with("test_experiment")
    
    def test_run_tracking(self, mock_mlflow):
        """Test run tracking functionality"""
        tracker = ExperimentTracker(experiment_name="test_experiment")
        
        with tracker.start_run("test_run"):
            tracker.log_params({'param1': 'value1', 'param2': 42})
            tracker.log_metrics({'accuracy': 0.85, 'loss': 0.3})
            tracker.log_artifacts(['model.pkl', 'results.json'])
        
        # Verify MLflow calls
        mock_mlflow.log_params.assert_called_once()
        mock_mlflow.log_metrics.assert_called_once()
        mock_mlflow.log_artifacts.assert_called_once()
    
    def test_model_logging(self, mock_mlflow):
        """Test model logging functionality"""
        tracker = ExperimentTracker(experiment_name="test_experiment")
        
        # Create mock model
        mock_model = Mock()
        mock_model.__class__.__name__ = "RandomForestModel"
        
        with tracker.start_run("test_run"):
            tracker.log_model(mock_model, "test_model")
        
        # Should have attempted to log model
        mock_mlflow.sklearn.log_model.assert_called_once()


class TestHyperparameterTuning:
    """Test hyperparameter tuning methods"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for tuning"""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y
    
    @pytest.fixture
    def rf_model_and_grid(self):
        """Create RF model and parameter grid"""
        config = RandomForestConfig(n_estimators=5, random_state=42)
        model = RandomForestModel(config)
        param_grid = {
            'n_estimators': [5, 10],
            'max_depth': [2, 3, None]
        }
        return model, param_grid
    
    def test_grid_search_tuner(self, sample_data, rf_model_and_grid):
        """Test grid search tuning"""
        X, y = sample_data
        model, param_grid = rf_model_and_grid
        
        tuner = GridSearchTuner()
        result = tuner.tune(
            model=model,
            param_grid=param_grid,
            X_train=X[:80],
            y_train=y[:80],
            X_val=X[80:],
            y_val=y[80:],
            cv=2
        )
        
        assert isinstance(result, TuningResult)
        assert result.best_params is not None
        assert 'n_estimators' in result.best_params
        assert 'max_depth' in result.best_params
        assert 0 <= result.best_score <= 1
        assert len(result.all_results) == len(param_grid['n_estimators']) * len(param_grid['max_depth'])
    
    def test_random_search_tuner(self, sample_data, rf_model_and_grid):
        """Test random search tuning"""
        X, y = sample_data
        model, param_grid = rf_model_and_grid
        
        tuner = RandomSearchTuner(n_trials=5)
        result = tuner.tune(
            model=model,
            param_distributions=param_grid,
            X_train=X[:80],
            y_train=y[:80],
            X_val=X[80:],
            y_val=y[80:],
            cv=2
        )
        
        assert isinstance(result, TuningResult)
        assert result.best_params is not None
        assert 0 <= result.best_score <= 1
        assert len(result.all_results) <= 5  # Limited by n_trials
    
    @pytest.mark.slow
    def test_bayesian_optimization_tuner(self, sample_data, rf_model_and_grid):
        """Test Bayesian optimization tuning"""
        X, y = sample_data
        model, param_grid = rf_model_and_grid
        
        # Mock optuna for testing
        with patch('training.hyperparameter_tuning.optuna') as mock_optuna:
            mock_study = Mock()
            mock_study.best_params = {'n_estimators': 10, 'max_depth': 3}
            mock_study.best_value = 0.85
            mock_study.trials = [Mock(params={'n_estimators': 5}, value=0.80)]
            mock_optuna.create_study.return_value = mock_study
            
            tuner = BayesianOptimizationTuner(n_trials=3)
            result = tuner.tune(
                model=model,
                param_space=param_grid,
                X_train=X[:80],
                y_train=y[:80],
                X_val=X[80:],
                y_val=y[80:]
            )
            
            assert isinstance(result, TuningResult)
            assert result.best_params == {'n_estimators': 10, 'max_depth': 3}
            assert result.best_score == 0.85
    
    def test_tuning_result_persistence(self, sample_data, rf_model_and_grid):
        """Test saving and loading tuning results"""
        X, y = sample_data
        model, param_grid = rf_model_and_grid
        
        tuner = GridSearchTuner()
        result = tuner.tune(
            model=model,
            param_grid=param_grid,
            X_train=X[:80],
            y_train=y[:80],
            X_val=X[80:],
            y_val=y[80:],
            cv=2
        )
        
        with tempfile.NamedTemporaryFile(suffix='.json') as tmp_file:
            # Save result
            result.save(tmp_file.name)
            
            # Load result
            loaded_result = TuningResult.load(tmp_file.name)
            
            assert loaded_result.best_params == result.best_params
            assert loaded_result.best_score == result.best_score
            assert len(loaded_result.all_results) == len(result.all_results)
    
    def test_create_tuner_factory(self):
        """Test tuner factory function"""
        # Test grid search
        grid_tuner = create_tuner('grid', n_trials=10)
        assert isinstance(grid_tuner, GridSearchTuner)
        
        # Test random search
        random_tuner = create_tuner('random', n_trials=5)
        assert isinstance(random_tuner, RandomSearchTuner)
        assert random_tuner.n_trials == 5
        
        # Test invalid tuner type
        with pytest.raises(ValueError):
            create_tuner('invalid_type')


class TestExperimentRunner:
    """Test ExperimentRunner functionality"""
    
    @pytest.fixture
    def sample_config(self):
        """Create sample experiment configuration"""
        from config import ExperimentConfig, DataConfig, TrainingConfig
        
        config = ExperimentConfig(
            name="test_experiment",
            data=DataConfig(
                airports=['EGLL'],
                start_date='2023-01-01',
                end_date='2023-01-10'
            ),
            training=TrainingConfig(
                test_size=0.2,
                random_state=42,
                cross_validation=True,
                cv_folds=3
            ),
            models={
                'random_forest': RandomForestConfig(n_estimators=5, random_state=42)
            }
        )
        return config
    
    @pytest.fixture
    def mock_data(self):
        """Mock data loading for experiments"""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 10), columns=[f'feature_{i}' for i in range(10)])
        y = pd.Series((X['feature_0'] + X['feature_1'] > 0).astype(int), name='target')
        return X, y
    
    def test_experiment_runner_initialization(self, sample_config):
        """Test ExperimentRunner initialization"""
        runner = ExperimentRunner(sample_config)
        assert runner.config == sample_config
        assert runner.results == {}
        assert runner.models == {}
    
    @patch('run_experiments.DataLoader')
    def test_data_loading(self, mock_data_loader, sample_config, mock_data):
        """Test data loading in experiment runner"""
        X, y = mock_data
        
        # Mock data loader
        mock_loader_instance = Mock()
        mock_loader_instance.create_features.return_value = pd.concat([X, y], axis=1)
        mock_data_loader.return_value = mock_loader_instance
        
        runner = ExperimentRunner(sample_config)
        features_df = runner._load_data()
        
        assert isinstance(features_df, pd.DataFrame)
        assert 'target' in features_df.columns
        mock_data_loader.assert_called_once()
    
    @patch('run_experiments.DataLoader')
    def test_single_model_experiment(self, mock_data_loader, sample_config, mock_data):
        """Test running single model experiment"""
        X, y = mock_data
        
        # Mock data loader
        mock_loader_instance = Mock()
        mock_loader_instance.create_features.return_value = pd.concat([X, y], axis=1)
        mock_data_loader.return_value = mock_loader_instance
        
        runner = ExperimentRunner(sample_config)
        results = runner.run_experiment('random_forest')
        
        assert 'random_forest' in results
        assert 'accuracy' in results['random_forest']
        assert 'training_time' in results['random_forest']
        assert 0 <= results['random_forest']['accuracy'] <= 1
    
    @patch('run_experiments.DataLoader')
    def test_hyperparameter_tuning_experiment(self, mock_data_loader, sample_config, mock_data):
        """Test experiment with hyperparameter tuning"""
        X, y = mock_data
        
        # Mock data loader
        mock_loader_instance = Mock()
        mock_loader_instance.create_features.return_value = pd.concat([X, y], axis=1)
        mock_data_loader.return_value = mock_loader_instance
        
        runner = ExperimentRunner(sample_config)
        
        # Define tuning parameters
        param_grid = {'n_estimators': [5, 10], 'max_depth': [2, 3]}
        
        results = runner.run_experiment_with_tuning(
            model_name='random_forest',
            param_grid=param_grid,
            tuning_method='grid',
            cv=2
        )
        
        assert 'random_forest' in results
        assert 'best_params' in results['random_forest']
        assert 'tuning_results' in results['random_forest']
        assert 'best_score' in results['random_forest']
    
    def test_result_aggregation(self, sample_config):
        """Test result aggregation and comparison"""
        runner = ExperimentRunner(sample_config)
        
        # Mock some results
        runner.results = {
            'random_forest': {
                'accuracy': 0.85,
                'precision': 0.83,
                'recall': 0.87,
                'f1_score': 0.85,
                'training_time': 1.5
            },
            'fnn': {
                'accuracy': 0.82,
                'precision': 0.80,
                'recall': 0.84,
                'f1_score': 0.82,
                'training_time': 2.1
            }
        }
        
        comparison_df = runner.compare_models()
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 2
        assert 'model' in comparison_df.columns
        assert 'accuracy' in comparison_df.columns
        assert 'training_time' in comparison_df.columns
        
        # Check sorting (should be by accuracy descending)
        assert comparison_df.iloc[0]['model'] == 'random_forest'  # Higher accuracy
        assert comparison_df.iloc[1]['model'] == 'fnn'


class TestExperimentSuite:
    """Test ExperimentSuite functionality"""
    
    def test_suite_initialization(self):
        """Test ExperimentSuite initialization"""
        suite = ExperimentSuite("test_suite")
        assert suite.suite_name == "test_suite"
        assert suite.experiments == {}
        assert suite.results == {}
    
    def test_experiment_addition(self):
        """Test adding experiments to suite"""
        suite = ExperimentSuite("test_suite")
        
        config1 = ExperimentConfig(name="exp1")
        config2 = ExperimentConfig(name="exp2")
        
        suite.add_experiment("exp1", config1)
        suite.add_experiment("exp2", config2)
        
        assert len(suite.experiments) == 2
        assert "exp1" in suite.experiments
        assert "exp2" in suite.experiments
    
    @patch('run_experiments.ExperimentRunner')
    def test_suite_execution(self, mock_runner_class):
        """Test suite execution"""
        # Mock experiment runner
        mock_runner = Mock()
        mock_runner.run_all_experiments.return_value = {
            'model1': {'accuracy': 0.85, 'training_time': 1.5}
        }
        mock_runner_class.return_value = mock_runner
        
        suite = ExperimentSuite("test_suite")
        config = ExperimentConfig(name="test_exp")
        suite.add_experiment("test_exp", config)
        
        results = suite.run_suite(parallel=False)
        
        assert "test_exp" in results
        assert results["test_exp"]["model1"]["accuracy"] == 0.85
        mock_runner_class.assert_called_once_with(config)
    
    def test_suite_comparison(self):
        """Test cross-experiment comparison"""
        suite = ExperimentSuite("test_suite")
        
        # Mock some results
        suite.results = {
            'exp1': {
                'model1': {'accuracy': 0.85, 'f1_score': 0.83},
                'model2': {'accuracy': 0.82, 'f1_score': 0.80}
            },
            'exp2': {
                'model1': {'accuracy': 0.87, 'f1_score': 0.85},
                'model2': {'accuracy': 0.79, 'f1_score': 0.77}
            }
        }
        
        comparison_df = suite.compare_experiments()
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert 'experiment' in comparison_df.columns
        assert 'model' in comparison_df.columns
        assert 'accuracy' in comparison_df.columns


class TestDistributedTraining:
    """Test distributed training capabilities"""
    
    def test_distributed_trainer_initialization(self):
        """Test DistributedTrainer initialization"""
        trainer = DistributedTrainer(strategy='mirrored')
        assert trainer.strategy_name == 'mirrored'
    
    @patch('training.trainer.tf')
    def test_strategy_setup(self, mock_tf):
        """Test TensorFlow strategy setup"""
        mock_strategy = Mock()
        mock_tf.distribute.MirroredStrategy.return_value = mock_strategy
        
        trainer = DistributedTrainer(strategy='mirrored')
        trainer._setup_strategy()
        
        assert trainer.strategy == mock_strategy
        mock_tf.distribute.MirroredStrategy.assert_called_once()
    
    def test_multi_gpu_detection(self):
        """Test multi-GPU detection"""
        trainer = DistributedTrainer()
        
        # Mock GPU detection
        with patch('training.trainer.tf') as mock_tf:
            mock_tf.config.list_physical_devices.return_value = ['GPU:0', 'GPU:1']
            
            gpu_count = trainer._get_gpu_count()
            assert gpu_count == 2


class TestErrorHandling:
    """Test error handling in training pipeline"""
    
    def test_invalid_model_name(self, sample_data):
        """Test handling of invalid model names"""
        X, y = sample_data
        trainer = Trainer()
        
        with pytest.raises(KeyError):
            trainer.predict("non_existent_model", X)
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient training data"""
        # Very small dataset
        X = np.random.randn(5, 3)
        y = np.random.randint(0, 2, 5)
        
        config = RandomForestConfig(n_estimators=10)
        model = RandomForestModel(config)
        trainer = Trainer()
        
        # Should handle gracefully (may warn but shouldn't crash)
        try:
            result = trainer.train_model(
                model=model,
                X_train=X[:3],
                y_train=y[:3],
                X_val=X[3:],
                y_val=y[3:],
                model_name="small_data_test"
            )
            # If successful, should have reasonable result
            assert 0 <= result['accuracy'] <= 1
        except (ValueError, Warning) as e:
            # Expected for very small datasets
            pass
    
    def test_invalid_hyperparameter_handling(self):
        """Test handling of invalid hyperparameters"""
        tuner = GridSearchTuner()
        
        # Invalid parameter grid (empty)
        with pytest.raises(ValueError):
            param_grid = {}
            # Should raise error for empty parameter grid
            tuner._validate_param_grid(param_grid)
    
    def test_experiment_configuration_validation(self):
        """Test experiment configuration validation"""
        # Test invalid configuration
        with pytest.raises((ValueError, TypeError)):
            invalid_config = ExperimentConfig(
                name="",  # Empty name should be invalid
                models={}  # No models defined
            )


# Performance and benchmark tests
class TestPerformanceBenchmarks:
    """Performance and benchmark tests"""
    
    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_training_performance_benchmark(self, benchmark):
        """Benchmark training performance"""
        np.random.seed(42)
        X = np.random.randn(1000, 20)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        config = RandomForestConfig(n_estimators=50, random_state=42)
        model = RandomForestModel(config)
        trainer = Trainer()
        
        def train_model():
            return trainer.train_model(
                model=model,
                X_train=X[:800],
                y_train=y[:800],
                X_val=X[800:],
                y_val=y[800:],
                model_name="benchmark_test"
            )
        
        result = benchmark(train_model)
        assert result['accuracy'] > 0  # Sanity check
    
    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_hyperparameter_tuning_benchmark(self, benchmark):
        """Benchmark hyperparameter tuning performance"""
        np.random.seed(42)
        X = np.random.randn(500, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        config = RandomForestConfig(n_estimators=10, random_state=42)
        model = RandomForestModel(config)
        param_grid = {
            'n_estimators': [5, 10, 15],
            'max_depth': [2, 3, None]
        }
        
        tuner = GridSearchTuner()
        
        def tune_hyperparameters():
            return tuner.tune(
                model=model,
                param_grid=param_grid,
                X_train=X[:400],
                y_train=y[:400],
                X_val=X[400:],
                y_val=y[400:],
                cv=2
            )
        
        result = benchmark(tune_hyperparameters)
        assert result.best_score > 0  # Sanity check


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])