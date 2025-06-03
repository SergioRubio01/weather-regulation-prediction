"""
Unit tests for configuration system.
Tests configuration loading, validation, parsing, and utilities.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import yaml
import json

from config import (
    ExperimentConfig, DataConfig, TrainingConfig,
    RandomForestConfig, LSTMConfig, CNNConfig,
    validate_config, create_default_config
)
from config_parser import ConfigParser, load_config, save_config, merge_configs
from config_utils import ConfigurationManager


class TestConfigClasses(unittest.TestCase):
    """Test configuration dataclasses."""
    
    def test_data_config_creation(self):
        """Test DataConfig creation and defaults."""
        config = DataConfig()
        
        # Check defaults
        self.assertEqual(config.airports, ["EGLL", "LSZH", "LFPG", "LOWW"])
        self.assertEqual(config.time_delta, 30)
        self.assertEqual(config.test_size, 0.2)
        self.assertEqual(config.random_state, 42)
        
    def test_data_config_datetime_conversion(self):
        """Test datetime string conversion in DataConfig."""
        config = DataConfig(
            time_init="2020-01-01 00:00:00",
            time_end="2020-12-31 23:59:59"
        )
        
        self.assertIsInstance(config.time_init, datetime)
        self.assertIsInstance(config.time_end, datetime)
        self.assertEqual(config.time_init.year, 2020)
        self.assertEqual(config.time_end.month, 12)
    
    def test_lstm_config_defaults(self):
        """Test LSTM configuration defaults."""
        config = LSTMConfig()
        
        self.assertEqual(config.units, [50, 100, 200])
        self.assertEqual(config.optimizer, "adam")
        self.assertEqual(config.activation, "tanh")
        self.assertFalse(config.bidirectional)
    
    def test_experiment_config_creation(self):
        """Test ExperimentConfig creation."""
        config = ExperimentConfig(
            name="test_experiment",
            description="Test description"
        )
        
        self.assertEqual(config.name, "test_experiment")
        self.assertIsNotNone(config.data)
        self.assertIsNotNone(config.training)
        self.assertIsNotNone(config.random_forest)
    
    def test_experiment_config_to_dict(self):
        """Test conversion to dictionary."""
        config = create_default_config()
        config_dict = config.to_dict()
        
        self.assertIn("name", config_dict)
        self.assertIn("data", config_dict)
        self.assertIn("models", config_dict)
        self.assertIn("experiment_settings", config_dict)


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation."""
    
    def test_valid_config(self):
        """Test validation of valid configuration."""
        config = create_default_config()
        errors = validate_config(config)
        
        self.assertEqual(len(errors), 0)
    
    def test_invalid_test_size(self):
        """Test validation with invalid test size."""
        config = create_default_config()
        config.data.test_size = 1.5
        
        errors = validate_config(config)
        self.assertTrue(any("test_size" in error for error in errors))
    
    def test_invalid_data_split(self):
        """Test validation with invalid train/val/test split."""
        config = create_default_config()
        config.data.test_size = 0.6
        config.data.validation_size = 0.5
        
        errors = validate_config(config)
        self.assertTrue(any("test_size + validation_size" in error for error in errors))
    
    def test_invalid_cv_folds(self):
        """Test validation with invalid cross-validation folds."""
        config = create_default_config()
        config.training.cv_folds = 1
        
        errors = validate_config(config)
        self.assertTrue(any("cv_folds" in error for error in errors))
    
    def test_invalid_tuning_trials(self):
        """Test validation with invalid tuning trials."""
        config = create_default_config()
        config.hyperparameter_tuning = True
        config.tuning_trials = 0
        
        errors = validate_config(config)
        self.assertTrue(any("tuning_trials" in error for error in errors))


class TestConfigParser(unittest.TestCase):
    """Test configuration parser functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = ConfigParser()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_load_yaml_config(self):
        """Test loading configuration from YAML."""
        # Create test YAML file
        config_dict = {
            'name': 'test_config',
            'description': 'Test configuration',
            'data': {
                'airports': ['EGLL'],
                'time_delta': 30,
                'test_size': 0.2
            }
        }
        
        yaml_path = Path(self.temp_dir) / "test.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f)
        
        # Load and verify
        config = self.parser.load_config(yaml_path)
        self.assertEqual(config.name, 'test_config')
        self.assertEqual(config.data.airports, ['EGLL'])
    
    def test_load_json_config(self):
        """Test loading configuration from JSON."""
        # Create test JSON file
        config_dict = {
            'name': 'test_json',
            'description': 'Test JSON configuration',
            'data': {
                'airports': ['LSZH'],
                'time_delta': 60
            }
        }
        
        json_path = Path(self.temp_dir) / "test.json"
        with open(json_path, 'w') as f:
            json.dump(config_dict, f)
        
        # Load and verify
        config = self.parser.load_config(json_path)
        self.assertEqual(config.name, 'test_json')
        self.assertEqual(config.data.time_delta, 60)
    
    def test_merge_configs(self):
        """Test configuration merging."""
        base_config = create_default_config()
        override_dict = {
            'name': 'merged_config',
            'data': {
                'airports': ['LFPG'],
                'test_size': 0.3
            }
        }
        
        merged = self.parser.merge_configs(base_config, override_dict)
        
        # Check merged values
        self.assertEqual(merged.name, 'merged_config')
        self.assertEqual(merged.data.airports, ['LFPG'])
        self.assertEqual(merged.data.test_size, 0.3)
        # Check preserved values
        self.assertEqual(merged.data.time_delta, base_config.data.time_delta)
    
    def test_generate_hyperparameter_grid(self):
        """Test hyperparameter grid generation."""
        config = create_default_config()
        config.lstm = LSTMConfig(
            units=[50, 100],
            epochs=[30, 60],
            batch_size=[32]
        )
        
        grid = self.parser.generate_hyperparameter_grid(config, 'lstm')
        
        # Should have 2 x 2 x 1 = 4 combinations
        self.assertEqual(len(grid), 4)
        
        # Check specific combinations
        combinations = [
            {'units': 50, 'epochs': 30, 'batch_size': 32},
            {'units': 50, 'epochs': 60, 'batch_size': 32},
            {'units': 100, 'epochs': 30, 'batch_size': 32},
            {'units': 100, 'epochs': 60, 'batch_size': 32}
        ]
        
        for combo in combinations:
            self.assertIn(combo, grid)
    
    def test_export_config(self):
        """Test configuration export."""
        config = create_default_config()
        config.name = "export_test"
        
        # Export to YAML
        yaml_path = Path(self.temp_dir) / "export.yaml"
        self.parser.export_config(config, yaml_path, format='yaml')
        self.assertTrue(yaml_path.exists())
        
        # Export to JSON
        json_path = Path(self.temp_dir) / "export.json"
        self.parser.export_config(config, json_path, format='json')
        self.assertTrue(json_path.exists())
        
        # Verify exported content
        reloaded = self.parser.load_config(yaml_path)
        self.assertEqual(reloaded.name, "export_test")
    
    def test_validate_and_fix_config(self):
        """Test configuration validation and fixing."""
        config = create_default_config()
        # Create invalid configuration
        config.data.test_size = 0.7
        config.data.validation_size = 0.5
        config.training.cv_folds = 1
        
        fixed_config, fixes = self.parser.validate_and_fix_config(config)
        
        # Check fixes were applied
        self.assertGreater(len(fixes), 0)
        self.assertEqual(fixed_config.data.test_size, 0.2)
        self.assertEqual(fixed_config.data.validation_size, 0.2)
        self.assertEqual(fixed_config.training.cv_folds, 5)
        
        # Validate fixed config
        errors = validate_config(fixed_config)
        self.assertEqual(len(errors), 0)
    
    def test_compare_configs(self):
        """Test configuration comparison."""
        config1 = create_default_config()
        config2 = create_default_config()
        config2.name = "different_name"
        config2.data.airports = ["LOWW"]
        
        differences = self.parser.compare_configs(config1, config2)
        
        self.assertIn("name", differences)
        self.assertIn("data.airports", differences)


class TestConfigurationManager(unittest.TestCase):
    """Test ConfigurationManager utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = ConfigurationManager()
        self.temp_configs_dir = tempfile.mkdtemp()
        self.manager.configs_dir = Path(self.temp_configs_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_configs_dir)
    
    def test_list_configs(self):
        """Test listing configuration files."""
        # Create test configs
        config1 = create_default_config()
        config1.name = "test1"
        save_config(config1, self.manager.configs_dir / "test1.yaml")
        
        config2 = create_default_config()
        config2.name = "test2"
        save_config(config2, self.manager.configs_dir / "test2.yaml")
        
        # List configs
        configs = self.manager.list_configs()
        
        self.assertEqual(len(configs), 2)
        names = [c['name'] for c in configs]
        self.assertIn("test1", names)
        self.assertIn("test2", names)
    
    def test_optimize_config_for_hardware(self):
        """Test hardware optimization."""
        config = create_default_config()
        
        # Optimize for 16GB GPU
        optimized = self.manager.optimize_config_for_hardware(config, gpu_memory_gb=16)
        
        # Check batch sizes were scaled
        self.assertGreater(optimized.lstm.batch_size[0], config.lstm.batch_size[0])
        self.assertTrue(optimized.training.mixed_precision)
    
    def test_estimate_training_time(self):
        """Test training time estimation."""
        config = create_default_config()
        config.models_to_train = ['random_forest', 'lstm']
        config.lstm.epochs = [50]
        config.hyperparameter_tuning = False
        
        estimates = self.manager.estimate_training_time(config)
        
        self.assertIn('random_forest', estimates)
        self.assertIn('lstm', estimates)
        self.assertGreater(estimates['lstm'], 0)
        self.assertGreater(estimates['random_forest'], 0)


class TestUtilityFunctions(unittest.TestCase):
    """Test standalone utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_load_save_config(self):
        """Test load_config and save_config utilities."""
        config = create_default_config()
        config.name = "utility_test"
        
        # Save
        filepath = Path(self.temp_dir) / "test.yaml"
        save_config(config, filepath)
        
        # Load
        loaded = load_config(filepath)
        
        self.assertEqual(loaded.name, "utility_test")
        self.assertEqual(loaded.data.airports, config.data.airports)
    
    def test_merge_configs_utility(self):
        """Test merge_configs utility function."""
        base = create_default_config()
        override = {'name': 'merged', 'data': {'test_size': 0.3}}
        
        merged = merge_configs(base, override)
        
        self.assertEqual(merged.name, 'merged')
        self.assertEqual(merged.data.test_size, 0.3)


if __name__ == '__main__':
    unittest.main()