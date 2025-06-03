"""
Configuration parser with advanced utilities for the weather regulation prediction system.
Handles loading, validation, merging, and manipulation of configuration files.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import yaml
import json
import copy
from datetime import datetime
import itertools
from dataclasses import fields, is_dataclass, asdict

from config import (
    ExperimentConfig, DataConfig, TrainingConfig,
    RandomForestConfig, LSTMConfig, CNNConfig, GRUConfig,
    TransformerConfig, EnsembleConfig, AutoencoderConfig,
    validate_config
)


class ConfigParser:
    """Advanced configuration parser with validation and utilities."""
    
    def __init__(self):
        self.config_classes = {
            'data': DataConfig,
            'training': TrainingConfig,
            'random_forest': RandomForestConfig,
            'lstm': LSTMConfig,
            'cnn': CNNConfig,
            'gru': GRUConfig,
            'transformer': TransformerConfig,
            'ensemble': EnsembleConfig,
            'autoencoder': AutoencoderConfig
        }
    
    def load_config(self, filepath: Union[str, Path]) -> ExperimentConfig:
        """
        Load configuration from YAML or JSON file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            ExperimentConfig instance
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        if filepath.suffix == '.yaml' or filepath.suffix == '.yml':
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        return self._parse_config_dict(config_dict)
    
    def _parse_config_dict(self, config_dict: Dict[str, Any]) -> ExperimentConfig:
        """
        Parse configuration dictionary into ExperimentConfig.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            ExperimentConfig instance
        """
        # Create base experiment config
        experiment_config = ExperimentConfig(
            name=config_dict.get('name', 'default'),
            description=config_dict.get('description', ''),
            version=config_dict.get('version', '1.0.0')
        )
        
        # Parse data configuration
        if 'data' in config_dict:
            experiment_config.data = self._parse_dataclass(
                config_dict['data'], DataConfig
            )
        
        # Parse training configuration
        if 'training' in config_dict:
            experiment_config.training = self._parse_dataclass(
                config_dict['training'], TrainingConfig
            )
        
        # Parse model configurations
        if 'models' in config_dict:
            for model_name, model_config in config_dict['models'].items():
                if model_name in self.config_classes:
                    config_class = self.config_classes[model_name]
                    parsed_config = self._parse_dataclass(model_config, config_class)
                    setattr(experiment_config, model_name, parsed_config)
        
        # Parse experiment settings
        if 'experiment_settings' in config_dict:
            settings = config_dict['experiment_settings']
            experiment_config.models_to_train = settings.get(
                'models_to_train', ['random_forest', 'lstm']
            )
            experiment_config.hyperparameter_tuning = settings.get(
                'hyperparameter_tuning', True
            )
            experiment_config.tuning_method = settings.get('tuning_method', 'grid')
            experiment_config.tuning_trials = settings.get('tuning_trials', 100)
        
        # Parse output settings
        if 'output_settings' in config_dict:
            settings = config_dict['output_settings']
            for key, value in settings.items():
                if hasattr(experiment_config, key):
                    setattr(experiment_config, key, value)
        
        return experiment_config
    
    def _parse_dataclass(self, config_dict: Dict[str, Any], dataclass_type: type) -> Any:
        """
        Parse dictionary into dataclass instance.
        
        Args:
            config_dict: Configuration dictionary
            dataclass_type: Target dataclass type
            
        Returns:
            Dataclass instance
        """
        # Get field names and types from dataclass
        field_names = {f.name for f in fields(dataclass_type)}
        
        # Filter config_dict to only include valid fields
        filtered_dict = {
            k: v for k, v in config_dict.items() 
            if k in field_names
        }
        
        # Handle special type conversions
        for field in fields(dataclass_type):
            if field.name in filtered_dict:
                value = filtered_dict[field.name]
                
                # Convert datetime strings
                if 'datetime' in str(field.type) and isinstance(value, str):
                    filtered_dict[field.name] = datetime.strptime(
                        value, "%Y-%m-%d %H:%M:%S"
                    )
        
        return dataclass_type(**filtered_dict)
    
    def merge_configs(
        self, 
        base_config: ExperimentConfig, 
        override_config: Union[ExperimentConfig, Dict[str, Any]]
    ) -> ExperimentConfig:
        """
        Merge two configurations, with override taking precedence.
        
        Args:
            base_config: Base configuration
            override_config: Configuration to override with
            
        Returns:
            Merged ExperimentConfig
        """
        # Convert to dictionaries
        base_dict = self._config_to_dict(base_config)
        
        if isinstance(override_config, ExperimentConfig):
            override_dict = self._config_to_dict(override_config)
        else:
            override_dict = override_config
        
        # Deep merge
        merged_dict = self._deep_merge(base_dict, override_dict)
        
        # Convert back to ExperimentConfig
        return self._parse_config_dict(merged_dict)
    
    def _config_to_dict(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Convert ExperimentConfig to dictionary."""
        result = {
            'name': config.name,
            'description': config.description,
            'version': config.version,
            'data': asdict(config.data) if config.data else {},
            'training': asdict(config.training) if config.training else {},
            'models': {},
            'experiment_settings': {
                'models_to_train': config.models_to_train,
                'hyperparameter_tuning': config.hyperparameter_tuning,
                'tuning_method': config.tuning_method,
                'tuning_trials': config.tuning_trials
            },
            'output_settings': {
                'save_predictions': config.save_predictions,
                'save_feature_importance': config.save_feature_importance,
                'save_confusion_matrix': config.save_confusion_matrix,
                'save_roc_curve': config.save_roc_curve,
                'save_training_history': config.save_training_history
            }
        }
        
        # Add model configurations
        for model_name in self.config_classes.keys():
            if hasattr(config, model_name):
                model_config = getattr(config, model_name)
                if model_config is not None:
                    result['models'][model_name] = asdict(model_config)
        
        return result
    
    def _deep_merge(self, dict1: Dict, dict2: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = copy.deepcopy(dict1)
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def generate_hyperparameter_grid(
        self, 
        config: ExperimentConfig, 
        model_name: str
    ) -> List[Dict[str, Any]]:
        """
        Generate all hyperparameter combinations for grid search.
        
        Args:
            config: Experiment configuration
            model_name: Name of the model
            
        Returns:
            List of hyperparameter dictionaries
        """
        model_config = getattr(config, model_name, None)
        if model_config is None:
            raise ValueError(f"No configuration found for model: {model_name}")
        
        # Get all fields that are lists (hyperparameter ranges)
        param_ranges = {}
        for field in fields(model_config):
            value = getattr(model_config, field.name)
            if isinstance(value, list) and len(value) > 0:
                param_ranges[field.name] = value
        
        # Generate all combinations
        if not param_ranges:
            return [{}]
        
        keys = list(param_ranges.keys())
        values = list(param_ranges.values())
        
        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def export_config(
        self, 
        config: ExperimentConfig, 
        filepath: Union[str, Path],
        format: str = 'yaml'
    ) -> None:
        """
        Export configuration to file.
        
        Args:
            config: Configuration to export
            filepath: Output file path
            format: Output format ('yaml' or 'json')
        """
        filepath = Path(filepath)
        config_dict = self._config_to_dict(config)
        
        if format == 'yaml':
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        elif format == 'json':
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def validate_and_fix_config(
        self, 
        config: ExperimentConfig
    ) -> Tuple[ExperimentConfig, List[str]]:
        """
        Validate configuration and attempt to fix common issues.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (fixed_config, list_of_fixes_applied)
        """
        fixes_applied = []
        fixed_config = copy.deepcopy(config)
        
        # Fix data configuration
        if fixed_config.data.test_size + fixed_config.data.validation_size >= 1.0:
            old_test = fixed_config.data.test_size
            old_val = fixed_config.data.validation_size
            fixed_config.data.test_size = 0.2
            fixed_config.data.validation_size = 0.2
            fixes_applied.append(
                f"Fixed test_size ({old_test}) + validation_size ({old_val}) >= 1.0"
            )
        
        # Fix training configuration
        if fixed_config.training.cv_folds < 2:
            old_folds = fixed_config.training.cv_folds
            fixed_config.training.cv_folds = 5
            fixes_applied.append(f"Fixed cv_folds from {old_folds} to 5")
        
        # Fix experiment settings
        if fixed_config.hyperparameter_tuning and fixed_config.tuning_trials <= 0:
            fixed_config.tuning_trials = 100
            fixes_applied.append("Fixed tuning_trials to 100")
        
        # Validate fixed configuration
        errors = validate_config(fixed_config)
        
        return fixed_config, fixes_applied
    
    def create_config_from_best_params(
        self, 
        base_config: ExperimentConfig,
        model_name: str,
        best_params: Dict[str, Any]
    ) -> ExperimentConfig:
        """
        Create a new configuration with best hyperparameters.
        
        Args:
            base_config: Base configuration
            model_name: Model name
            best_params: Best hyperparameters
            
        Returns:
            New configuration with best parameters
        """
        new_config = copy.deepcopy(base_config)
        model_config = getattr(new_config, model_name)
        
        # Update model config with best params
        for param_name, param_value in best_params.items():
            if hasattr(model_config, param_name):
                # Set single value instead of list
                setattr(model_config, param_name, [param_value])
        
        # Disable hyperparameter tuning
        new_config.hyperparameter_tuning = False
        new_config.tuning_trials = 0
        
        return new_config
    
    def compare_configs(
        self, 
        config1: ExperimentConfig, 
        config2: ExperimentConfig
    ) -> Dict[str, Any]:
        """
        Compare two configurations and return differences.
        
        Args:
            config1: First configuration
            config2: Second configuration
            
        Returns:
            Dictionary of differences
        """
        dict1 = self._config_to_dict(config1)
        dict2 = self._config_to_dict(config2)
        
        differences = self._find_differences(dict1, dict2)
        return differences
    
    def _find_differences(
        self, 
        dict1: Dict, 
        dict2: Dict, 
        path: str = ""
    ) -> Dict[str, Any]:
        """Recursively find differences between two dictionaries."""
        differences = {}
        
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        for key in all_keys:
            current_path = f"{path}.{key}" if path else key
            
            if key not in dict1:
                differences[current_path] = {"in_config2_only": dict2[key]}
            elif key not in dict2:
                differences[current_path] = {"in_config1_only": dict1[key]}
            elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                nested_diff = self._find_differences(
                    dict1[key], dict2[key], current_path
                )
                if nested_diff:
                    differences.update(nested_diff)
            elif dict1[key] != dict2[key]:
                differences[current_path] = {
                    "config1": dict1[key],
                    "config2": dict2[key]
                }
        
        return differences


# Utility functions for common operations
def load_config(filepath: Union[str, Path]) -> ExperimentConfig:
    """Load configuration from file."""
    parser = ConfigParser()
    return parser.load_config(filepath)


def save_config(config: ExperimentConfig, filepath: Union[str, Path], format: str = 'yaml'):
    """Save configuration to file."""
    parser = ConfigParser()
    parser.export_config(config, filepath, format)


def merge_configs(base: ExperimentConfig, override: Union[ExperimentConfig, Dict]) -> ExperimentConfig:
    """Merge two configurations."""
    parser = ConfigParser()
    return parser.merge_configs(base, override)


def generate_grid(config: ExperimentConfig, model: str) -> List[Dict[str, Any]]:
    """Generate hyperparameter grid for model."""
    parser = ConfigParser()
    return parser.generate_hyperparameter_grid(config, model)


if __name__ == "__main__":
    # Example usage
    parser = ConfigParser()
    
    # Load configuration
    config = parser.load_config("configs/default_config.yaml")
    print(f"Loaded config: {config.name}")
    
    # Validate and fix
    fixed_config, fixes = parser.validate_and_fix_config(config)
    if fixes:
        print("Applied fixes:")
        for fix in fixes:
            print(f"  - {fix}")
    
    # Generate hyperparameter grid for LSTM
    if 'lstm' in config.models_to_train:
        grid = parser.generate_hyperparameter_grid(config, 'lstm')
        print(f"\nLSTM hyperparameter grid size: {len(grid)}")
        print(f"First combination: {grid[0] if grid else 'None'}")
    
    # Compare with quick test config
    quick_config = parser.load_config("configs/quick_test.yaml")
    differences = parser.compare_configs(config, quick_config)
    print(f"\nNumber of differences: {len(differences)}")