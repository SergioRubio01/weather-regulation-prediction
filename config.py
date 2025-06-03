"""
Configuration module for weather regulation prediction system.
Provides structured configuration classes for all models and experiments.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import yaml
import json
from datetime import datetime
from enum import Enum


class ModelType(Enum):
    """Enumeration of available model types."""
    RANDOM_FOREST = "random_forest"
    LSTM = "lstm"
    GRU = "gru"
    CNN = "cnn"
    RNN = "rnn"
    FNN = "fnn"
    WAVENET = "wavenet"
    CNN_RNN = "cnn_rnn"
    CNN_LSTM = "cnn_lstm"
    TRANSFORMER = "transformer"
    ATTENTION_LSTM = "attention_lstm"
    ENSEMBLE = "ensemble"
    AUTOENCODER = "autoencoder"


class Criterion(Enum):
    """Enumeration of split criteria for tree-based models."""
    GINI = "gini"
    ENTROPY = "entropy"
    LOG_LOSS = "log_loss"


class Optimizer(Enum):
    """Enumeration of optimizers for neural networks."""
    ADAM = "adam"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAMW = "adamw"
    NADAM = "nadam"


class Activation(Enum):
    """Enumeration of activation functions."""
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    GELU = "gelu"


class Loss(Enum):
    """Enumeration of loss functions."""
    BINARY_CROSSENTROPY = "binary_crossentropy"
    CATEGORICAL_CROSSENTROPY = "categorical_crossentropy"
    MSE = "mse"
    MAE = "mae"
    HUBER = "huber"


@dataclass
class DataConfig:
    """Configuration for data processing and loading."""
    airports: List[str] = field(default_factory=lambda: ["EGLL", "LSZH", "LFPG", "LOWW"])
    time_init: str = "2017-01-01 00:50:00"
    time_end: str = "2019-12-08 23:50:00"
    time_delta: int = 30  # Time step in minutes
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    download_type: int = 1  # 0: web download, 1: local file
    data_path: str = "./Data"
    output_path: str = "./Output"
    
    # Feature engineering parameters
    use_minmax_scaler: bool = True
    use_label_binarizer: bool = True
    window_size: int = 1  # For time series models
    
    # Data augmentation
    use_oversampling: bool = False
    use_undersampling: bool = False
    smote_ratio: float = 1.0
    
    def __post_init__(self):
        """Convert string dates to datetime objects."""
        if isinstance(self.time_init, str):
            self.time_init = datetime.strptime(self.time_init, "%Y-%m-%d %H:%M:%S")
        if isinstance(self.time_end, str):
            self.time_end = datetime.strptime(self.time_end, "%Y-%m-%d %H:%M:%S")


@dataclass
class RandomForestConfig:
    """Configuration for Random Forest model."""
    n_estimators: List[int] = field(default_factory=lambda: [50, 100, 200])
    criterion: List[str] = field(default_factory=lambda: ["gini", "entropy", "log_loss"])
    max_depth: List[int] = field(default_factory=lambda: [5, 7, 10, 15])
    min_samples_split: List[int] = field(default_factory=lambda: [2, 5, 10])
    min_samples_leaf: List[int] = field(default_factory=lambda: [1, 2, 4])
    max_features: List[Union[str, float]] = field(default_factory=lambda: ["auto", "sqrt", 0.5])
    bootstrap: bool = True
    n_jobs: int = -1
    verbose: int = 0


@dataclass
class LSTMConfig:
    """Configuration for LSTM model."""
    units: List[int] = field(default_factory=lambda: [50, 100, 200])
    epochs: List[int] = field(default_factory=lambda: [30, 60, 120])
    batch_size: List[int] = field(default_factory=lambda: [32, 64, 128])
    dropout_rate: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.5])
    recurrent_dropout: List[float] = field(default_factory=lambda: [0.0, 0.2])
    activation: str = "tanh"
    recurrent_activation: str = "sigmoid"
    use_bias: bool = True
    return_sequences: bool = False
    stateful: bool = False
    optimizer: str = "adam"
    loss: str = "binary_crossentropy"
    learning_rate: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.1])
    
    # Architecture variations
    num_layers: List[int] = field(default_factory=lambda: [1, 2, 3])
    bidirectional: bool = False
    
    # Training parameters
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5


@dataclass
class CNNConfig:
    """Configuration for CNN model."""
    filters: List[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_size: List[int] = field(default_factory=lambda: [3, 5, 7])
    pool_size: List[int] = field(default_factory=lambda: [2, 3])
    dropout_rate: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.5])
    activation: str = "relu"
    padding: str = "same"
    epochs: List[int] = field(default_factory=lambda: [30, 60, 120])
    batch_size: List[int] = field(default_factory=lambda: [32, 64, 128])
    optimizer: str = "adam"
    loss: str = "binary_crossentropy"
    learning_rate: List[float] = field(default_factory=lambda: [0.001, 0.01])
    
    # Architecture parameters
    num_conv_layers: List[int] = field(default_factory=lambda: [1, 2, 3])
    use_batch_norm: bool = True
    dense_units: List[int] = field(default_factory=lambda: [64, 128, 256])


@dataclass
class TransformerConfig:
    """Configuration for Transformer model."""
    d_model: List[int] = field(default_factory=lambda: [64, 128, 256])
    num_heads: List[int] = field(default_factory=lambda: [4, 8])
    num_layers: List[int] = field(default_factory=lambda: [2, 4, 6])
    d_ff: List[int] = field(default_factory=lambda: [128, 256, 512])
    dropout_rate: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    attention_dropout: List[float] = field(default_factory=lambda: [0.1, 0.2])
    epochs: List[int] = field(default_factory=lambda: [50, 100, 200])
    batch_size: List[int] = field(default_factory=lambda: [16, 32, 64])
    optimizer: str = "adam"
    loss: str = "binary_crossentropy"
    learning_rate: List[float] = field(default_factory=lambda: [0.0001, 0.001])
    warmup_steps: int = 4000
    use_positional_encoding: bool = True


@dataclass
class GRUConfig:
    """Configuration for GRU model."""
    units: List[int] = field(default_factory=lambda: [50, 100, 200])
    epochs: List[int] = field(default_factory=lambda: [30, 60, 120])
    batch_size: List[int] = field(default_factory=lambda: [32, 64, 128])
    dropout_rate: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.5])
    recurrent_dropout: List[float] = field(default_factory=lambda: [0.0, 0.2])
    activation: str = "tanh"
    recurrent_activation: str = "sigmoid"
    optimizer: str = "adam"
    loss: str = "binary_crossentropy"
    learning_rate: List[float] = field(default_factory=lambda: [0.001, 0.01])
    num_layers: List[int] = field(default_factory=lambda: [1, 2, 3])
    bidirectional: bool = False


@dataclass
class EnsembleConfig:
    """Configuration for Ensemble models."""
    base_models: List[str] = field(default_factory=lambda: ["random_forest", "lstm", "cnn"])
    voting_type: str = "soft"  # "hard" or "soft"
    weights: Optional[List[float]] = None
    use_stacking: bool = False
    meta_learner: str = "logistic_regression"  # Used if use_stacking=True
    cv_folds: int = 5


@dataclass
class AutoencoderConfig:
    """Configuration for Autoencoder model."""
    encoding_dim: List[int] = field(default_factory=lambda: [8, 16, 32])
    hidden_layers: List[List[int]] = field(default_factory=lambda: [[64, 32], [128, 64, 32]])
    activation: str = "relu"
    output_activation: str = "sigmoid"
    epochs: List[int] = field(default_factory=lambda: [50, 100, 200])
    batch_size: List[int] = field(default_factory=lambda: [32, 64])
    optimizer: str = "adam"
    loss: str = "mse"
    learning_rate: List[float] = field(default_factory=lambda: [0.001, 0.01])
    use_for_feature_extraction: bool = True
    fine_tune_classifier: bool = True


@dataclass
class TrainingConfig:
    """General training configuration."""
    # Cross-validation
    use_cross_validation: bool = True
    cv_folds: int = 5
    stratified: bool = True
    
    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_monitor: str = "val_loss"
    early_stopping_mode: str = "min"
    
    # Learning rate scheduling
    use_lr_scheduler: bool = True
    lr_scheduler_type: str = "reduce_on_plateau"  # "cosine", "exponential", "step"
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    
    # Model checkpointing
    save_best_model: bool = True
    checkpoint_monitor: str = "val_accuracy"
    checkpoint_mode: str = "max"
    
    # Logging
    tensorboard: bool = True
    wandb: bool = False
    mlflow: bool = False
    log_interval: int = 10
    
    # Hardware
    use_gpu: bool = True
    mixed_precision: bool = False
    num_workers: int = 4


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str = "default_experiment"
    description: str = "Weather regulation prediction experiment"
    version: str = "1.0.0"
    
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Model configurations
    random_forest: Optional[RandomForestConfig] = field(default_factory=RandomForestConfig)
    lstm: Optional[LSTMConfig] = field(default_factory=LSTMConfig)
    cnn: Optional[CNNConfig] = field(default_factory=CNNConfig)
    gru: Optional[GRUConfig] = field(default_factory=GRUConfig)
    transformer: Optional[TransformerConfig] = field(default_factory=TransformerConfig)
    ensemble: Optional[EnsembleConfig] = field(default_factory=EnsembleConfig)
    autoencoder: Optional[AutoencoderConfig] = field(default_factory=AutoencoderConfig)
    
    # Experiment settings
    models_to_train: List[str] = field(default_factory=lambda: ["random_forest", "lstm"])
    hyperparameter_tuning: bool = True
    tuning_method: str = "grid"  # "grid", "random", "bayesian"
    tuning_trials: int = 100
    
    # Output settings
    save_predictions: bool = True
    save_feature_importance: bool = True
    save_confusion_matrix: bool = True
    save_roc_curve: bool = True
    save_training_history: bool = True
    
    def get_model_config(self, model_type: str) -> Any:
        """Get configuration for a specific model type."""
        return getattr(self, model_type, None)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "models": {
                model: getattr(self, model).__dict__ 
                for model in self.models_to_train 
                if hasattr(self, model) and getattr(self, model) is not None
            },
            "experiment_settings": {
                "models_to_train": self.models_to_train,
                "hyperparameter_tuning": self.hyperparameter_tuning,
                "tuning_method": self.tuning_method,
                "tuning_trials": self.tuning_trials
            }
        }
    
    def save(self, filepath: Path):
        """Save configuration to file."""
        filepath = Path(filepath)
        config_dict = self.to_dict()
        
        if filepath.suffix == ".yaml":
            with open(filepath, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif filepath.suffix == ".json":
            with open(filepath, "w") as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    @classmethod
    def load(cls, filepath: Path) -> "ExperimentConfig":
        """Load configuration from file."""
        filepath = Path(filepath)
        
        if filepath.suffix == ".yaml":
            with open(filepath, "r") as f:
                config_dict = yaml.safe_load(f)
        elif filepath.suffix == ".json":
            with open(filepath, "r") as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        # Create ExperimentConfig instance from dictionary
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Create ExperimentConfig from dictionary."""
        # This is a simplified version - you might need more sophisticated parsing
        experiment_config = cls(
            name=config_dict.get("name", "default"),
            description=config_dict.get("description", ""),
            version=config_dict.get("version", "1.0.0")
        )
        
        # Update sub-configurations
        if "data" in config_dict:
            experiment_config.data = DataConfig(**config_dict["data"])
        if "training" in config_dict:
            experiment_config.training = TrainingConfig(**config_dict["training"])
        
        # Update model configurations
        if "models" in config_dict:
            for model_name, model_config in config_dict["models"].items():
                if hasattr(experiment_config, model_name):
                    # Get the appropriate config class
                    config_class = globals()[f"{model_name.title().replace('_', '')}Config"]
                    setattr(experiment_config, model_name, config_class(**model_config))
        
        # Update experiment settings
        if "experiment_settings" in config_dict:
            for key, value in config_dict["experiment_settings"].items():
                if hasattr(experiment_config, key):
                    setattr(experiment_config, key, value)
        
        return experiment_config


def create_default_config() -> ExperimentConfig:
    """Create a default configuration."""
    return ExperimentConfig()


def validate_config(config: ExperimentConfig) -> List[str]:
    """
    Validate configuration and return list of issues.
    
    Args:
        config: ExperimentConfig instance to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Validate data configuration
    if config.data.test_size <= 0 or config.data.test_size >= 1:
        errors.append("test_size must be between 0 and 1")
    
    if config.data.validation_size < 0 or config.data.validation_size >= 1:
        errors.append("validation_size must be between 0 and 1")
    
    if config.data.test_size + config.data.validation_size >= 1:
        errors.append("test_size + validation_size must be less than 1")
    
    if config.data.time_delta <= 0:
        errors.append("time_delta must be positive")
    
    # Validate model configurations
    for model_name in config.models_to_train:
        model_config = config.get_model_config(model_name)
        if model_config is None:
            errors.append(f"No configuration found for model: {model_name}")
            continue
        
        # Check common neural network parameters
        if hasattr(model_config, "epochs"):
            if any(e <= 0 for e in model_config.epochs):
                errors.append(f"{model_name}: epochs must be positive")
        
        if hasattr(model_config, "batch_size"):
            if any(b <= 0 for b in model_config.batch_size):
                errors.append(f"{model_name}: batch_size must be positive")
        
        if hasattr(model_config, "learning_rate"):
            if any(lr <= 0 for lr in model_config.learning_rate):
                errors.append(f"{model_name}: learning_rate must be positive")
    
    # Validate training configuration
    if config.training.cv_folds < 2:
        errors.append("cv_folds must be at least 2")
    
    # Validate experiment settings
    if config.hyperparameter_tuning and config.tuning_trials <= 0:
        errors.append("tuning_trials must be positive when hyperparameter_tuning is enabled")
    
    valid_tuning_methods = ["grid", "random", "bayesian"]
    if config.tuning_method not in valid_tuning_methods:
        errors.append(f"tuning_method must be one of: {valid_tuning_methods}")
    
    return errors


if __name__ == "__main__":
    # Example usage
    config = create_default_config()
    config.name = "weather_prediction_experiment_v1"
    config.description = "Testing new configuration system"
    config.models_to_train = ["random_forest", "lstm", "transformer"]
    
    # Validate configuration
    errors = validate_config(config)
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration is valid!")
    
    # Save configuration
    config.save(Path("configs/example_config.yaml"))
    print(f"Configuration saved to configs/example_config.yaml")