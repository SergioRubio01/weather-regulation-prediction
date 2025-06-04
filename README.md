# Weather Regulation Prediction System

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](tests/)

A comprehensive machine learning system for predicting Air Traffic Flow Management (ATFM) regulations based on weather conditions. This system provides end-to-end capabilities for data processing, model training, hyperparameter optimization, and results analysis.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [Models](#models)
- [Data Pipeline](#data-pipeline)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results and Visualization](#results-and-visualization)
- [Performance](#performance)
- [Testing](#testing)
- [Balanced Dataset Approach](#balanced-dataset-approach)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Weather Regulation Prediction System is designed to predict ATFM regulations at European airports based on weather conditions from METAR reports and enhanced TAF predictions. The system supports 13 different machine learning models and provides comprehensive tools for experiment management, hyperparameter optimization, and results analysis.

### Key Capabilities

- **Multi-Model Support**: 13 different ML models including Random Forest, LSTM, CNN, Transformer, GRU, Ensemble methods, and more
- **Advanced Data Pipeline**: Automated data loading, validation, feature engineering, and preprocessing
- **Hyperparameter Optimization**: 6 different tuning methods including Bayesian optimization and distributed search
- **Experiment Management**: Complete experiment tracking with MLflow integration
- **Interactive Visualizations**: Comprehensive plotting and dashboard capabilities
- **Results Management**: Structured storage and comparison of experimental results
- **High Performance**: Optimized for scalability and efficient resource utilization

## Features

### 🚀 **13 Machine Learning Models**

- **Traditional ML**: Random Forest, Feedforward Neural Networks
- **Deep Learning**: LSTM, CNN, RNN, GRU, Transformer, Attention-LSTM
- **Advanced**: WaveNet, Autoencoder, Ensemble methods, Hybrid CNN-RNN/LSTM

### 🔧 **6 Hyperparameter Tuning Methods**

- Grid Search, Random Search, Bayesian Optimization
- Keras Tuner, Ray Tune, Multi-objective optimization

### 📊 **Comprehensive Data Pipeline**

- Intelligent data loading with caching
- Advanced data validation and quality checks
- Automated feature engineering for weather data
- Flexible preprocessing with sklearn integration

### 📈 **Results and Visualization**

- Interactive dashboards with Dash
- Advanced plotting with Plotly and Matplotlib
- Multi-format report generation (HTML, PDF, LaTeX, PowerPoint)
- MLflow experiment tracking

### ⚡ **High Performance**

- Distributed training support
- Parallel processing capabilities
- Memory optimization
- Comprehensive performance monitoring

## Installation

### Prerequisites

- Python 3.12 or higher
- Poetry (recommended) or pip

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd weather-regulation-prediction

# Install dependencies
poetry install

# Activate virtual environment
poetry env activate
```

### Using pip

```bash
# Clone the repository
git clone <repository-url>
cd weather-regulation-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Additional Dependencies

For full functionality, install optional dependencies:

```bash
# For PDF report generation
pip install reportlab

# For PowerPoint reports
pip install python-pptx

# For Bayesian optimization
pip install optuna

# For distributed tuning
pip install ray[tune]

# For advanced visualizations
pip install plotly dash
```

## Quick Start

### 1. Basic Model Training

```python
from src.config import ExperimentConfig, RandomForestConfig
from run_experiments import ExperimentRunner

# Create configuration
config = ExperimentConfig(
    name="quick_start_experiment",
    models={
        'random_forest': RandomForestConfig(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    }
)

# Run experiment
runner = ExperimentRunner(config)
results = runner.run_experiment('random_forest')
print(f"Accuracy: {results['random_forest']['accuracy']:.3f}")
```

### 2. Hyperparameter Tuning

```python
from training.hyperparameter_tuning import GridSearchTuner
from models.random_forest import RandomForestModel
from src.config import RandomForestConfig

# Create model
config = RandomForestConfig(random_state=42)
model = RandomForestModel(config)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None]
}

# Tune hyperparameters
tuner = GridSearchTuner()
result = tuner.tune(
    model=model,
    param_grid=param_grid,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    cv=5
)

print(f"Best parameters: {result.best_params}")
print(f"Best score: {result.best_score:.3f}")
```

### 3. Using Configuration Files

```yaml
# config.yaml
name: "weather_regulation_experiment"
data:
  airports: ["EGLL", "LSZH"]
  start_date: "2023-01-01"
  end_date: "2023-12-31"
training:
  test_size: 0.2
  validation_size: 0.2
  cross_validation: true
  cv_folds: 5
models:
  random_forest:
    n_estimators: 100
    max_depth: 15
    random_state: 42
  lstm:
    units: 64
    dropout: 0.3
    epochs: 50
    batch_size: 32
```

```python
from src.config_parser import ConfigParser
from run_experiments import ExperimentRunner

# Load configuration
parser = ConfigParser()
config = parser.load_config("config.yaml")

# Run all experiments
runner = ExperimentRunner(config)
results = runner.run_all_experiments()
```

## Architecture

The system follows a modular architecture with clear separation of concerns:

```bash
weather-regulation-prediction/
├── src/                       # Core source modules
│   ├── config.py              # Configuration classes
│   ├── config_parser.py       # YAML/JSON parsing
│   └── config_utils.py        # CLI utilities
├── data/                      # Data pipeline
│   ├── data_loader.py         # Data loading and caching
│   ├── data_validation.py     # Data quality checks
│   ├── feature_engineering.py # Feature creation
│   └── preprocessing.py       # Data preprocessing
├── models/                    # Model implementations
│   ├── base_model.py          # Abstract base class
│   ├── random_forest.py       # Random Forest
│   ├── lstm.py                # LSTM networks
│   ├── transformer.py         # Transformer architecture
│   └── ...                    # Other models
├── training/                  # Training pipeline
│   ├── trainer.py             # Training management
│   └── hyperparameter_tuning.py # HP optimization
├── results/                   # Results management
│   ├── results_manager.py     # Result storage/retrieval
│   └── report_generator.py    # Multi-format reporting
├── visualization/             # Visualization tools
│   ├── plots.py               # Plotting utilities
│   └── dashboard.py           # Interactive dashboards
├── scripts/                   # Executable scripts
│   ├── balanced_dataset/      # Balanced dataset pipeline
│   ├── utilities/             # General utilities
│   ├── legacy/                # Legacy compatibility
│   └── examples/              # Usage examples
├── tests/                     # Comprehensive test suite
└── configs/                   # Example configurations
```

For details on the script organization, see [scripts/README.md](scripts/README.md).

### Documentation

- **[CLAUDE.md](CLAUDE.md)**: Comprehensive guidance for Claude Code when working with this system
- **[BALANCED_DATASET_PROCEDURE.md](BALANCED_DATASET_PROCEDURE.md)**: Detailed documentation on the balanced dataset approach
- **[REFACTORING_PLAN.md](REFACTORING_PLAN.md)**: System refactoring plan and progress
- **[docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md)**: Complete API reference for all modules
- **[docs/PRECOMMIT_SETUP.md](docs/PRECOMMIT_SETUP.md)**: Pre-commit hooks configuration guide
- **[scripts/README.md](scripts/README.md)**: Guide to script organization and usage
- **[REORGANIZATION_SUMMARY.md](REORGANIZATION_SUMMARY.md)**: Summary of recent project reorganization

## Usage Guide

### Data Pipeline

#### Loading Data

```python
from data.data_loader import DataLoader

loader = DataLoader(data_path="./Data")

# Load weather data
metar_data = loader.load_metar_data("METAR_EGLL_filtered.csv")
taf_data = loader.load_taf_data("TAF_EGLL_enhanced.csv")
regulation_data = loader.load_regulation_data("Regulations_EGLL.csv")

# Create feature matrix
features = loader.create_features(
    metar_data=metar_data,
    regulation_data=regulation_data,
    target_column='has_regulation'
)
```

#### Data Validation

```python
from data.data_validation import DataValidator

validator = DataValidator()
report = validator.validate_weather_data(metar_data)

if report['errors']:
    print("Data quality issues found:")
    for error in report['errors']:
        print(f"- {error}")
```

#### Feature Engineering

```python
from data.feature_engineering import WeatherFeatureEngineer

engineer = WeatherFeatureEngineer()
enhanced_data = engineer.create_features(metar_data)

# New features include:
# - Flight categories (VFR, MVFR, IFR, LIFR)
# - Weather severity scores
# - Wind components (U/V)
# - Temporal features
```

### Model Training

#### Single Model Training

```python
from models.lstm import LSTMModel
from src.config import LSTMConfig
from training.trainer import Trainer

# Configure model
config = LSTMConfig(
    units=64,
    dropout=0.3,
    epochs=50,
    batch_size=32,
    sequence_length=24
)

# Train model
model = LSTMModel(config)
trainer = Trainer()

result = trainer.train_model(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    model_name="weather_lstm"
)
```

#### Ensemble Training

```python
from models.ensemble import EnsembleModel
from src.config import EnsembleConfig

config = EnsembleConfig(
    base_models=[
        {'type': 'random_forest', 'n_estimators': 100},
        {'type': 'lstm', 'units': 64, 'epochs': 30},
        {'type': 'fnn', 'hidden_layer_sizes': [100, 50]}
    ],
    ensemble_method='voting',
    voting_type='soft'
)

ensemble = EnsembleModel(config)
trainer.train_model(ensemble, X_train, y_train, X_val, y_val, "ensemble_model")
```

### Hyperparameter Tuning

#### Bayesian Optimization

```python
from training.hyperparameter_tuning import BayesianOptimizationTuner

tuner = BayesianOptimizationTuner(n_trials=100)

param_space = {
    'n_estimators': [50, 500],
    'max_depth': [3, 20],
    'learning_rate': [0.01, 0.3]
}

result = tuner.tune(
    model=model,
    param_space=param_space,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val
)
```

#### Distributed Tuning with Ray

```python
from training.hyperparameter_tuning import RayTuneTuner

tuner = RayTuneTuner(
    num_samples=50,
    max_concurrent=4,
    resources_per_trial={'cpu': 2, 'gpu': 0.5}
)

result = tuner.tune(model, param_space, X_train, y_train, X_val, y_val)
```

### Results Management

#### Saving and Loading Results

```python
from results.results_manager import ResultsManager, ExperimentResult
from datetime import datetime

manager = ResultsManager(base_path="./results")

# Create experiment result
experiment = ExperimentResult(
    experiment_id="weather_exp_001",
    experiment_name="Weather Regulation Prediction",
    timestamp=datetime.now(),
    config=config
)

# Save experiment
experiment_id = manager.save_experiment_result(experiment)

# Load and compare experiments
experiments_df = manager.list_experiments()
comparison = manager.compare_experiments(['exp_001', 'exp_002'])
```

#### Generating Reports

```python
from results.report_generator import ReportGenerator

generator = ReportGenerator()

# Generate comprehensive HTML report
html_report = generator.generate_report(
    experiment=experiment,
    format='html',
    output_path='weather_experiment_report.html',
    include_visualizations=True
)

# Generate PDF report for presentation
pdf_report = generator.generate_report(
    experiment=experiment,
    format='pdf',
    output_path='weather_experiment_report.pdf'
)
```

### Visualization

#### Model Performance Plots

```python
from visualization.plots import ModelVisualizer

visualizer = ModelVisualizer(save_path="./visualizations")

# Confusion matrix
cm_fig = visualizer.plot_confusion_matrix(
    y_true, y_pred,
    title="Weather Regulation Prediction",
    interactive=True,
    save_name="confusion_matrix"
)

# ROC curves comparison
roc_fig = visualizer.plot_roc_curves(
    y_true,
    {'RF': rf_proba, 'LSTM': lstm_proba, 'Ensemble': ensemble_proba},
    title="Model Comparison - ROC Curves",
    interactive=True
)

# Feature importance
importance_fig = visualizer.plot_feature_importance(
    {'Random Forest': rf_importance, 'LSTM': lstm_importance},
    top_n=15,
    title="Feature Importance Comparison"
)
```

#### Interactive Dashboard

```python
from visualization.dashboard import launch_dashboard

# Launch interactive model comparison dashboard
launch_dashboard(
    results_path="./results",
    port=8050
)
# Open browser to http://localhost:8050
```

## Configuration

The system uses a hierarchical configuration system with YAML/JSON support:

### Configuration Structure

```yaml
name: "experiment_name"
description: "Experiment description"

data:
  airports: ["EGLL", "LSZH", "LFPG", "LOWW"]
  start_date: "2023-01-01"
  end_date: "2023-12-31"
  weather_data_path: "./Data/METAR/"
  regulation_data_path: "./Data/Regulations/"

training:
  test_size: 0.2
  validation_size: 0.2
  random_state: 42
  cross_validation: true
  cv_folds: 5
  stratify: true

models:
  random_forest:
    n_estimators: 100
    max_depth: 15
    min_samples_split: 2
    min_samples_leaf: 1
    random_state: 42

  lstm:
    units: 64
    dropout: 0.3
    recurrent_dropout: 0.3
    batch_size: 32
    epochs: 50
    sequence_length: 24
    bidirectional: true

  transformer:
    d_model: 128
    num_heads: 8
    num_layers: 4
    dropout: 0.1
    sequence_length: 24

hyperparameter_tuning:
  enabled: true
  method: "bayesian"  # grid, random, bayesian
  n_trials: 100
  cv_folds: 3

experiment_tracking:
  enabled: true
  mlflow_uri: "./mlruns"
  experiment_name: "weather_regulations"
```

### Configuration Management

```python
from src.config_parser import ConfigParser

parser = ConfigParser()

# Load configuration
config = parser.load_config("experiment_config.yaml")

# Merge configurations
base_config = parser.load_config("base_config.yaml")
specific_config = parser.load_config("lstm_specific.yaml")
merged_config = parser.merge_configs(base_config, specific_config)

# Validate configuration
is_valid, errors = parser.validate_config(config)

# Generate parameter grid for tuning
param_grid = parser.generate_param_grid(config.models['random_forest'])
```

## Models

### Available Models

| Model | Type | Description | Key Features |
|-------|------|-------------|--------------|
| Random Forest | Traditional ML | Ensemble of decision trees | Fast training, feature importance, robust |
| FNN | Neural Network | Feedforward neural network | Simple, effective baseline |
| LSTM | RNN | Long Short-Term Memory | Sequential data, memory cells |
| GRU | RNN | Gated Recurrent Unit | Simpler than LSTM, good performance |
| CNN | Deep Learning | Convolutional Neural Network | Feature extraction, spatial patterns |
| RNN | Neural Network | Basic recurrent network | Simple sequential processing |
| Transformer | Deep Learning | Attention-based architecture | State-of-the-art for sequences |
| Attention-LSTM | Hybrid | LSTM with attention mechanism | Best of both worlds |
| WaveNet | Deep Learning | Dilated causal convolutions | Time series modeling |
| Autoencoder | Deep Learning | Encoder-decoder architecture | Feature learning, anomaly detection |
| Ensemble | Meta-model | Combination of multiple models | Improved robustness and accuracy |
| CNN-RNN | Hybrid | CNN feature extraction + RNN | Spatial and temporal features |
| CNN-LSTM | Hybrid | CNN feature extraction + LSTM | Advanced hybrid approach |

### Model Selection Guidelines

**For Quick Results:**

- Random Forest: Fast training, good baseline
- FNN: Simple neural network approach

**For Sequential Data:**

- LSTM: Complex temporal patterns
- GRU: Simpler alternative to LSTM
- Transformer: State-of-the-art attention-based

**For Feature Learning:**

- Autoencoder: Unsupervised feature extraction
- CNN: Spatial pattern recognition

**For Best Performance:**

- Ensemble: Combines multiple models
- Attention-LSTM: Advanced temporal modeling
- Transformer: Cutting-edge architecture

## Data Pipeline

### Data Loading

The data loader supports multiple formats and provides intelligent caching:

```python
from data.data_loader import DataLoader

loader = DataLoader(
    data_path="./Data",
    enable_cache=True,
    cache_size_gb=2.0,
    parallel_loading=True,
    n_jobs=4
)

# Supports CSV, Parquet, HDF5, JSON
weather_data = loader.load_metar_data("weather.parquet")
```

### Data Validation

Comprehensive validation ensures data quality:

```python
from data.data_validation import DataValidator, AnomalyDetector

validator = DataValidator()

# Weather-specific validation
report = validator.validate_weather_data(data)

# Anomaly detection
detector = AnomalyDetector()
anomalies = detector.detect_anomalies(
    data,
    columns=['temperature', 'pressure'],
    method='isolation_forest'
)

# Data drift detection
drift_report = detector.detect_drift(
    reference_data=train_data,
    current_data=new_data
)
```

### Feature Engineering

Automated feature engineering for weather data:

```python
from data.feature_engineering import (
    WeatherFeatureEngineer,
    TimeSeriesFeatureEngineer,
    AutomatedFeatureEngineer
)

# Weather-specific features
weather_eng = WeatherFeatureEngineer()
enhanced_data = weather_eng.create_features(weather_data)
# Adds: flight_category, weather_severity, wind_components, etc.

# Time series features
ts_eng = TimeSeriesFeatureEngineer()
ts_features = ts_eng.create_features(
    data,
    lags=[1, 3, 6, 12],
    rolling_windows=[6, 12, 24]
)
# Adds: lag features, rolling statistics, temporal features

# Automated feature selection
auto_eng = AutomatedFeatureEngineer()
selected_features = auto_eng.create_features(
    data,
    target_col='has_regulation',
    max_features=50
)
```

## Hyperparameter Tuning

### Tuning Methods

#### 1. Grid Search

```python
from training.hyperparameter_tuning import GridSearchTuner

tuner = GridSearchTuner()
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15]
}
result = tuner.tune(model, param_grid, X_train, y_train, X_val, y_val)
```

#### 2. Random Search

```python
from training.hyperparameter_tuning import RandomSearchTuner

tuner = RandomSearchTuner(n_trials=50)
param_distributions = {
    'n_estimators': [10, 500],
    'max_depth': [1, 20]
}
result = tuner.tune(model, param_distributions, X_train, y_train, X_val, y_val)
```

#### 3. Bayesian Optimization

```python
from training.hyperparameter_tuning import BayesianOptimizationTuner

tuner = BayesianOptimizationTuner(n_trials=100)
param_space = {
    'n_estimators': [50, 500],
    'max_depth': [3, 20],
    'learning_rate': [0.01, 0.3]
}
result = tuner.tune(model, param_space, X_train, y_train, X_val, y_val)
```

#### 4. Keras Tuner (for deep learning models)

```python
from training.hyperparameter_tuning import KerasTuner

tuner = KerasTuner(tuner_type='bayesian', max_trials=50)
result = tuner.tune(lstm_model, X_train, y_train, X_val, y_val)
```

#### 5. Ray Tune (distributed)

```python
from training.hyperparameter_tuning import RayTuneTuner

tuner = RayTuneTuner(
    num_samples=100,
    max_concurrent=8,
    resources_per_trial={'cpu': 2, 'gpu': 0.5}
)
result = tuner.tune(model, param_space, X_train, y_train, X_val, y_val)
```

#### 6. Multi-objective Optimization

```python
from training.hyperparameter_tuning import MultiObjectiveTuner

tuner = MultiObjectiveTuner(
    objectives=['accuracy', 'training_time'],
    directions=['maximize', 'minimize']
)
result = tuner.tune(model, param_space, X_train, y_train, X_val, y_val)
```

## Results and Visualization

### Results Management

Structured storage and retrieval of experimental results:

```python
from results.results_manager import ResultsManager

manager = ResultsManager(base_path="./results")

# List all experiments
experiments_df = manager.list_experiments()

# Compare multiple experiments
comparison = manager.compare_experiments(
    experiment_ids=['exp_001', 'exp_002', 'exp_003'],
    metric='test_accuracy'
)

# Get best models across all experiments
best_models = manager.get_best_models(n=10, metric='test_f1')

# Export results in various formats
manager.export_results('exp_001', format='excel')
manager.export_results('exp_002', format='latex')
```

### Visualization Tools

#### Static Plots

```python
from visualization.plots import ModelVisualizer, WeatherVisualizer

# Model performance visualization
model_viz = ModelVisualizer()

# Confusion matrix
cm_fig = model_viz.plot_confusion_matrix(y_true, y_pred, interactive=True)

# ROC curves
roc_fig = model_viz.plot_roc_curves(y_true, y_scores_dict, interactive=True)

# Feature importance
importance_fig = model_viz.plot_feature_importance(importance_dict)

# Training history
history_fig = model_viz.plot_training_history(history_dict)

# Weather-specific visualizations
weather_viz = WeatherVisualizer()

# Weather patterns
patterns_fig = weather_viz.plot_weather_patterns(weather_data)

# Regulation analysis
regulation_fig = weather_viz.plot_regulation_analysis(
    regulation_data, weather_data
)
```

#### Interactive Dashboard

```python
from visualization.dashboard import ModelComparisonDashboard

# Create dashboard
dashboard = ModelComparisonDashboard(results_path="./results")

# Run dashboard server
dashboard.run(debug=False, port=8050)
```

The dashboard provides:

- **Overview**: Experiment summaries and key metrics
- **Performance**: Model accuracy comparisons and radar charts
- **Comparison**: Side-by-side detailed analysis
- **Features**: Feature importance across models
- **Training**: Training curves and statistics
- **Weather**: Domain-specific weather analysis
- **Reports**: Export functionality

### Report Generation

Multi-format report generation:

```python
from results.report_generator import ReportGenerator

generator = ReportGenerator()

# HTML report with interactive charts
html_report = generator.generate_report(
    experiment=experiment_result,
    format='html',
    include_visualizations=True
)

# PDF report for presentations
pdf_report = generator.generate_report(
    experiment=experiment_result,
    format='pdf'
)

# LaTeX report for academic papers
latex_report = generator.generate_report(
    experiment=experiment_result,
    format='latex'
)

# PowerPoint presentation
pptx_report = generator.generate_report(
    experiment=experiment_result,
    format='pptx'
)
```

## Performance

### Optimization Features

- **Parallel Processing**: Multi-core training and data processing
- **Memory Management**: Intelligent caching and memory optimization
- **Distributed Training**: Multi-GPU support for deep learning models
- **Efficient Data Loading**: Lazy loading and format optimization

### Benchmarks

Performance benchmarks on standard hardware (16GB RAM, 8-core CPU):

| Dataset Size | Model | Training Time | Memory Usage |
|--------------|-------|---------------|--------------|
| 1K samples | Random Forest | 2.3s | 45MB |
| 1K samples | LSTM | 15.2s | 120MB |
| 10K samples | Random Forest | 8.7s | 180MB |
| 10K samples | LSTM | 89.4s | 650MB |
| 100K samples | Random Forest | 45.1s | 850MB |

### Performance Monitoring

```python
from tests.test_performance import PerformanceMonitor

monitor = PerformanceMonitor().start_monitoring()

# Run training
result = trainer.train_model(model, X_train, y_train, X_val, y_val)

metrics = monitor.stop_monitoring()
print(f"Execution time: {metrics['execution_time']:.2f}s")
print(f"Memory usage: {metrics['memory_usage']:.2f}MB")
print(f"Peak memory: {metrics['peak_memory']:.2f}MB")
```

## Testing

### Test Suite Structure

The project includes a comprehensive test suite:

```bash
tests/
├── test_config.py          # Configuration system tests
├── test_models.py          # Model implementation tests  
├── test_data_pipeline.py   # Data pipeline tests
├── test_training_pipeline.py # Training pipeline tests
├── test_integration.py     # End-to-end integration tests
└── test_performance.py     # Performance and benchmark tests
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_models.py -v
pytest tests/test_integration.py -v

# Run performance benchmarks
pytest tests/test_performance.py -m benchmark

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Speed and memory benchmarks
- **Compatibility Tests**: Legacy system compatibility

## Balanced Dataset Approach

For handling severe class imbalance in weather regulation prediction (often < 2% positive samples), we've developed a comprehensive balanced dataset approach that achieves dramatic improvements in model performance.

### Key Features

- **Intelligent Sampling**: Time-window based sampling around regulation events
- **Perfect Balance**: Achieves 50/50 class distribution
- **Multi-Airport Analysis**: Automatically selects airports with sufficient regulation data
- **Enhanced Features**: Weather severity indicators and variable wind handling

### Results

Using the balanced dataset approach on EGLL airport data:

- **Class Balance**: From 1.1% to 50% positive samples
- **F1 Score**: From 0.071 to 0.879 (1140% improvement)
- **Recall**: From 17% to 100%
- **AUC**: From 0.757 to 0.954

### Usage

```bash
# Run the complete balanced dataset pipeline
python scripts/balanced_dataset/run_balanced_pipeline.py

# Or run individual steps
python scripts/balanced_dataset/analyze_and_balance_data.py
python scripts/balanced_dataset/prepare_balanced_data.py
python scripts/balanced_dataset/train_balanced_model.py
```

For detailed documentation on the balanced dataset approach, including methodology, implementation details, and full results, see [BALANCED_DATASET_PROCEDURE.md](BALANCED_DATASET_PROCEDURE.md).

## Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd weather-regulation-prediction

# Install development dependencies
poetry install --with dev

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Check code style
black .
flake8 .
mypy .
```

### Adding New Models

1. Create model class inheriting from `BaseModel`
2. Implement required methods: `_build_model()`, `train()`, `predict()`
3. Add configuration class in `config.py`
4. Update model registry in `run_experiments.py`
5. Add tests in `tests/test_models.py`

Example:

```python
from models.base_model import BaseModel
from src.config import BaseModelConfig

class NewModelConfig(BaseModelConfig):
    param1: int = 10
    param2: float = 0.1

class NewModel(BaseModel):
    def __init__(self, config: NewModelConfig):
        super().__init__(config)

    def _build_model(self, input_shape):
        # Implement model architecture
        pass

    def train(self, X, y, X_val=None, y_val=None):
        # Implement training logic
        pass

    def predict(self, X):
        # Implement prediction logic
        pass
```

### Code Style Guidelines

- Follow PEP 8 style guide
- Use type hints for all function signatures
- Document all public functions and classes
- Write comprehensive tests for new features
- Use meaningful variable and function names

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Data Sources**: EUROCONTROL for ATFM regulation data, METAR/TAF weather data
- **Libraries**: TensorFlow, scikit-learn, Plotly, Dash, MLflow, and many others
- **Contributors**: Research team and open-source community

## Citation

If you use this system in your research, please cite:

```bibtex
@software{weather_regulation_prediction,
  title={Weather Regulation Prediction System},
  author={Sergio García},
  year={2025},
  url={https://github.com/sergiorubio01/weather-regulation-prediction}
}
```

## Support

For questions, issues, or contributions:

- **Documentation**: Check this README and inline code documentation
- **Issues**: Report bugs and feature requests via GitHub Issues
- **Discussions**: Use GitHub Discussions for questions and ideas

---

**Built with ❤️ for the aviation and machine learning communities**
