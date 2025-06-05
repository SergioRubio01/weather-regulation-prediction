# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive **Weather Regulation Prediction System** for airports in the NMOC (Network Manager Operations Centre) region. The project uses 13 different machine learning models to predict ATFM (Air Traffic Flow Management) regulations based on weather data from METAR reports and enhanced TAF predictions.

### System Capabilities

- **13 Machine Learning Models**: From traditional ML (Random Forest) to cutting-edge deep learning (Transformers, Attention-LSTM)
- **6 Hyperparameter Tuning Methods**: Grid search, Bayesian optimization, distributed tuning with Ray
- **Advanced Data Pipeline**: Intelligent loading, validation, feature engineering, and preprocessing
- **Comprehensive Results Management**: Structured storage, comparison, and multi-format reporting
- **Interactive Visualizations**: Dashboards, advanced plots, and domain-specific weather analysis
- **High Performance**: Optimized for scalability with parallel processing and memory management
- **Extensive Testing**: Unit, integration, performance, and compatibility tests

## Key Dependencies

### Core Dependencies

- **Python 3.12+**
- **scikit-learn**: Traditional ML models and preprocessing
- **tensorflow/keras**: Deep learning models (LSTM, CNN, Transformer, etc.)
- **pandas, numpy**: Data manipulation and numerical computing
- **PyYAML**: Configuration management
- **Poetry**: Dependency and environment management

### Visualization & Analysis

- **plotly**: Interactive visualizations
- **dash**: Interactive dashboards
- **matplotlib, seaborn**: Static plotting
- **mlflow**: Experiment tracking

### Optional Dependencies

- **optuna**: Bayesian optimization
- **ray[tune]**: Distributed hyperparameter tuning
- **reportlab**: PDF report generation
- **python-pptx**: PowerPoint presentations
- **pytaf, PythonMETAR**: Weather data parsing

## Common Commands

### Installation & Setup

```bash
# Install dependencies using Poetry
poetry install

# Install dev dependencies for linting and testing
poetry install --with dev

# Install pre-commit hooks
pre-commit install
```

### Development

```bash
# Run linting and code quality checks
ruff check .                    # Fast Python linter
black .                        # Code formatter
mypy .                         # Type checking
flake8 .                       # Style guide enforcement
pylint src/ models/ data/      # Code analysis

# Run tests
pytest                         # Run all tests
pytest -v                      # Verbose output
pytest tests/test_models.py    # Run specific test file
pytest -m "not slow"           # Skip slow tests
pytest --cov=. --cov-report=html  # With coverage report

# Run pre-commit on all files
pre-commit run --all-files
```

### Basic Model Training

```bash
# RECOMMENDED: Run balanced dataset pipeline (best results)
cd scripts/balanced_dataset
python run_balanced_pipeline.py

# For deep learning models on balanced data
python train_lstm_balanced.py
python train_additional_models.py
python visualize_dl_results.py

# Legacy: Run experiment with config (original system)
python run_experiments.py --config configs/quick_test.yaml
```

### Advanced Usage

```bash
# Run comprehensive experiment with all models
python run_experiments.py --config configs/production.yaml

# Generate performance report
python -c "from results.report_generator import ReportGenerator; ReportGenerator().generate_report('experiment_id', format='html')"

# Launch interactive dashboard
python -c "from visualization.dashboard import ModelComparisonDashboard; ModelComparisonDashboard().run()"

# Run performance benchmarks
pytest tests/test_performance.py -m benchmark

# Run integration tests
pytest tests/test_integration.py -v
```

### Legacy Compatibility

```bash
# Run legacy pipeline (maintains backward compatibility)
python scripts/legacy/model.py  # Uses legacy Dly_Classifier class

# Process TAF files (legacy)
python scripts/legacy/filter_TAFs.py

# Generate legacy plots
python scripts/legacy/plots.py
```

## Modern Architecture (Post-Refactoring)

### Configuration System

- **config.py**: Type-safe configuration classes with validation
- **config_parser.py**: YAML/JSON configuration parsing and merging
- **config_utils.py**: CLI utilities for configuration management
- **configs/**: Production-ready configuration files

### Data Pipeline (`data/`)

- **data_loader.py**: Intelligent data loading with caching and parallel processing
- **data_validation.py**: Comprehensive data quality checks and anomaly detection
- **feature_engineering.py**: Automated weather feature engineering and selection
- **preprocessing.py**: Flexible preprocessing pipelines with sklearn integration

### Model Implementations (`models/`)

- **base_model.py**: Abstract base class with common functionality
- **13 Model Classes**: RandomForest, LSTM, CNN, RNN, FNN, GRU, Transformer, Attention-LSTM, WaveNet, Autoencoder, Ensemble, CNN-RNN, CNN-LSTM
- **Advanced Features**: Bidirectional RNNs, attention mechanisms, ensemble methods

### Training Pipeline (`training/`)

- **trainer.py**: Centralized training management with experiment tracking
- **hyperparameter_tuning.py**: 6 different tuning methods including distributed optimization

### Results Management (`results/`)

- **results_manager.py**: Structured result storage, comparison, and export
- **report_generator.py**: Multi-format report generation (HTML, PDF, LaTeX, PowerPoint)

### Visualization (`visualization/`)

- **plots.py**: Advanced plotting with interactive and static options
- **dashboard.py**: Multi-tab interactive dashboard for model comparison

### Testing (`tests/`)

- **test_models.py**: Unit tests for all 13 models
- **test_data_pipeline.py**: Data pipeline testing
- **test_training_pipeline.py**: Training and tuning tests
- **test_integration.py**: End-to-end workflow tests
- **test_performance.py**: Comprehensive performance benchmarks

### Documentation

- **README.md**: Complete system documentation with examples
- **docs/API_DOCUMENTATION.md**: Detailed API reference
- **notebooks/**: Jupyter tutorials for quick start and advanced usage

## Data Flow & Pipeline

### 1. Configuration Loading

```python
from config_parser import ConfigParser
config = ConfigParser().load_config("configs/production.yaml")
```

### 2. Data Pipeline

```python
from data.data_loader import DataLoader
from data.feature_engineering import WeatherFeatureEngineer

# Load and enhance data
loader = DataLoader(enable_cache=True, parallel_loading=True)
engineer = WeatherFeatureEngineer()
enhanced_data = engineer.create_features(weather_data)
```

### 3. Model Training

```python
from run_experiments import ExperimentRunner
runner = ExperimentRunner(config)
results = runner.run_all_experiments()
```

### 4. Results Analysis

```python
from results.results_manager import ResultsManager
from visualization.dashboard import ModelComparisonDashboard

# Save and analyze results
manager = ResultsManager()
experiment_id = manager.save_experiment_result(results)

# Launch interactive dashboard
dashboard = ModelComparisonDashboard()
dashboard.run(port=8050)
```

## Supported Models

### Traditional Machine Learning

- **Random Forest**: Fast, interpretable, feature importance
- **FNN**: Feedforward neural networks for baseline comparison

### Recurrent Neural Networks

- **LSTM**: Long Short-Term Memory for sequential patterns
- **GRU**: Gated Recurrent Unit, simpler alternative to LSTM
- **RNN**: Basic recurrent networks for simple temporal modeling
- **Bidirectional variants**: Enhanced temporal understanding

### Convolutional Networks

- **CNN**: Spatial pattern recognition in weather data
- **CNN-RNN/CNN-LSTM**: Hybrid models combining spatial and temporal features

### Advanced Deep Learning

- **Transformer**: State-of-the-art attention-based architecture
- **Attention-LSTM**: LSTM enhanced with attention mechanisms
- **WaveNet**: Dilated causal convolutions for time series
- **Autoencoder**: Feature learning and anomaly detection

### Meta-Learning

- **Ensemble**: Combines multiple models for improved performance
- **Voting/Stacking**: Different ensemble strategies

## Hyperparameter Tuning Methods

1. **Grid Search**: Exhaustive search over parameter grid
2. **Random Search**: Random sampling from parameter distributions
3. **Bayesian Optimization**: Intelligent search using Optuna
4. **Keras Tuner**: Specialized tuning for deep learning models
5. **Ray Tune**: Distributed tuning with advanced schedulers
6. **Multi-objective**: Optimize multiple metrics simultaneously

## Performance Characteristics

### Benchmarks (16GB RAM, 8-core CPU)

- **Small dataset (1K samples)**: RF ~2s, LSTM ~15s
- **Medium dataset (10K samples)**: RF ~8s, LSTM ~90s
- **Large dataset (100K samples)**: RF ~45s

### Memory Optimization

- Intelligent caching system
- Lazy loading for large datasets
- Memory leak detection and prevention
- Peak memory monitoring

## Configuration Examples

### Quick Test Configuration

```yaml
name: "quick_test"
data:
  airports: ["EGLL"]
  start_date: "2023-01-01"
  end_date: "2023-03-31"
models:
  random_forest:
    n_estimators: 50
    max_depth: 10
  lstm:
    units: 32
    epochs: 10
hyperparameter_tuning:
  enabled: false
```

### Production Configuration

```yaml
name: "production_experiment"
data:
  airports: ["EGLL", "LSZH", "LFPG", "LOWW"]
  start_date: "2017-01-01"
  end_date: "2019-12-08"
models:
  random_forest:
    n_estimators: 200
    max_depth: 15
  lstm:
    units: 128
    epochs: 100
    bidirectional: true
  transformer:
    d_model: 256
    num_heads: 8
    num_layers: 6
hyperparameter_tuning:
  enabled: true
  method: "bayesian"
  n_trials: 100
```

## Data Sources & Paths

### Input Data Structure

```bash
./Data/
├── METAR/                    # Weather observations
│   ├── METAR_EGLL_filtered.csv
│   ├── METAR_LSZH_filtered.csv
│   └── ...
├── Regulations/              # ATFM regulation data
│   ├── Regulations_filtered_EGLL.csv
│   └── ...
└── TAF/                      # Weather forecasts (if available)
```

### Output Structure

```bash
./results/                     # Results management modules only
├── __init__.py
├── report_generator.py
└── results_manager.py

./visualizations/              # All generated plots and charts
├── traditional_ml/           # Traditional ML visualizations
│   ├── model_performance.png
│   ├── balanced_model_performance.png
│   ├── feature_importance.csv
│   └── balanced_feature_importance.csv
└── deep_learning/           # Deep learning visualizations
    ├── *_confusion_matrix.png
    ├── *_roc_pr_curves.png
    └── model_comparison.png

./experiment_results/          # Experiment outputs from run_experiments.py
./models/                      # Saved model artifacts
├── balanced_weather_regulation_model.pkl
└── deep_learning/            # Deep learning model checkpoints
```

## High-Level Architecture

### Two Workflow Systems

#### 1. Balanced Dataset Pipeline (Recommended)

Located in `scripts/balanced_dataset/`, this achieves dramatically better results:

- **Entry point**: `run_balanced_pipeline.py`
- **Key innovation**: Intelligent time-window sampling around regulation events
- **Performance**: F1 score improved from 0.071 → 0.879 (1,138% improvement)
- **Best for**: Production use and new experiments

#### 2. Original Experiment System (Legacy)

The `run_experiments.py` system offers more flexibility:

- **Entry point**: `run_experiments.py`
- **Features**: 13 models, 6 hyperparameter tuning methods, YAML configs
- **Best for**: Custom experiments and research

### Core Design Principles

1. **Modular Architecture**: Each component (data, models, training, visualization) is self-contained
2. **Configuration-Driven**: All experiments are controlled via YAML configurations
3. **Type Safety**: Extensive use of dataclasses and type hints throughout
4. **Backward Compatibility**: Legacy scripts maintained in `scripts/legacy/`

### Key Architectural Patterns

#### 1. Abstract Base Model Pattern

All models inherit from `BaseModel` which enforces a consistent interface:

- `train()`: Training logic with validation support
- `predict()`: Prediction with probability scores
- `_build_model()`: Model-specific architecture
- Automatic metrics calculation and history tracking

#### 2. Configuration Hierarchy

```
ExperimentConfig (top-level)
├── DataConfig (data sources and preprocessing)
├── TrainingConfig (training parameters)
├── Model-specific configs (RandomForestConfig, LSTMConfig, etc.)
└── HyperparameterTuningConfig (tuning settings)
```

#### 3. Pipeline Architecture

```
Data Loading → Validation → Feature Engineering → Preprocessing →
Model Training → Hyperparameter Tuning → Results Management → Visualization
```

#### 4. Experiment Management

- Each experiment gets a unique ID and timestamp
- Results are stored in structured format (JSON + artifacts)
- Support for experiment comparison and aggregation
- MLflow integration for tracking

### Model Registry

Models are dynamically loaded based on availability:

- **Always Available**: RandomForest, FNN (sklearn-based)
- **TensorFlow Required**: LSTM, CNN, GRU, RNN, Transformer, AttentionLSTM, WaveNet, Autoencoder, CNN-RNN, CNN-LSTM
- **Ensemble**: Requires at least base models to be available

### Data Flow

1. **Raw Data**: METAR CSVs + Regulation CSVs →
2. **DataLoader**: Merges and aligns time series →
3. **FeatureEngineer**: Creates weather features (severity, flight categories) →
4. **Preprocessor**: Scaling, encoding, sequencing →
5. **Model Input**: Ready for training

### Error Handling Strategy

- Graceful degradation when TensorFlow unavailable
- Comprehensive validation at each pipeline stage
- Detailed error messages with recovery suggestions
- Automatic fallback to simpler models when needed

## Important Implementation Details

### Adding New Models

1. Create model class inheriting from `BaseModel` in `models/`
2. Implement required methods: `train()`, `predict()`, `_build_model()`
3. Add configuration dataclass in `src/config.py`
4. Register model in `run_experiments.py` MODEL_REGISTRY
5. Add comprehensive tests in `tests/test_models.py`

### Working with Balanced Dataset

The balanced dataset approach is crucial for handling severe class imbalance:

- Original data: ~1% positive samples (regulations)
- Balanced data: 50% positive samples
- Uses time-window sampling around regulation events
- Dramatically improves model performance (F1: 0.07 → 0.88)

Key scripts in `scripts/balanced_dataset/`:

- `analyze_and_balance_data.py`: Initial analysis and balancing
- `prepare_balanced_data.py`: Feature engineering for balanced data
- `train_balanced_model.py`: Model training on balanced dataset
- `run_balanced_pipeline.py`: Complete pipeline execution

### Testing Best Practices

```bash
# Always run tests before committing
pytest tests/test_models.py::TestRandomForest  # Test specific model
pytest -k "test_train" -v                       # Test all training methods
pytest --cov=models --cov-report=term-missing   # Check coverage

# Integration tests are critical
pytest tests/test_integration.py::test_full_pipeline -v
```

### Performance Monitoring

For production experiments:

1. Enable MLflow tracking in configuration
2. Monitor memory usage with `psutil` integration
3. Use `tqdm` progress bars for long-running operations
4. Check `experiment_results/` for detailed logs

### Common Pitfalls to Avoid

1. **Data Leakage**: Always split data before any preprocessing
2. **Config Validation**: Use `ConfigParser.validate_config()` before experiments
3. **Memory Management**: Use `DataLoader(enable_cache=True)` for large datasets
4. **Model Persistence**: Always use `joblib` for sklearn models, `keras.save()` for DL models
