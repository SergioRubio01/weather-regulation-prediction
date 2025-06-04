# CLAUDE.md

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with this advanced weather regulation prediction system.

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

### Basic Usage

```bash
# Install dependencies using Poetry
poetry install

# Activate virtual environment
poetry env activate

# Run quick experiment with default config
python run_experiments.py --config configs/quick_test.yaml

# Run specific model
python run_experiments.py --models random_forest lstm

# Run with hyperparameter tuning
python run_experiments.py --tune --tuner bayesian --trials 50
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

# Run all tests
pytest tests/ --cov=. --cov-report=html
```

### Legacy Compatibility

```bash
# Run legacy pipeline (maintains backward compatibility)
python model.py  # Uses legacy Dly_Classifier class

# Process TAF files (legacy)
python filter_TAFs.py

# Generate legacy plots
python plots.py
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
./Output/
├── results/                  # Structured experiment results
├── visualizations/           # Generated plots and charts
├── reports/                  # Multi-format reports
└── models/                   # Saved model artifacts
```

## Important Notes for Development

### Adding New Models

1. Inherit from `BaseModel` class
2. Implement required methods: `train()`, `predict()`, `_build_model()`
3. Add configuration class in `config.py`
4. Register in `run_experiments.py`
5. Add tests in `tests/test_models.py`

### Configuration Management

- Use type-safe configuration classes
- Validate configurations before experiments
- Support environment-specific configs
- Enable configuration merging and inheritance

### Performance Considerations

- Enable caching for repeated data loading
- Use parallel processing where possible
- Monitor memory usage during experiments
- Implement early stopping for long training runs

### Testing Strategy

- Unit tests for individual components
- Integration tests for end-to-end workflows
- Performance benchmarks for scalability
- Compatibility tests for legacy support

### Legacy Support

- The system maintains full backward compatibility
- Legacy `functions.py` and `model.py` still functional
- Use `use_new_pipeline=True/False` to switch between old and new systems
- Gradual migration path available

## Troubleshooting

### Common Issues

- **Missing TensorFlow**: Some deep learning models will skip gracefully
- **Memory issues**: Reduce batch sizes or enable lazy loading
- **Performance**: Enable parallel processing and caching
- **Configuration errors**: Use schema validation for early detection

### Performance Optimization

- Use appropriate data types (float32 vs float64)
- Enable multiprocessing for CPU-intensive operations
- Use GPU acceleration when available
- Implement smart caching strategies

This system represents a complete evolution from the original codebase while maintaining compatibility and providing enterprise-grade capabilities for weather regulation prediction research.
