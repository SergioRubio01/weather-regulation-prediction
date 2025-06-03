# API Documentation

## Table of Contents

- [Configuration System](#configuration-system)
- [Data Pipeline](#data-pipeline)
- [Models](#models)
- [Training Pipeline](#training-pipeline)
- [Results Management](#results-management)
- [Visualization](#visualization)
- [Utilities](#utilities)

---

## Configuration System

### config.py

Configuration classes for all system components.

#### ExperimentConfig

Main configuration container for experiments.

```python
class ExperimentConfig:
    name: str
    description: Optional[str] = ""
    data: DataConfig
    training: TrainingConfig
    models: Dict[str, Any]
    hyperparameter_tuning: Optional[HyperparameterTuningConfig] = None
    experiment_tracking: Optional[ExperimentTrackingConfig] = None
```

**Parameters:**
- `name`: Unique experiment identifier
- `description`: Optional experiment description
- `data`: Data configuration (DataConfig instance)
- `training`: Training configuration (TrainingConfig instance)
- `models`: Dictionary of model configurations
- `hyperparameter_tuning`: Optional HP tuning configuration
- `experiment_tracking`: Optional MLflow tracking configuration

**Methods:**
- `to_dict() -> Dict[str, Any]`: Convert to dictionary
- `validate() -> bool`: Validate configuration
- `save(path: str)`: Save configuration to file
- `load(path: str) -> ExperimentConfig`: Load from file

#### DataConfig

Configuration for data processing.

```python
class DataConfig:
    airports: List[str]
    start_date: str
    end_date: str
    weather_data_path: Optional[str] = None
    regulation_data_path: Optional[str] = None
    taf_data_path: Optional[str] = None
    time_resolution: str = "30min"
    features_to_include: Optional[List[str]] = None
    target_column: str = "has_regulation"
```

#### TrainingConfig

Configuration for model training.

```python
class TrainingConfig:
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    cross_validation: bool = False
    cv_folds: int = 5
    early_stopping: bool = True
    patience: int = 10
```

#### Model Configurations

##### RandomForestConfig

```python
class RandomForestConfig(BaseModelConfig):
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str = "sqrt"
    bootstrap: bool = True
    n_jobs: Optional[int] = None
    random_state: Optional[int] = None
    class_weight: Optional[str] = None
```

##### LSTMConfig

```python
class LSTMConfig(BaseModelConfig):
    units: int = 50
    dropout: float = 0.2
    recurrent_dropout: float = 0.2
    return_sequences: bool = False
    bidirectional: bool = False
    batch_size: int = 32
    epochs: int = 100
    sequence_length: int = 24
    optimizer: Optimizer = Optimizer.ADAM
    loss: Loss = Loss.BINARY_CROSSENTROPY
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])
```

##### TransformerConfig

```python
class TransformerConfig(BaseModelConfig):
    d_model: int = 128
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    sequence_length: int = 24
    batch_size: int = 32
    epochs: int = 100
    optimizer: Optimizer = Optimizer.ADAM
    learning_rate: float = 0.001
    warmup_steps: int = 4000
```

### config_parser.py

Configuration parsing and management utilities.

#### ConfigParser

```python
class ConfigParser:
    def __init__(self, schema_validation: bool = True)
    
    def load_config(self, config_path: str) -> ExperimentConfig
    def save_config(self, config: ExperimentConfig, output_path: str)
    def parse_dict(self, config_dict: Dict[str, Any]) -> ExperimentConfig
    def merge_configs(self, base_config: Dict, override_config: Dict) -> Dict
    def validate_config(self, config: ExperimentConfig) -> Tuple[bool, List[str]]
    def generate_param_grid(self, model_config: BaseModelConfig) -> Dict[str, List]
```

**load_config(config_path: str) -> ExperimentConfig**

Load configuration from YAML or JSON file.

**Parameters:**
- `config_path`: Path to configuration file

**Returns:**
- Parsed ExperimentConfig object

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `ValidationError`: If configuration is invalid

**Example:**
```python
parser = ConfigParser()
config = parser.load_config("experiment_config.yaml")
```

---

## Data Pipeline

### data_loader.py

Data loading and caching utilities.

#### DataLoader

```python
class DataLoader:
    def __init__(self, 
                 data_path: str = "./Data",
                 enable_cache: bool = True,
                 cache_ttl_hours: int = 24,
                 parallel_loading: bool = False,
                 n_jobs: int = -1)
```

**Parameters:**
- `data_path`: Base path for data files
- `enable_cache`: Enable intelligent caching
- `cache_ttl_hours`: Cache time-to-live in hours
- `parallel_loading`: Enable parallel file loading
- `n_jobs`: Number of parallel jobs (-1 for all cores)

**Methods:**

**load_metar_data(file_path: str, **kwargs) -> pd.DataFrame**

Load METAR weather observation data.

**Parameters:**
- `file_path`: Path to METAR data file
- `**kwargs`: Additional pandas read parameters

**Returns:**
- DataFrame with parsed METAR data

**load_regulation_data(file_path: str, **kwargs) -> pd.DataFrame**

Load ATFM regulation data.

**Parameters:**
- `file_path`: Path to regulation data file
- `**kwargs`: Additional pandas read parameters

**Returns:**
- DataFrame with regulation data

**create_features(metar_data: pd.DataFrame, 
                 regulation_data: pd.DataFrame,
                 taf_data: Optional[pd.DataFrame] = None,
                 target_column: str = "has_regulation") -> pd.DataFrame**

Create combined feature matrix from all data sources.

**Parameters:**
- `metar_data`: Weather observation data
- `regulation_data`: Regulation data
- `taf_data`: Optional forecast data
- `target_column`: Target variable column name

**Returns:**
- Combined feature DataFrame

#### DataCache

```python
class DataCache:
    def __init__(self, cache_dir: str = "./cache", max_size_gb: float = 1.0, ttl_hours: int = 24)
    
    def get(self, key: str) -> Optional[Any]
    def set(self, key: str, value: Any)
    def exists(self, key: str) -> bool
    def clear(self)
    def cleanup_expired(self)
```

### data_validation.py

Data quality validation and monitoring.

#### DataValidator

```python
class DataValidator:
    def __init__(self, config: Optional[Dict] = None)
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]
    def validate_weather_data(self, data: pd.DataFrame) -> Dict[str, Any]
    def generate_validation_report(self, validation_results: Dict) -> str
```

**validate_weather_data(data: pd.DataFrame) -> Dict[str, Any]**

Validate weather data for quality and consistency.

**Parameters:**
- `data`: Weather DataFrame to validate

**Returns:**
- Validation report dictionary with errors, warnings, and summary

#### AnomalyDetector

```python
class AnomalyDetector:
    def __init__(self, contamination: float = 0.1)
    
    def detect_anomalies(self, 
                        data: pd.DataFrame,
                        columns: List[str],
                        method: str = "isolation_forest") -> List[int]
```

**detect_anomalies(data, columns, method) -> List[int]**

Detect anomalous data points.

**Parameters:**
- `data`: Input DataFrame
- `columns`: Columns to analyze for anomalies
- `method`: Detection method ("isolation_forest", "one_class_svm", "local_outlier_factor")

**Returns:**
- List of anomalous row indices

#### DataDriftDetector

```python
class DataDriftDetector:
    def __init__(self, alpha: float = 0.05)
    
    def detect_drift(self,
                    reference_data: pd.DataFrame,
                    current_data: pd.DataFrame,
                    columns: List[str]) -> Dict[str, Dict]
```

### feature_engineering.py

Automated feature engineering for weather data.

#### WeatherFeatureEngineer

```python
class WeatherFeatureEngineer:
    def __init__(self, include_derived: bool = True, include_interactions: bool = False)
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame
    def calculate_flight_category(self, visibility: pd.Series, ceiling: pd.Series) -> pd.Series
    def calculate_weather_severity(self, data: pd.DataFrame) -> pd.Series
    def create_wind_components(self, speed: pd.Series, direction: pd.Series) -> Tuple[pd.Series, pd.Series]
```

**create_features(data: pd.DataFrame) -> pd.DataFrame**

Create weather-specific features.

**Parameters:**
- `data`: Input weather DataFrame

**Returns:**
- Enhanced DataFrame with new features including:
  - Flight categories (VFR, MVFR, IFR, LIFR)
  - Weather severity scores
  - Wind components (U/V)
  - Dewpoint calculations
  - Derived meteorological parameters

#### TimeSeriesFeatureEngineer

```python
class TimeSeriesFeatureEngineer:
    def __init__(self, max_lag: int = 24, default_windows: List[int] = None)
    
    def create_features(self,
                       data: pd.DataFrame,
                       timestamp_col: str = "timestamp",
                       value_cols: List[str] = None,
                       lags: List[int] = None,
                       rolling_windows: List[int] = None) -> pd.DataFrame
```

#### AutomatedFeatureEngineer

```python
class AutomatedFeatureEngineer:
    def __init__(self, 
                 max_features: int = 100,
                 selection_method: str = "mutual_info",
                 create_polynomials: bool = True,
                 max_polynomial_degree: int = 2)
    
    def create_features(self,
                       data: pd.DataFrame,
                       target_col: str,
                       max_features: Optional[int] = None) -> pd.DataFrame
    def get_selected_features(self) -> List[str]
    def get_feature_importance(self) -> pd.DataFrame
```

### preprocessing.py

Data preprocessing pipeline components.

#### PreprocessingPipeline

```python
class PreprocessingPipeline:
    def __init__(self, steps: Optional[List[Tuple[str, Any]]] = None)
    
    def add_step(self, name: str, transformer: Any)
    def remove_step(self, name: str)
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs)
    def transform(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> pd.DataFrame
    def save(self, path: str)
    def load(path: str) -> 'PreprocessingPipeline'
```

**Example:**
```python
pipeline = PreprocessingPipeline()
pipeline.add_step('scaler', TimeSeriesScaler())
pipeline.add_step('encoder', CyclicalEncoder())
processed_data = pipeline.fit_transform(data)
```

#### TimeSeriesScaler

```python
class TimeSeriesScaler:
    def __init__(self, method: str = "standard", feature_range: Tuple[float, float] = (0, 1))
    
    def fit(self, X: pd.DataFrame) -> 'TimeSeriesScaler'
    def transform(self, X: pd.DataFrame) -> pd.DataFrame
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame
```

---

## Models

### base_model.py

Abstract base class for all models.

#### BaseModel

```python
class BaseModel(ABC):
    def __init__(self, config: BaseModelConfig)
    
    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None) -> Dict[str, Any]
    
    @abstractmethod  
    def predict(self, X) -> np.ndarray
    
    def predict_proba(self, X) -> np.ndarray
    def evaluate(self, X, y) -> ModelMetrics
    def cross_validate(self, X, y, cv: int = 5) -> Dict[str, List[float]]
    def save_model(self, path: str)
    def load_model(self, path: str)
    def get_feature_importance(self) -> Optional[pd.DataFrame]
```

#### ModelMetrics

```python
class ModelMetrics:
    def __init__(self)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None)
    def to_dict(self) -> Dict[str, float]
    def __str__(self) -> str
```

**Attributes:**
- `accuracy`: Classification accuracy
- `precision`: Precision score
- `recall`: Recall score
- `f1_score`: F1 score
- `auc_roc`: Area under ROC curve
- `confusion_matrix`: Confusion matrix

### Model Implementations

#### RandomForestModel

```python
class RandomForestModel(BaseModel):
    def __init__(self, config: RandomForestConfig)
    
    def train(self, X_train, y_train, X_val=None, y_val=None) -> Dict[str, Any]
    def predict(self, X) -> np.ndarray
    def tune_hyperparameters(self, X, y, param_grid: Dict, method: str = "grid", cv: int = 5) -> Dict
```

#### LSTMModel

```python
class LSTMModel(BaseModel):
    def __init__(self, config: LSTMConfig)
    
    def train(self, X_train, y_train, X_val=None, y_val=None) -> Dict[str, Any]
    def predict(self, X) -> np.ndarray
    def prepare_data(self, X, y=None) -> Tuple[np.ndarray, Optional[np.ndarray]]
    def _build_model(self, input_shape: Tuple[int, ...]) -> tf.keras.Model
```

#### EnsembleModel

```python
class EnsembleModel(BaseModel):
    def __init__(self, config: EnsembleConfig)
    
    def train(self, X_train, y_train, X_val=None, y_val=None) -> Dict[str, Any]
    def predict(self, X) -> np.ndarray
    def get_model_contributions(self) -> Dict[str, float]
```

---

## Training Pipeline

### trainer.py

Training management and orchestration.

#### Trainer

```python
class Trainer:
    def __init__(self, 
                 enable_logging: bool = True,
                 checkpoint_dir: Optional[str] = None,
                 experiment_tracker: Optional[ExperimentTracker] = None)
    
    def train_model(self,
                   model: BaseModel,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_val: Optional[np.ndarray] = None,
                   y_val: Optional[np.ndarray] = None,
                   model_name: str = "model") -> Dict[str, Any]
    
    def cross_validate(self,
                      model: BaseModel,
                      X: np.ndarray,
                      y: np.ndarray,
                      cv_folds: int = 5,
                      model_name: str = "model") -> Dict[str, Any]
    
    def train_ensemble(self,
                      models: Dict[str, BaseModel],
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_val: Optional[np.ndarray] = None,
                      y_val: Optional[np.ndarray] = None,
                      ensemble_method: str = "voting") -> Dict[str, Any]
```

**train_model(...) -> Dict[str, Any]**

Train a single model with validation.

**Parameters:**
- `model`: Model instance to train
- `X_train`: Training features
- `y_train`: Training targets
- `X_val`: Validation features (optional)
- `y_val`: Validation targets (optional)
- `model_name`: Name for the trained model

**Returns:**
- Dictionary with training results including metrics and timing

#### ExperimentTracker

```python
class ExperimentTracker:
    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None)
    
    def start_run(self, run_name: Optional[str] = None)
    def end_run(self)
    def log_params(self, params: Dict[str, Any])
    def log_metrics(self, metrics: Dict[str, float])
    def log_model(self, model: Any, model_name: str)
    def log_artifacts(self, artifacts: List[str])
```

### hyperparameter_tuning.py

Hyperparameter optimization methods.

#### GridSearchTuner

```python
class GridSearchTuner:
    def __init__(self, scoring: str = "accuracy", n_jobs: int = -1)
    
    def tune(self,
            model: BaseModel,
            param_grid: Dict[str, List],
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            cv: int = 5) -> TuningResult
```

#### BayesianOptimizationTuner

```python
class BayesianOptimizationTuner:
    def __init__(self, n_trials: int = 100, random_state: Optional[int] = None)
    
    def tune(self,
            model: BaseModel,
            param_space: Dict[str, List],
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray) -> TuningResult
```

#### TuningResult

```python
class TuningResult:
    def __init__(self,
                 best_params: Dict[str, Any],
                 best_score: float,
                 all_results: List[Dict[str, Any]])
    
    def save(self, path: str)
    def load(path: str) -> 'TuningResult'
    def plot_optimization_history(self) -> go.Figure
    def get_top_trials(self, n: int = 10) -> List[Dict[str, Any]]
```

---

## Results Management

### results_manager.py

Structured storage and retrieval of experimental results.

#### ResultsManager

```python
class ResultsManager:
    def __init__(self, base_path: str = "./results")
    
    def generate_experiment_id(self) -> str
    def save_model_result(self, result: ModelResult, experiment_id: Optional[str] = None) -> str
    def save_experiment_result(self, experiment: ExperimentResult) -> str
    def load_model_result(self, model_id: str) -> Optional[ModelResult]
    def load_experiment_result(self, experiment_id: str) -> Optional[ExperimentResult]
    def list_experiments(self, filter_by: Optional[Dict[str, Any]] = None) -> pd.DataFrame
    def compare_experiments(self, experiment_ids: List[str], metric: str = 'test_accuracy') -> pd.DataFrame
    def get_best_models(self, n: int = 10, metric: str = 'test_accuracy') -> pd.DataFrame
    def export_results(self, experiment_id: str, format: str = 'csv', output_path: Optional[str] = None) -> str
```

**save_experiment_result(experiment: ExperimentResult) -> str**

Save complete experiment results.

**Parameters:**
- `experiment`: ExperimentResult object containing all model results

**Returns:**
- Experiment ID string

**compare_experiments(experiment_ids: List[str], metric: str) -> pd.DataFrame**

Compare performance across multiple experiments.

**Parameters:**
- `experiment_ids`: List of experiment IDs to compare
- `metric`: Metric to compare ('test_accuracy', 'test_f1', etc.)

**Returns:**
- DataFrame with comparison results

#### ModelResult

```python
class ModelResult:
    model_name: str
    model_type: str
    timestamp: datetime
    config: Dict[str, Any]
    
    # Training metrics
    training_time: float
    training_history: Optional[Dict[str, List[float]]]
    
    # Test metrics  
    test_accuracy: Optional[float]
    test_precision: Optional[float]
    test_recall: Optional[float]
    test_f1: Optional[float]
    test_auc: Optional[float]
    
    # Additional data
    confusion_matrix: Optional[np.ndarray]
    feature_importance: Optional[pd.DataFrame]
    predictions: Optional[np.ndarray]
    
    def to_dict(self) -> Dict[str, Any]
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelResult'
```

#### ExperimentResult

```python
class ExperimentResult:
    experiment_id: str
    experiment_name: str
    timestamp: datetime
    config: ExperimentConfig
    model_results: Dict[str, ModelResult]
    
    # Aggregate metrics
    best_model: Optional[str]
    best_accuracy: Optional[float]
    comparison_metrics: Optional[pd.DataFrame]
    
    def add_model_result(self, result: ModelResult)
    def get_best_model_result(self) -> Optional[ModelResult]
    def generate_summary(self) -> Dict[str, Any]
```

### report_generator.py

Multi-format report generation.

#### ReportGenerator

```python
class ReportGenerator:
    def __init__(self, template_dir: Optional[str] = None)
    
    def generate_report(self,
                       experiment: ExperimentResult,
                       format: str = 'html',
                       output_path: Optional[str] = None,
                       include_visualizations: bool = True) -> str
```

**generate_report(...) -> str**

Generate comprehensive experiment report.

**Parameters:**
- `experiment`: ExperimentResult object
- `format`: Output format ('html', 'pdf', 'markdown', 'latex', 'pptx')
- `output_path`: Optional output file path
- `include_visualizations`: Whether to include charts and plots

**Returns:**
- Path to generated report file

**Supported Formats:**
- **HTML**: Interactive report with Bootstrap styling and Plotly charts
- **PDF**: Professional PDF using ReportLab
- **Markdown**: Documentation-friendly format
- **LaTeX**: Academic paper format
- **PowerPoint**: Presentation format

---

## Visualization

### plots.py

Advanced plotting and visualization utilities.

#### ModelVisualizer

```python
class ModelVisualizer:
    def __init__(self, save_path: str = "./visualizations")
    
    def plot_confusion_matrix(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             labels: Optional[List[str]] = None,
                             title: str = "Confusion Matrix",
                             normalize: bool = True,
                             interactive: bool = True,
                             save_name: Optional[str] = None) -> Union[go.Figure, plt.Figure]
    
    def plot_roc_curves(self,
                       y_true: np.ndarray,
                       y_scores: Dict[str, np.ndarray],
                       title: str = "ROC Curves Comparison",
                       interactive: bool = True,
                       save_name: Optional[str] = None) -> Union[go.Figure, plt.Figure]
    
    def plot_feature_importance(self,
                               importance_data: Dict[str, pd.DataFrame],
                               top_n: int = 20,
                               title: str = "Feature Importance Comparison",
                               interactive: bool = True,
                               save_name: Optional[str] = None) -> Union[go.Figure, plt.Figure]
    
    def plot_training_history(self,
                             histories: Dict[str, Dict[str, List[float]]],
                             metrics: List[str] = ['loss', 'accuracy'],
                             title: str = "Training History",
                             interactive: bool = True,
                             save_name: Optional[str] = None) -> Union[go.Figure, plt.Figure]
    
    def plot_model_comparison(self,
                             comparison_df: pd.DataFrame,
                             metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1_score'],
                             title: str = "Model Performance Comparison",
                             interactive: bool = True,
                             save_name: Optional[str] = None) -> Union[go.Figure, plt.Figure]
```

#### WeatherVisualizer

```python
class WeatherVisualizer:
    def __init__(self, save_path: str = "./visualizations")
    
    def plot_weather_patterns(self,
                             weather_data: pd.DataFrame,
                             features: List[str] = ['temperature', 'pressure', 'wind_speed'],
                             title: str = "Weather Patterns",
                             save_name: Optional[str] = None) -> go.Figure
    
    def plot_regulation_analysis(self,
                               regulation_data: pd.DataFrame,
                               weather_data: pd.DataFrame,
                               title: str = "Regulation vs Weather Analysis",
                               save_name: Optional[str] = None) -> go.Figure
    
    def plot_airport_comparison(self,
                              airport_data: Dict[str, pd.DataFrame],
                              metric: str = 'regulation_rate',
                              title: str = "Airport Comparison",
                              save_name: Optional[str] = None) -> go.Figure
```

### dashboard.py

Interactive dashboard for model comparison.

#### ModelComparisonDashboard

```python
class ModelComparisonDashboard:
    def __init__(self, results_path: str = "./results")
    
    def create_layout(self)
    def setup_callbacks(self)
    def run(self, debug: bool = False, port: int = 8050)
```

**Dashboard Tabs:**
- **Overview**: Experiment summaries and key metrics
- **Model Performance**: Accuracy comparisons and radar charts  
- **Detailed Comparison**: Side-by-side model analysis
- **Feature Analysis**: Feature importance across models
- **Training Analysis**: Training curves and statistics
- **Weather Analysis**: Domain-specific weather insights
- **Reports**: Export and download functionality

**Usage:**
```python
dashboard = ModelComparisonDashboard(results_path="./results")
dashboard.run(port=8050)
# Open browser to http://localhost:8050
```

---

## Utilities

### Utility Functions

#### create_model_result

```python
def create_model_result(model_name: str,
                       model_type: str,
                       y_true: np.ndarray,
                       y_pred: np.ndarray,
                       y_proba: Optional[np.ndarray] = None,
                       training_time: float = 0.0,
                       config: Optional[Dict[str, Any]] = None) -> ModelResult
```

Create ModelResult from predictions and true labels.

#### plot_model_results

```python
def plot_model_results(results: Dict[str, Any], 
                      save_path: str = "./visualizations") -> Dict[str, Any]
```

Quick function to generate all standard plots for model results.

#### create_tuner

```python
def create_tuner(tuner_type: str, **kwargs) -> BaseTuner
```

Factory function to create hyperparameter tuners.

**Parameters:**
- `tuner_type`: Type of tuner ('grid', 'random', 'bayesian', 'keras', 'ray', 'multi_objective')
- `**kwargs`: Tuner-specific parameters

### Error Handling

All API functions include comprehensive error handling:

- **Input Validation**: Check parameter types and ranges
- **File Operations**: Handle missing files and permissions
- **Model Training**: Catch and report training failures
- **Memory Management**: Monitor and prevent memory issues
- **Configuration**: Validate all configuration parameters

### Logging

The system provides detailed logging:

```python
import logging

# Configure logging level
logging.getLogger('weather_prediction').setLevel(logging.INFO)

# Enable specific module logging
logging.getLogger('training.trainer').setLevel(logging.DEBUG)
```

### Performance Monitoring

Built-in performance monitoring utilities:

```python
from tests.test_performance import PerformanceMonitor

monitor = PerformanceMonitor().start_monitoring()
# ... run operations ...
metrics = monitor.stop_monitoring()
```

---

## Examples

### Complete Workflow Example

```python
from config_parser import ConfigParser
from run_experiments import ExperimentRunner
from results.results_manager import ResultsManager
from results.report_generator import ReportGenerator
from visualization.dashboard import ModelComparisonDashboard

# 1. Load configuration
parser = ConfigParser()
config = parser.load_config("configs/experiment_config.yaml")

# 2. Run experiments
runner = ExperimentRunner(config)
results = runner.run_all_experiments()

# 3. Save results
results_manager = ResultsManager()
experiment_id = results_manager.save_experiment_result(runner.experiment_result)

# 4. Generate report
generator = ReportGenerator()
report_path = generator.generate_report(
    runner.experiment_result,
    format='html',
    include_visualizations=True
)

# 5. Launch dashboard
dashboard = ModelComparisonDashboard()
dashboard.run(port=8050)
```

### Custom Model Implementation Example

```python
from models.base_model import BaseModel
from config import BaseModelConfig
from dataclasses import dataclass

@dataclass
class CustomModelConfig(BaseModelConfig):
    param1: int = 10
    param2: float = 0.5

class CustomModel(BaseModel):
    def __init__(self, config: CustomModelConfig):
        super().__init__(config)
        self.model = None
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        # Implement training logic
        start_time = time.time()
        
        # Your training code here
        self.model = YourModelClass(**self.config.__dict__)
        self.model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Evaluate
        if X_val is not None and y_val is not None:
            y_pred = self.predict(X_val)
            metrics = self.evaluate(X_val, y_val)
            return {
                'accuracy': metrics.accuracy,
                'training_time': training_time
            }
        
        return {'training_time': training_time}
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model must be trained first")
        return self.model.predict(X)
```

This completes the comprehensive API documentation covering all major components of the Weather Regulation Prediction System.