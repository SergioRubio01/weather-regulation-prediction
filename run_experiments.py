"""
Experiment Runner for Weather Regulation Prediction Models

This module provides a comprehensive experiment management system that supports:
- Running multiple experiments with different configurations
- Parallel execution of experiments
- Automatic hyperparameter tuning
- Result aggregation and comparison
- Report generation
- Model ensemble creation
"""

import json
import logging
import multiprocessing as mp

# Import models with TensorFlow availability check
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from jinja2 import Template
from tqdm import tqdm

# Import our modules
from config import ExperimentConfig, ModelType
from config_parser import ConfigParser
from config_utils import ConfigurationManager
from models import TF_MODELS_AVAILABLE
from models.random_forest import RandomForestModel

# Try to import ensemble separately as it has conditional TF imports
try:
    from models.ensemble import EnsembleModel
except ImportError:
    EnsembleModel = None

if TF_MODELS_AVAILABLE:
    from models.attention_lstm import AttentionLSTMModel
    from models.autoencoder import AutoencoderModel
    from models.cnn import CNNModel
    from models.fnn import FNNModel
    from models.gru import GRUModel
    from models.hybrid_models import CNNLSTMModel, CNNRNNModel
    from models.lstm import LSTMModel
    from models.rnn import RNNModel
    from models.transformer import TransformerModel
    from models.wavenet import WaveNetModel
else:
    warnings.warn("TensorFlow not available. Only RandomForest model will be available.")
    # Set all TF models to None
    AttentionLSTMModel = None
    AutoencoderModel = None
    CNNModel = None
    FNNModel = None
    GRUModel = None
    CNNLSTMModel = None
    CNNRNNModel = None
    LSTMModel = None
    RNNModel = None
    TransformerModel = None
    WaveNetModel = None
from training.hyperparameter_tuning import create_tuner
from training.trainer import create_trainer


@dataclass
class ExperimentResult:
    """Container for experiment results"""

    config_name: str
    model_type: str
    metrics: dict[str, float]
    best_params: dict[str, Any] | None = None
    training_time: float = 0.0
    tuning_time: float = 0.0
    predictions: np.ndarray | None = None
    feature_importance: dict[str, float] | None = None
    model_path: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "config_name": self.config_name,
            "model_type": self.model_type,
            "metrics": self.metrics,
            "best_params": self.best_params,
            "training_time": self.training_time,
            "tuning_time": self.tuning_time,
            "model_path": self.model_path,
            "error_message": self.error_message,
        }


@dataclass
class ExperimentSuite:
    """Collection of experiments to run"""

    name: str
    experiments: list[tuple[str, ExperimentConfig]]
    output_dir: str
    parallel: bool = True
    max_workers: int | None = None
    tune_hyperparameters: bool = True
    tuning_method: str = "bayesian"
    tuning_trials: int = 50

    def __post_init__(self):
        """Initialize suite"""
        self.output_path = (
            Path(self.output_dir) / self.name / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(f"ExperimentSuite_{self.name}")
        logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(self.output_path / "experiment_suite.log")
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger


class ExperimentRunner:
    """Main experiment runner class"""

    def __init__(self, base_config: ExperimentConfig | None = None):
        self.base_config = base_config
        self.model_registry = self._create_model_registry()
        self.logger = logging.getLogger(self.__class__.__name__)

    def _create_model_registry(self) -> dict[str, Any]:
        """Create registry of available models"""
        registry = {
            ModelType.RANDOM_FOREST: RandomForestModel,
        }

        # Add Ensemble if available
        if EnsembleModel is not None:
            registry[ModelType.ENSEMBLE] = EnsembleModel

        # Add TensorFlow models only if available
        if TF_MODELS_AVAILABLE:
            registry.update(
                {
                    ModelType.LSTM: LSTMModel,
                    ModelType.CNN: CNNModel,
                    ModelType.RNN: RNNModel,
                    ModelType.FNN: FNNModel,
                    ModelType.WAVENET: WaveNetModel,
                    ModelType.GRU: GRUModel,
                    ModelType.TRANSFORMER: TransformerModel,
                    ModelType.ATTENTION_LSTM: AttentionLSTMModel,
                    ModelType.AUTOENCODER: AutoencoderModel,
                    ModelType.CNN_RNN: CNNRNNModel,
                    ModelType.CNN_LSTM: CNNLSTMModel,
                }
            )

        return registry

    def run_single_experiment(
        self,
        config: ExperimentConfig,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        tune_hyperparameters: bool = True,
        tuning_method: str = "bayesian",
        tuning_trials: int = 50,
    ) -> ExperimentResult:
        """
        Run a single experiment

        Args:
            config: Experiment configuration
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            tune_hyperparameters: Whether to perform hyperparameter tuning
            tuning_method: Method for hyperparameter tuning
            tuning_trials: Number of tuning trials

        Returns:
            ExperimentResult object
        """
        self.logger.info(f"Starting experiment: {config.name}")

        try:
            # Get model class
            model_class = self.model_registry.get(config.model_type)
            if model_class is None:
                raise ValueError(f"Unknown model type: {config.model_type}")

            # Initialize result
            result = ExperimentResult(
                config_name=config.name, model_type=config.model_type.value, metrics={}
            )

            # Hyperparameter tuning
            best_params = None
            tuning_time = 0.0

            if tune_hyperparameters and config.model_type != ModelType.ENSEMBLE:
                self.logger.info("Starting hyperparameter tuning...")
                tuning_start = datetime.now()

                # Get parameter space from config
                param_space = self._get_param_space(config)

                # Create tuner
                tuner = create_tuner(
                    method=tuning_method,
                    model_class=lambda **params: self._create_model_instance(
                        model_class, config, params
                    ),
                    param_space=param_space,
                    n_trials=tuning_trials,
                    scoring="f1_weighted",
                    cv=5,
                )

                # Perform tuning
                tuning_result = tuner.tune(X_train, y_train)
                best_params = tuning_result.best_params
                result.best_params = best_params

                tuning_time = (datetime.now() - tuning_start).total_seconds()
                result.tuning_time = tuning_time

                self.logger.info(f"Tuning completed. Best score: {tuning_result.best_score:.4f}")

            # Create model with best parameters
            model = self._create_model_instance(model_class, config, best_params)

            # Train model
            self.logger.info("Training model...")
            training_start = datetime.now()

            # Create trainer
            trainer = create_trainer(config, enable_tracking=True)

            # Train
            if hasattr(model, "train"):
                # Custom training method
                model.train(X_train, y_train, X_test, y_test)
            else:
                # Standard sklearn-style training
                training_history = trainer.train(
                    model=model.model if hasattr(model, "model") else model,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_test,
                    y_val=y_test,
                )

            training_time = (datetime.now() - training_start).total_seconds()
            result.training_time = training_time

            # Evaluate model
            self.logger.info("Evaluating model...")
            if hasattr(model, "evaluate"):
                metrics = model.evaluate(X_test, y_test)
                result.metrics = metrics.to_dict() if hasattr(metrics, "to_dict") else metrics
            else:
                # Manual evaluation
                predictions = model.predict(X_test)
                from sklearn.metrics import (
                    accuracy_score,
                    f1_score,
                    precision_score,
                    recall_score,
                    roc_auc_score,
                )

                result.metrics = {
                    "accuracy": accuracy_score(y_test, predictions),
                    "precision": precision_score(
                        y_test, predictions, average="weighted", zero_division=0
                    ),
                    "recall": recall_score(
                        y_test, predictions, average="weighted", zero_division=0
                    ),
                    "f1_score": f1_score(y_test, predictions, average="weighted", zero_division=0),
                }

                if len(np.unique(y_test)) == 2:
                    proba = (
                        model.predict_proba(X_test)[:, 1]
                        if hasattr(model, "predict_proba")
                        else predictions
                    )
                    result.metrics["auc_roc"] = roc_auc_score(y_test, proba)

            # Get predictions for ensemble
            result.predictions = model.predict(X_test)

            # Get feature importance if available
            if hasattr(model, "get_feature_importance"):
                result.feature_importance = model.get_feature_importance()

            # Save model
            model_path = self._save_model(model, config, result.metrics)
            result.model_path = model_path

            self.logger.info(
                f"Experiment completed successfully. "
                f"F1-Score: {result.metrics.get('f1_score', 0):.4f}"
            )

        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            result = ExperimentResult(
                config_name=config.name,
                model_type=config.model_type.value if config.model_type else "unknown",
                metrics={},
                error_message=str(e),
            )

        return result

    def run_experiment_suite(
        self,
        suite: ExperimentSuite,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> list[ExperimentResult]:
        """
        Run a suite of experiments

        Args:
            suite: Experiment suite configuration
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels

        Returns:
            List of ExperimentResult objects
        """
        suite.logger.info(f"Starting experiment suite with {len(suite.experiments)} experiments")

        results = []

        if suite.parallel and len(suite.experiments) > 1:
            # Parallel execution
            max_workers = suite.max_workers or min(mp.cpu_count(), len(suite.experiments))

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all experiments
                future_to_config = {
                    executor.submit(
                        self.run_single_experiment,
                        config,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        suite.tune_hyperparameters,
                        suite.tuning_method,
                        suite.tuning_trials,
                    ): (name, config)
                    for name, config in suite.experiments
                }

                # Process completed experiments
                for future in tqdm(
                    as_completed(future_to_config),
                    total=len(suite.experiments),
                    desc="Running experiments",
                ):
                    name, config = future_to_config[future]
                    try:
                        result = future.result()
                        results.append(result)
                        suite.logger.info(f"Completed: {name}")
                    except Exception as e:
                        suite.logger.error(f"Failed: {name} - {str(e)}")
                        results.append(
                            ExperimentResult(
                                config_name=name,
                                model_type=config.model_type.value,
                                metrics={},
                                error_message=str(e),
                            )
                        )
        else:
            # Sequential execution
            for name, config in tqdm(suite.experiments, desc="Running experiments"):
                result = self.run_single_experiment(
                    config,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    suite.tune_hyperparameters,
                    suite.tuning_method,
                    suite.tuning_trials,
                )
                results.append(result)

        # Save results
        self._save_suite_results(suite, results)

        # Generate report
        self.generate_report(suite, results)

        suite.logger.info("Experiment suite completed")

        return results

    def create_ensemble_from_results(
        self,
        results: list[ExperimentResult],
        ensemble_config: ExperimentConfig,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> ExperimentResult:
        """
        Create an ensemble model from experiment results

        Args:
            results: List of experiment results
            ensemble_config: Configuration for ensemble
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels

        Returns:
            ExperimentResult for ensemble
        """
        self.logger.info("Creating ensemble from experiment results")

        # Filter successful experiments
        successful_results = [
            r for r in results if r.error_message is None and r.model_path is not None
        ]

        if len(successful_results) < 2:
            raise ValueError("Need at least 2 successful models for ensemble")

        # Load trained models
        base_models = []
        for result in successful_results:
            model = joblib.load(result.model_path)
            base_models.append((result.model_type, model))

        # Update ensemble config
        ensemble_config.model_type = ModelType.ENSEMBLE
        ensemble_config.ensemble.base_models = [r.model_type for r in successful_results]

        # Create and train ensemble
        ensemble_model = EnsembleModel(ensemble_config)
        ensemble_model.set_base_models(base_models)

        # If using stacking, train meta-learner
        if ensemble_config.ensemble.method == "stacking":
            ensemble_model.train(X_train, y_train, X_test, y_test)

        # Evaluate ensemble
        ensemble_result = ExperimentResult(
            config_name="ensemble_from_results", model_type="ensemble", metrics={}
        )

        metrics = ensemble_model.evaluate(X_test, y_test)
        ensemble_result.metrics = metrics.to_dict()
        ensemble_result.predictions = ensemble_model.predict(X_test)

        # Save ensemble
        model_path = self._save_model(ensemble_model, ensemble_config, ensemble_result.metrics)
        ensemble_result.model_path = model_path

        self.logger.info(
            f"Ensemble created. F1-Score: {ensemble_result.metrics.get('f1_score', 0):.4f}"
        )

        return ensemble_result

    def compare_experiments(
        self, results: list[ExperimentResult], output_path: str | None = None
    ) -> pd.DataFrame:
        """
        Compare experiment results

        Args:
            results: List of experiment results
            output_path: Path to save comparison

        Returns:
            DataFrame with comparison
        """
        # Create comparison dataframe
        comparison_data = []

        for result in results:
            row = {
                "experiment": result.config_name,
                "model_type": result.model_type,
                "training_time": result.training_time,
                "tuning_time": result.tuning_time,
                **result.metrics,
            }

            if result.error_message:
                row["status"] = "failed"
                row["error"] = result.error_message
            else:
                row["status"] = "success"

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Sort by F1 score
        if "f1_score" in df.columns:
            df = df.sort_values("f1_score", ascending=False)

        # Save if requested
        if output_path:
            df.to_csv(output_path, index=False)

        return df

    def generate_report(self, suite: ExperimentSuite, results: list[ExperimentResult]) -> str:
        """
        Generate comprehensive HTML report

        Args:
            suite: Experiment suite
            results: List of results

        Returns:
            Path to generated report
        """
        self.logger.info("Generating experiment report")

        # Create comparison dataframe
        comparison_df = self.compare_experiments(results)

        # Generate plots
        plots = self._generate_plots(results, suite.output_path)

        # Create HTML report
        report_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Experiment Report: {{ suite_name }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric { font-weight: bold; }
                .best { background-color: #d4f4dd; }
                .failed { background-color: #f4d4d4; }
                .plot { margin: 20px 0; }
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>Experiment Report: {{ suite_name }}</h1>
            <p>Generated on: {{ timestamp }}</p>
            <p>Total experiments: {{ n_experiments }}</p>
            <p>Successful: {{ n_successful }}</p>
            <p>Failed: {{ n_failed }}</p>

            <h2>Summary Statistics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Best</th>
                    <th>Mean</th>
                    <th>Std</th>
                    <th>Best Model</th>
                </tr>
                {% for metric, stats in summary_stats.items() %}
                <tr>
                    <td class="metric">{{ metric }}</td>
                    <td>{{ "%.4f"|format(stats.best) }}</td>
                    <td>{{ "%.4f"|format(stats.mean) }}</td>
                    <td>{{ "%.4f"|format(stats.std) }}</td>
                    <td>{{ stats.best_model }}</td>
                </tr>
                {% endfor %}
            </table>

            <h2>Detailed Results</h2>
            {{ comparison_table }}

            <h2>Visualizations</h2>
            {% for plot_name, plot_html in plots.items() %}
            <div class="plot">
                <h3>{{ plot_name }}</h3>
                {{ plot_html|safe }}
            </div>
            {% endfor %}

            <h2>Configuration Details</h2>
            {% for name, config in experiments %}
            <details>
                <summary>{{ name }}</summary>
                <pre>{{ config }}</pre>
            </details>
            {% endfor %}
        </body>
        </html>
        """

        # Calculate summary statistics
        summary_stats = {}
        for metric in ["accuracy", "precision", "recall", "f1_score", "auc_roc"]:
            if metric in comparison_df.columns:
                valid_values = comparison_df[comparison_df[metric].notna()][metric]
                if len(valid_values) > 0:
                    best_idx = valid_values.idxmax()
                    summary_stats[metric] = {
                        "best": valid_values.max(),
                        "mean": valid_values.mean(),
                        "std": valid_values.std(),
                        "best_model": comparison_df.loc[best_idx, "experiment"],
                    }

        # Render template
        template = Template(report_template)
        report_html = template.render(
            suite_name=suite.name,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            n_experiments=len(results),
            n_successful=len([r for r in results if r.error_message is None]),
            n_failed=len([r for r in results if r.error_message is not None]),
            summary_stats=summary_stats,
            comparison_table=comparison_df.to_html(classes="comparison-table", index=False),
            plots=plots,
            experiments=[
                (name, yaml.dump(config.__dict__, default_flow_style=False))
                for name, config in suite.experiments
            ],
        )

        # Save report
        report_path = suite.output_path / "experiment_report.html"
        with open(report_path, "w") as f:
            f.write(report_html)

        self.logger.info(f"Report saved to: {report_path}")

        return str(report_path)

    def _get_param_space(self, config: ExperimentConfig) -> dict[str, Any]:
        """Extract parameter space from config"""
        # This is a simplified version - in practice, you'd extract from config
        param_spaces = {
            ModelType.RANDOM_FOREST: {
                "n_estimators": {"type": "int", "low": 50, "high": 500},
                "max_depth": {"type": "int", "low": 5, "high": 50},
                "min_samples_split": {"type": "int", "low": 2, "high": 20},
                "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
            },
            ModelType.LSTM: {
                "units": {"type": "categorical", "choices": [32, 64, 128, 256]},
                "dropout": {"type": "float", "low": 0.1, "high": 0.5},
                "recurrent_dropout": {"type": "float", "low": 0.1, "high": 0.5},
                "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            },
            ModelType.CNN: {
                "filters": {"type": "categorical", "choices": [[32, 64], [64, 128], [32, 64, 128]]},
                "kernel_size": {"type": "categorical", "choices": [3, 5, 7]},
                "dropout": {"type": "float", "low": 0.1, "high": 0.5},
            },
        }

        return param_spaces.get(config.model_type, {})

    def _create_model_instance(
        self, model_class: Any, config: ExperimentConfig, params: dict[str, Any] | None = None
    ) -> Any:
        """Create model instance with parameters"""
        # Update config with tuned parameters if available
        if params and hasattr(config, config.model_type.value):
            model_config = getattr(config, config.model_type.value)
            for key, value in params.items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)

        return model_class(config)

    def _save_model(self, model: Any, config: ExperimentConfig, metrics: dict[str, float]) -> str:
        """Save trained model"""
        output_dir = Path(config.output_dir or "./models")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{config.name}_{timestamp}.pkl"
        filepath = output_dir / filename

        # Save model and metadata
        save_data = {"model": model, "config": config, "metrics": metrics, "timestamp": timestamp}

        joblib.dump(save_data, filepath)

        return str(filepath)

    def _save_suite_results(self, suite: ExperimentSuite, results: list[ExperimentResult]) -> None:
        """Save suite results to file"""
        results_data = {
            "suite_name": suite.name,
            "timestamp": datetime.now().isoformat(),
            "results": [r.to_dict() for r in results],
        }

        results_path = suite.output_path / "results.json"
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2)

    def _generate_plots(self, results: list[ExperimentResult], output_path: Path) -> dict[str, str]:
        """Generate interactive plots"""
        plots = {}

        # Filter successful results
        successful_results = [r for r in results if r.error_message is None]

        if not successful_results:
            return plots

        # 1. Model performance comparison
        fig = go.Figure()

        metrics = ["accuracy", "precision", "recall", "f1_score"]
        for metric in metrics:
            values = [r.metrics.get(metric, 0) for r in successful_results]
            names = [r.config_name for r in successful_results]

            fig.add_trace(go.Bar(name=metric.replace("_", " ").title(), x=names, y=values))

        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode="group",
        )

        plots["Performance Comparison"] = fig.to_html(include_plotlyjs=False)

        # 2. Training time comparison
        fig = go.Figure()

        training_times = [r.training_time for r in successful_results]
        tuning_times = [r.tuning_time for r in successful_results]
        names = [r.config_name for r in successful_results]

        fig.add_trace(go.Bar(name="Training Time", x=names, y=training_times))
        fig.add_trace(go.Bar(name="Tuning Time", x=names, y=tuning_times))

        fig.update_layout(
            title="Time Comparison",
            xaxis_title="Model",
            yaxis_title="Time (seconds)",
            barmode="stack",
        )

        plots["Time Comparison"] = fig.to_html(include_plotlyjs=False)

        # 3. ROC curves if available
        if any(r.metrics.get("auc_roc") for r in successful_results):
            # This would require storing the actual ROC curve data
            # For now, we'll create a simple AUC comparison
            fig = go.Figure()

            auc_values = [r.metrics.get("auc_roc", 0) for r in successful_results]
            names = [r.config_name for r in successful_results]

            fig.add_trace(
                go.Bar(
                    x=names,
                    y=auc_values,
                    text=[f"{v:.3f}" for v in auc_values],
                    textposition="auto",
                )
            )

            fig.update_layout(
                title="AUC-ROC Comparison",
                xaxis_title="Model",
                yaxis_title="AUC-ROC",
                yaxis_range=[0, 1],
            )

            plots["AUC-ROC Comparison"] = fig.to_html(include_plotlyjs=False)

        return plots


def create_experiment_suite_from_yaml(yaml_path: str) -> ExperimentSuite:
    """
    Create experiment suite from YAML configuration

    Args:
        yaml_path: Path to YAML file

    Returns:
        ExperimentSuite object
    """
    with open(yaml_path) as f:
        suite_config = yaml.safe_load(f)

    # Load experiments
    experiments = []
    parser = ConfigParser()

    for exp_config in suite_config["experiments"]:
        if "config_file" in exp_config:
            # Load from file
            config = parser.load_config(exp_config["config_file"])
            name = exp_config.get("name", config.name)
        else:
            # Inline configuration
            config = parser.parse_config(exp_config["config"])
            name = exp_config["name"]

        experiments.append((name, config))

    # Create suite
    suite = ExperimentSuite(
        name=suite_config["name"],
        experiments=experiments,
        output_dir=suite_config.get("output_dir", "./experiment_results"),
        parallel=suite_config.get("parallel", True),
        max_workers=suite_config.get("max_workers"),
        tune_hyperparameters=suite_config.get("tune_hyperparameters", True),
        tuning_method=suite_config.get("tuning_method", "bayesian"),
        tuning_trials=suite_config.get("tuning_trials", 50),
    )

    return suite


# Example usage functions
def run_quick_experiment(model_type: str = "random_forest"):
    """Run a quick experiment with default settings"""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Generate sample data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create config
    parser = ConfigParser()
    config = parser.create_default_config()
    config.name = f"quick_{model_type}_experiment"
    config.model_type = ModelType(model_type)

    # Run experiment
    runner = ExperimentRunner()
    result = runner.run_single_experiment(
        config, X_train, y_train, X_test, y_test, tune_hyperparameters=False
    )

    print("Experiment completed!")
    print(f"F1-Score: {result.metrics.get('f1_score', 0):.4f}")
    print(f"Training time: {result.training_time:.2f} seconds")

    return result


def run_model_comparison(output_dir: str = "./model_comparison"):
    """Run comparison of all available models"""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Generate sample data
    X, y = make_classification(
        n_samples=2000, n_features=30, n_informative=20, n_redundant=10, random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create experiments for each model type
    experiments = []
    parser = ConfigParser()
    config_manager = ConfigurationManager()

    for model_type in ModelType:
        if model_type == ModelType.ENSEMBLE:
            continue  # Skip ensemble for now

        config = parser.create_default_config()
        config.name = f"{model_type.value}_comparison"
        config.model_type = model_type
        config = config_manager.optimize_config_for_hardware(config)

        experiments.append((config.name, config))

    # Create suite
    suite = ExperimentSuite(
        name="model_comparison",
        experiments=experiments,
        output_dir=output_dir,
        parallel=True,
        tune_hyperparameters=True,
        tuning_trials=20,  # Reduced for quick comparison
    )

    # Run experiments
    runner = ExperimentRunner()
    results = runner.run_experiment_suite(suite, X_train, y_train, X_test, y_test)

    # Create ensemble from best models
    if len(results) > 2:
        ensemble_config = parser.create_default_config()
        ensemble_config.name = "ensemble_from_comparison"

        ensemble_result = runner.create_ensemble_from_results(
            results, ensemble_config, X_train, y_train, X_test, y_test
        )

        print(f"\nEnsemble F1-Score: {ensemble_result.metrics.get('f1_score', 0):.4f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("--quick", action="store_true", help="Run quick experiment")
    parser.add_argument("--compare", action="store_true", help="Run model comparison")
    parser.add_argument("--suite", type=str, help="Path to experiment suite YAML")
    parser.add_argument(
        "--model", type=str, default="random_forest", help="Model type for quick experiment"
    )
    parser.add_argument(
        "--output", type=str, default="./experiment_results", help="Output directory"
    )

    args = parser.parse_args()

    if args.quick:
        run_quick_experiment(args.model)
    elif args.compare:
        run_model_comparison(args.output)
    elif args.suite:
        # Load data (this would be replaced with actual data loading)
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split

        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Run suite
        suite = create_experiment_suite_from_yaml(args.suite)
        runner = ExperimentRunner()
        results = runner.run_experiment_suite(suite, X_train, y_train, X_test, y_test)
    else:
        parser.print_help()
