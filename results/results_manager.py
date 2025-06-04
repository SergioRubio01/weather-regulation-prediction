"""
Advanced Results Management System for Weather Regulation Prediction

This module provides comprehensive result storage, retrieval, analysis, and comparison
capabilities for machine learning experiments. It includes:
- Standardized result storage in multiple formats
- Result versioning and metadata tracking
- Performance metric aggregation and analysis
- Model comparison utilities
- Result export and sharing capabilities
"""

import hashlib
import json
import logging
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import ExperimentConfig

warnings.filterwarnings("ignore")


@dataclass
class ModelResult:
    """Container for single model results"""

    model_name: str
    model_type: str
    timestamp: datetime
    config: dict[str, Any]

    # Training metrics
    training_time: float
    training_history: dict[str, list[float]] | None = None

    # Validation metrics
    val_accuracy: float | None = None
    val_loss: float | None = None
    val_metrics: dict[str, float] = field(default_factory=dict)

    # Test metrics
    test_accuracy: float | None = None
    test_precision: float | None = None
    test_recall: float | None = None
    test_f1: float | None = None
    test_auc: float | None = None
    test_metrics: dict[str, float] = field(default_factory=dict)

    # Additional data
    confusion_matrix: np.ndarray | None = None
    feature_importance: pd.DataFrame | None = None
    predictions: np.ndarray | None = None
    prediction_probabilities: np.ndarray | None = None

    # Model artifacts
    model_path: str | None = None
    checkpoint_paths: list[str] = field(default_factory=list)

    # Metadata
    dataset_info: dict[str, Any] = field(default_factory=dict)
    hardware_info: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        if self.confusion_matrix is not None:
            data["confusion_matrix"] = self.confusion_matrix.tolist()
        if self.predictions is not None:
            data["predictions"] = self.predictions.tolist()
        if self.prediction_probabilities is not None:
            data["prediction_probabilities"] = self.prediction_probabilities.tolist()
        if self.feature_importance is not None:
            data["feature_importance"] = self.feature_importance.to_dict()
        # Convert datetime
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelResult":
        """Create from dictionary"""
        # Convert lists back to numpy arrays
        if "confusion_matrix" in data and data["confusion_matrix"] is not None:
            data["confusion_matrix"] = np.array(data["confusion_matrix"])
        if "predictions" in data and data["predictions"] is not None:
            data["predictions"] = np.array(data["predictions"])
        if "prediction_probabilities" in data and data["prediction_probabilities"] is not None:
            data["prediction_probabilities"] = np.array(data["prediction_probabilities"])
        if "feature_importance" in data and data["feature_importance"] is not None:
            data["feature_importance"] = pd.DataFrame.from_dict(data["feature_importance"])
        # Convert timestamp
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class ExperimentResult:
    """Container for complete experiment results"""

    experiment_id: str
    experiment_name: str
    timestamp: datetime
    config: ExperimentConfig

    # Model results
    model_results: dict[str, ModelResult] = field(default_factory=dict)

    # Aggregate metrics
    best_model: str | None = None
    best_accuracy: float | None = None
    average_accuracy: float | None = None

    # Comparison data
    comparison_metrics: pd.DataFrame | None = None
    ranking: pd.DataFrame | None = None

    # Metadata
    total_training_time: float = 0.0
    dataset_hash: str | None = None
    git_commit: str | None = None
    environment_info: dict[str, Any] = field(default_factory=dict)

    def add_model_result(self, result: ModelResult):
        """Add a model result to the experiment"""
        self.model_results[result.model_name] = result
        self.total_training_time += result.training_time
        self._update_aggregate_metrics()

    def _update_aggregate_metrics(self):
        """Update aggregate metrics based on model results"""
        if not self.model_results:
            return

        accuracies = []
        for name, result in self.model_results.items():
            if result.test_accuracy is not None:
                accuracies.append((name, result.test_accuracy))

        if accuracies:
            accuracies.sort(key=lambda x: x[1], reverse=True)
            self.best_model = accuracies[0][0]
            self.best_accuracy = accuracies[0][1]
            self.average_accuracy = np.mean([acc[1] for acc in accuracies])

            # Create comparison dataframe
            metrics_data = []
            for name, result in self.model_results.items():
                metrics_data.append(
                    {
                        "model": name,
                        "accuracy": result.test_accuracy,
                        "precision": result.test_precision,
                        "recall": result.test_recall,
                        "f1_score": result.test_f1,
                        "auc": result.test_auc,
                        "training_time": result.training_time,
                    }
                )
            self.comparison_metrics = pd.DataFrame(metrics_data)

            # Create ranking
            self.ranking = self.comparison_metrics.sort_values(
                "accuracy", ascending=False
            ).reset_index(drop=True)
            self.ranking["rank"] = range(1, len(self.ranking) + 1)


class ResultsManager:
    """Main results management class"""

    def __init__(self, base_path: str = "./results"):
        """Initialize results manager"""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.experiments_path = self.base_path / "experiments"
        self.models_path = self.base_path / "models"
        self.reports_path = self.base_path / "reports"
        self.visualizations_path = self.base_path / "visualizations"

        for path in [
            self.experiments_path,
            self.models_path,
            self.reports_path,
            self.visualizations_path,
        ]:
            path.mkdir(exist_ok=True)

        self.logger = self._setup_logger()

        # Results index
        self.index_path = self.base_path / "results_index.json"
        self.index = self._load_index()

    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _load_index(self) -> dict[str, Any]:
        """Load results index"""
        if self.index_path.exists():
            with open(self.index_path) as f:
                return json.load(f)
        return {"experiments": {}, "models": {}}

    def _save_index(self):
        """Save results index"""
        with open(self.index_path, "w") as f:
            json.dump(self.index, f, indent=2)

    def generate_experiment_id(self) -> str:
        """Generate unique experiment ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(
            str(datetime.now()).encode(), usedforsecurity=False
        ).hexdigest()[:6]
        return f"exp_{timestamp}_{random_suffix}"

    def save_model_result(self, result: ModelResult, experiment_id: str | None = None) -> str:
        """Save individual model result"""
        # Generate filename
        model_id = f"{result.model_name}_{result.timestamp.strftime('%Y%m%d_%H%M%S')}"

        # Save model result
        result_path = self.models_path / f"{model_id}.json"
        with open(result_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Update index
        self.index["models"][model_id] = {
            "path": str(result_path),
            "model_name": result.model_name,
            "model_type": result.model_type,
            "timestamp": result.timestamp.isoformat(),
            "test_accuracy": result.test_accuracy,
            "experiment_id": experiment_id,
        }
        self._save_index()

        self.logger.info(f"Saved model result: {model_id}")
        return model_id

    def save_experiment_result(self, experiment: ExperimentResult) -> str:
        """Save complete experiment result"""
        # Save experiment
        exp_path = self.experiments_path / f"{experiment.experiment_id}.json"

        # Convert to dict for serialization
        exp_data = {
            "experiment_id": experiment.experiment_id,
            "experiment_name": experiment.experiment_name,
            "timestamp": experiment.timestamp.isoformat(),
            "config": (
                experiment.config.to_dict()
                if hasattr(experiment.config, "to_dict")
                else experiment.config
            ),
            "model_results": {
                name: result.to_dict() for name, result in experiment.model_results.items()
            },
            "best_model": experiment.best_model,
            "best_accuracy": experiment.best_accuracy,
            "average_accuracy": experiment.average_accuracy,
            "total_training_time": experiment.total_training_time,
            "dataset_hash": experiment.dataset_hash,
            "git_commit": experiment.git_commit,
            "environment_info": experiment.environment_info,
        }

        if experiment.comparison_metrics is not None:
            exp_data["comparison_metrics"] = experiment.comparison_metrics.to_dict()
        if experiment.ranking is not None:
            exp_data["ranking"] = experiment.ranking.to_dict()

        with open(exp_path, "w") as f:
            json.dump(exp_data, f, indent=2)

        # Save individual model results
        for model_name, model_result in experiment.model_results.items():
            self.save_model_result(model_result, experiment.experiment_id)

        # Update index
        self.index["experiments"][experiment.experiment_id] = {
            "path": str(exp_path),
            "name": experiment.experiment_name,
            "timestamp": experiment.timestamp.isoformat(),
            "best_model": experiment.best_model,
            "best_accuracy": experiment.best_accuracy,
            "n_models": len(experiment.model_results),
        }
        self._save_index()

        self.logger.info(f"Saved experiment: {experiment.experiment_id}")
        return experiment.experiment_id

    def load_model_result(self, model_id: str) -> ModelResult | None:
        """Load individual model result"""
        if model_id not in self.index["models"]:
            self.logger.error(f"Model {model_id} not found in index")
            return None

        model_path = Path(self.index["models"][model_id]["path"])
        with open(model_path) as f:
            data = json.load(f)

        return ModelResult.from_dict(data)

    def load_experiment_result(self, experiment_id: str) -> ExperimentResult | None:
        """Load complete experiment result"""
        if experiment_id not in self.index["experiments"]:
            self.logger.error(f"Experiment {experiment_id} not found in index")
            return None

        exp_path = Path(self.index["experiments"][experiment_id]["path"])
        with open(exp_path) as f:
            data = json.load(f)

        # Reconstruct experiment
        experiment = ExperimentResult(
            experiment_id=data["experiment_id"],
            experiment_name=data["experiment_name"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            config=data["config"],  # Would need proper deserialization
            best_model=data.get("best_model"),
            best_accuracy=data.get("best_accuracy"),
            average_accuracy=data.get("average_accuracy"),
            total_training_time=data.get("total_training_time", 0),
            dataset_hash=data.get("dataset_hash"),
            git_commit=data.get("git_commit"),
            environment_info=data.get("environment_info", {}),
        )

        # Reconstruct model results
        for name, model_data in data.get("model_results", {}).items():
            experiment.model_results[name] = ModelResult.from_dict(model_data)

        # Reconstruct dataframes
        if "comparison_metrics" in data:
            experiment.comparison_metrics = pd.DataFrame.from_dict(data["comparison_metrics"])
        if "ranking" in data:
            experiment.ranking = pd.DataFrame.from_dict(data["ranking"])

        return experiment

    def list_experiments(self, filter_by: dict[str, Any] | None = None) -> pd.DataFrame:
        """List all experiments with optional filtering"""
        experiments = []

        for exp_id, exp_info in self.index["experiments"].items():
            exp_data = {
                "experiment_id": exp_id,
                "name": exp_info["name"],
                "timestamp": exp_info["timestamp"],
                "best_model": exp_info.get("best_model"),
                "best_accuracy": exp_info.get("best_accuracy"),
                "n_models": exp_info.get("n_models", 0),
            }

            # Apply filters
            if filter_by:
                match = True
                for key, value in filter_by.items():
                    if key in exp_data and exp_data[key] != value:
                        match = False
                        break
                if not match:
                    continue

            experiments.append(exp_data)

        df = pd.DataFrame(experiments)
        if len(df) > 0:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp", ascending=False)

        return df

    def list_models(self, experiment_id: str | None = None) -> pd.DataFrame:
        """List all models, optionally filtered by experiment"""
        models = []

        for model_id, model_info in self.index["models"].items():
            if experiment_id and model_info.get("experiment_id") != experiment_id:
                continue

            model_data = {
                "model_id": model_id,
                "model_name": model_info["model_name"],
                "model_type": model_info["model_type"],
                "timestamp": model_info["timestamp"],
                "test_accuracy": model_info.get("test_accuracy"),
                "experiment_id": model_info.get("experiment_id"),
            }
            models.append(model_data)

        df = pd.DataFrame(models)
        if len(df) > 0:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp", ascending=False)

        return df

    def compare_experiments(
        self, experiment_ids: list[str], metric: str = "test_accuracy"
    ) -> pd.DataFrame:
        """Compare multiple experiments"""
        comparison_data = []

        for exp_id in experiment_ids:
            experiment = self.load_experiment_result(exp_id)
            if experiment:
                for model_name, model_result in experiment.model_results.items():
                    row = {
                        "experiment_id": exp_id,
                        "experiment_name": experiment.experiment_name,
                        "model": model_name,
                        "metric_value": getattr(model_result, metric, None),
                    }

                    # Add all test metrics
                    for metric_name in [
                        "test_accuracy",
                        "test_precision",
                        "test_recall",
                        "test_f1",
                        "test_auc",
                    ]:
                        row[metric_name] = getattr(model_result, metric_name, None)

                    row["training_time"] = model_result.training_time
                    comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        # Pivot for easier comparison
        if len(comparison_df) > 0:
            pivot_df = comparison_df.pivot_table(
                index="model", columns="experiment_name", values=metric, aggfunc="mean"
            )
            return pivot_df

        return comparison_df

    def get_best_models(self, n: int = 10, metric: str = "test_accuracy") -> pd.DataFrame:
        """Get top N best models across all experiments"""
        all_models = []

        for model_id, model_info in self.index["models"].items():
            model_result = self.load_model_result(model_id)
            if model_result:
                model_data = {
                    "model_id": model_id,
                    "model_name": model_result.model_name,
                    "model_type": model_result.model_type,
                    "experiment_id": model_info.get("experiment_id"),
                    "timestamp": model_result.timestamp,
                    metric: getattr(model_result, metric, None),
                    "test_accuracy": model_result.test_accuracy,
                    "test_f1": model_result.test_f1,
                    "training_time": model_result.training_time,
                }
                all_models.append(model_data)

        df = pd.DataFrame(all_models)
        if len(df) > 0:
            df = df.dropna(subset=[metric])
            df = df.sort_values(metric, ascending=False).head(n)

        return df

    def export_results(
        self, experiment_id: str, format: str = "csv", output_path: str | None = None
    ) -> str:
        """Export experiment results in various formats"""
        experiment = self.load_experiment_result(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        if output_path is None:
            output_path = self.reports_path / f"{experiment_id}_results.{format}"
        else:
            output_path = Path(output_path)

        if format == "csv":
            # Export comparison metrics
            if experiment.comparison_metrics is not None:
                experiment.comparison_metrics.to_csv(output_path, index=False)

        elif format == "excel":
            # Export to Excel with multiple sheets
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                # Summary sheet
                summary_data = {
                    "Experiment ID": experiment.experiment_id,
                    "Name": experiment.experiment_name,
                    "Timestamp": experiment.timestamp,
                    "Best Model": experiment.best_model,
                    "Best Accuracy": experiment.best_accuracy,
                    "Average Accuracy": experiment.average_accuracy,
                    "Total Training Time": experiment.total_training_time,
                }
                pd.DataFrame([summary_data]).to_excel(writer, sheet_name="Summary", index=False)

                # Model comparison
                if experiment.comparison_metrics is not None:
                    experiment.comparison_metrics.to_excel(
                        writer, sheet_name="Model Comparison", index=False
                    )

                # Ranking
                if experiment.ranking is not None:
                    experiment.ranking.to_excel(writer, sheet_name="Ranking", index=False)

        elif format == "json":
            # Export as JSON
            exp_data = {
                "experiment": {
                    "id": experiment.experiment_id,
                    "name": experiment.experiment_name,
                    "timestamp": experiment.timestamp.isoformat(),
                    "best_model": experiment.best_model,
                    "best_accuracy": experiment.best_accuracy,
                },
                "results": {},
            }

            for name, result in experiment.model_results.items():
                exp_data["results"][name] = {
                    "accuracy": result.test_accuracy,
                    "precision": result.test_precision,
                    "recall": result.test_recall,
                    "f1_score": result.test_f1,
                    "auc": result.test_auc,
                    "training_time": result.training_time,
                }

            with open(output_path, "w") as f:
                json.dump(exp_data, f, indent=2)

        elif format == "latex":
            # Export as LaTeX table
            if experiment.comparison_metrics is not None:
                latex_content = experiment.comparison_metrics.to_latex(
                    index=False,
                    float_format="%.4f",
                    caption=f"Model comparison for {experiment.experiment_name}",
                    label=f"tab:{experiment.experiment_id}",
                )
                with open(output_path, "w") as f:
                    f.write(latex_content)

        else:
            raise ValueError(f"Unsupported format: {format}")

        self.logger.info(f"Exported results to {output_path}")
        return str(output_path)

    def cleanup_old_results(self, days: int = 30):
        """Remove results older than specified days"""
        cutoff_date = datetime.now() - pd.Timedelta(days=days)
        removed_experiments = []
        removed_models = []

        # Check experiments
        for exp_id, exp_info in list(self.index["experiments"].items()):
            exp_date = datetime.fromisoformat(exp_info["timestamp"])
            if exp_date < cutoff_date:
                # Remove experiment files
                exp_path = Path(exp_info["path"])
                if exp_path.exists():
                    exp_path.unlink()
                removed_experiments.append(exp_id)
                del self.index["experiments"][exp_id]

        # Check models
        for model_id, model_info in list(self.index["models"].items()):
            model_date = datetime.fromisoformat(model_info["timestamp"])
            if model_date < cutoff_date:
                # Remove model files
                model_path = Path(model_info["path"])
                if model_path.exists():
                    model_path.unlink()
                removed_models.append(model_id)
                del self.index["models"][model_id]

        self._save_index()

        self.logger.info(
            f"Cleaned up {len(removed_experiments)} experiments and "
            f"{len(removed_models)} models older than {days} days"
        )

        return removed_experiments, removed_models


# Utility functions
def create_model_result(
    model_name: str,
    model_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    training_time: float = 0.0,
    config: dict[str, Any] | None = None,
) -> ModelResult:
    """Create ModelResult from predictions"""

    # Calculate metrics
    result = ModelResult(
        model_name=model_name,
        model_type=model_type,
        timestamp=datetime.now(),
        config=config or {},
        training_time=training_time,
        predictions=y_pred,
        prediction_probabilities=y_proba,
    )

    # Classification metrics
    result.test_accuracy = accuracy_score(y_true, y_pred)
    result.test_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    result.test_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    result.test_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    if y_proba is not None and len(np.unique(y_true)) == 2:
        # Binary classification AUC
        result.test_auc = roc_auc_score(y_true, y_proba[:, 1])

    # Confusion matrix
    result.confusion_matrix = confusion_matrix(y_true, y_pred)

    # Store all metrics
    result.test_metrics = {
        "accuracy": result.test_accuracy,
        "precision": result.test_precision,
        "recall": result.test_recall,
        "f1_score": result.test_f1,
        "auc": result.test_auc,
    }

    return result


def aggregate_cross_validation_results(
    cv_results: list[ModelResult], model_name: str, model_type: str
) -> ModelResult:
    """Aggregate results from cross-validation folds"""

    # Aggregate metrics
    aggregated = ModelResult(
        model_name=f"{model_name}_cv",
        model_type=model_type,
        timestamp=datetime.now(),
        config={},
        training_time=sum(r.training_time for r in cv_results),
    )

    # Average test metrics
    metrics_to_average = ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_auc"]

    for metric in metrics_to_average:
        values = [getattr(r, metric) for r in cv_results if getattr(r, metric) is not None]
        if values:
            setattr(aggregated, metric, np.mean(values))
            aggregated.test_metrics[f"{metric}_std"] = np.std(values)

    # Combine confusion matrices
    cms = [r.confusion_matrix for r in cv_results if r.confusion_matrix is not None]
    if cms:
        aggregated.confusion_matrix = np.sum(cms, axis=0)

    return aggregated
