"""
Hyperparameter Tuning Module for Weather Regulation Prediction Models

This module provides comprehensive hyperparameter optimization including:
- Grid Search
- Random Search
- Bayesian Optimization (using Optuna)
- Hyperband
- Population-based training
- Multi-objective optimization
"""

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import keras_tuner as kt
import numpy as np
import optuna
import ray
import tensorflow as tf
from optuna.pruners import MedianPruner
from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler
from ray import tune
from ray.tune.schedulers import HyperBandScheduler, PopulationBasedTraining
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

from config_parser import ConfigParser


@dataclass
class TuningResult:
    """Container for hyperparameter tuning results"""

    best_params: dict[str, Any]
    best_score: float
    all_trials: list[dict[str, Any]]
    tuning_history: dict[str, list[float]]
    elapsed_time: float
    n_trials: int
    search_method: str
    model_type: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": self.n_trials,
            "elapsed_time": self.elapsed_time,
            "search_method": self.search_method,
            "model_type": self.model_type,
            "tuning_history": self.tuning_history,
        }

    def save(self, filepath: str) -> None:
        """Save results to file"""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "TuningResult":
        """Load results from file"""
        with open(filepath) as f:
            data = json.load(f)
        return cls(**data)


class HyperparameterTuner:
    """Base class for hyperparameter tuning"""

    def __init__(
        self,
        model_class: Any,
        param_space: dict[str, Any],
        scoring: str = "accuracy",
        cv: int = 5,
        n_jobs: int = -1,
    ):
        self.model_class = model_class
        self.param_space = param_space
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.logger = self._setup_logger()

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

    def tune(self, X: np.ndarray, y: np.ndarray, **kwargs) -> TuningResult:
        """Perform hyperparameter tuning"""
        raise NotImplementedError("Subclasses must implement tune method")


class GridSearchTuner(HyperparameterTuner):
    """Grid search hyperparameter tuning"""

    def tune(self, X: np.ndarray, y: np.ndarray, **kwargs) -> TuningResult:
        """Perform grid search"""
        self.logger.info("Starting Grid Search")
        start_time = datetime.now()

        # Create base model
        model = self.model_class()

        # Perform grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=self.param_space,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
            verbose=2,
            return_train_score=True,
        )

        grid_search.fit(X, y)

        # Extract results
        elapsed_time = (datetime.now() - start_time).total_seconds()

        # Process all trials
        all_trials = []
        for i in range(len(grid_search.cv_results_["params"])):
            trial = {
                "params": grid_search.cv_results_["params"][i],
                "mean_test_score": grid_search.cv_results_["mean_test_score"][i],
                "std_test_score": grid_search.cv_results_["std_test_score"][i],
                "mean_train_score": grid_search.cv_results_["mean_train_score"][i],
                "rank": grid_search.cv_results_["rank_test_score"][i],
            }
            all_trials.append(trial)

        # Create tuning history
        tuning_history = {
            "scores": grid_search.cv_results_["mean_test_score"].tolist(),
            "train_scores": grid_search.cv_results_["mean_train_score"].tolist(),
        }

        result = TuningResult(
            best_params=grid_search.best_params_,
            best_score=grid_search.best_score_,
            all_trials=all_trials,
            tuning_history=tuning_history,
            elapsed_time=elapsed_time,
            n_trials=len(all_trials),
            search_method="grid_search",
            model_type=self.model_class.__name__,
        )

        self.logger.info(f"Grid Search completed. Best score: {result.best_score:.4f}")

        return result


class RandomSearchTuner(HyperparameterTuner):
    """Random search hyperparameter tuning"""

    def __init__(
        self, model_class: Any, param_distributions: dict[str, Any], n_iter: int = 100, **kwargs
    ):
        super().__init__(model_class, param_distributions, **kwargs)
        self.n_iter = n_iter

    def tune(self, X: np.ndarray, y: np.ndarray, **kwargs) -> TuningResult:
        """Perform random search"""
        self.logger.info(f"Starting Random Search with {self.n_iter} iterations")
        start_time = datetime.now()

        # Create base model
        model = self.model_class()

        # Perform random search
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=self.param_space,
            n_iter=self.n_iter,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
            verbose=2,
            return_train_score=True,
            random_state=42,
        )

        random_search.fit(X, y)

        # Extract results
        elapsed_time = (datetime.now() - start_time).total_seconds()

        # Process all trials
        all_trials = []
        for i in range(len(random_search.cv_results_["params"])):
            trial = {
                "params": random_search.cv_results_["params"][i],
                "mean_test_score": random_search.cv_results_["mean_test_score"][i],
                "std_test_score": random_search.cv_results_["std_test_score"][i],
                "mean_train_score": random_search.cv_results_["mean_train_score"][i],
                "rank": random_search.cv_results_["rank_test_score"][i],
            }
            all_trials.append(trial)

        # Create tuning history
        tuning_history = {
            "scores": random_search.cv_results_["mean_test_score"].tolist(),
            "train_scores": random_search.cv_results_["mean_train_score"].tolist(),
        }

        result = TuningResult(
            best_params=random_search.best_params_,
            best_score=random_search.best_score_,
            all_trials=all_trials,
            tuning_history=tuning_history,
            elapsed_time=elapsed_time,
            n_trials=len(all_trials),
            search_method="random_search",
            model_type=self.model_class.__name__,
        )

        self.logger.info(f"Random Search completed. Best score: {result.best_score:.4f}")

        return result


class BayesianOptimizationTuner(HyperparameterTuner):
    """Bayesian optimization using Optuna"""

    def __init__(
        self,
        model_class: Any,
        param_space: dict[str, Any],
        n_trials: int = 100,
        sampler: str = "tpe",
        **kwargs,
    ):
        super().__init__(model_class, param_space, **kwargs)
        self.n_trials = n_trials
        self.sampler = self._get_sampler(sampler)
        self.study = None

    def _get_sampler(self, sampler_name: str) -> optuna.samplers.BaseSampler:
        """Get Optuna sampler"""
        samplers = {
            "tpe": TPESampler(seed=42),
            "random": RandomSampler(seed=42),
            "cmaes": CmaEsSampler(seed=42),
        }
        return samplers.get(sampler_name, TPESampler(seed=42))

    def _create_objective(self, X: np.ndarray, y: np.ndarray) -> Callable:
        """Create objective function for Optuna"""

        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            params = {}
            for param_name, param_config in self.param_space.items():
                if param_config["type"] == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        step=param_config.get("step", 1),
                    )
                elif param_config["type"] == "float":
                    if param_config.get("log", False):
                        params[param_name] = trial.suggest_loguniform(
                            param_name, param_config["low"], param_config["high"]
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name, param_config["low"], param_config["high"]
                        )
                elif param_config["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config["choices"]
                    )

            # Create and evaluate model
            model = self.model_class(**params)
            scores = cross_val_score(
                model,
                X,
                y,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=1,  # Avoid nested parallelism
            )

            return scores.mean()

        return objective

    def tune(self, X: np.ndarray, y: np.ndarray, **kwargs) -> TuningResult:
        """Perform Bayesian optimization"""
        self.logger.info(f"Starting Bayesian Optimization with {self.n_trials} trials")
        start_time = datetime.now()

        # Create study
        self.study = optuna.create_study(
            direction="maximize",
            sampler=self.sampler,
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        )

        # Create objective
        objective = self._create_objective(X, y)

        # Optimize
        self.study.optimize(
            objective, n_trials=self.n_trials, n_jobs=self.n_jobs, show_progress_bar=True
        )

        # Extract results
        elapsed_time = (datetime.now() - start_time).total_seconds()

        # Process all trials
        all_trials = []
        for trial in self.study.trials:
            trial_dict = {
                "params": trial.params,
                "value": trial.value,
                "state": str(trial.state),
                "number": trial.number,
            }
            all_trials.append(trial_dict)

        # Create tuning history
        tuning_history = {
            "scores": [t.value for t in self.study.trials if t.value is not None],
            "best_scores": [self.study.best_value] * len(self.study.trials),
        }

        result = TuningResult(
            best_params=self.study.best_params,
            best_score=self.study.best_value,
            all_trials=all_trials,
            tuning_history=tuning_history,
            elapsed_time=elapsed_time,
            n_trials=len(self.study.trials),
            search_method="bayesian_optimization",
            model_type=self.model_class.__name__,
        )

        self.logger.info(f"Bayesian Optimization completed. Best score: {result.best_score:.4f}")

        return result

    def visualize_optimization(self, output_dir: str) -> None:
        """Generate Optuna visualization plots"""
        if self.study is None:
            raise ValueError("No study available. Run tune() first.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Import visualization modules
        from optuna.visualization import (
            plot_optimization_history,
            plot_parallel_coordinate,
            plot_param_importances,
        )

        # Generate plots
        plots = {
            "optimization_history": plot_optimization_history(self.study),
            "param_importances": plot_param_importances(self.study),
            "parallel_coordinate": plot_parallel_coordinate(self.study),
        }

        # Save plots
        for name, fig in plots.items():
            fig.write_html(str(output_path / f"{name}.html"))

        self.logger.info(f"Visualization plots saved to {output_dir}")


class KerasTuner(HyperparameterTuner):
    """Keras Tuner for neural network hyperparameter optimization"""

    def __init__(
        self,
        build_model_fn: Callable,
        param_space: dict[str, Any],
        tuner_type: str = "random",
        max_trials: int = 50,
        **kwargs,
    ):
        super().__init__(None, param_space, **kwargs)
        self.build_model_fn = build_model_fn
        self.tuner_type = tuner_type
        self.max_trials = max_trials
        self.tuner = None

    def _create_tuner(self, hp: kt.HyperParameters) -> kt.Tuner:
        """Create Keras Tuner instance"""
        project_dir = Path("./keras_tuner_projects")
        project_dir.mkdir(exist_ok=True)

        tuner_classes = {
            "random": kt.RandomSearch,
            "bayesian": kt.BayesianOptimization,
            "hyperband": kt.Hyperband,
        }

        tuner_class = tuner_classes.get(self.tuner_type, kt.RandomSearch)

        if self.tuner_type == "hyperband":
            return tuner_class(
                self.build_model_fn,
                objective=kt.Objective("val_accuracy", direction="max"),
                max_epochs=100,
                factor=3,
                directory=str(project_dir),
                project_name=f"tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
        else:
            return tuner_class(
                self.build_model_fn,
                objective=kt.Objective("val_accuracy", direction="max"),
                max_trials=self.max_trials,
                directory=str(project_dir),
                project_name=f"tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )

    def tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        **kwargs,
    ) -> TuningResult:
        """Perform Keras hyperparameter tuning"""
        self.logger.info(f"Starting Keras Tuner ({self.tuner_type}) with {self.max_trials} trials")
        start_time = datetime.now()

        # Create tuner
        self.tuner = self._create_tuner(None)

        # Prepare validation data
        if validation_data is None:
            from sklearn.model_selection import train_test_split

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = validation_data

        # Search
        self.tuner.search(
            X_train,
            y_train,
            epochs=50,
            validation_data=(X_val, y_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=5, restore_best_weights=True
                )
            ],
        )

        # Get results
        best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = self.tuner.get_best_models(num_trials=1)[0]

        # Evaluate best model
        val_loss, val_acc = best_model.evaluate(X_val, y_val, verbose=0)

        # Extract all trials
        all_trials = []
        for trial in self.tuner.oracle.get_best_trials(num_trials=self.max_trials):
            trial_dict = {
                "params": trial.hyperparameters.values,
                "score": trial.score,
                "trial_id": trial.trial_id,
            }
            all_trials.append(trial_dict)

        elapsed_time = (datetime.now() - start_time).total_seconds()

        result = TuningResult(
            best_params=best_hps.values,
            best_score=val_acc,
            all_trials=all_trials,
            tuning_history={"scores": [t["score"] for t in all_trials if t["score"] is not None]},
            elapsed_time=elapsed_time,
            n_trials=len(all_trials),
            search_method=f"keras_tuner_{self.tuner_type}",
            model_type="keras_model",
        )

        self.logger.info(
            f"Keras Tuner completed. Best validation accuracy: {result.best_score:.4f}"
        )

        return result


class RayTuneTuner(HyperparameterTuner):
    """Ray Tune for distributed hyperparameter optimization"""

    def __init__(
        self,
        trainable_fn: Callable,
        param_space: dict[str, Any],
        num_samples: int = 100,
        scheduler: str = "hyperband",
        **kwargs,
    ):
        super().__init__(None, param_space, **kwargs)
        self.trainable_fn = trainable_fn
        self.num_samples = num_samples
        self.scheduler = self._get_scheduler(scheduler)

    def _get_scheduler(self, scheduler_name: str):
        """Get Ray Tune scheduler"""
        schedulers = {
            "hyperband": HyperBandScheduler(metric="accuracy", mode="max", max_t=100),
            "pbt": PopulationBasedTraining(
                metric="accuracy",
                mode="max",
                perturbation_interval=10,
                hyperparam_mutations={
                    "lr": tune.loguniform(1e-5, 1e-1),
                    "batch_size": [16, 32, 64, 128],
                },
            ),
        }
        return schedulers.get(scheduler_name)

    def tune(self, X: np.ndarray, y: np.ndarray, **kwargs) -> TuningResult:
        """Perform Ray Tune optimization"""
        self.logger.info(f"Starting Ray Tune with {self.num_samples} samples")
        start_time = datetime.now()

        # Initialize Ray
        if not ray.is_initialized():
            ray.init(num_cpus=os.cpu_count())

        # Convert param space to Ray Tune format
        ray_param_space = self._convert_param_space()

        # Run optimization
        analysis = tune.run(
            self.trainable_fn,
            config=ray_param_space,
            num_samples=self.num_samples,
            scheduler=self.scheduler,
            progress_reporter=tune.CLIReporter(metric_columns=["accuracy", "training_iteration"]),
            resources_per_trial={
                "cpu": 1,
                "gpu": 0.25 if tf.config.list_physical_devices("GPU") else 0,
            },
        )

        # Get best trial
        best_trial = analysis.get_best_trial("accuracy", "max")

        # Extract all trials
        all_trials = []
        for trial in analysis.trials:
            trial_dict = {
                "params": trial.config,
                "score": trial.last_result.get("accuracy", 0),
                "trial_id": trial.trial_id,
            }
            all_trials.append(trial_dict)

        elapsed_time = (datetime.now() - start_time).total_seconds()

        result = TuningResult(
            best_params=best_trial.config,
            best_score=best_trial.last_result["accuracy"],
            all_trials=all_trials,
            tuning_history={"scores": [t["score"] for t in all_trials]},
            elapsed_time=elapsed_time,
            n_trials=len(all_trials),
            search_method="ray_tune",
            model_type="ray_trainable",
        )

        # Shutdown Ray
        ray.shutdown()

        self.logger.info(f"Ray Tune completed. Best accuracy: {result.best_score:.4f}")

        return result

    def _convert_param_space(self) -> dict[str, Any]:
        """Convert parameter space to Ray Tune format"""
        ray_space = {}

        for param_name, param_config in self.param_space.items():
            if param_config["type"] == "int":
                ray_space[param_name] = tune.randint(param_config["low"], param_config["high"])
            elif param_config["type"] == "float":
                if param_config.get("log", False):
                    ray_space[param_name] = tune.loguniform(
                        param_config["low"], param_config["high"]
                    )
                else:
                    ray_space[param_name] = tune.uniform(param_config["low"], param_config["high"])
            elif param_config["type"] == "categorical":
                ray_space[param_name] = tune.choice(param_config["choices"])

        return ray_space


class MultiObjectiveTuner(BayesianOptimizationTuner):
    """Multi-objective hyperparameter optimization"""

    def __init__(
        self,
        model_class: Any,
        param_space: dict[str, Any],
        objectives: list[str],
        n_trials: int = 100,
        **kwargs,
    ):
        super().__init__(model_class, param_space, n_trials=n_trials, **kwargs)
        self.objectives = objectives

    def _create_multi_objective(self, X: np.ndarray, y: np.ndarray) -> Callable:
        """Create multi-objective function"""
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            make_scorer,
            precision_score,
            recall_score,
        )

        # Define scorers
        scorers = {
            "accuracy": make_scorer(accuracy_score),
            "f1": make_scorer(f1_score, average="weighted"),
            "recall": make_scorer(recall_score, average="weighted"),
            "precision": make_scorer(precision_score, average="weighted", zero_division=0),
        }

        def objective(trial: optuna.Trial) -> tuple[float, ...]:
            # Sample hyperparameters
            params = {}
            for param_name, param_config in self.param_space.items():
                if param_config["type"] == "int":
                    params[param_name] = trial.suggest_int(
                        param_name, param_config["low"], param_config["high"]
                    )
                elif param_config["type"] == "float":
                    if param_config.get("log", False):
                        params[param_name] = trial.suggest_loguniform(
                            param_name, param_config["low"], param_config["high"]
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name, param_config["low"], param_config["high"]
                        )
                elif param_config["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config["choices"]
                    )

            # Create model
            model = self.model_class(**params)

            # Evaluate multiple objectives
            objective_values = []
            for obj_name in self.objectives:
                if obj_name in scorers:
                    scores = cross_val_score(
                        model, X, y, cv=self.cv, scoring=scorers[obj_name], n_jobs=1
                    )
                    objective_values.append(scores.mean())

            return tuple(objective_values)

        return objective

    def tune(self, X: np.ndarray, y: np.ndarray, **kwargs) -> TuningResult:
        """Perform multi-objective optimization"""
        self.logger.info(f"Starting Multi-Objective Optimization for {self.objectives}")
        start_time = datetime.now()

        # Create multi-objective study
        self.study = optuna.create_study(
            directions=["maximize"] * len(self.objectives), sampler=self.sampler
        )

        # Create objective
        objective = self._create_multi_objective(X, y)

        # Optimize
        self.study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs)

        # Get Pareto front
        pareto_trials = self.study.best_trials

        # Select best trial based on primary objective
        best_trial = max(pareto_trials, key=lambda t: t.values[0])

        elapsed_time = (datetime.now() - start_time).total_seconds()

        # Process all trials
        all_trials = []
        for trial in self.study.trials:
            trial_dict = {
                "params": trial.params,
                "values": trial.values,
                "state": str(trial.state),
                "number": trial.number,
            }
            all_trials.append(trial_dict)

        result = TuningResult(
            best_params=best_trial.params,
            best_score=best_trial.values[0],
            all_trials=all_trials,
            tuning_history={
                f"{obj}_scores": [
                    t.values[i] for t in self.study.trials if t.values and len(t.values) > i
                ]
                for i, obj in enumerate(self.objectives)
            },
            elapsed_time=elapsed_time,
            n_trials=len(self.study.trials),
            search_method="multi_objective_optimization",
            model_type=self.model_class.__name__,
        )

        self.logger.info(
            f"Multi-Objective Optimization completed. "
            f"Best {self.objectives[0]}: {result.best_score:.4f}"
        )

        return result


def create_tuner(
    method: str, model_class: Any, param_space: dict[str, Any], **kwargs
) -> HyperparameterTuner:
    """
    Factory function to create appropriate tuner

    Args:
        method: Tuning method ('grid', 'random', 'bayesian', 'keras', 'ray', 'multi_objective')
        model_class: Model class or build function
        param_space: Parameter search space
        **kwargs: Additional arguments for specific tuners

    Returns:
        HyperparameterTuner instance
    """
    tuners = {
        "grid": GridSearchTuner,
        "random": RandomSearchTuner,
        "bayesian": BayesianOptimizationTuner,
        "keras": KerasTuner,
        "ray": RayTuneTuner,
        "multi_objective": MultiObjectiveTuner,
    }

    tuner_class = tuners.get(method)
    if tuner_class is None:
        raise ValueError(f"Unknown tuning method: {method}")

    return tuner_class(model_class, param_space, **kwargs)


def tune_from_config(
    config_path: str, model_class: Any, X: np.ndarray, y: np.ndarray, method: str = "bayesian"
) -> TuningResult:
    """
    Perform hyperparameter tuning using configuration file

    Args:
        config_path: Path to configuration file
        model_class: Model class to tune
        X: Training features
        y: Training labels
        method: Tuning method

    Returns:
        TuningResult with best parameters
    """
    # Load configuration
    parser = ConfigParser()
    config = parser.load_config(config_path)

    # Extract parameter space from config
    param_space = parser.get_hyperparameter_grid(config)

    # Create tuner
    tuner = create_tuner(
        method=method,
        model_class=model_class,
        param_space=param_space,
        scoring="accuracy",
        cv=5,
        n_trials=100 if method in ["bayesian", "ray"] else None,
    )

    # Perform tuning
    result = tuner.tune(X, y)

    # Save results
    output_dir = Path(config.output_dir or "./tuning_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    result_file = output_dir / f"{config.experiment_name}_{method}_results.json"
    result.save(str(result_file))

    return result
