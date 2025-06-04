"""
Random Forest model implementation for weather regulation prediction.
"""

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.config import ExperimentConfig, RandomForestConfig

from .base_model import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest classification model."""

    def __init__(self, config: ExperimentConfig):
        """
        Initialize Random Forest model.

        Args:
            config: Experiment configuration
        """
        super().__init__(config, "random_forest")
        self.model_config: RandomForestConfig = config.random_forest

    def build_model(self, input_shape: tuple[int, ...], **kwargs) -> RandomForestClassifier:
        """
        Build Random Forest model.

        Args:
            input_shape: Shape of input data (not used for RF)
            **kwargs: Additional arguments

        Returns:
            RandomForestClassifier instance
        """
        # Get fixed hyperparameters
        fixed_params = {
            "bootstrap": self.model_config.bootstrap,
            "n_jobs": self.model_config.n_jobs,
            "verbose": self.model_config.verbose,
            "random_state": self.random_state,
        }

        # If hyperparameter tuning is disabled, use first values
        if not self.config.hyperparameter_tuning:
            self.model = RandomForestClassifier(
                n_estimators=self.model_config.n_estimators[0],
                criterion=self.model_config.criterion[0],
                max_depth=self.model_config.max_depth[0],
                min_samples_split=self.model_config.min_samples_split[0],
                min_samples_leaf=self.model_config.min_samples_leaf[0],
                max_features=self.model_config.max_features[0],
                **fixed_params,
            )
        else:
            # Create base model for hyperparameter tuning
            self.model = RandomForestClassifier(**fixed_params)

        return self.model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Train Random Forest model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (not used for RF)
            y_val: Validation labels (not used for RF)
            **kwargs: Additional training arguments

        Returns:
            Training history
        """
        self.logger.info("Starting Random Forest training...")
        start_time = time.time()

        # Build model if not already built
        if self.model is None:
            self.build_model(X_train.shape)

        # Perform hyperparameter tuning if enabled
        if self.config.hyperparameter_tuning:
            self.logger.info("Performing hyperparameter tuning...")
            self._tune_hyperparameters(X_train, y_train)
        else:
            # Direct training
            self.model.fit(X_train, y_train)

        # Record training time
        self.metrics.training_time = time.time() - start_time
        self.is_trained = True

        # Calculate training metrics
        train_score = self.model.score(X_train, y_train)
        self.training_history = {
            "train_accuracy": train_score,
            "training_time": self.metrics.training_time,
        }

        # If validation data provided, calculate validation score
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            self.training_history["val_accuracy"] = val_score

        # Get feature importance
        self.metrics.feature_importance = self.model.feature_importances_

        self.logger.info(f"Training completed in {self.metrics.training_time:.2f} seconds")
        self.logger.info(f"Training accuracy: {train_score:.4f}")

        return self.training_history

    def _tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Perform hyperparameter tuning.

        Args:
            X_train: Training features
            y_train: Training labels
        """
        # Define parameter grid
        param_grid = {
            "n_estimators": self.model_config.n_estimators,
            "criterion": self.model_config.criterion,
            "max_depth": self.model_config.max_depth,
            "min_samples_split": self.model_config.min_samples_split,
            "min_samples_leaf": self.model_config.min_samples_leaf,
            "max_features": self.model_config.max_features,
        }

        # Remove parameters with only one value
        param_grid = {k: v for k, v in param_grid.items() if len(v) > 1}

        if not param_grid:
            # No hyperparameters to tune
            self.model.fit(X_train, y_train)
            return

        # Choose search method
        if self.config.tuning_method == "grid":
            search = GridSearchCV(
                self.model,
                param_grid,
                cv=self.config.training.cv_folds,
                scoring="accuracy",
                n_jobs=-1,
                verbose=1,
            )
        elif self.config.tuning_method == "random":
            search = RandomizedSearchCV(
                self.model,
                param_grid,
                n_iter=min(self.config.tuning_trials, 50),  # Limit iterations for RF
                cv=self.config.training.cv_folds,
                scoring="accuracy",
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state,
            )
        else:
            # For bayesian optimization, fall back to random search
            self.logger.warning("Bayesian optimization not implemented for RF, using random search")
            search = RandomizedSearchCV(
                self.model,
                param_grid,
                n_iter=min(self.config.tuning_trials, 50),
                cv=self.config.training.cv_folds,
                scoring="accuracy",
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state,
            )

        # Perform search
        search.fit(X_train, y_train)

        # Store best parameters and model
        self.best_params = search.best_params_
        self.model = search.best_estimator_

        # Store CV results
        cv_results = pd.DataFrame(search.cv_results_)
        self.training_history["cv_results"] = cv_results.to_dict()
        self.training_history["best_params"] = self.best_params
        self.training_history["best_score"] = search.best_score_

        self.logger.info(f"Best parameters: {self.best_params}")
        self.logger.info(f"Best CV score: {search.best_score_:.4f}")

    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with Random Forest.

        Args:
            X: Input features

        Returns:
            Binary predictions
        """
        return self.model.predict(X)

    def _predict_proba_implementation(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities with Random Forest.

        Args:
            X: Input features

        Returns:
            Prediction probabilities
        """
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance from Random Forest.

        Returns:
            Feature importance array
        """
        if self.is_trained:
            return self.model.feature_importances_
        return None

    def plot_feature_importance(self, feature_names: list | None = None, top_n: int = 20):
        """
        Plot feature importance.

        Args:
            feature_names: Optional list of feature names
            top_n: Number of top features to show
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting feature importance")

        import matplotlib.pyplot as plt

        # Get feature importance
        importance = self.get_feature_importance()

        # Create DataFrame for easier plotting
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importance))]

        importance_df = (
            pd.DataFrame({"feature": feature_names, "importance": importance})
            .sort_values("importance", ascending=False)
            .head(top_n)
        )

        # Create plot
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df["feature"], importance_df["importance"])
        plt.xlabel("Feature Importance")
        plt.title(f"Top {top_n} Feature Importances - Random Forest")
        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / "feature_importance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Feature importance plot saved to {plot_path}")

    def _save_model_implementation(self, filepath: Path):
        """
        Save Random Forest model.

        Args:
            filepath: Path to save model
        """
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "best_params": self.best_params,
                    "feature_importance": self.metrics.feature_importance,
                },
                f,
            )

    def _load_model_implementation(self, filepath: Path):
        """
        Load Random Forest model.

        Args:
            filepath: Path to load model from
        """
        with open(filepath, "rb") as f:
            saved_data = pickle.load(f)
            self.model = saved_data["model"]
            self.best_params = saved_data.get("best_params")
            if "feature_importance" in saved_data:
                self.metrics.feature_importance = saved_data["feature_importance"]

    def get_model_params(self) -> dict[str, Any]:
        """
        Get current model parameters.

        Returns:
            Dictionary of model parameters
        """
        if self.model is None:
            return {}

        return self.model.get_params()

    def print_trees_info(self):
        """Print information about the trees in the forest."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        n_trees = len(self.model.estimators_)
        tree_depths = [tree.tree_.max_depth for tree in self.model.estimators_]

        print("\nRandom Forest Information:")
        print(f"Number of trees: {n_trees}")
        print(f"Average tree depth: {np.mean(tree_depths):.2f}")
        print(f"Max tree depth: {np.max(tree_depths)}")
        print(f"Min tree depth: {np.min(tree_depths)}")
        print(f"Number of features: {self.model.n_features_in_}")
        print(f"Number of classes: {self.model.n_classes_}")
