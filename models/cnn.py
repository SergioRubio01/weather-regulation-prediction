"""
CNN model implementation for weather regulation prediction.
"""

import json
import time
from pathlib import Path
from typing import Any

import keras_tuner as kt
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    MaxPooling1D,
)
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from tensorflow import keras

from src.config import CNNConfig, ExperimentConfig

from .base_model import BaseModel


class CNNModel(BaseModel):
    """Convolutional Neural Network model for time series classification."""

    def __init__(self, config: ExperimentConfig):
        """
        Initialize CNN model.

        Args:
            config: Experiment configuration
        """
        super().__init__(config, "cnn")
        self.model_config: CNNConfig = config.cnn
        self.callbacks = []
        self.sequence_length = config.data.window_size

        # Set TensorFlow seed
        tf.random.set_seed(self.random_state)

    def build_model(
        self,
        input_shape: tuple[int, ...],
        filters: int | None = None,
        kernel_size: int | None = None,
        pool_size: int | None = None,
        num_conv_layers: int | None = None,
        dropout_rate: float | None = None,
        dense_units: int | None = None,
        learning_rate: float | None = None,
        use_batch_norm: bool | None = None,
        **kwargs,
    ) -> Sequential:
        """
        Build CNN model architecture.

        Args:
            input_shape: Shape of input data
            filters: Number of convolutional filters
            kernel_size: Size of convolutional kernel
            pool_size: Size of pooling window
            num_conv_layers: Number of convolutional layers
            dropout_rate: Dropout rate
            dense_units: Number of units in dense layer
            learning_rate: Learning rate
            use_batch_norm: Whether to use batch normalization
            **kwargs: Additional arguments

        Returns:
            Keras Sequential model
        """
        # Use provided parameters or defaults from config
        filters = filters or self.model_config.filters[0]
        kernel_size = kernel_size or self.model_config.kernel_size[0]
        pool_size = pool_size or self.model_config.pool_size[0]
        num_conv_layers = num_conv_layers or self.model_config.num_conv_layers[0]
        dropout_rate = dropout_rate or self.model_config.dropout_rate[0]
        dense_units = dense_units or self.model_config.dense_units[0]
        learning_rate = learning_rate or self.model_config.learning_rate[0]
        use_batch_norm = (
            use_batch_norm if use_batch_norm is not None else self.model_config.use_batch_norm
        )

        # Reshape input if needed
        if len(input_shape) == 1:
            input_shape = (input_shape[0], 1)

        model = Sequential()

        # Add convolutional layers
        for i in range(num_conv_layers):
            # Calculate filters for this layer (increasing pattern)
            layer_filters = filters * (2**i)

            # Convolutional layer
            if i == 0:
                model.add(
                    Conv1D(
                        filters=layer_filters,
                        kernel_size=kernel_size,
                        padding=self.model_config.padding,
                        input_shape=input_shape,
                        name=f"conv1d_{i}",
                    )
                )
            else:
                model.add(
                    Conv1D(
                        filters=layer_filters,
                        kernel_size=kernel_size,
                        padding=self.model_config.padding,
                        name=f"conv1d_{i}",
                    )
                )

            # Batch normalization
            if use_batch_norm:
                model.add(BatchNormalization(name=f"batch_norm_{i}"))

            # Activation
            model.add(Activation(self.model_config.activation, name=f"activation_{i}"))

            # Pooling (skip on last layer to preserve more features)
            if i < num_conv_layers - 1:
                model.add(MaxPooling1D(pool_size=pool_size, name=f"max_pool_{i}"))

            # Dropout
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate, name=f"dropout_conv_{i}"))

        # Global pooling or flatten
        if num_conv_layers > 2:
            model.add(GlobalAveragePooling1D(name="global_avg_pool"))
        else:
            model.add(Flatten(name="flatten"))

        # Dense layers
        model.add(Dense(dense_units, name="dense_1"))
        if use_batch_norm:
            model.add(BatchNormalization(name="batch_norm_dense"))
        model.add(Activation(self.model_config.activation, name="activation_dense"))

        if dropout_rate > 0:
            model.add(Dropout(dropout_rate, name="dropout_dense"))

        # Output layer
        model.add(Dense(1, activation="sigmoid", name="output"))

        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=self.model_config.loss, metrics=["accuracy", "AUC"])

        self.model = model
        return model

    def _prepare_data_for_cnn(self, X: np.ndarray) -> np.ndarray:
        """
        Prepare data for CNN input.

        Args:
            X: Input features

        Returns:
            Reshaped data for CNN
        """
        if len(X.shape) == 2:
            # Add channel dimension
            return X.reshape(X.shape[0], X.shape[1], 1)
        return X

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Train CNN model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            **kwargs: Additional training arguments

        Returns:
            Training history
        """
        self.logger.info("Starting CNN training...")
        start_time = time.time()

        # Prepare data
        X_train = self._prepare_data_for_cnn(X_train)
        if X_val is not None:
            X_val = self._prepare_data_for_cnn(X_val)
            validation_data = (X_val, y_val)
        else:
            validation_data = None

        # Perform hyperparameter tuning if enabled
        if self.config.hyperparameter_tuning:
            self.logger.info("Performing hyperparameter tuning...")
            best_hp = self._tune_hyperparameters(X_train, y_train, validation_data)
            self.best_params = best_hp
        else:
            # Build model with default parameters
            if self.model is None:
                self.build_model(X_train.shape[1:])

        # Setup callbacks
        self._setup_callbacks()

        # Get training parameters
        epochs = self.model_config.epochs[0]
        batch_size = self.model_config.batch_size[0]

        # Train model
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.callbacks,
            verbose=1,
        )

        # Record training time
        self.metrics.training_time = time.time() - start_time
        self.is_trained = True

        # Store training history
        self.training_history = {
            "loss": history.history["loss"],
            "accuracy": history.history["accuracy"],
            "auc": history.history.get("auc", []),
        }

        if validation_data is not None:
            self.training_history["val_loss"] = history.history["val_loss"]
            self.training_history["val_accuracy"] = history.history["val_accuracy"]
            self.training_history["val_auc"] = history.history.get("val_auc", [])

        self.logger.info(f"Training completed in {self.metrics.training_time:.2f} seconds")
        self.logger.info(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")

        return self.training_history

    def _setup_callbacks(self):
        """Setup training callbacks."""
        self.callbacks = []

        # Early stopping
        if self.config.training.use_early_stopping:
            early_stopping = EarlyStopping(
                monitor=self.config.training.early_stopping_monitor,
                patience=self.config.training.early_stopping_patience,
                restore_best_weights=True,
                mode=self.config.training.early_stopping_mode,
                verbose=1,
            )
            self.callbacks.append(early_stopping)

        # Learning rate reduction
        if self.config.training.use_lr_scheduler:
            lr_scheduler = ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
            )
            self.callbacks.append(lr_scheduler)

        # Model checkpointing
        if self.config.training.save_best_model:
            checkpoint_path = self.output_dir / "best_model.h5"
            checkpoint = ModelCheckpoint(
                checkpoint_path,
                monitor=self.config.training.checkpoint_monitor,
                save_best_only=True,
                mode=self.config.training.checkpoint_mode,
                verbose=1,
            )
            self.callbacks.append(checkpoint)

    def _tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """
        Perform hyperparameter tuning using Keras Tuner.

        Args:
            X_train: Training features
            y_train: Training labels
            validation_data: Validation data tuple

        Returns:
            Best hyperparameters
        """

        def build_model(hp):
            """Build model for hyperparameter tuning."""
            model = Sequential()

            # Hyperparameters to tune
            filters = hp.Choice("filters", self.model_config.filters)
            kernel_size = hp.Choice("kernel_size", self.model_config.kernel_size)
            pool_size = hp.Choice("pool_size", self.model_config.pool_size)
            num_conv_layers = hp.Choice("num_conv_layers", self.model_config.num_conv_layers)
            dropout_rate = hp.Choice("dropout_rate", self.model_config.dropout_rate)
            dense_units = hp.Choice("dense_units", self.model_config.dense_units)
            learning_rate = hp.Choice("learning_rate", self.model_config.learning_rate)

            # Build convolutional layers
            for i in range(num_conv_layers):
                layer_filters = filters * (2**i)

                if i == 0:
                    model.add(
                        Conv1D(
                            filters=layer_filters,
                            kernel_size=kernel_size,
                            padding=self.model_config.padding,
                            activation=self.model_config.activation,
                            input_shape=X_train.shape[1:],
                        )
                    )
                else:
                    model.add(
                        Conv1D(
                            filters=layer_filters,
                            kernel_size=kernel_size,
                            padding=self.model_config.padding,
                            activation=self.model_config.activation,
                        )
                    )

                if self.model_config.use_batch_norm:
                    model.add(BatchNormalization())

                if i < num_conv_layers - 1:
                    model.add(MaxPooling1D(pool_size=pool_size))

                if dropout_rate > 0:
                    model.add(Dropout(dropout_rate))

            # Flatten and dense layers
            if num_conv_layers > 2:
                model.add(GlobalAveragePooling1D())
            else:
                model.add(Flatten())

            model.add(Dense(dense_units, activation=self.model_config.activation))
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))

            model.add(Dense(1, activation="sigmoid"))

            # Compile
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss=self.model_config.loss,
                metrics=["accuracy"],
            )

            return model

        # Choose tuner based on method
        if self.config.tuning_method == "bayesian":
            tuner = kt.BayesianOptimization(
                build_model,
                objective="val_accuracy",
                max_trials=self.config.tuning_trials,
                directory=self.output_dir / "tuning",
                project_name="cnn_tuning",
            )
        elif self.config.tuning_method == "random":
            tuner = kt.RandomSearch(
                build_model,
                objective="val_accuracy",
                max_trials=self.config.tuning_trials,
                directory=self.output_dir / "tuning",
                project_name="cnn_tuning",
            )
        else:  # grid search
            tuner = kt.GridSearch(
                build_model,
                objective="val_accuracy",
                directory=self.output_dir / "tuning",
                project_name="cnn_tuning",
            )

        # Search
        tuner.search(
            X_train,
            y_train,
            epochs=min(self.model_config.epochs),
            batch_size=self.model_config.batch_size[0],
            validation_data=validation_data,
            callbacks=(
                [EarlyStopping(patience=5)] if self.config.training.use_early_stopping else []
            ),
            verbose=1,
        )

        # Get best hyperparameters
        best_hp = tuner.get_best_hyperparameters()[0]
        self.model = tuner.get_best_models()[0]

        # Convert to dictionary
        best_params = {
            "filters": best_hp.get("filters"),
            "kernel_size": best_hp.get("kernel_size"),
            "pool_size": best_hp.get("pool_size"),
            "num_conv_layers": best_hp.get("num_conv_layers"),
            "dropout_rate": best_hp.get("dropout_rate"),
            "dense_units": best_hp.get("dense_units"),
            "learning_rate": best_hp.get("learning_rate"),
        }

        self.logger.info(f"Best hyperparameters: {best_params}")

        return best_params

    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with CNN.

        Args:
            X: Input features

        Returns:
            Binary predictions
        """
        X = self._prepare_data_for_cnn(X)
        predictions = self.model.predict(X, verbose=0)
        return (predictions > 0.5).astype(int).flatten()

    def _predict_proba_implementation(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities with CNN.

        Args:
            X: Input features

        Returns:
            Prediction probabilities
        """
        X = self._prepare_data_for_cnn(X)
        proba = self.model.predict(X, verbose=0)

        # Convert to 2D array with probabilities for both classes
        proba_2d = np.zeros((len(proba), 2))
        proba_2d[:, 1] = proba.flatten()
        proba_2d[:, 0] = 1 - proba.flatten()

        return proba_2d

    def get_feature_maps(self, X: np.ndarray, layer_index: int = 0) -> np.ndarray:
        """
        Get feature maps from a specific convolutional layer.

        Args:
            X: Input sample(s)
            layer_index: Index of convolutional layer

        Returns:
            Feature maps
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        X = self._prepare_data_for_cnn(X)

        # Find convolutional layers
        conv_layers = [layer for layer in self.model.layers if "conv" in layer.name]

        if layer_index >= len(conv_layers):
            raise ValueError(
                f"Layer index {layer_index} out of range. Model has {len(conv_layers)} conv layers."
            )

        # Create intermediate model
        intermediate_model = keras.Model(
            inputs=self.model.input, outputs=conv_layers[layer_index].output
        )

        # Get feature maps
        feature_maps = intermediate_model.predict(X)

        return feature_maps

    def visualize_filters(self, layer_index: int = 0):
        """
        Visualize convolutional filters.

        Args:
            layer_index: Index of convolutional layer
        """
        import matplotlib.pyplot as plt

        if not self.is_trained:
            raise ValueError("Model must be trained first")

        # Find convolutional layers
        conv_layers = [layer for layer in self.model.layers if "conv" in layer.name]

        if layer_index >= len(conv_layers):
            raise ValueError(f"Layer index {layer_index} out of range.")

        # Get filters
        filters, biases = conv_layers[layer_index].get_weights()

        # Normalize filter values
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)

        # Plot filters
        n_filters = filters.shape[2]
        n_cols = 8
        n_rows = (n_filters + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        axes = axes.flatten()

        for i in range(n_filters):
            ax = axes[i]
            ax.plot(filters[:, 0, i])
            ax.set_title(f"Filter {i}")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_filters, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(f"Convolutional Filters - Layer {layer_index}")
        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / f"filters_layer_{layer_index}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Filter visualization saved to {plot_path}")

    def _save_model_implementation(self, filepath: Path):
        """
        Save CNN model.

        Args:
            filepath: Path to save model
        """
        # Save model
        self.model.save(str(filepath.with_suffix(".h5")))

        # Save additional info
        info = {"best_params": self.best_params, "training_history": self.training_history}

        with open(filepath.with_suffix(".json"), "w") as f:
            json.dump(info, f, indent=2)

    def _load_model_implementation(self, filepath: Path):
        """
        Load CNN model.

        Args:
            filepath: Path to load model from
        """
        # Load model
        self.model = load_model(str(filepath.with_suffix(".h5")))

        # Load additional info
        info_path = filepath.with_suffix(".json")
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
                self.best_params = info.get("best_params")
                self.training_history = info.get("training_history", {})
