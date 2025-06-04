"""
LSTM model implementation for weather regulation prediction.
"""

import json
import time
from pathlib import Path
from typing import Any

import keras_tuner as kt
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers import LSTM, Bidirectional, Dense, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import SGD, Adam, Nadam, RMSprop
from tensorflow import keras

from config import ExperimentConfig, LSTMConfig

from .base_model import BaseModel


class LSTMModel(BaseModel):
    """LSTM model for time series classification."""

    def __init__(self, config: ExperimentConfig):
        """
        Initialize LSTM model.

        Args:
            config: Experiment configuration
        """
        super().__init__(config, "lstm")
        self.model_config: LSTMConfig = config.lstm
        self.callbacks = []
        self.sequence_length = config.data.window_size

        # Set TensorFlow seed
        tf.random.set_seed(self.random_state)

    def build_model(
        self,
        input_shape: tuple[int, ...],
        units: int | None = None,
        num_layers: int | None = None,
        dropout_rate: float | None = None,
        learning_rate: float | None = None,
        bidirectional: bool | None = None,
        **kwargs,
    ) -> Sequential:
        """
        Build LSTM model architecture.

        Args:
            input_shape: Shape of input data (timesteps, features)
            units: Number of LSTM units
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            bidirectional: Whether to use bidirectional LSTM
            **kwargs: Additional arguments

        Returns:
            Keras Sequential model
        """
        # Use provided parameters or defaults from config
        units = units or self.model_config.units[0]
        num_layers = num_layers or self.model_config.num_layers[0]
        dropout_rate = dropout_rate or self.model_config.dropout_rate[0]
        learning_rate = learning_rate or self.model_config.learning_rate[0]
        bidirectional = (
            bidirectional if bidirectional is not None else self.model_config.bidirectional
        )

        # Reshape input if needed
        if len(input_shape) == 1:
            input_shape = (self.sequence_length, input_shape[0])

        model = Sequential()

        # Add LSTM layers
        for i in range(num_layers):
            return_sequences = i < num_layers - 1  # All but last layer return sequences

            # Create LSTM layer
            lstm_layer = LSTM(
                units=units,
                activation=self.model_config.activation,
                recurrent_activation=self.model_config.recurrent_activation,
                use_bias=self.model_config.use_bias,
                return_sequences=return_sequences,
                recurrent_dropout=self.model_config.recurrent_dropout[0],
                name=f"lstm_{i}",
            )

            # Add bidirectional wrapper if needed
            if bidirectional:
                lstm_layer = Bidirectional(lstm_layer, name=f"bidirectional_lstm_{i}")

            # Add to model
            if i == 0:
                model.add(lstm_layer(shape=input_shape))
            else:
                model.add(lstm_layer)

            # Add dropout
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate, name=f"dropout_{i}"))

        # Output layer
        model.add(Dense(1, activation="sigmoid", name="output"))

        # Compile model
        optimizer = self._get_optimizer(learning_rate)
        model.compile(optimizer=optimizer, loss=self.model_config.loss, metrics=["accuracy", "AUC"])

        self.model = model
        return model

    def _get_optimizer(self, learning_rate: float):
        """Get optimizer based on configuration."""
        optimizer_name = self.model_config.optimizer.lower()

        if optimizer_name == "adam":
            return Adam(learning_rate=learning_rate)
        elif optimizer_name == "sgd":
            return SGD(learning_rate=learning_rate)
        elif optimizer_name == "rmsprop":
            return RMSprop(learning_rate=learning_rate)
        elif optimizer_name == "nadam":
            return Nadam(learning_rate=learning_rate)
        elif optimizer_name == "adamw":
            return keras.optimizers.AdamW(learning_rate=learning_rate)
        else:
            self.logger.warning(f"Unknown optimizer {optimizer_name}, using Adam")
            return Adam(learning_rate=learning_rate)

    def _prepare_sequences(
        self, X: np.ndarray, y: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Prepare sequences for LSTM input.

        Args:
            X: Input features (samples, features)
            y: Labels (optional)

        Returns:
            X_seq: Sequenced input (samples, timesteps, features)
            y_seq: Sequenced labels (if y provided)
        """
        if len(X.shape) == 3:
            # Already in sequence format
            return X, y

        # Create sequences
        n_samples = X.shape[0] - self.sequence_length + 1
        n_features = X.shape[1]

        X_seq = np.zeros((n_samples, self.sequence_length, n_features))

        for i in range(n_samples):
            X_seq[i] = X[i : i + self.sequence_length]

        # Adjust labels if provided
        if y is not None:
            y_seq = y[self.sequence_length - 1 :]
            return X_seq, y_seq

        return X_seq, None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Train LSTM model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            **kwargs: Additional training arguments

        Returns:
            Training history
        """
        self.logger.info("Starting LSTM training...")
        start_time = time.time()

        # Prepare sequences
        X_train_seq, y_train_seq = self._prepare_sequences(X_train, y_train)
        if X_val is not None:
            X_val_seq, y_val_seq = self._prepare_sequences(X_val, y_val)
            validation_data = (X_val_seq, y_val_seq)
        else:
            validation_data = None

        # Perform hyperparameter tuning if enabled
        if self.config.hyperparameter_tuning:
            self.logger.info("Performing hyperparameter tuning...")
            best_hp = self._tune_hyperparameters(X_train_seq, y_train_seq, validation_data)
            self.best_params = best_hp
        else:
            # Build model with default parameters
            if self.model is None:
                self.build_model(X_train_seq.shape[1:])

        # Setup callbacks
        self._setup_callbacks()

        # Get training parameters
        epochs = self.model_config.epochs[0]
        batch_size = self.model_config.batch_size[0]

        # Train model
        history = self.model.fit(
            X_train_seq,
            y_train_seq,
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
                patience=self.model_config.early_stopping_patience,
                restore_best_weights=True,
                mode=self.config.training.early_stopping_mode,
                verbose=1,
            )
            self.callbacks.append(early_stopping)

        # Learning rate reduction
        if self.config.training.use_lr_scheduler:
            lr_scheduler = ReduceLROnPlateau(
                monitor="val_loss",
                factor=self.model_config.reduce_lr_factor,
                patience=self.model_config.reduce_lr_patience,
                min_lr=1e-7,
                verbose=1,
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

        # TensorBoard
        if self.config.training.tensorboard:
            tensorboard = TensorBoard(
                log_dir=self.output_dir / "logs",
                histogram_freq=1,
                write_graph=True,
                update_freq="epoch",
            )
            self.callbacks.append(tensorboard)

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
            units = hp.Choice("units", self.model_config.units)
            num_layers = hp.Choice("num_layers", self.model_config.num_layers)
            dropout_rate = hp.Choice("dropout_rate", self.model_config.dropout_rate)
            learning_rate = hp.Choice("learning_rate", self.model_config.learning_rate)
            bidirectional = hp.Boolean("bidirectional", default=self.model_config.bidirectional)

            # Build layers
            for i in range(num_layers):
                return_sequences = i < num_layers - 1

                lstm_layer = LSTM(
                    units=units,
                    activation=self.model_config.activation,
                    recurrent_activation=self.model_config.recurrent_activation,
                    return_sequences=return_sequences,
                    recurrent_dropout=hp.Choice(
                        "recurrent_dropout", self.model_config.recurrent_dropout
                    ),
                )

                if bidirectional:
                    lstm_layer = Bidirectional(lstm_layer)

                if i == 0:
                    model.add(lstm_layer)
                else:
                    model.add(lstm_layer)

                if dropout_rate > 0:
                    model.add(Dropout(dropout_rate))

            model.add(Dense(1, activation="sigmoid"))

            # Compile
            optimizer = self._get_optimizer(learning_rate)
            model.compile(optimizer=optimizer, loss=self.model_config.loss, metrics=["accuracy"])

            return model

        # Choose tuner based on method
        if self.config.tuning_method == "bayesian":
            tuner = kt.BayesianOptimization(
                build_model,
                objective="val_accuracy",
                max_trials=self.config.tuning_trials,
                directory=self.output_dir / "tuning",
                project_name="lstm_tuning",
            )
        elif self.config.tuning_method == "random":
            tuner = kt.RandomSearch(
                build_model,
                objective="val_accuracy",
                max_trials=self.config.tuning_trials,
                directory=self.output_dir / "tuning",
                project_name="lstm_tuning",
            )
        else:  # grid search
            tuner = kt.GridSearch(
                build_model,
                objective="val_accuracy",
                directory=self.output_dir / "tuning",
                project_name="lstm_tuning",
            )

        # Setup callbacks for tuning
        tuning_callbacks = []
        if self.config.training.use_early_stopping:
            tuning_callbacks.append(EarlyStopping(patience=5, restore_best_weights=True))

        # Search
        tuner.search(
            X_train,
            y_train,
            epochs=min(self.model_config.epochs),  # Use minimum epochs for tuning
            batch_size=self.model_config.batch_size[0],
            validation_data=validation_data,
            callbacks=tuning_callbacks,
            verbose=1,
        )

        # Get best hyperparameters
        best_hp = tuner.get_best_hyperparameters()[0]
        self.model = tuner.get_best_models()[0]

        # Convert to dictionary
        best_params = {
            "units": best_hp.get("units"),
            "num_layers": best_hp.get("num_layers"),
            "dropout_rate": best_hp.get("dropout_rate"),
            "learning_rate": best_hp.get("learning_rate"),
            "bidirectional": best_hp.get("bidirectional"),
            "recurrent_dropout": best_hp.get("recurrent_dropout"),
        }

        self.logger.info(f"Best hyperparameters: {best_params}")

        return best_params

    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with LSTM.

        Args:
            X: Input features

        Returns:
            Binary predictions
        """
        X_seq, _ = self._prepare_sequences(X)
        predictions = self.model.predict(X_seq, verbose=0)
        return (predictions > 0.5).astype(int).flatten()

    def _predict_proba_implementation(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities with LSTM.

        Args:
            X: Input features

        Returns:
            Prediction probabilities
        """
        X_seq, _ = self._prepare_sequences(X)
        proba = self.model.predict(X_seq, verbose=0)

        # Convert to 2D array with probabilities for both classes
        proba_2d = np.zeros((len(proba), 2))
        proba_2d[:, 1] = proba.flatten()
        proba_2d[:, 0] = 1 - proba.flatten()

        return proba_2d

    def plot_training_history(self):
        """Plot training history."""
        if not self.training_history:
            raise ValueError("No training history available")

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Loss plot
        ax = axes[0]
        ax.plot(self.training_history["loss"], label="Training Loss")
        if "val_loss" in self.training_history:
            ax.plot(self.training_history["val_loss"], label="Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Model Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Accuracy plot
        ax = axes[1]
        ax.plot(self.training_history["accuracy"], label="Training Accuracy")
        if "val_accuracy" in self.training_history:
            ax.plot(self.training_history["val_accuracy"], label="Validation Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / "training_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Training history plot saved to {plot_path}")

    def _save_model_implementation(self, filepath: Path):
        """
        Save LSTM model.

        Args:
            filepath: Path to save model
        """
        # Save model
        self.model.save(str(filepath.with_suffix(".h5")))

        # Save additional info
        info = {
            "best_params": self.best_params,
            "sequence_length": self.sequence_length,
            "training_history": self.training_history,
        }

        with open(filepath.with_suffix(".json"), "w") as f:
            json.dump(info, f, indent=2)

    def _load_model_implementation(self, filepath: Path):
        """
        Load LSTM model.

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
                self.sequence_length = info.get("sequence_length", 1)
                self.training_history = info.get("training_history", {})

    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if self.model is None:
            return "Model not built yet"

        from io import StringIO

        stream = StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + "\n"))
        return stream.getvalue()
