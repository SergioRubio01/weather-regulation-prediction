"""
RNN model implementation for weather regulation prediction.
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Dropout, SimpleRNN
from keras.models import Sequential, load_model
from keras.optimizers import Adam

from src.config import ExperimentConfig

from .base_model import BaseModel


class RNNModel(BaseModel):
    """Simple RNN model for time series classification."""

    def __init__(self, config: ExperimentConfig):
        """
        Initialize RNN model.

        Args:
            config: Experiment configuration
        """
        super().__init__(config, "rnn")
        # RNN uses basic LSTM config for now
        self.model_config = config.lstm
        self.callbacks = []
        self.sequence_length = config.data.window_size

        # Set TensorFlow seed
        tf.random.set_seed(self.random_state)

    def build_model(
        self,
        input_shape: tuple[int, ...],
        units: int = 50,
        num_layers: int = 2,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        **kwargs,
    ) -> Sequential:
        """
        Build RNN model architecture.

        Args:
            input_shape: Shape of input data (timesteps, features)
            units: Number of RNN units
            num_layers: Number of RNN layers
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            **kwargs: Additional arguments

        Returns:
            Keras Sequential model
        """
        # Reshape input if needed
        if len(input_shape) == 1:
            input_shape = (self.sequence_length, input_shape[0])

        model = Sequential()

        # Add RNN layers
        for i in range(num_layers):
            return_sequences = i < num_layers - 1

            if i == 0:
                model.add(
                    SimpleRNN(
                        units=units,
                        activation="tanh",
                        return_sequences=return_sequences,
                        input_shape=input_shape,
                        name=f"rnn_{i}",
                    )
                )
            else:
                model.add(
                    SimpleRNN(
                        units=units,
                        activation="tanh",
                        return_sequences=return_sequences,
                        name=f"rnn_{i}",
                    )
                )

            # Add dropout
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate, name=f"dropout_{i}"))

        # Output layer
        model.add(Dense(1, activation="sigmoid", name="output"))

        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

        self.model = model
        return model

    def _prepare_sequences(
        self, X: np.ndarray, y: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Prepare sequences for RNN input.

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
        Train RNN model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            **kwargs: Additional training arguments

        Returns:
            Training history
        """
        self.logger.info("Starting RNN training...")
        start_time = time.time()

        # Prepare sequences
        X_train_seq, y_train_seq = self._prepare_sequences(X_train, y_train)
        if X_val is not None:
            X_val_seq, y_val_seq = self._prepare_sequences(X_val, y_val)
            validation_data = (X_val_seq, y_val_seq)
        else:
            validation_data = None

        # Build model if not already built
        if self.model is None:
            self.build_model(X_train_seq.shape[1:])

        # Setup callbacks
        self._setup_callbacks()

        # Train model - using hardcoded parameters for RNN
        epochs = 50
        batch_size = 32

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
        }

        if validation_data is not None:
            self.training_history["val_loss"] = history.history["val_loss"]
            self.training_history["val_accuracy"] = history.history["val_accuracy"]

        self.logger.info(f"Training completed in {self.metrics.training_time:.2f} seconds")
        self.logger.info(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")

        return self.training_history

    def _setup_callbacks(self):
        """Setup training callbacks."""
        self.callbacks = []

        # Early stopping
        if self.config.training.use_early_stopping:
            early_stopping = EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
            )
            self.callbacks.append(early_stopping)

        # Learning rate reduction
        if self.config.training.use_lr_scheduler:
            lr_scheduler = ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
            )
            self.callbacks.append(lr_scheduler)

    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with RNN.

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
        Predict probabilities with RNN.

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

    def _save_model_implementation(self, filepath: Path):
        """
        Save RNN model.

        Args:
            filepath: Path to save model
        """
        self.model.save(str(filepath.with_suffix(".h5")))

        # Save additional info
        info = {"sequence_length": self.sequence_length, "training_history": self.training_history}

        with open(filepath.with_suffix(".json"), "w") as f:
            json.dump(info, f, indent=2)

    def _load_model_implementation(self, filepath: Path):
        """
        Load RNN model.

        Args:
            filepath: Path to load model from
        """
        self.model = load_model(str(filepath.with_suffix(".h5")))

        # Load additional info
        info_path = filepath.with_suffix(".json")
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
                self.sequence_length = info.get("sequence_length", 1)
                self.training_history = info.get("training_history", {})
