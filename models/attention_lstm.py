"""
Attention LSTM model implementation for weather regulation prediction.
Combines LSTM with attention mechanism for improved performance.
"""

import json
import time
from pathlib import Path
from typing import Any

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import (
    LSTM,
    BatchNormalization,
    Bidirectional,
    Concatenate,
    Dense,
    Dropout,
    Input,
    Lambda,
    Layer,
)
from keras.models import Model, load_model
from keras.optimizers import Adam
from tensorflow import keras

from src.config import ExperimentConfig

from .base_model import BaseModel


class AttentionLayer(Layer):
    """Attention mechanism layer for LSTM."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # Input shape: (batch_size, time_steps, features)
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], input_shape[-1]),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias", shape=(input_shape[-1],), initializer="zeros", trainable=True
        )
        self.u = self.add_weight(
            name="attention_u",
            shape=(input_shape[-1],),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        # Compute attention scores
        uit = K.tanh(K.dot(inputs, self.W) + self.b)
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        # Apply softmax
        ait = K.softmax(ait, axis=1)
        ait = K.expand_dims(ait, axis=-1)

        # Apply attention weights
        weighted_input = inputs * ait
        output = K.sum(weighted_input, axis=1)

        return output, ait

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super().get_config()


class AttentionLSTMModel(BaseModel):
    """LSTM model with attention mechanism."""

    def __init__(self, config: ExperimentConfig):
        """
        Initialize Attention LSTM model.

        Args:
            config: Experiment configuration
        """
        super().__init__(config, "attention_lstm")
        # Use LSTM config as base
        self.model_config = config.lstm
        self.callbacks = []
        self.sequence_length = config.data.window_size

        # Set TensorFlow seed
        tf.random.set_seed(self.random_state)

    def build_model(
        self,
        input_shape: tuple[int, ...],
        units: int = 128,
        num_layers: int = 2,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        bidirectional: bool = True,
        attention_type: str = "bahdanau",
        **kwargs,
    ) -> Model:
        """
        Build Attention LSTM model architecture.

        Args:
            input_shape: Shape of input data (timesteps, features)
            units: Number of LSTM units
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            bidirectional: Whether to use bidirectional LSTM
            attention_type: Type of attention ('bahdanau' or 'custom')
            **kwargs: Additional arguments

        Returns:
            Keras Model
        """
        # Reshape input if needed
        if len(input_shape) == 1:
            input_shape = (self.sequence_length, input_shape[0])

        # Input layer
        inputs = Input(shape=input_shape, name="input")

        # LSTM layers with return sequences for attention
        x = inputs
        lstm_outputs = []

        for i in range(num_layers):
            return_sequences = True  # Always return sequences for attention

            # Create LSTM layer
            lstm_layer = LSTM(
                units=units, return_sequences=return_sequences, return_state=False, name=f"lstm_{i}"
            )

            # Add bidirectional wrapper if needed
            if bidirectional:
                lstm_layer = Bidirectional(lstm_layer, name=f"bidirectional_lstm_{i}")

            x = lstm_layer(x)

            # Batch normalization
            x = BatchNormalization()(x)

            # Dropout
            if dropout_rate > 0:
                x = Dropout(dropout_rate)(x)

            # Store LSTM output for skip connections
            lstm_outputs.append(x)

        if attention_type == "bahdanau":
            # Bahdanau attention mechanism
            attention_output, attention_weights = self._bahdanau_attention(x, units)
        else:
            # Custom attention layer
            attention_layer = AttentionLayer(name="attention")
            attention_output, attention_weights = attention_layer(x)

        # Combine attention output with last LSTM hidden state
        last_hidden = Lambda(lambda x: x[:, -1, :], name="last_hidden")(x)
        combined = Concatenate(name="combine_attention")([attention_output, last_hidden])

        # Dense layers
        dense = Dense(64, activation="relu", name="dense_1")(combined)
        if dropout_rate > 0:
            dense = Dropout(dropout_rate)(dense)

        dense = Dense(32, activation="relu", name="dense_2")(dense)
        if dropout_rate > 0:
            dense = Dropout(dropout_rate / 2)(dense)

        # Output layer
        outputs = Dense(1, activation="sigmoid", name="output")(dense)

        # Create model
        model = Model(inputs=inputs, outputs=outputs, name="AttentionLSTM")

        # Also create a model for extracting attention weights
        self.attention_model = Model(inputs=inputs, outputs=attention_weights)

        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy", "AUC", "Precision", "Recall"],
        )

        self.model = model
        return model

    def _bahdanau_attention(self, lstm_output, units):
        """
        Bahdanau attention mechanism.

        Args:
            lstm_output: LSTM output tensor (batch_size, time_steps, features)
            units: Number of units for attention

        Returns:
            Tuple of (context_vector, attention_weights)
        """
        # Hidden state shape: (batch_size, time_steps, features)
        hidden_size = int(lstm_output.shape[-1])

        # Score function
        score_first_layer = Dense(hidden_size, name="attention_score_1")
        score_second_layer = Dense(1, name="attention_score_2")

        # Compute scores
        scores = score_second_layer(keras.activations.tanh(score_first_layer(lstm_output)))

        # Attention weights
        attention_weights = keras.activations.softmax(scores, axis=1)

        # Context vector
        context_vector = attention_weights * lstm_output
        context_vector = Lambda(lambda x: K.sum(x, axis=1), name="context_vector")(context_vector)

        return context_vector, attention_weights

    def _prepare_sequences(
        self, X: np.ndarray, y: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Prepare sequences for LSTM input."""
        if len(X.shape) == 3:
            return X, y

        # Create sequences
        n_samples = X.shape[0] - self.sequence_length + 1
        n_features = X.shape[1]

        X_seq = np.zeros((n_samples, self.sequence_length, n_features))

        for i in range(n_samples):
            X_seq[i] = X[i : i + self.sequence_length]

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
        Train Attention LSTM model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            **kwargs: Additional training arguments

        Returns:
            Training history
        """
        self.logger.info("Starting Attention LSTM training...")
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

        # Get training parameters
        epochs = self.model_config.epochs[0] if hasattr(self.model_config, "epochs") else 100
        batch_size = (
            self.model_config.batch_size[0] if hasattr(self.model_config, "batch_size") else 32
        )

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
            "precision": history.history.get("precision", []),
            "recall": history.history.get("recall", []),
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
                monitor="val_loss", factor=0.5, patience=7, min_lr=1e-7, verbose=1
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

    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Get attention weights for input sequences.

        Args:
            X: Input features

        Returns:
            Attention weights
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        X_seq, _ = self._prepare_sequences(X)
        attention_weights = self.attention_model.predict(X_seq, verbose=0)

        return attention_weights

    def visualize_attention(self, X: np.ndarray, sample_idx: int = 0):
        """
        Visualize attention weights for a specific sample.

        Args:
            X: Input features
            sample_idx: Index of sample to visualize
        """
        import matplotlib.pyplot as plt

        # Get attention weights
        attention_weights = self.get_attention_weights(X)

        if sample_idx >= len(attention_weights):
            raise ValueError(f"Sample index {sample_idx} out of range")

        # Get weights for specific sample
        sample_weights = attention_weights[sample_idx].squeeze()

        # Create visualization
        plt.figure(figsize=(12, 4))

        # Heatmap
        plt.subplot(1, 2, 1)
        plt.imshow(sample_weights.reshape(1, -1), cmap="Blues", aspect="auto")
        plt.colorbar()
        plt.xlabel("Time Steps")
        plt.ylabel("Sample")
        plt.title("Attention Weights Heatmap")

        # Line plot
        plt.subplot(1, 2, 2)
        plt.plot(sample_weights)
        plt.xlabel("Time Steps")
        plt.ylabel("Attention Weight")
        plt.title("Attention Weights Over Time")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / f"attention_weights_sample_{sample_idx}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Attention visualization saved to {plot_path}")

    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with Attention LSTM."""
        X_seq, _ = self._prepare_sequences(X)
        predictions = self.model.predict(X_seq, verbose=0)
        return (predictions > 0.5).astype(int).flatten()

    def _predict_proba_implementation(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities with Attention LSTM."""
        X_seq, _ = self._prepare_sequences(X)
        proba = self.model.predict(X_seq, verbose=0)

        proba_2d = np.zeros((len(proba), 2))
        proba_2d[:, 1] = proba.flatten()
        proba_2d[:, 0] = 1 - proba.flatten()

        return proba_2d

    def _save_model_implementation(self, filepath: Path):
        """Save Attention LSTM model."""
        # Custom objects for loading
        custom_objects = {"AttentionLayer": AttentionLayer}

        # Save main model
        self.model.save(str(filepath.with_suffix(".h5")), save_traces=False)

        # Save attention model
        self.attention_model.save(str(filepath.with_suffix("_attention.h5")), save_traces=False)

        # Save additional info
        info = {
            "best_params": self.best_params,
            "sequence_length": self.sequence_length,
            "training_history": self.training_history,
            "custom_objects": list(custom_objects.keys()),
        }

        with open(filepath.with_suffix(".json"), "w") as f:
            json.dump(info, f, indent=2)

    def _load_model_implementation(self, filepath: Path):
        """Load Attention LSTM model."""
        # Custom objects for loading
        custom_objects = {"AttentionLayer": AttentionLayer}

        # Load main model
        self.model = load_model(str(filepath.with_suffix(".h5")), custom_objects=custom_objects)

        # Load attention model
        attention_path = filepath.with_suffix("_attention.h5")
        if attention_path.exists():
            self.attention_model = load_model(str(attention_path), custom_objects=custom_objects)

        # Load additional info
        info_path = filepath.with_suffix(".json")
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
                self.best_params = info.get("best_params")
                self.sequence_length = info.get("sequence_length", 1)
                self.training_history = info.get("training_history", {})
