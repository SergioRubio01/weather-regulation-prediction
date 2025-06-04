"""
WaveNet model implementation for weather regulation prediction.
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Activation, Add, Conv1D, Dense, Dropout, Input, Layer, Multiply
from keras.models import Model, load_model
from keras.optimizers import Adam
from tensorflow import keras

from src.config import ExperimentConfig

from .base_model import BaseModel


class CausalConv1D(Layer):
    """Causal convolution layer for WaveNet."""

    def __init__(self, filters, kernel_size, dilation_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate

    def build(self, input_shape):
        self.conv = Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding="causal",
            dilation_rate=self.dilation_rate,
        )
        super().build(input_shape)

    def call(self, inputs):
        return self.conv(inputs)


class WaveNetModel(BaseModel):
    """WaveNet model for time series classification."""

    def __init__(self, config: ExperimentConfig):
        """
        Initialize WaveNet model.

        Args:
            config: Experiment configuration
        """
        super().__init__(config, "wavenet")
        # WaveNet uses LSTM config as base
        self.model_config = config.lstm
        self.callbacks = []

        # WaveNet specific parameters
        self.n_filters = 32
        self.filter_width = 2
        self.dilations = [1, 2, 4, 8, 16, 32]
        self.residual_channels = 32
        self.skip_channels = 32

        # Set TensorFlow seed
        tf.random.set_seed(self.random_state)

    def _wavenet_residual_block(self, inputs, dilation_rate, filters, kernel_size, name_prefix):
        """
        Create a WaveNet residual block.

        Args:
            inputs: Input tensor
            dilation_rate: Dilation rate for convolution
            filters: Number of filters
            kernel_size: Kernel size
            name_prefix: Prefix for layer names

        Returns:
            Tuple of (residual_output, skip_connection)
        """
        # Dilated causal convolution
        conv = CausalConv1D(
            filters=2 * filters,  # Double for gated activation
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            name=f"{name_prefix}_dilated_conv",
        )(inputs)

        # Gated activation unit
        tanh_out = Activation("tanh")(conv[:, :, :filters])
        sigm_out = Activation("sigmoid")(conv[:, :, filters:])
        acts = Multiply()([tanh_out, sigm_out])

        # Skip connection
        skip_out = Conv1D(self.skip_channels, 1, name=f"{name_prefix}_skip")(acts)

        # Residual connection
        res_out = Conv1D(self.residual_channels, 1, name=f"{name_prefix}_residual")(acts)
        res_out = Add()([res_out, inputs])

        return res_out, skip_out

    def build_model(
        self,
        input_shape: tuple[int, ...],
        n_filters: int | None = None,
        kernel_size: int = 2,
        n_blocks: int = 6,
        n_layers_per_block: int = 1,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        **kwargs,
    ) -> Model:
        """
        Build WaveNet model architecture.

        Args:
            input_shape: Shape of input data
            n_filters: Number of filters
            kernel_size: Kernel size
            n_blocks: Number of residual blocks
            n_layers_per_block: Layers per block
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            **kwargs: Additional arguments

        Returns:
            Keras Model
        """
        n_filters = n_filters or self.n_filters

        # Reshape input if needed
        if len(input_shape) == 1:
            input_shape = (input_shape[0], 1)

        # Input layer
        inputs = Input(shape=input_shape, name="input")

        # Initial causal convolution
        x = CausalConv1D(self.residual_channels, kernel_size, name="initial_causal_conv")(inputs)

        # WaveNet blocks
        skip_connections = []
        for i in range(n_blocks):
            for j in range(n_layers_per_block):
                dilation_rate = self.dilations[i % len(self.dilations)]
                x, skip = self._wavenet_residual_block(
                    x,
                    dilation_rate=dilation_rate,
                    filters=n_filters,
                    kernel_size=kernel_size,
                    name_prefix=f"block_{i}_layer_{j}",
                )
                skip_connections.append(skip)

        # Combine skip connections
        if len(skip_connections) > 1:
            out = Add(name="skip_connections")(skip_connections)
        else:
            out = skip_connections[0]

        # Final processing
        out = Activation("relu")(out)
        out = Conv1D(n_filters, 1, activation="relu", name="conv_1x1_1")(out)

        if dropout_rate > 0:
            out = Dropout(dropout_rate)(out)

        out = Conv1D(n_filters, 1, activation="relu", name="conv_1x1_2")(out)

        # Global average pooling
        out = keras.layers.GlobalAveragePooling1D()(out)

        # Output layer
        outputs = Dense(1, activation="sigmoid", name="output")(out)

        # Create model
        model = Model(inputs=inputs, outputs=outputs, name="WaveNet")

        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", "AUC"])

        self.model = model
        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Train WaveNet model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            **kwargs: Additional training arguments

        Returns:
            Training history
        """
        self.logger.info("Starting WaveNet training...")
        start_time = time.time()

        # Prepare data
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        if X_val is not None and len(X_val.shape) == 2:
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            validation_data = (X_val, y_val)
        else:
            validation_data = None

        # Build model if not already built
        if self.model is None:
            self.build_model(X_train.shape[1:])

        # Setup callbacks
        self._setup_callbacks()

        # Train model
        epochs = 100  # Fixed for WaveNet
        batch_size = 32

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
                monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
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
                checkpoint_path, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1
            )
            self.callbacks.append(checkpoint)

    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with WaveNet.

        Args:
            X: Input features

        Returns:
            Binary predictions
        """
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        predictions = self.model.predict(X, verbose=0)
        return (predictions > 0.5).astype(int).flatten()

    def _predict_proba_implementation(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities with WaveNet.

        Args:
            X: Input features

        Returns:
            Prediction probabilities
        """
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        proba = self.model.predict(X, verbose=0)

        # Convert to 2D array with probabilities for both classes
        proba_2d = np.zeros((len(proba), 2))
        proba_2d[:, 1] = proba.flatten()
        proba_2d[:, 0] = 1 - proba.flatten()

        return proba_2d

    def get_receptive_field(self) -> int:
        """Calculate the receptive field of the WaveNet model."""
        receptive_field = 1
        for dilation in self.dilations:
            receptive_field += (self.filter_width - 1) * dilation
        return receptive_field

    def _save_model_implementation(self, filepath: Path):
        """
        Save WaveNet model.

        Args:
            filepath: Path to save model
        """
        # Save model
        self.model.save(str(filepath.with_suffix(".h5")))

        # Save additional info
        info = {
            "n_filters": self.n_filters,
            "filter_width": self.filter_width,
            "dilations": self.dilations,
            "training_history": self.training_history,
        }

        with open(filepath.with_suffix(".json"), "w") as f:
            json.dump(info, f, indent=2)

    def _load_model_implementation(self, filepath: Path):
        """
        Load WaveNet model.

        Args:
            filepath: Path to load model from
        """
        # Custom objects for loading
        custom_objects = {"CausalConv1D": CausalConv1D}

        # Load model
        self.model = load_model(str(filepath.with_suffix(".h5")), custom_objects=custom_objects)

        # Load additional info
        info_path = filepath.with_suffix(".json")
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
                self.n_filters = info.get("n_filters", 32)
                self.filter_width = info.get("filter_width", 2)
                self.dilations = info.get("dilations", [1, 2, 4, 8, 16, 32])
                self.training_history = info.get("training_history", {})
