"""
Autoencoder model implementation for weather regulation prediction.
Uses autoencoder for feature learning and anomaly detection.
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import BatchNormalization, Dense, Dropout, Input
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.regularizers import l1_l2

from src.config import AutoencoderConfig, ExperimentConfig

from .base_model import BaseModel


class AutoencoderModel(BaseModel):
    """Autoencoder model for feature extraction and classification."""

    def __init__(self, config: ExperimentConfig):
        """
        Initialize Autoencoder model.

        Args:
            config: Experiment configuration
        """
        super().__init__(config, "autoencoder")
        self.model_config: AutoencoderConfig = config.autoencoder
        self.callbacks = []

        # Autoencoder components
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.classifier = None

        # Set TensorFlow seed
        tf.random.set_seed(self.random_state)

    def build_model(
        self,
        input_shape: tuple[int, ...],
        encoding_dim: int | None = None,
        hidden_layers: list[int] | None = None,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        l1_reg: float = 0.0,
        l2_reg: float = 0.01,
        **kwargs,
    ) -> Model:
        """
        Build Autoencoder model architecture.

        Args:
            input_shape: Shape of input data
            encoding_dim: Dimension of encoding layer
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            l1_reg: L1 regularization
            l2_reg: L2 regularization
            **kwargs: Additional arguments

        Returns:
            Keras Model (classifier)
        """
        # Use provided parameters or defaults from config
        encoding_dim = encoding_dim or self.model_config.encoding_dim[0]
        hidden_layers = hidden_layers or self.model_config.hidden_layers[0]

        # Flatten input shape if needed
        if len(input_shape) > 1:
            input_dim = np.prod(input_shape)
        else:
            input_dim = input_shape[0]

        # Build encoder
        encoder_input = Input(shape=(input_dim,), name="encoder_input")
        x = encoder_input

        # Hidden layers for encoder
        for i, units in enumerate(hidden_layers):
            x = Dense(
                units,
                activation="relu",
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                name=f"encoder_hidden_{i}",
            )(x)
            x = BatchNormalization()(x)
            if dropout_rate > 0:
                x = Dropout(dropout_rate)(x)

        # Encoding layer
        encoded = Dense(
            encoding_dim,
            activation="relu",
            kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
            name="encoding",
        )(x)

        self.encoder = Model(encoder_input, encoded, name="encoder")

        # Build decoder
        decoder_input = Input(shape=(encoding_dim,), name="decoder_input")
        x = decoder_input

        # Hidden layers for decoder (reverse of encoder)
        for i, units in enumerate(reversed(hidden_layers)):
            x = Dense(
                units,
                activation="relu",
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                name=f"decoder_hidden_{i}",
            )(x)
            x = BatchNormalization()(x)
            if dropout_rate > 0:
                x = Dropout(dropout_rate)(x)

        # Output layer
        decoded = Dense(
            input_dim, activation=self.model_config.output_activation, name="reconstruction"
        )(x)

        self.decoder = Model(decoder_input, decoded, name="decoder")

        # Build full autoencoder
        autoencoder_input = Input(shape=(input_dim,), name="autoencoder_input")
        encoded_repr = self.encoder(autoencoder_input)
        decoded_output = self.decoder(encoded_repr)

        self.autoencoder = Model(autoencoder_input, decoded_output, name="autoencoder")

        # Compile autoencoder
        self.autoencoder.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=self.model_config.loss,
            metrics=["mae"],
        )

        # Build classifier on top of encoder
        if self.model_config.use_for_feature_extraction:
            classifier_input = Input(shape=(input_dim,), name="classifier_input")

            # Use pre-trained encoder for feature extraction
            encoded_features = self.encoder(classifier_input)

            # Classification layers
            x = Dense(32, activation="relu", name="classifier_hidden")(encoded_features)
            x = Dropout(dropout_rate)(x)
            x = Dense(16, activation="relu", name="classifier_hidden_2")(x)
            output = Dense(1, activation="sigmoid", name="classifier_output")(x)

            self.classifier = Model(classifier_input, output, name="classifier")

            # Compile classifier
            self.classifier.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss="binary_crossentropy",
                metrics=["accuracy", "AUC"],
            )

            # Return classifier as main model
            self.model = self.classifier
        else:
            # Return autoencoder as main model
            self.model = self.autoencoder

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
        Train Autoencoder model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            **kwargs: Additional training arguments

        Returns:
            Training history
        """
        self.logger.info("Starting Autoencoder training...")
        start_time = time.time()

        # Flatten data if needed
        if len(X_train.shape) > 2:
            X_train = X_train.reshape(X_train.shape[0], -1)
        if X_val is not None and len(X_val.shape) > 2:
            X_val = X_val.reshape(X_val.shape[0], -1)

        # Build model if not already built
        if self.model is None:
            self.build_model((X_train.shape[1],))

        # Setup callbacks
        self._setup_callbacks()

        # Get training parameters
        epochs = self.model_config.epochs[0]
        batch_size = self.model_config.batch_size[0]

        # Step 1: Pre-train autoencoder (unsupervised)
        self.logger.info("Pre-training autoencoder...")

        # For autoencoder training, we use X as both input and output
        ae_history = self.autoencoder.fit(
            X_train,
            X_train,
            validation_data=(X_val, X_val) if X_val is not None else None,
            epochs=epochs // 2,  # Use half epochs for pre-training
            batch_size=batch_size,
            callbacks=self.callbacks,
            verbose=1,
        )

        # Step 2: Train classifier if using for feature extraction
        if self.model_config.use_for_feature_extraction:
            self.logger.info("Training classifier with encoded features...")

            # Freeze encoder weights initially if not fine-tuning
            if not self.model_config.fine_tune_classifier:
                for layer in self.encoder.layers:
                    layer.trainable = False

            # Train classifier
            validation_data = (X_val, y_val) if X_val is not None else None

            clf_history = self.classifier.fit(
                X_train,
                y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=self.callbacks,
                verbose=1,
            )

            # Fine-tune if enabled
            if (
                self.model_config.fine_tune_classifier
                and not self.model_config.fine_tune_classifier
            ):
                self.logger.info("Fine-tuning encoder and classifier...")

                # Unfreeze encoder
                for layer in self.encoder.layers:
                    layer.trainable = True

                # Recompile with lower learning rate
                self.classifier.compile(
                    optimizer=Adam(learning_rate=0.0001),
                    loss="binary_crossentropy",
                    metrics=["accuracy", "AUC"],
                )

                # Fine-tune
                self.classifier.fit(
                    X_train,
                    y_train,
                    validation_data=validation_data,
                    epochs=epochs // 4,
                    batch_size=batch_size,
                    callbacks=self.callbacks,
                    verbose=1,
                )

            # Combine histories
            history = clf_history
        else:
            history = ae_history

        # Record training time
        self.metrics.training_time = time.time() - start_time
        self.is_trained = True

        # Store training history
        self.training_history = {
            "loss": history.history["loss"],
            "autoencoder_loss": ae_history.history["loss"],
        }

        if self.model_config.use_for_feature_extraction:
            self.training_history["accuracy"] = history.history.get("accuracy", [])
            self.training_history["auc"] = history.history.get("auc", [])

        if validation_data is not None:
            self.training_history["val_loss"] = history.history.get("val_loss", [])
            if self.model_config.use_for_feature_extraction:
                self.training_history["val_accuracy"] = history.history.get("val_accuracy", [])
                self.training_history["val_auc"] = history.history.get("val_auc", [])

        self.logger.info(f"Training completed in {self.metrics.training_time:.2f} seconds")

        return self.training_history

    def _setup_callbacks(self):
        """Setup training callbacks."""
        self.callbacks = []

        # Early stopping
        if self.config.training.use_early_stopping:
            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=self.config.training.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            )
            self.callbacks.append(early_stopping)

        # Learning rate reduction
        if self.config.training.use_lr_scheduler:
            lr_scheduler = ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
            )
            self.callbacks.append(lr_scheduler)

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encode input data to latent representation.

        Args:
            X: Input data

        Returns:
            Encoded representation
        """
        if self.encoder is None:
            raise ValueError("Encoder not built yet")

        # Flatten if needed
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)

        return self.encoder.predict(X, verbose=0)

    def decode(self, encoded: np.ndarray) -> np.ndarray:
        """
        Decode latent representation to reconstructed data.

        Args:
            encoded: Encoded representation

        Returns:
            Reconstructed data
        """
        if self.decoder is None:
            raise ValueError("Decoder not built yet")

        return self.decoder.predict(encoded, verbose=0)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstruct input data through autoencoder.

        Args:
            X: Input data

        Returns:
            Reconstructed data
        """
        if self.autoencoder is None:
            raise ValueError("Autoencoder not built yet")

        # Flatten if needed
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)

        return self.autoencoder.predict(X, verbose=0)

    def compute_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error for anomaly detection.

        Args:
            X: Input data

        Returns:
            Reconstruction error per sample
        """
        reconstructed = self.reconstruct(X)

        # Compute MSE per sample
        mse = np.mean((X.reshape(X.shape[0], -1) - reconstructed) ** 2, axis=1)

        return mse

    def detect_anomalies(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """
        Detect anomalies based on reconstruction error.

        Args:
            X: Input data
            threshold: Anomaly threshold (if None, use 95th percentile)

        Returns:
            Binary array indicating anomalies
        """
        errors = self.compute_reconstruction_error(X)

        if threshold is None:
            # Use 95th percentile as threshold
            threshold = np.percentile(errors, 95)

        return errors > threshold

    def visualize_reconstruction(self, X: np.ndarray, n_samples: int = 5):
        """
        Visualize original vs reconstructed samples.

        Args:
            X: Input data
            n_samples: Number of samples to visualize
        """
        import matplotlib.pyplot as plt

        # Get reconstructions
        reconstructed = self.reconstruct(X[:n_samples])

        # Flatten for visualization
        X_flat = X[:n_samples].reshape(n_samples, -1)

        fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_samples):
            # Original
            axes[i, 0].plot(X_flat[i])
            axes[i, 0].set_title(f"Original Sample {i+1}")
            axes[i, 0].grid(True, alpha=0.3)

            # Reconstructed
            axes[i, 1].plot(reconstructed[i])
            axes[i, 1].set_title(f"Reconstructed Sample {i+1}")
            axes[i, 1].grid(True, alpha=0.3)

            # Add MSE
            mse = np.mean((X_flat[i] - reconstructed[i]) ** 2)
            axes[i, 1].text(
                0.02,
                0.98,
                f"MSE: {mse:.4f}",
                transform=axes[i, 1].transAxes,
                verticalalignment="top",
            )

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / "reconstruction_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Reconstruction comparison saved to {plot_path}")

    def visualize_latent_space(self, X: np.ndarray, y: np.ndarray):
        """
        Visualize the latent space representation.

        Args:
            X: Input data
            y: Labels for coloring
        """
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        # Encode data
        encoded = self.encode(X)

        # If encoding dim > 2, use PCA
        if encoded.shape[1] > 2:
            pca = PCA(n_components=2)
            encoded_2d = pca.fit_transform(encoded)
            explained_var = pca.explained_variance_ratio_
        else:
            encoded_2d = encoded
            explained_var = [1.0, 0.0]

        # Create scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            encoded_2d[:, 0],
            encoded_2d[:, 1],
            c=y,
            cmap="viridis",
            alpha=0.6,
            edgecolors="k",
            linewidth=0.5,
        )
        plt.colorbar(scatter, label="Class")
        plt.xlabel(f"Latent Dim 1 ({explained_var[0]:.2%} var)")
        plt.ylabel(f"Latent Dim 2 ({explained_var[1]:.2%} var)")
        plt.title("Latent Space Representation")
        plt.grid(True, alpha=0.3)

        # Save plot
        plot_path = self.output_dir / "latent_space.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Latent space visualization saved to {plot_path}")

    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with autoencoder-based classifier."""
        if not self.model_config.use_for_feature_extraction:
            raise ValueError("Autoencoder not configured for classification")

        # Flatten if needed
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)

        predictions = self.classifier.predict(X, verbose=0)
        return (predictions > 0.5).astype(int).flatten()

    def _predict_proba_implementation(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities with autoencoder-based classifier."""
        if not self.model_config.use_for_feature_extraction:
            raise ValueError("Autoencoder not configured for classification")

        # Flatten if needed
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)

        proba = self.classifier.predict(X, verbose=0)

        proba_2d = np.zeros((len(proba), 2))
        proba_2d[:, 1] = proba.flatten()
        proba_2d[:, 0] = 1 - proba.flatten()

        return proba_2d

    def _save_model_implementation(self, filepath: Path):
        """Save autoencoder model components."""
        # Save encoder
        self.encoder.save(str(filepath.with_suffix("_encoder.h5")))

        # Save decoder
        self.decoder.save(str(filepath.with_suffix("_decoder.h5")))

        # Save autoencoder
        self.autoencoder.save(str(filepath.with_suffix("_autoencoder.h5")))

        # Save classifier if exists
        if self.classifier is not None:
            self.classifier.save(str(filepath.with_suffix("_classifier.h5")))

        # Save additional info
        info = {
            "model_config": self.model_config.__dict__,
            "training_history": self.training_history,
            "use_for_feature_extraction": self.model_config.use_for_feature_extraction,
        }

        with open(filepath.with_suffix(".json"), "w") as f:
            json.dump(info, f, indent=2)

    def _load_model_implementation(self, filepath: Path):
        """Load autoencoder model components."""
        # Load encoder
        self.encoder = load_model(str(filepath.with_suffix("_encoder.h5")))

        # Load decoder
        self.decoder = load_model(str(filepath.with_suffix("_decoder.h5")))

        # Load autoencoder
        self.autoencoder = load_model(str(filepath.with_suffix("_autoencoder.h5")))

        # Load classifier if exists
        classifier_path = filepath.with_suffix("_classifier.h5")
        if classifier_path.exists():
            self.classifier = load_model(str(classifier_path))
            self.model = self.classifier
        else:
            self.model = self.autoencoder

        # Load additional info
        info_path = filepath.with_suffix(".json")
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
                self.training_history = info.get("training_history", {})
