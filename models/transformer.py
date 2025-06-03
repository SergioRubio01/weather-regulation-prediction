"""
Transformer model implementation for weather regulation prediction.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import time
import json

import tensorflow as tf
from tensorflow import keras
from keras.models import Model, load_model
from keras.layers import (
    Input, Dense, Dropout, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, Embedding, Layer
)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
import keras_tuner as kt

from .base_model import BaseModel
from config import ExperimentConfig, TransformerConfig


class TransformerBlock(Layer):
    """Transformer block with multi-head attention and feed-forward network."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )
        self.dropout1 = Dropout(dropout_rate)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        
        self.ff1 = Dense(d_ff, activation='relu')
        self.ff2 = Dense(d_model)
        self.dropout2 = Dropout(dropout_rate)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
    
    def call(self, inputs, training=None):
        # Multi-head attention
        attn_output = self.attention(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network
        ff_output = self.ff1(out1)
        ff_output = self.ff2(ff_output)
        ff_output = self.dropout2(ff_output, training=training)
        out2 = self.layernorm2(out1 + ff_output)
        
        return out2
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate
        })
        return config


class PositionalEncoding(Layer):
    """Positional encoding layer for transformer."""
    
    def __init__(self, position, d_model, **kwargs):
        super().__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)
    
    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles
    
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )
        
        # Apply sin to even indices
        sines = tf.math.sin(angle_rads[:, 0::2])
        
        # Apply cos to odd indices
        cosines = tf.math.cos(angle_rads[:, 1::2])
        
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return tf.cast(pos_encoding, tf.float32)
    
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class TransformerModel(BaseModel):
    """Transformer model for time series classification."""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize Transformer model.
        
        Args:
            config: Experiment configuration
        """
        super().__init__(config, 'transformer')
        self.model_config: TransformerConfig = config.transformer
        self.callbacks = []
        
        # Set TensorFlow seed
        tf.random.set_seed(self.random_state)
    
    def build_model(
        self, 
        input_shape: Tuple[int, ...],
        d_model: Optional[int] = None,
        num_heads: Optional[int] = None,
        num_layers: Optional[int] = None,
        d_ff: Optional[int] = None,
        dropout_rate: Optional[float] = None,
        learning_rate: Optional[float] = None,
        use_positional_encoding: Optional[bool] = None,
        **kwargs
    ) -> Model:
        """
        Build Transformer model architecture.
        
        Args:
            input_shape: Shape of input data
            d_model: Dimension of the model
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Dimension of feed-forward network
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            use_positional_encoding: Whether to use positional encoding
            **kwargs: Additional arguments
            
        Returns:
            Keras Model
        """
        # Use provided parameters or defaults from config
        d_model = d_model or self.model_config.d_model[0]
        num_heads = num_heads or self.model_config.num_heads[0]
        num_layers = num_layers or self.model_config.num_layers[0]
        d_ff = d_ff or self.model_config.d_ff[0]
        dropout_rate = dropout_rate or self.model_config.dropout_rate[0]
        learning_rate = learning_rate or self.model_config.learning_rate[0]
        use_positional_encoding = use_positional_encoding if use_positional_encoding is not None else self.model_config.use_positional_encoding
        
        # Input layer
        inputs = Input(shape=input_shape, name='input')
        
        # Project input to d_model dimensions
        x = Dense(d_model, name='input_projection')(inputs)
        
        # Add positional encoding
        if use_positional_encoding:
            x = PositionalEncoding(input_shape[0], d_model)(x)
        
        # Add transformer blocks
        for i in range(num_layers):
            x = TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_rate=dropout_rate,
                name=f'transformer_block_{i}'
            )(x)
        
        # Global average pooling
        x = GlobalAveragePooling1D(name='global_avg_pool')(x)
        
        # Classification head
        x = Dense(64, activation='relu', name='dense_1')(x)
        x = Dropout(dropout_rate, name='dropout_dense')(x)
        x = Dense(32, activation='relu', name='dense_2')(x)
        outputs = Dense(1, activation='sigmoid', name='output')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='Transformer')
        
        # Compile with custom learning rate schedule if specified
        if self.model_config.warmup_steps > 0:
            learning_rate = self._create_learning_rate_schedule(
                d_model, self.model_config.warmup_steps
            )
        
        optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        model.compile(
            optimizer=optimizer,
            loss=self.model_config.loss,
            metrics=['accuracy', 'AUC']
        )
        
        self.model = model
        return model
    
    def _create_learning_rate_schedule(self, d_model: int, warmup_steps: int = 4000):
        """Create learning rate schedule with warmup."""
        class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, d_model, warmup_steps=4000):
                super().__init__()
                self.d_model = d_model
                self.d_model = tf.cast(self.d_model, tf.float32)
                self.warmup_steps = warmup_steps
            
            def __call__(self, step):
                arg1 = tf.math.rsqrt(step)
                arg2 = step * (self.warmup_steps ** -1.5)
                return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        
        return CustomSchedule(d_model, warmup_steps)
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train Transformer model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            **kwargs: Additional training arguments
            
        Returns:
            Training history
        """
        self.logger.info("Starting Transformer training...")
        start_time = time.time()
        
        # Validation data
        validation_data = (X_val, y_val) if X_val is not None else None
        
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
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.callbacks,
            verbose=1
        )
        
        # Record training time
        self.metrics.training_time = time.time() - start_time
        self.is_trained = True
        
        # Store training history
        self.training_history = {
            'loss': history.history['loss'],
            'accuracy': history.history['accuracy'],
            'auc': history.history.get('auc', [])
        }
        
        if validation_data is not None:
            self.training_history['val_loss'] = history.history['val_loss']
            self.training_history['val_accuracy'] = history.history['val_accuracy']
            self.training_history['val_auc'] = history.history.get('val_auc', [])
        
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
                verbose=1
            )
            self.callbacks.append(early_stopping)
        
        # Learning rate reduction (only if not using warmup schedule)
        if self.config.training.use_lr_scheduler and self.model_config.warmup_steps == 0:
            lr_scheduler = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )
            self.callbacks.append(lr_scheduler)
        
        # Model checkpointing
        if self.config.training.save_best_model:
            checkpoint_path = self.output_dir / 'best_model.h5'
            checkpoint = ModelCheckpoint(
                checkpoint_path,
                monitor=self.config.training.checkpoint_monitor,
                save_best_only=True,
                mode=self.config.training.checkpoint_mode,
                verbose=1
            )
            self.callbacks.append(checkpoint)
    
    def _tune_hyperparameters(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Dict[str, Any]:
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
            # Hyperparameters to tune
            d_model = hp.Choice('d_model', self.model_config.d_model)
            num_heads = hp.Choice('num_heads', self.model_config.num_heads)
            num_layers = hp.Choice('num_layers', self.model_config.num_layers)
            d_ff = hp.Choice('d_ff', self.model_config.d_ff)
            dropout_rate = hp.Choice('dropout_rate', self.model_config.dropout_rate)
            learning_rate = hp.Choice('learning_rate', self.model_config.learning_rate)
            
            # Build model
            return self.build_model(
                input_shape=X_train.shape[1:],
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                d_ff=d_ff,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate
            )
        
        # Choose tuner
        if self.config.tuning_method == 'bayesian':
            tuner = kt.BayesianOptimization(
                build_model,
                objective='val_accuracy',
                max_trials=self.config.tuning_trials,
                directory=self.output_dir / 'tuning',
                project_name='transformer_tuning'
            )
        elif self.config.tuning_method == 'random':
            tuner = kt.RandomSearch(
                build_model,
                objective='val_accuracy',
                max_trials=self.config.tuning_trials,
                directory=self.output_dir / 'tuning',
                project_name='transformer_tuning'
            )
        else:  # grid search
            tuner = kt.GridSearch(
                build_model,
                objective='val_accuracy',
                directory=self.output_dir / 'tuning',
                project_name='transformer_tuning'
            )
        
        # Search
        tuner.search(
            X_train, y_train,
            epochs=min(self.model_config.epochs),
            batch_size=self.model_config.batch_size[0],
            validation_data=validation_data,
            callbacks=[EarlyStopping(patience=10)] if self.config.training.use_early_stopping else [],
            verbose=1
        )
        
        # Get best hyperparameters
        best_hp = tuner.get_best_hyperparameters()[0]
        self.model = tuner.get_best_models()[0]
        
        # Convert to dictionary
        best_params = {
            'd_model': best_hp.get('d_model'),
            'num_heads': best_hp.get('num_heads'),
            'num_layers': best_hp.get('num_layers'),
            'd_ff': best_hp.get('d_ff'),
            'dropout_rate': best_hp.get('dropout_rate'),
            'learning_rate': best_hp.get('learning_rate')
        }
        
        self.logger.info(f"Best hyperparameters: {best_params}")
        
        return best_params
    
    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with Transformer."""
        predictions = self.model.predict(X, verbose=0)
        return (predictions > 0.5).astype(int).flatten()
    
    def _predict_proba_implementation(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities with Transformer."""
        proba = self.model.predict(X, verbose=0)
        
        proba_2d = np.zeros((len(proba), 2))
        proba_2d[:, 1] = proba.flatten()
        proba_2d[:, 0] = 1 - proba.flatten()
        
        return proba_2d
    
    def get_attention_weights(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get attention weights from all transformer blocks.
        
        Args:
            X: Input data
            
        Returns:
            Dictionary with attention weights for each layer
        """
        attention_weights = {}
        
        # Create intermediate models to extract attention weights
        for i, layer in enumerate(self.model.layers):
            if 'transformer_block' in layer.name:
                # Get attention layer output
                attention_model = Model(
                    inputs=self.model.input,
                    outputs=layer.attention.output
                )
                weights = attention_model.predict(X)
                attention_weights[f'layer_{i}'] = weights
        
        return attention_weights
    
    def _save_model_implementation(self, filepath: Path):
        """Save Transformer model."""
        # Custom objects for loading
        custom_objects = {
            'TransformerBlock': TransformerBlock,
            'PositionalEncoding': PositionalEncoding
        }
        
        # Save model with custom objects
        self.model.save(str(filepath.with_suffix('.h5')), save_traces=False)
        
        # Save additional info
        info = {
            'best_params': self.best_params,
            'training_history': self.training_history,
            'custom_objects': list(custom_objects.keys())
        }
        
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(info, f, indent=2)
    
    def _load_model_implementation(self, filepath: Path):
        """Load Transformer model."""
        # Custom objects for loading
        custom_objects = {
            'TransformerBlock': TransformerBlock,
            'PositionalEncoding': PositionalEncoding
        }
        
        # Load model
        self.model = load_model(str(filepath.with_suffix('.h5')), custom_objects=custom_objects)
        
        # Load additional info
        info_path = filepath.with_suffix('.json')
        if info_path.exists():
            with open(info_path, 'r') as f:
                info = json.load(f)
                self.best_params = info.get('best_params')
                self.training_history = info.get('training_history', {})