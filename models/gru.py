"""
GRU (Gated Recurrent Unit) model implementation for weather regulation prediction.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import time
import json

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import GRU, Dense, Dropout, Bidirectional, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
import keras_tuner as kt

from .base_model import BaseModel
from config import ExperimentConfig, GRUConfig


class GRUModel(BaseModel):
    """GRU model for time series classification."""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize GRU model.
        
        Args:
            config: Experiment configuration
        """
        super().__init__(config, 'gru')
        self.model_config: GRUConfig = config.gru
        self.callbacks = []
        self.sequence_length = config.data.window_size
        
        # Set TensorFlow seed
        tf.random.set_seed(self.random_state)
    
    def build_model(
        self, 
        input_shape: Tuple[int, ...],
        units: Optional[int] = None,
        num_layers: Optional[int] = None,
        dropout_rate: Optional[float] = None,
        learning_rate: Optional[float] = None,
        bidirectional: Optional[bool] = None,
        recurrent_dropout: Optional[float] = None,
        **kwargs
    ) -> Sequential:
        """
        Build GRU model architecture.
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            units: Number of GRU units
            num_layers: Number of GRU layers
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            bidirectional: Whether to use bidirectional GRU
            recurrent_dropout: Recurrent dropout rate
            **kwargs: Additional arguments
            
        Returns:
            Keras Sequential model
        """
        # Use provided parameters or defaults from config
        units = units or self.model_config.units[0]
        num_layers = num_layers or self.model_config.num_layers[0]
        dropout_rate = dropout_rate or self.model_config.dropout_rate[0]
        learning_rate = learning_rate or self.model_config.learning_rate[0]
        bidirectional = bidirectional if bidirectional is not None else self.model_config.bidirectional
        recurrent_dropout = recurrent_dropout or self.model_config.recurrent_dropout[0]
        
        # Reshape input if needed
        if len(input_shape) == 1:
            input_shape = (self.sequence_length, input_shape[0])
        
        model = Sequential()
        
        # Add GRU layers
        for i in range(num_layers):
            return_sequences = i < num_layers - 1
            
            # Create GRU layer
            gru_layer = GRU(
                units=units,
                activation=self.model_config.activation,
                recurrent_activation=self.model_config.recurrent_activation,
                return_sequences=return_sequences,
                recurrent_dropout=recurrent_dropout,
                reset_after=True,  # GRU specific parameter
                name=f'gru_{i}'
            )
            
            # Add bidirectional wrapper if needed
            if bidirectional:
                gru_layer = Bidirectional(gru_layer, name=f'bidirectional_gru_{i}')
            
            # Add to model
            if i == 0:
                model.add(keras.layers.Input(shape=input_shape))
                model.add(gru_layer)
            else:
                model.add(gru_layer)
            
            # Add batch normalization (helps with gradient flow)
            model.add(BatchNormalization())
            
            # Add dropout
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate, name=f'dropout_{i}'))
        
        # Dense layers
        model.add(Dense(32, activation='relu', name='dense_1'))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate / 2, name='dropout_dense'))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid', name='output'))
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=self.model_config.loss,
            metrics=['accuracy', 'AUC', 'Precision', 'Recall']
        )
        
        self.model = model
        return model
    
    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare sequences for GRU input.
        
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
            X_seq[i] = X[i:i + self.sequence_length]
        
        # Adjust labels if provided
        if y is not None:
            y_seq = y[self.sequence_length - 1:]
            return X_seq, y_seq
        
        return X_seq, None
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train GRU model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            **kwargs: Additional training arguments
            
        Returns:
            Training history
        """
        self.logger.info("Starting GRU training...")
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
            X_train_seq, y_train_seq,
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
            'auc': history.history.get('auc', []),
            'precision': history.history.get('precision', []),
            'recall': history.history.get('recall', [])
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
        
        # Learning rate reduction
        if self.config.training.use_lr_scheduler:
            lr_scheduler = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
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
            model = Sequential()
            
            # Hyperparameters to tune
            units = hp.Choice('units', self.model_config.units)
            num_layers = hp.Choice('num_layers', self.model_config.num_layers)
            dropout_rate = hp.Choice('dropout_rate', self.model_config.dropout_rate)
            learning_rate = hp.Choice('learning_rate', self.model_config.learning_rate)
            bidirectional = hp.Boolean('bidirectional', default=self.model_config.bidirectional)
            recurrent_dropout = hp.Choice('recurrent_dropout', self.model_config.recurrent_dropout)
            
            # Build layers
            for i in range(num_layers):
                return_sequences = i < num_layers - 1
                
                gru_layer = GRU(
                    units=units,
                    activation=self.model_config.activation,
                    return_sequences=return_sequences,
                    recurrent_dropout=recurrent_dropout
                )
                
                if bidirectional:
                    gru_layer = Bidirectional(gru_layer)
                
                if i == 0:
                    model.add(gru_layer)
                else:
                    model.add(gru_layer)
                
                model.add(BatchNormalization())
                
                if dropout_rate > 0:
                    model.add(Dropout(dropout_rate))
            
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            
            # Compile
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(
                optimizer=optimizer,
                loss=self.model_config.loss,
                metrics=['accuracy']
            )
            
            return model
        
        # Choose tuner based on method
        if self.config.tuning_method == 'bayesian':
            tuner = kt.BayesianOptimization(
                build_model,
                objective='val_accuracy',
                max_trials=self.config.tuning_trials,
                directory=self.output_dir / 'tuning',
                project_name='gru_tuning'
            )
        elif self.config.tuning_method == 'random':
            tuner = kt.RandomSearch(
                build_model,
                objective='val_accuracy',
                max_trials=self.config.tuning_trials,
                directory=self.output_dir / 'tuning',
                project_name='gru_tuning'
            )
        else:  # grid search
            tuner = kt.GridSearch(
                build_model,
                objective='val_accuracy',
                directory=self.output_dir / 'tuning',
                project_name='gru_tuning'
            )
        
        # Search
        tuner.search(
            X_train, y_train,
            epochs=min(self.model_config.epochs),
            batch_size=self.model_config.batch_size[0],
            validation_data=validation_data,
            callbacks=[EarlyStopping(patience=5)] if self.config.training.use_early_stopping else [],
            verbose=1
        )
        
        # Get best hyperparameters
        best_hp = tuner.get_best_hyperparameters()[0]
        self.model = tuner.get_best_models()[0]
        
        # Convert to dictionary
        best_params = {
            'units': best_hp.get('units'),
            'num_layers': best_hp.get('num_layers'),
            'dropout_rate': best_hp.get('dropout_rate'),
            'learning_rate': best_hp.get('learning_rate'),
            'bidirectional': best_hp.get('bidirectional'),
            'recurrent_dropout': best_hp.get('recurrent_dropout')
        }
        
        self.logger.info(f"Best hyperparameters: {best_params}")
        
        return best_params
    
    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with GRU."""
        X_seq, _ = self._prepare_sequences(X)
        predictions = self.model.predict(X_seq, verbose=0)
        return (predictions > 0.5).astype(int).flatten()
    
    def _predict_proba_implementation(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities with GRU."""
        X_seq, _ = self._prepare_sequences(X)
        proba = self.model.predict(X_seq, verbose=0)
        
        proba_2d = np.zeros((len(proba), 2))
        proba_2d[:, 1] = proba.flatten()
        proba_2d[:, 0] = 1 - proba.flatten()
        
        return proba_2d
    
    def _save_model_implementation(self, filepath: Path):
        """Save GRU model."""
        self.model.save(str(filepath.with_suffix('.h5')))
        
        info = {
            'best_params': self.best_params,
            'sequence_length': self.sequence_length,
            'training_history': self.training_history
        }
        
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(info, f, indent=2)
    
    def _load_model_implementation(self, filepath: Path):
        """Load GRU model."""
        self.model = load_model(str(filepath.with_suffix('.h5')))
        
        info_path = filepath.with_suffix('.json')
        if info_path.exists():
            with open(info_path, 'r') as f:
                info = json.load(f)
                self.best_params = info.get('best_params')
                self.sequence_length = info.get('sequence_length', 1)
                self.training_history = info.get('training_history', {})