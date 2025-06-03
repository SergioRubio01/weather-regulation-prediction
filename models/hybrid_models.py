"""
Hybrid model implementations (CNN-RNN and CNN-LSTM) for weather regulation prediction.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import time
import json

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model, load_model
from keras.layers import (
    Conv1D, MaxPooling1D, LSTM, SimpleRNN, Dense, 
    Dropout, BatchNormalization, Flatten, Concatenate,
    Input, GlobalAveragePooling1D
)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam

from .base_model import BaseModel
from config import ExperimentConfig


class CNNRNNModel(BaseModel):
    """CNN-RNN hybrid model for time series classification."""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize CNN-RNN model.
        
        Args:
            config: Experiment configuration
        """
        super().__init__(config, 'cnn_rnn')
        # Use CNN config as base
        self.model_config = config.cnn
        self.callbacks = []
        
        # Set TensorFlow seed
        tf.random.set_seed(self.random_state)
    
    def build_model(
        self, 
        input_shape: Tuple[int, ...],
        cnn_filters: int = 64,
        kernel_size: int = 3,
        pool_size: int = 2,
        rnn_units: int = 50,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        **kwargs
    ) -> Sequential:
        """
        Build CNN-RNN hybrid model architecture.
        
        Args:
            input_shape: Shape of input data
            cnn_filters: Number of CNN filters
            kernel_size: CNN kernel size
            pool_size: Pooling size
            rnn_units: Number of RNN units
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            **kwargs: Additional arguments
            
        Returns:
            Keras Sequential model
        """
        # Reshape input if needed
        if len(input_shape) == 1:
            input_shape = (input_shape[0], 1)
        
        model = Sequential()
        
        # CNN layers
        model.add(Conv1D(
            filters=cnn_filters,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            input_shape=input_shape,
            name='cnn_1'
        ))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=pool_size))
        
        model.add(Conv1D(
            filters=cnn_filters * 2,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            name='cnn_2'
        ))
        model.add(BatchNormalization())
        
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
        
        # RNN layers
        model.add(SimpleRNN(
            units=rnn_units,
            return_sequences=True,
            name='rnn_1'
        ))
        model.add(SimpleRNN(
            units=rnn_units // 2,
            return_sequences=False,
            name='rnn_2'
        ))
        
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
        
        # Dense layers
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        self.model = model
        return model
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train CNN-RNN model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            **kwargs: Additional training arguments
            
        Returns:
            Training history
        """
        self.logger.info("Starting CNN-RNN training...")
        start_time = time.time()
        
        # Prepare data
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        if X_val is not None:
            if len(X_val.shape) == 2:
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
        epochs = 60
        batch_size = 32
        
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
        
        if self.config.training.use_early_stopping:
            self.callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ))
        
        if self.config.training.use_lr_scheduler:
            self.callbacks.append(ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                verbose=1
            ))
    
    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        predictions = self.model.predict(X, verbose=0)
        return (predictions > 0.5).astype(int).flatten()
    
    def _predict_proba_implementation(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        proba = self.model.predict(X, verbose=0)
        
        proba_2d = np.zeros((len(proba), 2))
        proba_2d[:, 1] = proba.flatten()
        proba_2d[:, 0] = 1 - proba.flatten()
        
        return proba_2d
    
    def _save_model_implementation(self, filepath: Path):
        """Save model."""
        self.model.save(str(filepath.with_suffix('.h5')))
        
        info = {
            'training_history': self.training_history
        }
        
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(info, f, indent=2)
    
    def _load_model_implementation(self, filepath: Path):
        """Load model."""
        self.model = load_model(str(filepath.with_suffix('.h5')))
        
        info_path = filepath.with_suffix('.json')
        if info_path.exists():
            with open(info_path, 'r') as f:
                info = json.load(f)
                self.training_history = info.get('training_history', {})


class CNNLSTMModel(BaseModel):
    """CNN-LSTM hybrid model for time series classification."""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize CNN-LSTM model.
        
        Args:
            config: Experiment configuration
        """
        super().__init__(config, 'cnn_lstm')
        # Use LSTM config as base
        self.model_config = config.lstm
        self.callbacks = []
        
        # Set TensorFlow seed
        tf.random.set_seed(self.random_state)
    
    def build_model(
        self, 
        input_shape: Tuple[int, ...],
        cnn_filters: int = 64,
        kernel_size: int = 3,
        pool_size: int = 2,
        lstm_units: int = 100,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        use_parallel: bool = True,
        **kwargs
    ) -> Model:
        """
        Build CNN-LSTM hybrid model architecture.
        
        Args:
            input_shape: Shape of input data
            cnn_filters: Number of CNN filters
            kernel_size: CNN kernel size
            pool_size: Pooling size
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            use_parallel: Whether to use parallel CNN and LSTM paths
            **kwargs: Additional arguments
            
        Returns:
            Keras Model
        """
        # Reshape input if needed
        if len(input_shape) == 1:
            input_shape = (input_shape[0], 1)
        
        inputs = Input(shape=input_shape)
        
        if use_parallel:
            # Parallel architecture: CNN and LSTM process input separately
            
            # CNN path
            cnn = Conv1D(cnn_filters, kernel_size, padding='same', activation='relu')(inputs)
            cnn = BatchNormalization()(cnn)
            cnn = MaxPooling1D(pool_size)(cnn)
            cnn = Conv1D(cnn_filters * 2, kernel_size, padding='same', activation='relu')(cnn)
            cnn = GlobalAveragePooling1D()(cnn)
            if dropout_rate > 0:
                cnn = Dropout(dropout_rate)(cnn)
            
            # LSTM path
            lstm = LSTM(lstm_units, return_sequences=True)(inputs)
            lstm = LSTM(lstm_units // 2)(lstm)
            if dropout_rate > 0:
                lstm = Dropout(dropout_rate)(lstm)
            
            # Merge paths
            merged = Concatenate()([cnn, lstm])
            
        else:
            # Sequential architecture: CNN feeds into LSTM
            
            # CNN layers
            x = Conv1D(cnn_filters, kernel_size, padding='same', activation='relu')(inputs)
            x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size)(x)
            x = Conv1D(cnn_filters * 2, kernel_size, padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            if dropout_rate > 0:
                x = Dropout(dropout_rate)(x)
            
            # LSTM layers
            x = LSTM(lstm_units, return_sequences=True)(x)
            x = LSTM(lstm_units // 2)(x)
            if dropout_rate > 0:
                x = Dropout(dropout_rate)(x)
            
            merged = x
        
        # Dense layers
        dense = Dense(64, activation='relu')(merged)
        if dropout_rate > 0:
            dense = Dropout(dropout_rate)(dense)
        outputs = Dense(1, activation='sigmoid')(dense)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='CNN_LSTM')
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        self.model = model
        return model
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train CNN-LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            **kwargs: Additional training arguments
            
        Returns:
            Training history
        """
        self.logger.info("Starting CNN-LSTM training...")
        start_time = time.time()
        
        # Prepare data
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        if X_val is not None:
            if len(X_val.shape) == 2:
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
        epochs = self.model_config.epochs[0] if hasattr(self.model_config, 'epochs') else 60
        batch_size = self.model_config.batch_size[0] if hasattr(self.model_config, 'batch_size') else 32
        
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
        
        if self.config.training.use_early_stopping:
            self.callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ))
        
        if self.config.training.use_lr_scheduler:
            self.callbacks.append(ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                verbose=1
            ))
        
        if self.config.training.save_best_model:
            self.callbacks.append(ModelCheckpoint(
                self.output_dir / 'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ))
    
    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        predictions = self.model.predict(X, verbose=0)
        return (predictions > 0.5).astype(int).flatten()
    
    def _predict_proba_implementation(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        proba = self.model.predict(X, verbose=0)
        
        proba_2d = np.zeros((len(proba), 2))
        proba_2d[:, 1] = proba.flatten()
        proba_2d[:, 0] = 1 - proba.flatten()
        
        return proba_2d
    
    def _save_model_implementation(self, filepath: Path):
        """Save model."""
        self.model.save(str(filepath.with_suffix('.h5')))
        
        info = {
            'training_history': self.training_history
        }
        
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(info, f, indent=2)
    
    def _load_model_implementation(self, filepath: Path):
        """Load model."""
        self.model = load_model(str(filepath.with_suffix('.h5')))
        
        info_path = filepath.with_suffix('.json')
        if info_path.exists():
            with open(info_path, 'r') as f:
                info = json.load(f)
                self.training_history = info.get('training_history', {})