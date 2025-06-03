"""
Unified Training Manager for Weather Regulation Prediction Models

This module provides a comprehensive training interface that supports:
- Multiple model architectures
- Hyperparameter tuning
- Cross-validation
- Early stopping
- Model checkpointing
- Experiment tracking
- Distributed training support
"""

import os
import json
import pickle
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, 
    TensorBoard, CSVLogger
)
import mlflow
import mlflow.sklearn
import mlflow.keras
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import yaml
import joblib
from tqdm import tqdm

from config import ExperimentConfig, ModelType
from config_parser import ConfigParser


class TrainingCallback(ABC):
    """Abstract base class for custom training callbacks"""
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, logs: Dict[str, float]) -> None:
        pass
    
    @abstractmethod
    def on_training_end(self, logs: Dict[str, float]) -> None:
        pass


class ExperimentTracker:
    """Tracks experiments using MLflow"""
    
    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "file:./mlruns"
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(experiment_name)
        
    def start_run(self, run_name: str) -> str:
        """Start a new MLflow run"""
        mlflow.start_run(run_name=run_name)
        return mlflow.active_run().info.run_id
        
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters"""
        for key, value in params.items():
            mlflow.log_param(key, value)
            
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
            
    def log_model(self, model: Any, artifact_path: str, model_type: str) -> None:
        """Log model artifact"""
        if model_type in ['sklearn', 'random_forest', 'fnn']:
            mlflow.sklearn.log_model(model, artifact_path)
        else:
            mlflow.keras.log_model(model, artifact_path)
            
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log artifact file"""
        mlflow.log_artifact(local_path, artifact_path)
        
    def end_run(self) -> None:
        """End the current run"""
        mlflow.end_run()


class ModelCheckpointer:
    """Handles model checkpointing during training"""
    
    def __init__(self, checkpoint_dir: str, model_name: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        
    def get_keras_checkpoint_callback(self, monitor: str = 'val_loss') -> ModelCheckpoint:
        """Get Keras ModelCheckpoint callback"""
        filepath = self.checkpoint_dir / f"{self.model_name}_best.h5"
        return ModelCheckpoint(
            filepath=str(filepath),
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            mode='min' if 'loss' in monitor else 'max',
            verbose=1
        )
        
    def save_sklearn_model(self, model: Any, metrics: Dict[str, float]) -> str:
        """Save sklearn model with metrics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.checkpoint_dir / f"{self.model_name}_{timestamp}.pkl"
        
        checkpoint_data = {
            'model': model,
            'metrics': metrics,
            'timestamp': timestamp
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint_data, f)
            
        return str(filepath)
        
    def load_best_model(self, model_type: str) -> Any:
        """Load the best saved model"""
        if model_type in ['sklearn', 'random_forest', 'fnn']:
            # Find the latest checkpoint
            checkpoints = list(self.checkpoint_dir.glob(f"{self.model_name}_*.pkl"))
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found for {self.model_name}")
            
            latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
            with open(latest_checkpoint, 'rb') as f:
                data = pickle.load(f)
                return data['model']
        else:
            # Keras model
            filepath = self.checkpoint_dir / f"{self.model_name}_best.h5"
            if not filepath.exists():
                raise FileNotFoundError(f"No checkpoint found at {filepath}")
            return tf.keras.models.load_model(str(filepath))


class Trainer:
    """Unified trainer for all model types"""
    
    def __init__(self, config: ExperimentConfig, experiment_tracker: Optional[ExperimentTracker] = None):
        self.config = config
        self.experiment_tracker = experiment_tracker
        self.logger = self._setup_logger()
        self.checkpointer = ModelCheckpointer(
            checkpoint_dir=config.output_dir or "./checkpoints",
            model_name=config.experiment_name
        )
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def train(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              callbacks: Optional[List[TrainingCallback]] = None) -> Dict[str, Any]:
        """
        Train a model with the specified configuration
        
        Args:
            model: The model instance to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            callbacks: Additional custom callbacks
            
        Returns:
            Dictionary containing training history and metrics
        """
        self.logger.info(f"Starting training for {self.config.experiment_name}")
        
        # Start experiment tracking
        if self.experiment_tracker:
            run_id = self.experiment_tracker.start_run(
                run_name=f"{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            self.experiment_tracker.log_params(self._get_config_params())
        
        try:
            # Determine model type
            model_type = self._get_model_type(model)
            
            if model_type in ['sklearn', 'random_forest', 'fnn']:
                history = self._train_sklearn_model(model, X_train, y_train, X_val, y_val)
            else:
                history = self._train_keras_model(model, X_train, y_train, X_val, y_val, callbacks)
                
            # Evaluate final model
            if X_val is not None and y_val is not None:
                val_metrics = self._evaluate_model(model, X_val, y_val, model_type)
                history['val_metrics'] = val_metrics
                
                if self.experiment_tracker:
                    self.experiment_tracker.log_metrics(val_metrics)
                    
            # Save model
            if self.experiment_tracker:
                self.experiment_tracker.log_model(
                    model, 
                    artifact_path="model",
                    model_type=model_type
                )
                
            self.logger.info("Training completed successfully")
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
            
        finally:
            if self.experiment_tracker:
                self.experiment_tracker.end_run()
                
        return history
        
    def _train_sklearn_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: Optional[np.ndarray] = None, 
                            y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train sklearn-based models"""
        self.logger.info("Training sklearn model")
        
        # For models with partial_fit, we can implement mini-batch training
        if hasattr(model, 'partial_fit'):
            history = self._train_with_partial_fit(model, X_train, y_train, X_val, y_val)
        else:
            # Standard fit
            model.fit(X_train, y_train)
            history = {'training_completed': True}
            
        # Save checkpoint
        metrics = {}
        if X_val is not None and y_val is not None:
            metrics = self._evaluate_model(model, X_val, y_val, 'sklearn')
            
        checkpoint_path = self.checkpointer.save_sklearn_model(model, metrics)
        history['checkpoint_path'] = checkpoint_path
        
        return history
        
    def _train_with_partial_fit(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: Optional[np.ndarray] = None, 
                               y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train models that support partial_fit with mini-batches"""
        batch_size = self.config.training.batch_size
        n_epochs = self.config.training.epochs
        
        history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': []
        }
        
        n_batches = int(np.ceil(len(X_train) / batch_size))
        
        for epoch in range(n_epochs):
            # Shuffle data
            indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            epoch_loss = 0
            epoch_acc = 0
            
            # Mini-batch training
            for i in tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{n_epochs}"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_train))
                
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                # Get unique classes for this batch
                classes = np.unique(y_batch)
                if len(classes) < len(np.unique(y_train)):
                    # Skip batch if it doesn't contain all classes
                    continue
                    
                model.partial_fit(X_batch, y_batch, classes=np.unique(y_train))
                
                # Calculate batch metrics
                y_pred = model.predict(X_batch)
                batch_acc = np.mean(y_pred == y_batch)
                epoch_acc += batch_acc
                
            # Calculate epoch metrics
            epoch_acc /= n_batches
            history['accuracy'].append(epoch_acc)
            
            # Validation metrics
            if X_val is not None and y_val is not None:
                val_pred = model.predict(X_val)
                val_acc = np.mean(val_pred == y_val)
                history['val_accuracy'].append(val_acc)
                
                self.logger.info(
                    f"Epoch {epoch+1}/{n_epochs} - "
                    f"accuracy: {epoch_acc:.4f} - "
                    f"val_accuracy: {val_acc:.4f}"
                )
                
                # Early stopping check
                if self.config.training.early_stopping and epoch > 10:
                    if len(history['val_accuracy']) > 10:
                        recent_val_acc = history['val_accuracy'][-10:]
                        if max(recent_val_acc) == recent_val_acc[0]:
                            self.logger.info("Early stopping triggered")
                            break
                            
        return history
        
    def _train_keras_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                          custom_callbacks: Optional[List[TrainingCallback]] = None) -> Dict[str, Any]:
        """Train Keras-based models"""
        self.logger.info("Training Keras model")
        
        # Prepare callbacks
        callbacks = self._get_keras_callbacks()
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            batch_size=self.config.training.batch_size,
            epochs=self.config.training.epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        # Process custom callbacks
        if custom_callbacks:
            for callback in custom_callbacks:
                callback.on_training_end(history.history)
                
        return {
            'history': history.history,
            'model': model,
            'checkpoint_path': str(self.checkpointer.checkpoint_dir / f"{self.config.experiment_name}_best.h5")
        }
        
    def _get_keras_callbacks(self) -> List:
        """Get Keras callbacks based on configuration"""
        callbacks = []
        
        # Model checkpoint
        callbacks.append(self.checkpointer.get_keras_checkpoint_callback())
        
        # Early stopping
        if self.config.training.early_stopping:
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=self.config.training.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ))
            
        # Reduce learning rate
        callbacks.append(ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ))
        
        # TensorBoard
        if self.config.output_dir:
            log_dir = Path(self.config.output_dir) / "logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
            callbacks.append(TensorBoard(
                log_dir=str(log_dir),
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            ))
            
        # CSV logger
        if self.config.output_dir:
            csv_path = Path(self.config.output_dir) / f"{self.config.experiment_name}_training.csv"
            callbacks.append(CSVLogger(str(csv_path)))
            
        return callbacks
        
    def cross_validate(self, model_class: Any, X: np.ndarray, y: np.ndarray,
                      cv_folds: int = 5, **model_kwargs) -> Dict[str, Any]:
        """
        Perform cross-validation
        
        Args:
            model_class: Model class to instantiate
            X: Features
            y: Labels
            cv_folds: Number of CV folds
            **model_kwargs: Arguments to pass to model constructor
            
        Returns:
            Dictionary with CV results
        """
        self.logger.info(f"Starting {cv_folds}-fold cross-validation")
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'fold_models': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            self.logger.info(f"Training fold {fold + 1}/{cv_folds}")
            
            # Split data
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]
            
            # Create and train model
            model = model_class(**model_kwargs)
            
            # Train based on model type
            model_type = self._get_model_type(model)
            if model_type in ['sklearn', 'random_forest', 'fnn']:
                model.fit(X_train_fold, y_train_fold)
            else:
                # For Keras models, we need to compile first
                model.compile(
                    optimizer=self.config.training.optimizer,
                    loss=self.config.training.loss,
                    metrics=['accuracy']
                )
                model.fit(
                    X_train_fold, y_train_fold,
                    batch_size=self.config.training.batch_size,
                    epochs=self.config.training.epochs,
                    validation_data=(X_val_fold, y_val_fold),
                    verbose=0
                )
            
            # Evaluate
            metrics = self._evaluate_model(model, X_val_fold, y_val_fold, model_type)
            
            # Store results
            cv_results['accuracy'].append(metrics.get('accuracy', 0))
            cv_results['precision'].append(metrics.get('precision', 0))
            cv_results['recall'].append(metrics.get('recall', 0))
            cv_results['f1'].append(metrics.get('f1_score', 0))
            cv_results['fold_models'].append(model)
            
        # Calculate statistics
        cv_results['mean_accuracy'] = np.mean(cv_results['accuracy'])
        cv_results['std_accuracy'] = np.std(cv_results['accuracy'])
        cv_results['mean_precision'] = np.mean(cv_results['precision'])
        cv_results['std_precision'] = np.std(cv_results['precision'])
        cv_results['mean_recall'] = np.mean(cv_results['recall'])
        cv_results['std_recall'] = np.std(cv_results['recall'])
        cv_results['mean_f1'] = np.mean(cv_results['f1'])
        cv_results['std_f1'] = np.std(cv_results['f1'])
        
        self.logger.info(
            f"Cross-validation completed. "
            f"Mean accuracy: {cv_results['mean_accuracy']:.4f} "
            f"(+/- {cv_results['std_accuracy']:.4f})"
        )
        
        return cv_results
        
    def train_ensemble(self, models: List[Tuple[str, Any]], X_train: np.ndarray, y_train: np.ndarray,
                      X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                      n_jobs: int = -1) -> Dict[str, Any]:
        """
        Train multiple models in parallel for ensemble
        
        Args:
            models: List of (name, model) tuples
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            n_jobs: Number of parallel jobs (-1 for all cores)
            
        Returns:
            Dictionary with trained models and results
        """
        self.logger.info(f"Training ensemble with {len(models)} models")
        
        if n_jobs == -1:
            n_jobs = os.cpu_count()
            
        results = {
            'models': {},
            'metrics': {},
            'training_time': {}
        }
        
        def train_single_model(model_info: Tuple[str, Any]) -> Tuple[str, Any, Dict[str, float], float]:
            """Train a single model"""
            name, model = model_info
            start_time = datetime.now()
            
            self.logger.info(f"Training {name}")
            
            # Train model
            model_type = self._get_model_type(model)
            if model_type in ['sklearn', 'random_forest', 'fnn']:
                model.fit(X_train, y_train)
            else:
                model.compile(
                    optimizer=self.config.training.optimizer,
                    loss=self.config.training.loss,
                    metrics=['accuracy']
                )
                model.fit(
                    X_train, y_train,
                    batch_size=self.config.training.batch_size,
                    epochs=self.config.training.epochs,
                    validation_data=(X_val, y_val) if X_val is not None else None,
                    verbose=0
                )
                
            # Evaluate
            metrics = {}
            if X_val is not None and y_val is not None:
                metrics = self._evaluate_model(model, X_val, y_val, model_type)
                
            training_time = (datetime.now() - start_time).total_seconds()
            
            return name, model, metrics, training_time
            
        # Train models in parallel
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(train_single_model, model_info) for model_info in models]
            
            for future in futures:
                name, model, metrics, training_time = future.result()
                results['models'][name] = model
                results['metrics'][name] = metrics
                results['training_time'][name] = training_time
                
        self.logger.info("Ensemble training completed")
        
        return results
        
    def _evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray, 
                       model_type: str) -> Dict[str, float]:
        """Evaluate model performance"""
        # Get predictions
        if model_type in ['sklearn', 'random_forest', 'fnn']:
            y_pred = model.predict(X)
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X)[:, 1]
            else:
                y_proba = y_pred
        else:
            y_proba = model.predict(X).flatten()
            y_pred = (y_proba > 0.5).astype(int)
            
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y, y_pred, average='weighted', zero_division=0)
        }
        
        # Add AUC if binary classification
        if len(np.unique(y)) == 2:
            metrics['auc_roc'] = roc_auc_score(y, y_proba)
            
        return metrics
        
    def _get_model_type(self, model: Any) -> str:
        """Determine model type from instance"""
        model_class_name = model.__class__.__name__.lower()
        
        if 'randomforest' in model_class_name:
            return 'random_forest'
        elif 'mlp' in model_class_name:
            return 'fnn'
        elif hasattr(model, 'fit') and hasattr(model, 'predict') and not hasattr(model, 'compile'):
            return 'sklearn'
        else:
            return 'keras'
            
    def _get_config_params(self) -> Dict[str, Any]:
        """Extract parameters from config for logging"""
        params = {
            'experiment_name': self.config.experiment_name,
            'model_type': self.config.model_type.value if self.config.model_type else 'unknown',
            'batch_size': self.config.training.batch_size,
            'epochs': self.config.training.epochs,
            'learning_rate': self.config.training.learning_rate,
            'optimizer': self.config.training.optimizer.value,
            'early_stopping': self.config.training.early_stopping,
            'validation_split': self.config.training.validation_split
        }
        
        # Add model-specific params
        if hasattr(self.config, self.config.model_type.value):
            model_config = getattr(self.config, self.config.model_type.value)
            if model_config:
                for key, value in model_config.__dict__.items():
                    if not key.startswith('_'):
                        params[f"{self.config.model_type.value}_{key}"] = value
                        
        return params


class DistributedTrainer(Trainer):
    """Extended trainer with distributed training support"""
    
    def __init__(self, config: ExperimentConfig, experiment_tracker: Optional[ExperimentTracker] = None,
                 strategy: Optional[tf.distribute.Strategy] = None):
        super().__init__(config, experiment_tracker)
        self.strategy = strategy or self._get_distribution_strategy()
        
    def _get_distribution_strategy(self) -> tf.distribute.Strategy:
        """Get appropriate distribution strategy"""
        # Check for multiple GPUs
        gpus = tf.config.list_physical_devices('GPU')
        
        if len(gpus) > 1:
            self.logger.info(f"Using MirroredStrategy with {len(gpus)} GPUs")
            return tf.distribute.MirroredStrategy()
        elif len(gpus) == 1:
            self.logger.info("Using single GPU")
            return tf.distribute.OneDeviceStrategy(device="/gpu:0")
        else:
            self.logger.info("Using CPU")
            return tf.distribute.OneDeviceStrategy(device="/cpu:0")
            
    def train_distributed(self, model_fn: callable, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train model with distribution strategy
        
        Args:
            model_fn: Function that returns a compiled model
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Training history
        """
        self.logger.info("Starting distributed training")
        
        # Create model within strategy scope
        with self.strategy.scope():
            model = model_fn()
            
        # Create distributed datasets
        train_dataset = self._create_distributed_dataset(X_train, y_train)
        val_dataset = None
        if X_val is not None and y_val is not None:
            val_dataset = self._create_distributed_dataset(X_val, y_val)
            
        # Train using parent class method
        return self.train(model, train_dataset, None, val_dataset, None)
        
    def _create_distributed_dataset(self, X: np.ndarray, y: np.ndarray) -> tf.data.Dataset:
        """Create distributed dataset"""
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.batch(self.config.training.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        # Distribute dataset
        dataset = self.strategy.experimental_distribute_dataset(dataset)
        
        return dataset


def create_trainer(config: Union[str, Dict, ExperimentConfig], 
                  enable_tracking: bool = True,
                  distributed: bool = False) -> Trainer:
    """
    Factory function to create appropriate trainer
    
    Args:
        config: Configuration (path, dict, or ExperimentConfig)
        enable_tracking: Whether to enable experiment tracking
        distributed: Whether to use distributed training
        
    Returns:
        Trainer instance
    """
    # Load config if needed
    if isinstance(config, str):
        parser = ConfigParser()
        config = parser.load_config(config)
    elif isinstance(config, dict):
        parser = ConfigParser()
        config = parser.parse_config(config)
        
    # Create experiment tracker
    tracker = None
    if enable_tracking:
        tracker = ExperimentTracker(
            experiment_name=config.experiment_name,
            tracking_uri=config.output_dir
        )
        
    # Create trainer
    if distributed:
        return DistributedTrainer(config, tracker)
    else:
        return Trainer(config, tracker)