"""
Feedforward Neural Network (FNN) model implementation for weather regulation prediction.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import time
import pickle

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from .base_model import BaseModel
from config import ExperimentConfig


class FNNModel(BaseModel):
    """Feedforward Neural Network (Multi-layer Perceptron) model."""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize FNN model.
        
        Args:
            config: Experiment configuration
        """
        super().__init__(config, 'fnn')
        # FNN doesn't have dedicated config, use default parameters
        self.scaler = StandardScaler()
        self.is_scaled = False
    
    def build_model(
        self, 
        input_shape: Tuple[int, ...],
        hidden_layer_sizes: Tuple[int, ...] = (100, 50),
        activation: str = 'relu',
        solver: str = 'adam',
        alpha: float = 0.0001,
        learning_rate_init: float = 0.001,
        max_iter: int = 1000,
        **kwargs
    ) -> MLPClassifier:
        """
        Build FNN model.
        
        Args:
            input_shape: Shape of input data (not used directly)
            hidden_layer_sizes: Tuple of hidden layer sizes
            activation: Activation function
            solver: Optimization solver
            alpha: L2 regularization parameter
            learning_rate_init: Initial learning rate
            max_iter: Maximum iterations
            **kwargs: Additional arguments
            
        Returns:
            MLPClassifier instance
        """
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            shuffle=True,
            random_state=self.random_state,
            tol=0.0001,
            verbose=False,
            warm_start=False,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
        
        return self.model
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train FNN model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (not used - FNN uses internal validation)
            y_val: Validation labels (not used)
            **kwargs: Additional training arguments
            
        Returns:
            Training history
        """
        self.logger.info("Starting FNN training...")
        start_time = time.time()
        
        # Scale data
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.is_scaled = True
        
        # Build model if not already built
        if self.model is None:
            self.build_model(X_train.shape)
        
        # Perform hyperparameter tuning if enabled
        if self.config.hyperparameter_tuning:
            self.logger.info("Performing hyperparameter tuning...")
            self._tune_hyperparameters(X_train_scaled, y_train)
        else:
            # Direct training
            self.model.fit(X_train_scaled, y_train)
        
        # Record training time
        self.metrics.training_time = time.time() - start_time
        self.is_trained = True
        
        # Calculate training metrics
        train_score = self.model.score(X_train_scaled, y_train)
        self.training_history = {
            'train_accuracy': train_score,
            'training_time': self.metrics.training_time,
            'n_iter': self.model.n_iter_,
            'loss_curve': self.model.loss_curve_ if hasattr(self.model, 'loss_curve_') else []
        }
        
        # If validation data provided, calculate validation score
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_score = self.model.score(X_val_scaled, y_val)
            self.training_history['val_accuracy'] = val_score
        
        self.logger.info(f"Training completed in {self.metrics.training_time:.2f} seconds")
        self.logger.info(f"Training accuracy: {train_score:.4f}")
        self.logger.info(f"Iterations: {self.model.n_iter_}")
        
        return self.training_history
    
    def _tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Perform hyperparameter tuning for FNN.
        
        Args:
            X_train: Training features (already scaled)
            y_train: Training labels
        """
        # Define parameter grid
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100), (100, 50, 25)],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['adam', 'lbfgs', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01, 0.1]
        }
        
        # Choose search method
        if self.config.tuning_method == 'grid':
            # Limit grid search to most important parameters
            limited_param_grid = {
                'hidden_layer_sizes': [(50, 50), (100, 50), (100, 100)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001]
            }
            search = GridSearchCV(
                self.model,
                limited_param_grid,
                cv=3,  # Reduced CV folds for speed
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
        else:  # random search
            search = RandomizedSearchCV(
                self.model,
                param_grid,
                n_iter=min(self.config.tuning_trials, 20),
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state
            )
        
        # Perform search
        search.fit(X_train, y_train)
        
        # Store best parameters and model
        self.best_params = search.best_params_
        self.model = search.best_estimator_
        
        # Store CV results
        self.training_history['best_params'] = self.best_params
        self.training_history['best_score'] = search.best_score_
        
        self.logger.info(f"Best parameters: {self.best_params}")
        self.logger.info(f"Best CV score: {search.best_score_:.4f}")
    
    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with FNN.
        
        Args:
            X: Input features
            
        Returns:
            Binary predictions
        """
        if self.is_scaled:
            X = self.scaler.transform(X)
        return self.model.predict(X)
    
    def _predict_proba_implementation(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities with FNN.
        
        Args:
            X: Input features
            
        Returns:
            Prediction probabilities
        """
        if self.is_scaled:
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)
    
    def plot_loss_curve(self):
        """Plot training loss curve if available."""
        import matplotlib.pyplot as plt
        
        if not hasattr(self.model, 'loss_curve_') or not self.model.loss_curve_:
            self.logger.warning("No loss curve available")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.model.loss_curve_, label='Training Loss')
        if hasattr(self.model, 'validation_scores_'):
            plt.plot(self.model.validation_scores_, label='Validation Score')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('FNN Training Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = self.output_dir / 'loss_curve.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Loss curve saved to {plot_path}")
    
    def get_network_architecture(self) -> Dict[str, Any]:
        """Get information about network architecture."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        return {
            'n_layers': self.model.n_layers_,
            'hidden_layer_sizes': self.model.hidden_layer_sizes,
            'n_outputs': self.model.n_outputs_,
            'activation': self.model.activation,
            'solver': self.model.solver,
            'loss': self.model.loss_,
            'n_iter': self.model.n_iter_
        }
    
    def _save_model_implementation(self, filepath: Path):
        """
        Save FNN model.
        
        Args:
            filepath: Path to save model
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'is_scaled': self.is_scaled,
                'best_params': self.best_params
            }, f)
    
    def _load_model_implementation(self, filepath: Path):
        """
        Load FNN model.
        
        Args:
            filepath: Path to load model from
        """
        with open(filepath, 'rb') as f:
            saved_data = pickle.load(f)
            self.model = saved_data['model']
            self.scaler = saved_data['scaler']
            self.is_scaled = saved_data['is_scaled']
            self.best_params = saved_data.get('best_params')