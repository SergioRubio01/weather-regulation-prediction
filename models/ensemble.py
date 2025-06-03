"""
Ensemble model implementation for weather regulation prediction.
Combines multiple models for improved performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import time
import pickle
import json

from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
import joblib

from .base_model import BaseModel
from .random_forest import RandomForestModel
from .lstm import LSTMModel
from .cnn import CNNModel
from .gru import GRUModel
from .fnn import FNNModel
from config import ExperimentConfig, EnsembleConfig


class EnsembleModel(BaseModel):
    """Ensemble model combining multiple base models."""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize Ensemble model.
        
        Args:
            config: Experiment configuration
        """
        super().__init__(config, 'ensemble')
        self.model_config: EnsembleConfig = config.ensemble
        self.base_models = {}
        self.model_weights = None
        self.meta_learner = None
        self.use_stacking = self.model_config.use_stacking
        
        # Initialize base models
        self._initialize_base_models()
    
    def _initialize_base_models(self):
        """Initialize base models specified in configuration."""
        model_classes = {
            'random_forest': RandomForestModel,
            'lstm': LSTMModel,
            'cnn': CNNModel,
            'gru': GRUModel,
            'fnn': FNNModel
        }
        
        for model_name in self.model_config.base_models:
            if model_name in model_classes:
                self.logger.info(f"Initializing {model_name} for ensemble")
                self.base_models[model_name] = model_classes[model_name](self.config)
            else:
                self.logger.warning(f"Unknown model type: {model_name}")
    
    def build_model(self, input_shape: Tuple[int, ...], **kwargs):
        """
        Build ensemble model.
        
        Args:
            input_shape: Shape of input data
            **kwargs: Additional arguments
            
        Returns:
            Ensemble model (VotingClassifier or StackingClassifier)
        """
        if self.use_stacking:
            # Build stacking classifier
            self._build_stacking_model()
        else:
            # Build voting classifier
            self._build_voting_model()
        
        return self.model
    
    def _build_voting_model(self):
        """Build voting ensemble."""
        # For sklearn compatibility, we need sklearn models
        # This is a simplified version - in practice, you'd wrap neural networks
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        
        estimators = []
        
        for model_name in self.model_config.base_models:
            if model_name == 'random_forest':
                estimators.append((
                    model_name,
                    RandomForestClassifier(n_estimators=100, random_state=self.random_state)
                ))
            else:
                # Use MLP as proxy for neural networks
                estimators.append((
                    model_name,
                    MLPClassifier(
                        hidden_layer_sizes=(100, 50),
                        max_iter=1000,
                        random_state=self.random_state
                    )
                ))
        
        # Set weights if provided
        weights = self.model_config.weights
        
        self.model = VotingClassifier(
            estimators=estimators,
            voting=self.model_config.voting_type,
            weights=weights,
            n_jobs=-1
        )
    
    def _build_stacking_model(self):
        """Build stacking ensemble."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        
        estimators = []
        
        for model_name in self.model_config.base_models:
            if model_name == 'random_forest':
                estimators.append((
                    model_name,
                    RandomForestClassifier(n_estimators=100, random_state=self.random_state)
                ))
            else:
                estimators.append((
                    model_name,
                    MLPClassifier(
                        hidden_layer_sizes=(100, 50),
                        max_iter=1000,
                        random_state=self.random_state
                    )
                ))
        
        # Choose meta-learner
        if self.model_config.meta_learner == 'logistic_regression':
            final_estimator = LogisticRegression(random_state=self.random_state)
        elif self.model_config.meta_learner == 'gradient_boosting':
            final_estimator = GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state
            )
        else:
            final_estimator = LogisticRegression(random_state=self.random_state)
        
        self.model = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=self.model_config.cv_folds,
            n_jobs=-1
        )
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train ensemble model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            **kwargs: Additional training arguments
            
        Returns:
            Training history
        """
        self.logger.info("Starting Ensemble training...")
        start_time = time.time()
        
        # Option 1: Train each base model individually (more control)
        if self.config.training.save_best_model:
            self._train_base_models_individually(X_train, y_train, X_val, y_val)
        else:
            # Option 2: Use sklearn ensemble directly
            if self.model is None:
                self.build_model(X_train.shape)
            
            self.model.fit(X_train, y_train)
        
        # Record training time
        self.metrics.training_time = time.time() - start_time
        self.is_trained = True
        
        # Calculate training metrics
        train_score = self.score(X_train, y_train)
        self.training_history = {
            'train_accuracy': train_score,
            'training_time': self.metrics.training_time
        }
        
        # If validation data provided, calculate validation score
        if X_val is not None and y_val is not None:
            val_score = self.score(X_val, y_val)
            self.training_history['val_accuracy'] = val_score
        
        self.logger.info(f"Training completed in {self.metrics.training_time:.2f} seconds")
        self.logger.info(f"Training accuracy: {train_score:.4f}")
        
        return self.training_history
    
    def _train_base_models_individually(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """Train each base model individually for better control."""
        self.training_history['base_models'] = {}
        
        for model_name, model in self.base_models.items():
            self.logger.info(f"Training {model_name}...")
            
            # Train base model
            history = model.train(X_train, y_train, X_val, y_val)
            self.training_history['base_models'][model_name] = history
            
            # Evaluate base model
            if X_val is not None and y_val is not None:
                metrics = model.evaluate(X_val, y_val, save_plots=False)
                self.logger.info(
                    f"{model_name} validation accuracy: {metrics.accuracy:.4f}"
                )
        
        # If using stacking, train meta-learner
        if self.use_stacking:
            self._train_meta_learner(X_train, y_train, X_val, y_val)
    
    def _train_meta_learner(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """Train meta-learner for stacking ensemble."""
        self.logger.info("Training meta-learner...")
        
        # Get predictions from base models using cross-validation
        base_predictions = []
        
        for model_name, model in self.base_models.items():
            # Get out-of-fold predictions
            if hasattr(model.model, 'predict_proba'):
                # Use cross_val_predict to get out-of-fold predictions
                # This is a simplified version
                preds = model.predict_proba(X_train)[:, 1]
            else:
                preds = model.predict(X_train)
            
            base_predictions.append(preds)
        
        # Stack predictions
        X_meta = np.column_stack(base_predictions)
        
        # Train meta-learner
        if self.model_config.meta_learner == 'logistic_regression':
            self.meta_learner = LogisticRegression(random_state=self.random_state)
        elif self.model_config.meta_learner == 'gradient_boosting':
            self.meta_learner = GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state
            )
        
        self.meta_learner.fit(X_meta, y_train)
        
        # Evaluate on validation set
        if X_val is not None and y_val is not None:
            val_base_predictions = []
            for model_name, model in self.base_models.items():
                if hasattr(model.model, 'predict_proba'):
                    preds = model.predict_proba(X_val)[:, 1]
                else:
                    preds = model.predict(X_val)
                val_base_predictions.append(preds)
            
            X_val_meta = np.column_stack(val_base_predictions)
            val_score = self.meta_learner.score(X_val_meta, y_val)
            self.logger.info(f"Meta-learner validation accuracy: {val_score:.4f}")
    
    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with ensemble.
        
        Args:
            X: Input features
            
        Returns:
            Binary predictions
        """
        if hasattr(self, 'model') and self.model is not None:
            # Use sklearn ensemble
            return self.model.predict(X)
        else:
            # Use custom ensemble logic
            if self.use_stacking:
                return self._predict_stacking(X)
            else:
                return self._predict_voting(X)
    
    def _predict_voting(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using voting."""
        predictions = []
        weights = self.model_config.weights or [1] * len(self.base_models)
        
        for i, (model_name, model) in enumerate(self.base_models.items()):
            pred = model.predict(X)
            predictions.append(pred * weights[i])
        
        # Weighted voting
        weighted_sum = np.sum(predictions, axis=0)
        threshold = np.sum(weights) / 2
        
        return (weighted_sum > threshold).astype(int)
    
    def _predict_stacking(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using stacking."""
        # Get base model predictions
        base_predictions = []
        
        for model_name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                preds = model.predict_proba(X)[:, 1]
            else:
                preds = model.predict(X)
            base_predictions.append(preds)
        
        # Stack predictions
        X_meta = np.column_stack(base_predictions)
        
        # Use meta-learner
        return self.meta_learner.predict(X_meta)
    
    def _predict_proba_implementation(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities with ensemble.
        
        Args:
            X: Input features
            
        Returns:
            Prediction probabilities
        """
        if hasattr(self, 'model') and self.model is not None:
            # Use sklearn ensemble
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
        
        # Custom implementation
        if self.use_stacking:
            return self._predict_proba_stacking(X)
        else:
            return self._predict_proba_voting(X)
    
    def _predict_proba_voting(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using voting."""
        probas = []
        weights = self.model_config.weights or [1] * len(self.base_models)
        
        for i, (model_name, model) in enumerate(self.base_models.items()):
            proba = model.predict_proba(X)
            probas.append(proba * weights[i])
        
        # Weighted average
        weighted_proba = np.sum(probas, axis=0) / np.sum(weights)
        
        return weighted_proba
    
    def _predict_proba_stacking(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using stacking."""
        # Get base model predictions
        base_predictions = []
        
        for model_name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                preds = model.predict_proba(X)[:, 1]
            else:
                preds = model.predict(X)
            base_predictions.append(preds)
        
        # Stack predictions
        X_meta = np.column_stack(base_predictions)
        
        # Use meta-learner
        if hasattr(self.meta_learner, 'predict_proba'):
            return self.meta_learner.predict_proba(X_meta)
        else:
            # Convert predictions to probabilities
            preds = self.meta_learner.predict(X_meta)
            proba = np.zeros((len(preds), 2))
            proba[:, 1] = preds
            proba[:, 0] = 1 - preds
            return proba
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_base_model_scores(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Get individual scores for each base model.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Dictionary of model names and scores
        """
        scores = {}
        
        for model_name, model in self.base_models.items():
            if model.is_trained:
                pred = model.predict(X)
                score = np.mean(pred == y)
                scores[model_name] = score
        
        return scores
    
    def plot_model_contributions(self, X: np.ndarray, y: np.ndarray):
        """Plot contribution of each model to ensemble performance."""
        import matplotlib.pyplot as plt
        
        # Get individual model scores
        base_scores = self.get_base_model_scores(X, y)
        
        # Get ensemble score
        ensemble_score = self.score(X, y)
        
        # Prepare data for plotting
        models = list(base_scores.keys()) + ['ensemble']
        scores = list(base_scores.values()) + [ensemble_score]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, scores)
        
        # Highlight ensemble bar
        bars[-1].set_color('red')
        bars[-1].set_alpha(0.7)
        
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.title('Model Contributions to Ensemble')
        plt.ylim(0, 1.1)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'model_contributions.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Model contributions plot saved to {plot_path}")
    
    def _save_model_implementation(self, filepath: Path):
        """Save ensemble model."""
        save_dict = {
            'use_stacking': self.use_stacking,
            'model_config': self.model_config.__dict__,
            'training_history': self.training_history
        }
        
        # Save sklearn model if available
        if hasattr(self, 'model') and self.model is not None:
            save_dict['sklearn_model'] = self.model
        
        # Save meta-learner if using custom stacking
        if hasattr(self, 'meta_learner') and self.meta_learner is not None:
            save_dict['meta_learner'] = self.meta_learner
        
        # Save base models
        save_dict['base_model_paths'] = {}
        for model_name, model in self.base_models.items():
            if model.is_trained:
                model_path = filepath.parent / f'{model_name}_ensemble.pkl'
                model.save_model(model_path)
                save_dict['base_model_paths'][model_name] = str(model_path)
        
        # Save ensemble data
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
    
    def _load_model_implementation(self, filepath: Path):
        """Load ensemble model."""
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.use_stacking = save_dict['use_stacking']
        self.training_history = save_dict.get('training_history', {})
        
        # Load sklearn model if available
        if 'sklearn_model' in save_dict:
            self.model = save_dict['sklearn_model']
        
        # Load meta-learner
        if 'meta_learner' in save_dict:
            self.meta_learner = save_dict['meta_learner']
        
        # Load base models
        if 'base_model_paths' in save_dict:
            for model_name, model_path in save_dict['base_model_paths'].items():
                if model_name in self.base_models:
                    self.base_models[model_name].load_model(Path(model_path))