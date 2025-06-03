"""
Base model class for weather regulation prediction system.
Provides a standard interface for all models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split, cross_val_score
import logging

from config import ExperimentConfig


class ModelMetrics:
    """Container for model evaluation metrics."""
    
    def __init__(self):
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0
        self.auc_roc = 0.0
        self.confusion_matrix = None
        self.fpr = None
        self.tpr = None
        self.thresholds = None
        self.training_time = 0.0
        self.prediction_time = 0.0
        self.cv_scores = []
        self.feature_importance = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'auc_roc': self.auc_roc,
            'confusion_matrix': self.confusion_matrix.tolist() if self.confusion_matrix is not None else None,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            'cv_scores': self.cv_scores
        }
    
    def __str__(self) -> str:
        """String representation of metrics."""
        return (
            f"Accuracy: {self.accuracy:.4f}\n"
            f"Precision: {self.precision:.4f}\n"
            f"Recall: {self.recall:.4f}\n"
            f"F1-Score: {self.f1_score:.4f}\n"
            f"AUC-ROC: {self.auc_roc:.4f}\n"
            f"Training Time: {self.training_time:.2f}s\n"
            f"Prediction Time: {self.prediction_time:.2f}s"
        )


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, config: ExperimentConfig, model_type: str):
        """
        Initialize base model.
        
        Args:
            config: Experiment configuration
            model_type: Type of model (e.g., 'lstm', 'random_forest')
        """
        self.config = config
        self.model_type = model_type
        self.model_config = getattr(config, model_type)
        self.model = None
        self.metrics = ModelMetrics()
        self.is_trained = False
        self.training_history = {}
        self.best_params = None
        
        # Set up logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Set random seed for reproducibility
        self.random_state = config.data.random_state
        np.random.seed(self.random_state)
        
        # Output paths
        self.output_dir = Path(config.data.output_path) / model_type
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def build_model(self, input_shape: Tuple[int, ...], **kwargs) -> Any:
        """
        Build the model architecture.
        
        Args:
            input_shape: Shape of input data
            **kwargs: Additional arguments
            
        Returns:
            Built model
        """
        pass
    
    @abstractmethod
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional training arguments
            
        Returns:
            Training history
        """
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self._predict_implementation(X)
    
    @abstractmethod
    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        """
        Implementation-specific prediction method.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        pass
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self._predict_proba_implementation(X)
    
    def _predict_proba_implementation(self, X: np.ndarray) -> np.ndarray:
        """
        Implementation-specific probability prediction.
        Default implementation for models without probability support.
        
        Args:
            X: Input features
            
        Returns:
            Prediction probabilities
        """
        # Default: return binary predictions as probabilities
        predictions = self.predict(X)
        proba = np.zeros((len(predictions), 2))
        proba[:, 1] = predictions
        proba[:, 0] = 1 - predictions
        return proba
    
    def evaluate(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        save_plots: bool = True
    ) -> ModelMetrics:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            save_plots: Whether to save evaluation plots
            
        Returns:
            Model metrics
        """
        import time
        
        # Make predictions
        start_time = time.time()
        y_pred = self.predict(X_test)
        self.metrics.prediction_time = time.time() - start_time
        
        # Get probabilities if available
        try:
            y_proba = self.predict_proba(X_test)[:, 1]
        except:
            y_proba = y_pred
        
        # Calculate metrics
        self.metrics.accuracy = accuracy_score(y_test, y_pred)
        self.metrics.precision = precision_score(y_test, y_pred, zero_division=0)
        self.metrics.recall = recall_score(y_test, y_pred, zero_division=0)
        self.metrics.f1_score = f1_score(y_test, y_pred, zero_division=0)
        self.metrics.confusion_matrix = confusion_matrix(y_test, y_pred)
        
        # ROC curve
        try:
            self.metrics.auc_roc = roc_auc_score(y_test, y_proba)
            self.metrics.fpr, self.metrics.tpr, self.metrics.thresholds = roc_curve(y_test, y_proba)
        except:
            self.logger.warning("Could not calculate ROC metrics")
        
        # Save plots if requested
        if save_plots:
            self.plot_metrics()
        
        return self.metrics
    
    def cross_validate(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        cv: int = 5
    ) -> np.ndarray:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Labels
            cv: Number of folds
            
        Returns:
            Cross-validation scores
        """
        if not hasattr(self.model, 'fit'):
            raise NotImplementedError("Cross-validation not implemented for this model type")
        
        scores = cross_val_score(
            self.model, X, y, 
            cv=cv, 
            scoring='accuracy',
            n_jobs=-1
        )
        self.metrics.cv_scores = scores.tolist()
        return scores
    
    def plot_metrics(self):
        """Plot evaluation metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Confusion Matrix
        ax = axes[0, 0]
        sns.heatmap(
            self.metrics.confusion_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Inactive', 'Active'],
            yticklabels=['Inactive', 'Active'],
            ax=ax
        )
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        # ROC Curve
        ax = axes[0, 1]
        if self.metrics.fpr is not None:
            ax.plot(self.metrics.fpr, self.metrics.tpr, 'b-', 
                   label=f'ROC (AUC = {self.metrics.auc_roc:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Metrics Bar Chart
        ax = axes[1, 0]
        metrics_dict = {
            'Accuracy': self.metrics.accuracy,
            'Precision': self.metrics.precision,
            'Recall': self.metrics.recall,
            'F1-Score': self.metrics.f1_score
        }
        ax.bar(metrics_dict.keys(), metrics_dict.values())
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics')
        for i, (k, v) in enumerate(metrics_dict.items()):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        # Feature Importance (if available)
        ax = axes[1, 1]
        if hasattr(self, 'get_feature_importance'):
            importance = self.get_feature_importance()
            if importance is not None:
                # Show top 10 features
                top_features = pd.DataFrame({
                    'feature': range(len(importance)),
                    'importance': importance
                }).nlargest(10, 'importance')
                
                ax.barh(top_features['feature'], top_features['importance'])
                ax.set_xlabel('Importance')
                ax.set_ylabel('Feature Index')
                ax.set_title('Top 10 Feature Importances')
        else:
            ax.text(0.5, 0.5, 'Feature importance\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Importance')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'{self.model_type}_metrics.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Metrics plot saved to {plot_path}")
    
    def save_results(self, prefix: str = ''):
        """
        Save model results and metrics.
        
        Args:
            prefix: Prefix for output files
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save metrics
        metrics_path = self.output_dir / f'{prefix}{self.model_type}_metrics_{timestamp}.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics.to_dict(), f, indent=2)
        
        # Save training history
        if self.training_history:
            history_path = self.output_dir / f'{prefix}{self.model_type}_history_{timestamp}.json'
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
        
        # Save best parameters if available
        if self.best_params:
            params_path = self.output_dir / f'{prefix}{self.model_type}_best_params_{timestamp}.json'
            with open(params_path, 'w') as f:
                json.dump(self.best_params, f, indent=2)
        
        self.logger.info(f"Results saved to {self.output_dir}")
    
    def save_model(self, filepath: Optional[Path] = None):
        """
        Save trained model.
        
        Args:
            filepath: Optional custom filepath
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = self.output_dir / f'{self.model_type}_model_{timestamp}.pkl'
        
        self._save_model_implementation(filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    @abstractmethod
    def _save_model_implementation(self, filepath: Path):
        """Implementation-specific model saving."""
        pass
    
    def load_model(self, filepath: Path):
        """
        Load trained model.
        
        Args:
            filepath: Path to saved model
        """
        self._load_model_implementation(filepath)
        self.is_trained = True
        self.logger.info(f"Model loaded from {filepath}")
    
    @abstractmethod
    def _load_model_implementation(self, filepath: Path):
        """Implementation-specific model loading."""
        pass
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance if available.
        
        Returns:
            Feature importance array or None
        """
        return None
    
    def prepare_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Prepare data for training.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            X_train, X_test, y_train, y_test, X_val, y_val
        """
        # Split data
        test_size = self.config.data.test_size
        val_size = self.config.data.validation_size
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: train vs val (if validation size > 0)
        if val_size > 0:
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, 
                random_state=self.random_state, stratify=y_temp
            )
        else:
            X_train, y_train = X_temp, y_temp
            X_val, y_val = None, None
        
        return X_train, X_test, y_train, y_test, X_val, y_val
    
    def __repr__(self) -> str:
        """String representation of model."""
        return f"{self.__class__.__name__}(model_type={self.model_type}, trained={self.is_trained})"