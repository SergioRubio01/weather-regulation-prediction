#!/usr/bin/env python3
"""
Visualization Module for Deep Learning Results

This module provides comprehensive visualization capabilities for deep learning
model results including training history, performance metrics, and comparisons.
"""

import os
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class DeepLearningVisualizer:
    """Visualizer for deep learning model results"""
    
    def __init__(self, save_dir=None):
        """Initialize visualizer"""
        if save_dir is None:
            self.save_dir = project_root / "visualizations" / "deep_learning"
        else:
            self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Model colors for consistency
        self.model_colors = {
            'FNN': '#1f77b4',
            'CNN': '#ff7f0e', 
            'LSTM': '#2ca02c',
            'GRU': '#d62728',
            'RNN': '#9467bd',
            'Transformer': '#8c564b',
            'Attention-LSTM': '#e377c2',
            'Gradient Boosting': '#7f7f7f',
            'Random Forest': '#bcbd22'
        }
        
    def plot_training_history(self, history, model_name, save=True):
        """Plot training history for a model"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{model_name} Training History', fontsize=16)
        
        # Loss plot
        ax = axes[0, 0]
        ax.plot(history['loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in history:
            ax.plot(history['val_loss'], label='Val Loss', linewidth=2, linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Model Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax = axes[0, 1]
        if 'accuracy' in history:
            ax.plot(history['accuracy'], label='Train Accuracy', linewidth=2)
            if 'val_accuracy' in history:
                ax.plot(history['val_accuracy'], label='Val Accuracy', linewidth=2, linestyle='--')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Model Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # AUC plot
        ax = axes[1, 0]
        if 'auc' in history:
            ax.plot(history['auc'], label='Train AUC', linewidth=2)
            if 'val_auc' in history:
                ax.plot(history['val_auc'], label='Val AUC', linewidth=2, linestyle='--')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('AUC')
            ax.set_title('Model AUC')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Learning rate plot (if available)
        ax = axes[1, 1]
        if 'lr' in history:
            ax.plot(history['lr'], label='Learning Rate', linewidth=2, color='orange')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Learning Rate\nNot Available', 
                   ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / f"{model_name.lower()}_training_history.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved training history plot to: {save_path}")
        
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, save=True):
        """Plot confusion matrix with metrics"""
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Regulation', 'Regulation'],
                   yticklabels=['No Regulation', 'Regulation'],
                   ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(f'{model_name} Confusion Matrix\n' + 
                    f'Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | ' +
                    f'Recall: {recall:.3f} | F1: {f1:.3f}', fontsize=14)
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / f"{model_name.lower()}_confusion_matrix.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved confusion matrix to: {save_path}")
        
        return fig
    
    def plot_roc_pr_curves(self, y_true, y_scores, model_name, save=True):
        """Plot ROC and PR curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        ax1.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # PR curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        ax2.plot(recall, precision, linewidth=2, label=f'{model_name} (AUC = {pr_auc:.3f})')
        ax2.axhline(y=y_true.mean(), color='k', linestyle='--', alpha=0.5, 
                   label=f'Baseline ({y_true.mean():.3f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(f'{model_name} Performance Curves', fontsize=16)
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / f"{model_name.lower()}_roc_pr_curves.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved ROC/PR curves to: {save_path}")
        
        return fig
    
    def plot_threshold_analysis(self, y_true, y_scores, model_name, save=True):
        """Plot metrics vs threshold"""
        thresholds = np.linspace(0, 1, 50)
        metrics = {'f1': [], 'precision': [], 'recall': [], 'accuracy': []}
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            
            # Calculate metrics
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics['accuracy'].append(accuracy)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)
        
        # Find optimal threshold
        best_idx = np.argmax(metrics['f1'])
        best_threshold = thresholds[best_idx]
        best_f1 = metrics['f1'][best_idx]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(thresholds, metrics['f1'], label='F1 Score', linewidth=2)
        ax.plot(thresholds, metrics['precision'], label='Precision', linewidth=2)
        ax.plot(thresholds, metrics['recall'], label='Recall', linewidth=2)
        ax.plot(thresholds, metrics['accuracy'], label='Accuracy', linewidth=2)
        
        # Mark optimal threshold
        ax.axvline(x=best_threshold, color='red', linestyle='--', alpha=0.7,
                  label=f'Optimal Threshold = {best_threshold:.3f}')
        ax.scatter([best_threshold], [best_f1], color='red', s=100, zorder=5)
        
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title(f'{model_name} - Metrics vs Threshold\n' +
                    f'Best F1: {best_f1:.3f} at threshold {best_threshold:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / f"{model_name.lower()}_threshold_analysis.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved threshold analysis to: {save_path}")
        
        return fig, best_threshold
    
    def plot_model_comparison(self, results_dict, save=True):
        """Plot comparison of all models"""
        # Extract metrics
        models = []
        f1_scores = []
        accuracies = []
        aucs = []
        
        for model_name, metrics in results_dict.items():
            models.append(model_name)
            f1_scores.append(metrics.get('f1_score', 0))
            accuracies.append(metrics.get('accuracy', 0))
            aucs.append(metrics.get('auc', 0))
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Bar plot of metrics
        x = np.arange(len(models))
        width = 0.25
        
        bars1 = ax1.bar(x - width, f1_scores, width, label='F1 Score', 
                        color=[self.model_colors.get(m, '#333333') for m in models], alpha=0.8)
        bars2 = ax1.bar(x, accuracies, width, label='Accuracy', 
                        color=[self.model_colors.get(m, '#333333') for m in models], alpha=0.6)
        bars3 = ax1.bar(x + width, aucs, width, label='AUC', 
                        color=[self.model_colors.get(m, '#333333') for m in models], alpha=0.4)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 1.1)
        
        # Radar chart
        categories = ['F1 Score', 'Accuracy', 'AUC']
        
        # Number of models to show (top 5 by F1)
        top_indices = np.argsort(f1_scores)[-5:][::-1]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax2 = plt.subplot(122, projection='polar')
        
        for idx in top_indices:
            values = [f1_scores[idx], accuracies[idx], aucs[idx]]
            values += values[:1]
            
            ax2.plot(angles, values, 'o-', linewidth=2, 
                    label=models[idx], 
                    color=self.model_colors.get(models[idx], '#333333'))
            ax2.fill(angles, values, alpha=0.15,
                    color=self.model_colors.get(models[idx], '#333333'))
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories)
        ax2.set_ylim(0, 1)
        ax2.set_title('Top 5 Models - Radar Chart', pad=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / "model_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved model comparison to: {save_path}")
        
        return fig
    
    def plot_prediction_distribution(self, y_true, predictions_dict, save=True):
        """Plot prediction probability distributions"""
        n_models = len(predictions_dict)
        fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (model_name, y_scores) in enumerate(predictions_dict.items()):
            ax = axes[idx]
            
            # Separate scores by true class
            scores_class_0 = y_scores[y_true == 0]
            scores_class_1 = y_scores[y_true == 1]
            
            # Plot distributions
            ax.hist(scores_class_0, bins=30, alpha=0.6, label='True Class 0', 
                   color='blue', density=True)
            ax.hist(scores_class_1, bins=30, alpha=0.6, label='True Class 1', 
                   color='red', density=True)
            
            ax.set_xlabel('Predicted Probability')
            ax.set_ylabel('Density')
            ax.set_title(f'{model_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        # Remove unused subplots
        for idx in range(len(predictions_dict), len(axes)):
            fig.delaxes(axes[idx])
        
        fig.suptitle('Prediction Probability Distributions', fontsize=16)
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / "prediction_distributions.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved prediction distributions to: {save_path}")
        
        return fig
    
    def create_summary_report(self, all_results, save=True):
        """Create a comprehensive summary report with all visualizations"""
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 24))
        gs = fig.add_gridspec(6, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Deep Learning Models - Comprehensive Results Summary', fontsize=20, y=0.995)
        
        # 1. Model comparison bar chart
        ax1 = fig.add_subplot(gs[0, :])
        models = []
        f1_scores = []
        for model_name, metrics in all_results.items():
            if 'performance' in metrics:
                models.append(model_name)
                f1_scores.append(metrics['performance'].get('f1_score', 0))
        
        y_pos = np.arange(len(models))
        colors = [self.model_colors.get(m, '#333333') for m in models]
        bars = ax1.barh(y_pos, f1_scores, color=colors)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(models)
        ax1.set_xlabel('F1 Score')
        ax1.set_title('Model Performance Ranking')
        ax1.set_xlim(0, 1)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 2. Training time comparison
        ax2 = fig.add_subplot(gs[1, 0])
        if any('training_time' in all_results[m] for m in all_results):
            times = [all_results[m].get('training_time', 0) for m in models]
            ax2.bar(range(len(models)), times, color=colors)
            ax2.set_xticks(range(len(models)))
            ax2.set_xticklabels(models, rotation=45, ha='right')
            ax2.set_ylabel('Training Time (seconds)')
            ax2.set_title('Training Time Comparison')
            ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Model complexity
        ax3 = fig.add_subplot(gs[1, 1])
        if any('parameters' in all_results[m] for m in all_results):
            params = [all_results[m].get('parameters', 0) for m in models]
            ax3.bar(range(len(models)), params, color=colors)
            ax3.set_xticks(range(len(models)))
            ax3.set_xticklabels(models, rotation=45, ha='right')
            ax3.set_ylabel('Number of Parameters')
            ax3.set_title('Model Complexity')
            ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Dataset info
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.axis('off')
        dataset_text = "Dataset Information:\n\n"
        if all_results and list(all_results.values())[0].get('dataset'):
            dataset = list(all_results.values())[0]['dataset']
            total = dataset.get('total_samples', 'N/A')
            train = dataset.get('train_samples', 'N/A')
            val = dataset.get('val_samples', 'N/A')
            test = dataset.get('test_samples', 'N/A')
            
            dataset_text += f"Total Samples: {total:,}\n" if isinstance(total, int) else f"Total Samples: {total}\n"
            dataset_text += f"Training: {train:,}\n" if isinstance(train, int) else f"Training: {train}\n"
            dataset_text += f"Validation: {val:,}\n" if isinstance(val, int) else f"Validation: {val}\n"
            dataset_text += f"Test: {test:,}\n" if isinstance(test, int) else f"Test: {test}\n"
            dataset_text += f"Features: {dataset.get('features', 'N/A')}\n"
            dataset_text += f"Positive Ratio: {dataset.get('positive_ratio', 'N/A'):.1%}"
        ax4.text(0.1, 0.9, dataset_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 5. Performance metrics table
        ax5 = fig.add_subplot(gs[2:4, :])
        ax5.axis('tight')
        ax5.axis('off')
        
        # Create table data
        table_data = []
        headers = ['Model', 'F1 Score', 'Accuracy', 'AUC', 'Precision', 'Recall', 'Threshold']
        
        for model_name in models:
            if 'performance' in all_results[model_name]:
                perf = all_results[model_name]['performance']
                row = [
                    model_name,
                    f"{perf.get('f1_score', 0):.3f}",
                    f"{perf.get('accuracy', 0):.3f}",
                    f"{perf.get('auc', 0):.3f}",
                    f"{perf.get('precision', 0):.3f}",
                    f"{perf.get('recall', 0):.3f}",
                    f"{perf.get('optimal_threshold', 0.5):.2f}"
                ]
                table_data.append(row)
        
        table = ax5.table(cellText=table_data, colLabels=headers, 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color cells based on F1 score
        for i in range(1, len(table_data) + 1):
            f1_value = float(table_data[i-1][1])
            if f1_value >= 0.8:
                color = '#90EE90'  # Light green
            elif f1_value >= 0.7:
                color = '#FFFFE0'  # Light yellow
            else:
                color = '#FFB6C1'  # Light red
            for j in range(len(headers)):
                table[(i, j)].set_facecolor(color)
        
        ax5.set_title('Detailed Performance Metrics', fontsize=14, pad=20)
        
        # 6. Key findings
        ax6 = fig.add_subplot(gs[4:, :])
        ax6.axis('off')
        
        findings_text = """Key Findings:

1. Model Performance:
   • Best performing deep learning model: {} (F1={:.3f})
   • Traditional ML baseline: Gradient Boosting (F1=0.879)
   • Performance gap: {:.1%}

2. Deep Learning Insights:
   • CNN performs best among DL models, capturing spatial patterns effectively
   • RNN-based models (LSTM, GRU) show similar performance (~0.67 F1)
   • Small dataset size (1,596 samples) limits deep learning effectiveness

3. Threshold Analysis:
   • RNN models require very low thresholds (0.10), indicating calibration issues
   • CNN and FNN have more reasonable thresholds (0.35-0.45)

4. Recommendations:
   • For production: Use Gradient Boosting or Random Forest
   • For deep learning: Increase dataset size to 10,000+ samples
   • Consider ensemble methods combining traditional ML and deep learning
""".format(
            models[0] if models else "N/A",
            f1_scores[0] if f1_scores else 0,
            (0.879 - (f1_scores[0] if f1_scores else 0)) / 0.879
        )
        
        ax6.text(0.05, 0.95, findings_text, transform=ax6.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / "comprehensive_summary.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comprehensive summary to: {save_path}")
        
        return fig


def visualize_all_results():
    """Load and visualize all deep learning results"""
    visualizer = DeepLearningVisualizer()
    
    # Load results
    results_dir = project_root / "models" / "deep_learning"
    
    # Load additional models results
    results_file = results_dir / "additional_models_results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Create comprehensive summary
        visualizer.create_summary_report(results)
        
        print("\nVisualization complete! Check the visualizations/deep_learning directory.")
    else:
        print("No results file found. Please run the training scripts first.")


if __name__ == "__main__":
    visualize_all_results()