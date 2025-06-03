"""
Advanced Visualization Module for Weather Regulation Prediction

This module provides comprehensive visualization capabilities including:
- Interactive plots with Plotly
- Static plots with Matplotlib/Seaborn
- Model performance visualizations
- Feature importance analysis
- Training history plots
- Model comparison dashboards
- Custom weather-specific visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Set default style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ModelVisualizer:
    """Visualization tools for machine learning models"""
    
    def __init__(self, save_path: str = "./visualizations"):
        """Initialize visualizer"""
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Color schemes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9800',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
        self.model_colors = px.colors.qualitative.Plotly
        
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            labels: Optional[List[str]] = None,
                            title: str = "Confusion Matrix",
                            normalize: bool = True,
                            interactive: bool = True,
                            save_name: Optional[str] = None) -> Union[go.Figure, plt.Figure]:
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
        if labels is None:
            labels = [f"Class {i}" for i in range(len(cm))]
            
        if interactive:
            # Plotly version
            text = np.around(cm, decimals=2) if normalize else cm
            
            fig = ff.create_annotated_heatmap(
                z=cm,
                x=labels,
                y=labels,
                annotation_text=text,
                colorscale='Blues',
                showscale=True
            )
            
            fig.update_layout(
                title=title,
                xaxis=dict(title='Predicted Label', side='bottom'),
                yaxis=dict(title='True Label'),
                width=600,
                height=500
            )
            
            fig.update_xaxes(side="bottom")
            
            if save_name:
                fig.write_html(self.save_path / f"{save_name}.html")
                fig.write_image(self.save_path / f"{save_name}.png")
                
            return fig
            
        else:
            # Matplotlib version
            fig, ax = plt.subplots(figsize=(8, 6))
            
            sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                       cmap='Blues', square=True, cbar=True,
                       xticklabels=labels, yticklabels=labels, ax=ax)
            
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title(title)
            
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_path / f"{save_name}.png", dpi=300, bbox_inches='tight')
                
            return fig
    
    def plot_roc_curves(self, y_true: np.ndarray, y_scores: Dict[str, np.ndarray],
                       title: str = "ROC Curves Comparison",
                       interactive: bool = True,
                       save_name: Optional[str] = None) -> Union[go.Figure, plt.Figure]:
        """Plot ROC curves for multiple models"""
        
        if interactive:
            fig = go.Figure()
            
            for i, (model_name, scores) in enumerate(y_scores.items()):
                fpr, tpr, _ = roc_curve(y_true, scores)
                roc_auc = auc(fpr, tpr)
                
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{model_name} (AUC = {roc_auc:.3f})',
                    line=dict(color=self.model_colors[i % len(self.model_colors)], width=2)
                ))
            
            # Add diagonal line
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(color='gray', width=1, dash='dash')
            ))
            
            fig.update_layout(
                title=title,
                xaxis=dict(title='False Positive Rate', range=[0, 1]),
                yaxis=dict(title='True Positive Rate', range=[0, 1]),
                width=800,
                height=600,
                legend=dict(x=0.7, y=0.1),
                template='plotly_white'
            )
            
            if save_name:
                fig.write_html(self.save_path / f"{save_name}.html")
                fig.write_image(self.save_path / f"{save_name}.png")
                
            return fig
            
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            for i, (model_name, scores) in enumerate(y_scores.items()):
                fpr, tpr, _ = roc_curve(y_true, scores)
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})',
                       color=self.model_colors[i % len(self.model_colors)], linewidth=2)
            
            ax.plot([0, 1], [0, 1], 'k--', label='Random')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(title)
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_path / f"{save_name}.png", dpi=300, bbox_inches='tight')
                
            return fig
    
    def plot_feature_importance(self, importance_data: Dict[str, pd.DataFrame],
                              top_n: int = 20,
                              title: str = "Feature Importance Comparison",
                              interactive: bool = True,
                              save_name: Optional[str] = None) -> Union[go.Figure, plt.Figure]:
        """Plot feature importance for multiple models"""
        
        if interactive:
            fig = make_subplots(
                rows=1, cols=len(importance_data),
                subplot_titles=list(importance_data.keys()),
                horizontal_spacing=0.1
            )
            
            for i, (model_name, importance_df) in enumerate(importance_data.items()):
                # Get top features
                top_features = importance_df.nlargest(top_n, 'importance')
                
                fig.add_trace(
                    go.Bar(
                        x=top_features['importance'],
                        y=top_features['feature'],
                        orientation='h',
                        name=model_name,
                        marker_color=self.model_colors[i % len(self.model_colors)],
                        showlegend=False
                    ),
                    row=1, col=i+1
                )
                
            fig.update_layout(
                title=title,
                height=600,
                width=400 * len(importance_data),
                template='plotly_white'
            )
            
            fig.update_xaxes(title_text="Importance Score")
            fig.update_yaxes(title_text="Features", col=1)
            
            if save_name:
                fig.write_html(self.save_path / f"{save_name}.html")
                fig.write_image(self.save_path / f"{save_name}.png")
                
            return fig
            
        else:
            n_models = len(importance_data)
            fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
            
            if n_models == 1:
                axes = [axes]
                
            for i, (model_name, importance_df) in enumerate(importance_data.items()):
                ax = axes[i]
                
                # Get top features
                top_features = importance_df.nlargest(top_n, 'importance')
                
                ax.barh(top_features['feature'], top_features['importance'],
                       color=self.model_colors[i % len(self.model_colors)])
                ax.set_xlabel('Importance Score')
                ax.set_title(model_name)
                ax.grid(True, alpha=0.3)
                
                if i == 0:
                    ax.set_ylabel('Features')
                    
            plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_path / f"{save_name}.png", dpi=300, bbox_inches='tight')
                
            return fig
    
    def plot_training_history(self, histories: Dict[str, Dict[str, List[float]]],
                            metrics: List[str] = ['loss', 'accuracy'],
                            title: str = "Training History",
                            interactive: bool = True,
                            save_name: Optional[str] = None) -> Union[go.Figure, plt.Figure]:
        """Plot training history for multiple models"""
        
        if interactive:
            fig = make_subplots(
                rows=len(metrics), cols=1,
                subplot_titles=[f'{metric.capitalize()} History' for metric in metrics],
                vertical_spacing=0.1
            )
            
            for i, metric in enumerate(metrics):
                for j, (model_name, history) in enumerate(histories.items()):
                    if metric in history:
                        epochs = list(range(1, len(history[metric]) + 1))
                        
                        # Training metric
                        fig.add_trace(
                            go.Scatter(
                                x=epochs,
                                y=history[metric],
                                mode='lines',
                                name=f'{model_name} - Train',
                                line=dict(color=self.model_colors[j % len(self.model_colors)]),
                                legendgroup=model_name,
                                showlegend=(i == 0)
                            ),
                            row=i+1, col=1
                        )
                        
                        # Validation metric
                        val_metric = f'val_{metric}'
                        if val_metric in history:
                            fig.add_trace(
                                go.Scatter(
                                    x=epochs,
                                    y=history[val_metric],
                                    mode='lines',
                                    name=f'{model_name} - Val',
                                    line=dict(
                                        color=self.model_colors[j % len(self.model_colors)],
                                        dash='dash'
                                    ),
                                    legendgroup=model_name,
                                    showlegend=(i == 0)
                                ),
                                row=i+1, col=1
                            )
                            
            fig.update_xaxes(title_text="Epoch")
            fig.update_yaxes(title_text="Value")
            
            fig.update_layout(
                title=title,
                height=400 * len(metrics),
                width=900,
                template='plotly_white',
                legend=dict(x=0.7, y=1)
            )
            
            if save_name:
                fig.write_html(self.save_path / f"{save_name}.html")
                fig.write_image(self.save_path / f"{save_name}.png")
                
            return fig
            
        else:
            fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 5*len(metrics)))
            
            if len(metrics) == 1:
                axes = [axes]
                
            for i, metric in enumerate(metrics):
                ax = axes[i]
                
                for j, (model_name, history) in enumerate(histories.items()):
                    if metric in history:
                        epochs = list(range(1, len(history[metric]) + 1))
                        
                        # Training metric
                        ax.plot(epochs, history[metric],
                               label=f'{model_name} - Train',
                               color=self.model_colors[j % len(self.model_colors)],
                               linewidth=2)
                        
                        # Validation metric
                        val_metric = f'val_{metric}'
                        if val_metric in history:
                            ax.plot(epochs, history[val_metric],
                                   label=f'{model_name} - Val',
                                   color=self.model_colors[j % len(self.model_colors)],
                                   linestyle='--', linewidth=2)
                            
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric.capitalize())
                ax.set_title(f'{metric.capitalize()} History')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_path / f"{save_name}.png", dpi=300, bbox_inches='tight')
                
            return fig
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame,
                            metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1_score'],
                            title: str = "Model Performance Comparison",
                            interactive: bool = True,
                            save_name: Optional[str] = None) -> Union[go.Figure, plt.Figure]:
        """Plot model comparison radar chart or bar chart"""
        
        if interactive:
            # Radar chart for interactive
            fig = go.Figure()
            
            for i, row in comparison_df.iterrows():
                model_name = row['model']
                values = [row[metric] for metric in metrics]
                values.append(values[0])  # Complete the circle
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics + [metrics[0]],
                    fill='toself',
                    name=model_name,
                    line=dict(color=self.model_colors[i % len(self.model_colors)])
                ))
                
            fig.update_layout(
                title=title,
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=True,
                width=800,
                height=600,
                template='plotly_white'
            )
            
            if save_name:
                fig.write_html(self.save_path / f"{save_name}.html")
                fig.write_image(self.save_path / f"{save_name}.png")
                
            return fig
            
        else:
            # Grouped bar chart for static
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = np.arange(len(comparison_df))
            width = 0.8 / len(metrics)
            
            for i, metric in enumerate(metrics):
                offset = (i - len(metrics)/2) * width + width/2
                bars = ax.bar(x + offset, comparison_df[metric], width,
                             label=metric.replace('_', ' ').title())
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
                    
            ax.set_xlabel('Models')
            ax.set_ylabel('Score')
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(comparison_df['model'], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.1)
            
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_path / f"{save_name}.png", dpi=300, bbox_inches='tight')
                
            return fig
    
    def plot_prediction_distribution(self, predictions: Dict[str, np.ndarray],
                                   true_labels: np.ndarray,
                                   title: str = "Prediction Distribution",
                                   interactive: bool = True,
                                   save_name: Optional[str] = None) -> Union[go.Figure, plt.Figure]:
        """Plot distribution of predictions"""
        
        if interactive:
            fig = make_subplots(
                rows=len(predictions), cols=2,
                subplot_titles=[f"{model} - Class 0", f"{model} - Class 1" 
                              for model in predictions.keys()],
                vertical_spacing=0.1
            )
            
            for i, (model_name, probs) in enumerate(predictions.items()):
                # Class 0 predictions
                class_0_probs = probs[true_labels == 0][:, 1] if len(probs.shape) > 1 else probs[true_labels == 0]
                class_1_probs = probs[true_labels == 1][:, 1] if len(probs.shape) > 1 else probs[true_labels == 1]
                
                fig.add_trace(
                    go.Histogram(
                        x=class_0_probs,
                        name='True Class 0',
                        marker_color='blue',
                        opacity=0.7,
                        nbinsx=30,
                        showlegend=(i == 0)
                    ),
                    row=i+1, col=1
                )
                
                fig.add_trace(
                    go.Histogram(
                        x=class_1_probs,
                        name='True Class 1',
                        marker_color='red',
                        opacity=0.7,
                        nbinsx=30,
                        showlegend=(i == 0)
                    ),
                    row=i+1, col=2
                )
                
            fig.update_xaxes(title_text="Predicted Probability")
            fig.update_yaxes(title_text="Count")
            
            fig.update_layout(
                title=title,
                height=300 * len(predictions),
                width=900,
                template='plotly_white',
                barmode='overlay'
            )
            
            if save_name:
                fig.write_html(self.save_path / f"{save_name}.html")
                fig.write_image(self.save_path / f"{save_name}.png")
                
            return fig
            
        else:
            n_models = len(predictions)
            fig, axes = plt.subplots(n_models, 2, figsize=(12, 4*n_models))
            
            if n_models == 1:
                axes = axes.reshape(1, -1)
                
            for i, (model_name, probs) in enumerate(predictions.items()):
                # Class 0 predictions
                class_0_probs = probs[true_labels == 0][:, 1] if len(probs.shape) > 1 else probs[true_labels == 0]
                class_1_probs = probs[true_labels == 1][:, 1] if len(probs.shape) > 1 else probs[true_labels == 1]
                
                axes[i, 0].hist(class_0_probs, bins=30, alpha=0.7, color='blue', label='True Class 0')
                axes[i, 0].set_title(f'{model_name} - Predictions for True Class 0')
                axes[i, 0].set_xlabel('Predicted Probability of Class 1')
                axes[i, 0].set_ylabel('Count')
                axes[i, 0].grid(True, alpha=0.3)
                
                axes[i, 1].hist(class_1_probs, bins=30, alpha=0.7, color='red', label='True Class 1')
                axes[i, 1].set_title(f'{model_name} - Predictions for True Class 1')
                axes[i, 1].set_xlabel('Predicted Probability of Class 1')
                axes[i, 1].set_ylabel('Count')
                axes[i, 1].grid(True, alpha=0.3)
                
            plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_path / f"{save_name}.png", dpi=300, bbox_inches='tight')
                
            return fig
    
    def plot_learning_curves(self, train_sizes: np.ndarray,
                           train_scores: Dict[str, np.ndarray],
                           val_scores: Dict[str, np.ndarray],
                           title: str = "Learning Curves",
                           interactive: bool = True,
                           save_name: Optional[str] = None) -> Union[go.Figure, plt.Figure]:
        """Plot learning curves for multiple models"""
        
        if interactive:
            fig = go.Figure()
            
            for i, model_name in enumerate(train_scores.keys()):
                # Training scores
                train_mean = np.mean(train_scores[model_name], axis=1)
                train_std = np.std(train_scores[model_name], axis=1)
                
                # Validation scores
                val_mean = np.mean(val_scores[model_name], axis=1)
                val_std = np.std(val_scores[model_name], axis=1)
                
                color = self.model_colors[i % len(self.model_colors)]
                
                # Training curve with confidence interval
                fig.add_trace(go.Scatter(
                    x=train_sizes, y=train_mean,
                    mode='lines',
                    name=f'{model_name} - Train',
                    line=dict(color=color, width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=np.concatenate([train_sizes, train_sizes[::-1]]),
                    y=np.concatenate([train_mean - train_std, (train_mean + train_std)[::-1]]),
                    fill='toself',
                    fillcolor=color,
                    opacity=0.2,
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Validation curve with confidence interval
                fig.add_trace(go.Scatter(
                    x=train_sizes, y=val_mean,
                    mode='lines',
                    name=f'{model_name} - Val',
                    line=dict(color=color, width=2, dash='dash')
                ))
                
                fig.add_trace(go.Scatter(
                    x=np.concatenate([train_sizes, train_sizes[::-1]]),
                    y=np.concatenate([val_mean - val_std, (val_mean + val_std)[::-1]]),
                    fill='toself',
                    fillcolor=color,
                    opacity=0.2,
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
            fig.update_layout(
                title=title,
                xaxis=dict(title='Training Set Size'),
                yaxis=dict(title='Score'),
                width=900,
                height=600,
                template='plotly_white'
            )
            
            if save_name:
                fig.write_html(self.save_path / f"{save_name}.html")
                fig.write_image(self.save_path / f"{save_name}.png")
                
            return fig
            
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for i, model_name in enumerate(train_scores.keys()):
                # Training scores
                train_mean = np.mean(train_scores[model_name], axis=1)
                train_std = np.std(train_scores[model_name], axis=1)
                
                # Validation scores
                val_mean = np.mean(val_scores[model_name], axis=1)
                val_std = np.std(val_scores[model_name], axis=1)
                
                color = self.model_colors[i % len(self.model_colors)]
                
                # Plot with confidence intervals
                ax.plot(train_sizes, train_mean, 'o-', color=color,
                       label=f'{model_name} - Train')
                ax.fill_between(train_sizes, train_mean - train_std,
                              train_mean + train_std, alpha=0.2, color=color)
                
                ax.plot(train_sizes, val_mean, 's--', color=color,
                       label=f'{model_name} - Val')
                ax.fill_between(train_sizes, val_mean - val_std,
                              val_mean + val_std, alpha=0.2, color=color)
                
            ax.set_xlabel('Training Set Size')
            ax.set_ylabel('Score')
            ax.set_title(title)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_path / f"{save_name}.png", dpi=300, bbox_inches='tight')
                
            return fig


class WeatherVisualizer:
    """Weather-specific visualizations"""
    
    def __init__(self, save_path: str = "./visualizations"):
        """Initialize weather visualizer"""
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
    def plot_weather_patterns(self, weather_data: pd.DataFrame,
                            features: List[str] = ['temperature', 'pressure', 'wind_speed'],
                            title: str = "Weather Patterns",
                            save_name: Optional[str] = None) -> go.Figure:
        """Plot weather patterns over time"""
        
        fig = make_subplots(
            rows=len(features), cols=1,
            subplot_titles=[f'{feat.replace("_", " ").title()}' for feat in features],
            vertical_spacing=0.05,
            shared_xaxes=True
        )
        
        for i, feature in enumerate(features):
            if feature in weather_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=weather_data.index,
                        y=weather_data[feature],
                        mode='lines',
                        name=feature.replace('_', ' ').title(),
                        line=dict(width=1),
                        showlegend=False
                    ),
                    row=i+1, col=1
                )
                
                # Add rolling average
                if len(weather_data) > 24:
                    rolling_avg = weather_data[feature].rolling(24).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=weather_data.index,
                            y=rolling_avg,
                            mode='lines',
                            name=f'{feature} (24h avg)',
                            line=dict(width=2, dash='dash'),
                            showlegend=False
                        ),
                        row=i+1, col=1
                    )
                    
        fig.update_xaxes(title_text="Time", row=len(features))
        fig.update_layout(
            title=title,
            height=300 * len(features),
            width=1000,
            template='plotly_white'
        )
        
        if save_name:
            fig.write_html(self.save_path / f"{save_name}.html")
            fig.write_image(self.save_path / f"{save_name}.png")
            
        return fig
    
    def plot_regulation_analysis(self, regulation_data: pd.DataFrame,
                               weather_data: pd.DataFrame,
                               title: str = "Regulation vs Weather Analysis",
                               save_name: Optional[str] = None) -> go.Figure:
        """Plot regulation occurrence analysis with weather conditions"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Regulations by Hour', 'Regulations by Weather Severity',
                          'Weather Conditions During Regulations', 'Regulation Duration Distribution'],
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                  [{"type": "box"}, {"type": "histogram"}]]
        )
        
        # 1. Regulations by hour of day
        if 'hour' in regulation_data.columns:
            hourly_regs = regulation_data.groupby('hour')['has_regulation'].sum()
            fig.add_trace(
                go.Bar(x=hourly_regs.index, y=hourly_regs.values, name='Regulations'),
                row=1, col=1
            )
            
        # 2. Regulations vs weather severity
        if 'weather_severity' in weather_data.columns and 'has_regulation' in weather_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=weather_data['weather_severity'],
                    y=weather_data['has_regulation'],
                    mode='markers',
                    marker=dict(size=5, opacity=0.5),
                    name='Severity vs Regulation'
                ),
                row=1, col=2
            )
            
        # 3. Weather conditions during regulations
        weather_features = ['temperature', 'wind_speed', 'visibility', 'pressure']
        for feature in weather_features:
            if feature in weather_data.columns and 'has_regulation' in weather_data.columns:
                reg_weather = weather_data[weather_data['has_regulation'] == 1][feature]
                no_reg_weather = weather_data[weather_data['has_regulation'] == 0][feature]
                
                fig.add_trace(
                    go.Box(y=reg_weather, name=f'{feature} (Reg)', showlegend=False),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Box(y=no_reg_weather, name=f'{feature} (No Reg)', showlegend=False),
                    row=2, col=1
                )
                
        # 4. Regulation duration distribution
        if 'duration' in regulation_data.columns:
            fig.add_trace(
                go.Histogram(
                    x=regulation_data['duration'],
                    nbinsx=30,
                    name='Duration'
                ),
                row=2, col=2
            )
            
        fig.update_layout(
            title=title,
            height=800,
            width=1200,
            showlegend=False,
            template='plotly_white'
        )
        
        if save_name:
            fig.write_html(self.save_path / f"{save_name}.html")
            fig.write_image(self.save_path / f"{save_name}.png")
            
        return fig
    
    def plot_airport_comparison(self, airport_data: Dict[str, pd.DataFrame],
                              metric: str = 'regulation_rate',
                              title: str = "Airport Comparison",
                              save_name: Optional[str] = None) -> go.Figure:
        """Compare metrics across multiple airports"""
        
        fig = go.Figure()
        
        for i, (airport, data) in enumerate(airport_data.items()):
            if metric in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[metric],
                        mode='lines',
                        name=airport,
                        line=dict(width=2)
                    )
                )
                
        fig.update_layout(
            title=title,
            xaxis=dict(title='Time'),
            yaxis=dict(title=metric.replace('_', ' ').title()),
            width=1000,
            height=600,
            template='plotly_white',
            hovermode='x unified'
        )
        
        if save_name:
            fig.write_html(self.save_path / f"{save_name}.html")
            fig.write_image(self.save_path / f"{save_name}.png")
            
        return fig


# Utility functions for quick plotting
def plot_model_results(results: Dict[str, Any], save_path: str = "./visualizations"):
    """Quick function to plot all standard model results"""
    visualizer = ModelVisualizer(save_path)
    
    plots = {}
    
    # Confusion matrix
    if 'y_true' in results and 'y_pred' in results:
        plots['confusion_matrix'] = visualizer.plot_confusion_matrix(
            results['y_true'], results['y_pred'],
            title=f"Confusion Matrix - {results.get('model_name', 'Model')}",
            save_name=f"cm_{results.get('model_name', 'model')}"
        )
        
    # ROC curves
    if 'y_true' in results and 'y_scores' in results:
        plots['roc_curves'] = visualizer.plot_roc_curves(
            results['y_true'], results['y_scores'],
            save_name=f"roc_{results.get('experiment_id', 'exp')}"
        )
        
    # Feature importance
    if 'feature_importance' in results:
        plots['feature_importance'] = visualizer.plot_feature_importance(
            results['feature_importance'],
            save_name=f"importance_{results.get('experiment_id', 'exp')}"
        )
        
    # Training history
    if 'training_history' in results:
        plots['training_history'] = visualizer.plot_training_history(
            results['training_history'],
            save_name=f"history_{results.get('experiment_id', 'exp')}"
        )
        
    return plots


def create_experiment_dashboard(experiment_results: Dict[str, Any], 
                              save_path: str = "./visualizations") -> str:
    """Create comprehensive dashboard for experiment results"""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    # Create figure with subplots
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=['Model Comparison', 'Best Model Confusion Matrix', 'ROC Curves',
                       'Feature Importance', 'Training History', 'Prediction Distribution',
                       'Learning Curves', 'Performance by Dataset Size', 'Time Analysis'],
        specs=[[{"type": "bar"}, {"type": "heatmap"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "scatter"}, {"type": "histogram"}],
               [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}]],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # Add plots based on available data
    # ... (implementation would add various plots to the dashboard)
    
    fig.update_layout(
        title="Experiment Results Dashboard",
        height=1500,
        width=1500,
        showlegend=True,
        template='plotly_white'
    )
    
    # Save dashboard
    dashboard_path = Path(save_path) / f"dashboard_{experiment_results.get('experiment_id', 'exp')}.html"
    fig.write_html(dashboard_path)
    
    return str(dashboard_path)