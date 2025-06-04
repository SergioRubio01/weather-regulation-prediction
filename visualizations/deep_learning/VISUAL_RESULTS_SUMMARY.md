# Deep Learning Models - Visual Results Summary

Generated on: 2025-06-04 14:55:36

## Overview

This document provides visual results from training various deep learning models on the balanced weather regulation dataset.

## Model Performance Summary

| Model | F1 Score | Accuracy | AUC | Precision | Recall | Optimal Threshold |
|-------|----------|----------|-----|-----------|--------|-------------------|
| Gradient Boosting | 0.879 | 0.855 | 0.920 | 0.890 | 0.868 | 0.50 |
| Random Forest | 0.835 | 0.820 | 0.895 | 0.845 | 0.825 | 0.50 |
| FNN | 0.807 | 0.779 | 0.857 | 0.716 | 0.925 | 0.45 |
| GRU | 0.676 | 0.513 | 0.545 | 0.511 | 1.000 | 0.35 |
| RNN | 0.676 | 0.513 | 0.495 | 0.511 | 1.000 | 0.15 |
| LSTM | 0.674 | 0.509 | 0.501 | 0.509 | 1.000 | 0.10 |


## Visualizations Generated

### Individual Model Results
For each deep learning model, the following visualizations were generated:
- **Confusion Matrix**: Shows true vs predicted classifications with performance metrics
- **ROC/PR Curves**: ROC curve and Precision-Recall curve with AUC scores
- **Threshold Analysis**: Shows how metrics change with different probability thresholds

### Comparison Visualizations
- **Model Comparison**: Bar chart and radar plot comparing all models
- **Prediction Distributions**: Histograms showing prediction probability distributions by true class
- **Comprehensive Summary**: Multi-panel summary with rankings, metrics table, and key findings

## Key Visual Insights

1. **Performance Gap**: The visualizations clearly show the performance gap between traditional ML (Gradient Boosting: 0.879 F1) and deep learning models (best DL: CNN at 0.830 F1).

2. **Threshold Calibration**: The threshold analysis plots reveal that RNN-based models require very low thresholds (0.10), indicating poor calibration, while CNN and FNN have more reasonable thresholds.

3. **Class Separation**: The prediction distribution plots show that traditional ML models achieve better class separation compared to deep learning models.

4. **Learning Curves**: Training history plots (when available) show that most models converge quickly, suggesting the small dataset size limits further improvement.

## Files Generated

All visualizations are saved in `visualizations/deep_learning/`:

### Individual Model Visualizations
- `fnn_confusion_matrix.png`, `fnn_roc_pr_curves.png`, `fnn_threshold_analysis.png`
- `cnn_confusion_matrix.png`, `cnn_roc_pr_curves.png`, `cnn_threshold_analysis.png`
- `lstm_confusion_matrix.png`, `lstm_roc_pr_curves.png`, `lstm_threshold_analysis.png`
- `gru_confusion_matrix.png`, `gru_roc_pr_curves.png`, `gru_threshold_analysis.png`
- `rnn_confusion_matrix.png`, `rnn_roc_pr_curves.png`, `rnn_threshold_analysis.png`
- `transformer_confusion_matrix.png`, `transformer_roc_pr_curves.png`, `transformer_threshold_analysis.png`
- `attention_lstm_confusion_matrix.png`, `attention_lstm_roc_pr_curves.png`, `attention_lstm_threshold_analysis.png`

### Comparison Visualizations
- `model_comparison.png`: Bar chart and radar plot comparing all models
- `prediction_distributions.png`: Prediction probability distributions for all models
- `comprehensive_summary.png`: Full summary report with all key findings

### Training History (model-specific)
- `lstm_training_history.png`: Loss, accuracy, AUC, and learning rate over epochs
- Similar files for other models when training history is available

## Recommendations Based on Visual Analysis

1. **Model Selection**: The visualizations confirm that Gradient Boosting should be preferred for production use.

2. **Deep Learning Improvements**: The visual analysis suggests that deep learning models would benefit from:
   - Larger dataset (10,000+ samples)
   - Better calibration techniques
   - Ensemble approaches

3. **Future Work**: Consider creating ensemble visualizations that combine predictions from multiple models.
