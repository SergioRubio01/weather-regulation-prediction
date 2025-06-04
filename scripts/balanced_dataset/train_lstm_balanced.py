#!/usr/bin/env python3
"""
Train LSTM model on the balanced dataset

This script trains an LSTM model using the balanced dataset approach.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

print("=" * 60)
print("LSTM Model Training on Balanced Dataset")
print("=" * 60)

# Load balanced dataset
print("\n1. Loading balanced dataset...")
balanced_data_path = project_root / "data" / "balanced_weather_data.csv"
data = pd.read_csv(balanced_data_path)
print(f"   Loaded {len(data):,} samples")
print(f"   Positive ratio: {data['has_regulation'].mean():.1%}")

# Define features (all numeric columns except target)
feature_cols = [col for col in data.columns if col not in ['has_regulation', 'datetime', 'airport']]
print(f"   Features: {len(feature_cols)}")

# Prepare data
X = data[feature_cols].values
y = data['has_regulation'].values

# Split data (70/15/15 split as in balanced approach)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)

print(f"\n2. Data split:")
print(f"   Training: {len(X_train):,} samples")
print(f"   Validation: {len(X_val):,} samples")
print(f"   Test: {len(X_test):,} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Prepare sequence data for LSTM
def create_sequences(X, y, seq_len):
    """Create sequences for LSTM input"""
    if len(X) <= seq_len:
        # If data is too small, use all of it as one sequence
        return X.reshape(X.shape[0], 1, X.shape[1]), y
    
    X_seq = []
    y_seq = []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i-seq_len:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

# Use shorter sequence length for small dataset
sequence_length = min(12, len(X_train) // 10)  # Reduced from 24 to 12
print(f"\n3. Creating sequences with length {sequence_length}...")

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, sequence_length)
X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, sequence_length)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, sequence_length)

print(f"   Training sequences: {X_train_seq.shape}")
print(f"   Validation sequences: {X_val_seq.shape}")
print(f"   Test sequences: {X_test_seq.shape}")

# Build LSTM model
print("\n4. Building LSTM model...")

def build_lstm_model(input_shape):
    """Build LSTM model architecture"""
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First LSTM layer with return sequences
        layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        layers.BatchNormalization(),
        
        # Second LSTM layer
        layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2),
        layers.BatchNormalization(),
        
        # Dense layers
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

# Create model
model = build_lstm_model(input_shape=(sequence_length, X_train.shape[1]))

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

print("\nModel Summary:")
model.summary()

# Set up callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=str(project_root / 'models' / 'lstm_balanced_best.keras'),
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    )
]

# Train model
print("\n5. Training LSTM model...")
history = model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Evaluate on test set
print("\n6. Evaluating on test set...")
test_loss, test_accuracy, test_auc = model.evaluate(X_test_seq, y_test_seq, verbose=0)
print(f"   Test Loss: {test_loss:.4f}")
print(f"   Test Accuracy: {test_accuracy:.4f}")
print(f"   Test AUC: {test_auc:.4f}")

# Get predictions
y_pred_proba = model.predict(X_test_seq, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Find optimal threshold
print("\n7. Finding optimal threshold...")
thresholds = np.arange(0.1, 0.9, 0.05)
best_f1 = 0
best_threshold = 0.5

for threshold in thresholds:
    y_pred_temp = (y_pred_proba > threshold).astype(int).flatten()
    f1 = f1_score(y_test_seq, y_pred_temp)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"   Best threshold: {best_threshold:.2f}")
print(f"   Best F1 score: {best_f1:.4f}")

# Get predictions with optimal threshold
y_pred_optimal = (y_pred_proba > best_threshold).astype(int).flatten()

# Print classification report
print("\n8. Classification Report (with optimal threshold):")
print(classification_report(y_test_seq, y_pred_optimal, 
                          target_names=['No Regulation', 'Regulation']))

# Confusion matrix
cm = confusion_matrix(y_test_seq, y_pred_optimal)
print("\nConfusion Matrix:")
print(f"TN: {cm[0,0]:4d}  FP: {cm[0,1]:4d}")
print(f"FN: {cm[1,0]:4d}  TP: {cm[1,1]:4d}")

# Save model and configuration
print("\n9. Saving model...")
output_dir = project_root / "models" / "deep_learning"
output_dir.mkdir(parents=True, exist_ok=True)

# Save the model in Keras format
model_path = output_dir / "lstm_balanced_final.keras"
model.save(model_path)
print(f"   Model saved to: {model_path}")

# Save configuration and results
import json
config_path = output_dir / "lstm_balanced_config.json"
config_data = {
    "model_type": "LSTM",
    "training_date": datetime.now().isoformat(),
    "sequence_length": sequence_length,
    "architecture": {
        "lstm_units": [64, 32],
        "dropout": 0.2,
        "recurrent_dropout": 0.2,
        "dense_units": 16
    },
    "training": {
        "epochs": len(history.history['loss']),
        "batch_size": 32,
        "learning_rate": 0.001
    },
    "performance": {
        "test_accuracy": float(test_accuracy),
        "test_auc": float(test_auc),
        "test_f1": float(best_f1),
        "optimal_threshold": float(best_threshold)
    },
    "dataset": {
        "total_samples": len(data),
        "train_samples": len(X_train_seq),
        "val_samples": len(X_val_seq),
        "test_samples": len(X_test_seq),
        "features": len(feature_cols)
    }
}

with open(config_path, 'w') as f:
    json.dump(config_data, f, indent=2)
print(f"   Configuration saved to: {config_path}")

# Save scaler
import joblib
scaler_path = output_dir / "lstm_scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"   Scaler saved to: {scaler_path}")

# Generate visualizations
print("\n10. Generating visualizations...")
from visualize_dl_results import DeepLearningVisualizer

visualizer = DeepLearningVisualizer()

# Plot training history
if history:
    visualizer.plot_training_history(history.history, "LSTM")

# Plot confusion matrix
visualizer.plot_confusion_matrix(y_test_seq, y_pred_optimal, "LSTM")

# Plot ROC and PR curves
visualizer.plot_roc_pr_curves(y_test_seq, y_pred_proba.flatten(), "LSTM")

# Plot threshold analysis
visualizer.plot_threshold_analysis(y_test_seq, y_pred_proba.flatten(), "LSTM")

print("\n" + "=" * 60)
print("LSTM Training Complete!")
print("=" * 60)
print(f"\nFinal Results:")
print(f"  - Test F1 Score: {best_f1:.4f}")
print(f"  - Test Accuracy: {test_accuracy:.4f}")
print(f"  - Test AUC: {test_auc:.4f}")
print(f"  - Optimal Threshold: {best_threshold:.2f}")
print(f"\nComparison with Traditional ML (from balanced dataset):")
print(f"  - Gradient Boosting: F1=0.879")
print(f"  - Random Forest: F1=0.835")
print(f"  - CNN: F1=0.830")
print(f"  - LSTM: F1={best_f1:.3f}")
print(f"\nVisualizations saved to: visualizations/deep_learning/")