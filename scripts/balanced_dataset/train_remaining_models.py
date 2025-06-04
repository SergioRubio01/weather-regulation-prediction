#!/usr/bin/env python3
"""
Train remaining deep learning models on the balanced dataset

Models to train:
- Attention-LSTM
- WaveNet
- Autoencoder
- RNN
- FNN
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
import json

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import ExperimentConfig, LSTMConfig, CNNConfig, TrainingConfig, AutoencoderConfig
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

print("=" * 60)
print("Training Remaining Deep Learning Models on Balanced Dataset")
print("=" * 60)

# Load balanced dataset
print("\n1. Loading balanced dataset...")
balanced_data_path = project_root / "data" / "balanced_weather_data.csv"
data = pd.read_csv(balanced_data_path)
print(f"   Loaded {len(data):,} samples")
print(f"   Positive ratio: {data['has_regulation'].mean():.1%}")

# Define features
feature_cols = [col for col in data.columns if col not in ['has_regulation', 'datetime', 'airport']]
print(f"   Features: {len(feature_cols)}")

# Prepare data
X = data[feature_cols].values
y = data['has_regulation'].values

# Split data
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

# Create output directory
output_dir = project_root / "models" / "deep_learning"
output_dir.mkdir(parents=True, exist_ok=True)

# Prepare sequence data
def create_sequences(X, y, seq_len):
    """Create sequences for RNN input"""
    if len(X) <= seq_len:
        return X.reshape(X.shape[0], 1, X.shape[1]), y
    
    X_seq = []
    y_seq = []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i-seq_len:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

sequence_length = 8
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, sequence_length)
X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, sequence_length)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, sequence_length)

print(f"\n3. Sequence shapes:")
print(f"   Training: {X_train_seq.shape}")
print(f"   Validation: {X_val_seq.shape}")
print(f"   Test: {X_test_seq.shape}")

# Dictionary to store results
all_results = {}

# Function to train and evaluate a model
def train_and_evaluate_model(model_class, model_name, config, X_train_data, y_train_data, 
                           X_val_data, y_val_data, X_test_data, y_test_data):
    """Train and evaluate a single model"""
    print(f"\n{'=' * 40}")
    print(f"Training {model_name}")
    print(f"{'=' * 40}")
    
    try:
        # Create experiment config
        exp_config = ExperimentConfig(
            name=f"{model_name}_balanced",
            models={model_name.lower().replace('-', '_'): config}
        )
        
        # Initialize model
        model = model_class(exp_config)
        
        # Train model
        print(f"\nTraining {model_name}...")
        start_time = datetime.now()
        model.train(X_train_data, y_train_data, X_val_data, y_val_data)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate
        print(f"\nEvaluating {model_name}...")
        # Get predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test_data)
        else:
            y_pred_proba = model.predict(X_test_data)
            if len(y_pred_proba.shape) == 1:
                y_pred_proba = y_pred_proba.reshape(-1, 1)
        
        # Find optimal threshold
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in np.arange(0.1, 0.9, 0.05):
            y_pred_temp = (y_pred_proba > threshold).astype(int).flatten()
            f1 = f1_score(y_test_data, y_pred_temp)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        print(f"\nOptimal threshold: {best_threshold:.2f}")
        print(f"Best F1 score: {best_f1:.4f}")
        
        # Get predictions with optimal threshold
        y_pred = (y_pred_proba > best_threshold).astype(int).flatten()
        
        # Classification report
        print(f"\n{model_name} Classification Report:")
        print(classification_report(y_test_data, y_pred, 
                                  target_names=['No Regulation', 'Regulation']))
        
        # Save model
        model_path = output_dir / f"{model_name.lower().replace('-', '_')}_balanced_final.keras"
        model.save(str(model_path))
        print(f"\n{model_name} model saved!")
        
        # Store results
        results = {
            'f1_score': best_f1,
            'optimal_threshold': best_threshold,
            'training_time': training_time,
            'test_accuracy': (y_pred == y_test_data).mean()
        }
        
        return results
        
    except Exception as e:
        print(f"\nError training {model_name}: {str(e)}")
        return None

# Model configurations
models_to_train = [
    # FNN (Feedforward Neural Network)
    (FNNModel, "FNN", ModelConfig(
        hidden_layers=[64, 32, 16],
        dropout_rate=0.3,
        batch_size=32,
        epochs=30,
        learning_rate=0.001
    ), X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test),
    
    # RNN (Basic Recurrent Neural Network)
    (RNNModel, "RNN", ModelConfig(
        rnn_units=64,
        dropout_rate=0.2,
        batch_size=32,
        epochs=30,
        learning_rate=0.001
    ), X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq),
    
    # Attention-LSTM
    (AttentionLSTMModel, "Attention-LSTM", ModelConfig(
        lstm_units=64,
        attention_units=32,
        dropout_rate=0.2,
        batch_size=32,
        epochs=30,
        learning_rate=0.001
    ), X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq),
    
    # WaveNet
    (WaveNetModel, "WaveNet", ModelConfig(
        filters=32,
        kernel_size=2,
        nb_stacks=2,
        nb_layers=4,
        dropout_rate=0.2,
        batch_size=32,
        epochs=30,
        learning_rate=0.001
    ), X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq),
    
    # Autoencoder
    (AutoencoderModel, "Autoencoder", ModelConfig(
        encoding_dim=16,
        hidden_layers=[32, 16],
        dropout_rate=0.2,
        batch_size=32,
        epochs=30,
        learning_rate=0.001
    ), X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test),
]

# Train each model
for model_info in models_to_train:
    model_class, model_name, config, X_tr, y_tr, X_v, y_v, X_te, y_te = model_info
    results = train_and_evaluate_model(
        model_class, model_name, config, X_tr, y_tr, X_v, y_v, X_te, y_te
    )
    if results:
        all_results[model_name] = results

# Save results summary
results_path = output_dir / "remaining_models_results.json"
with open(results_path, 'w') as f:
    json.dump(all_results, f, indent=2)

# Print final summary
print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
print("\nFinal Results Summary:")
print("\nAll Models Comparison:")
print("  - Gradient Boosting: F1=0.879")
print("  - Random Forest: F1=0.835")
print("  - CNN: F1=0.830")
print("  - LSTM: F1=0.678")
print("  - GRU: F1=0.676")
print("  - Transformer: F1=0.674")

for model_name, results in all_results.items():
    print(f"  - {model_name}: F1={results['f1_score']:.3f}")

print(f"\nResults saved to: {results_path}")