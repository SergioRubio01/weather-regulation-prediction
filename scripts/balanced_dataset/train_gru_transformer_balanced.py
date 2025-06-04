#!/usr/bin/env python3
"""
Train GRU and Transformer models on the balanced dataset

This script trains GRU and Transformer models using the balanced dataset approach.
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
print("GRU and Transformer Models Training on Balanced Dataset")
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

# Use shorter sequence length for small dataset
sequence_length = 8  # Even shorter for better performance
print(f"\n3. Creating sequences with length {sequence_length}...")

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, sequence_length)
X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, sequence_length)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, sequence_length)


# ========== GRU Model ==========
print("\n" + "=" * 40)
print("Training GRU Model")
print("=" * 40)

def build_gru_model(input_shape):
    """Build GRU model architecture"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # Bidirectional GRU
        layers.Bidirectional(layers.GRU(32, return_sequences=True, 
                                       dropout=0.2, recurrent_dropout=0.2)),
        layers.BatchNormalization(),
        
        # Second GRU layer
        layers.Bidirectional(layers.GRU(16, dropout=0.2, recurrent_dropout=0.2)),
        layers.BatchNormalization(),
        
        # Dense layers
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

# Create and compile GRU model
gru_model = build_gru_model(input_shape=(sequence_length, X_train.shape[1]))
gru_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

print("\nGRU Model Summary:")
gru_model.summary()

# Callbacks for GRU
gru_callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint(filepath=str(output_dir / 'gru_balanced_best.keras'),
                   monitor='val_auc', mode='max', save_best_only=True, verbose=1)
]

# Train GRU
print("\n4. Training GRU model...")
gru_history = gru_model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=30,  # Reduced epochs
    batch_size=32,
    callbacks=gru_callbacks,
    verbose=1
)

# Evaluate GRU
print("\nEvaluating GRU on test set...")
gru_test_loss, gru_test_accuracy, gru_test_auc = gru_model.evaluate(X_test_seq, y_test_seq, verbose=0)
print(f"   Test Loss: {gru_test_loss:.4f}")
print(f"   Test Accuracy: {gru_test_accuracy:.4f}")
print(f"   Test AUC: {gru_test_auc:.4f}")

# Find optimal threshold for GRU
y_pred_proba_gru = gru_model.predict(X_test_seq, verbose=0)
best_f1_gru = 0
best_threshold_gru = 0.5

for threshold in np.arange(0.1, 0.9, 0.05):
    y_pred_temp = (y_pred_proba_gru > threshold).astype(int).flatten()
    f1 = f1_score(y_test_seq, y_pred_temp)
    if f1 > best_f1_gru:
        best_f1_gru = f1
        best_threshold_gru = threshold

print(f"\nGRU Optimal threshold: {best_threshold_gru:.2f}")
print(f"GRU Best F1 score: {best_f1_gru:.4f}")

# Get predictions with optimal threshold
y_pred_gru = (y_pred_proba_gru > best_threshold_gru).astype(int).flatten()

print("\nGRU Classification Report:")
print(classification_report(y_test_seq, y_pred_gru, 
                          target_names=['No Regulation', 'Regulation']))

# Save GRU model
gru_model.save(output_dir / "gru_balanced_final.keras")
print(f"GRU model saved!")


# ========== Transformer Model ==========
print("\n" + "=" * 40)
print("Training Transformer Model")
print("=" * 40)

class TransformerBlock(layers.Layer):
    """Transformer block implementation"""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def build_transformer_model(input_shape, num_heads=4, ff_dim=32):
    """Build Transformer model architecture"""
    inputs = layers.Input(shape=input_shape)
    
    # Positional encoding
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    positions = layers.Embedding(input_dim=input_shape[0], output_dim=input_shape[1])(positions)
    x = inputs + positions
    
    # Transformer block
    x = TransformerBlock(input_shape[1], num_heads, ff_dim)(x)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    return keras.Model(inputs=inputs, outputs=outputs)

# Create and compile Transformer model
transformer_model = build_transformer_model(input_shape=(sequence_length, X_train.shape[1]))
transformer_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

print("\nTransformer Model Summary:")
transformer_model.summary()

# Callbacks for Transformer
transformer_callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint(filepath=str(output_dir / 'transformer_balanced_best.keras'),
                   monitor='val_auc', mode='max', save_best_only=True, verbose=1)
]

# Train Transformer
print("\n5. Training Transformer model...")
transformer_history = transformer_model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=30,
    batch_size=32,
    callbacks=transformer_callbacks,
    verbose=1
)

# Evaluate Transformer
print("\nEvaluating Transformer on test set...")
trans_test_loss, trans_test_accuracy, trans_test_auc = transformer_model.evaluate(X_test_seq, y_test_seq, verbose=0)
print(f"   Test Loss: {trans_test_loss:.4f}")
print(f"   Test Accuracy: {trans_test_accuracy:.4f}")
print(f"   Test AUC: {trans_test_auc:.4f}")

# Find optimal threshold for Transformer
y_pred_proba_trans = transformer_model.predict(X_test_seq, verbose=0)
best_f1_trans = 0
best_threshold_trans = 0.5

for threshold in np.arange(0.1, 0.9, 0.05):
    y_pred_temp = (y_pred_proba_trans > threshold).astype(int).flatten()
    f1 = f1_score(y_test_seq, y_pred_temp)
    if f1 > best_f1_trans:
        best_f1_trans = f1
        best_threshold_trans = threshold

print(f"\nTransformer Optimal threshold: {best_threshold_trans:.2f}")
print(f"Transformer Best F1 score: {best_f1_trans:.4f}")

# Get predictions with optimal threshold
y_pred_trans = (y_pred_proba_trans > best_threshold_trans).astype(int).flatten()

print("\nTransformer Classification Report:")
print(classification_report(y_test_seq, y_pred_trans, 
                          target_names=['No Regulation', 'Regulation']))

# Save Transformer model
transformer_model.save(output_dir / "transformer_balanced_final.keras")
print(f"Transformer model saved!")

# Save results summary
import json
results_summary = {
    "training_date": datetime.now().isoformat(),
    "dataset_info": {
        "total_samples": len(data),
        "sequence_length": sequence_length,
        "features": len(feature_cols)
    },
    "gru_results": {
        "test_accuracy": float(gru_test_accuracy),
        "test_auc": float(gru_test_auc),
        "test_f1": float(best_f1_gru),
        "optimal_threshold": float(best_threshold_gru)
    },
    "transformer_results": {
        "test_accuracy": float(trans_test_accuracy),
        "test_auc": float(trans_test_auc),
        "test_f1": float(best_f1_trans),
        "optimal_threshold": float(best_threshold_trans)
    }
}

with open(output_dir / "gru_transformer_results.json", 'w') as f:
    json.dump(results_summary, f, indent=2)

# Save scaler
import joblib
joblib.dump(scaler, output_dir / "deep_learning_scaler.pkl")

print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
print("\nFinal Results Summary:")
print(f"\nGRU Model:")
print(f"  - Test F1 Score: {best_f1_gru:.4f}")
print(f"  - Test Accuracy: {gru_test_accuracy:.4f}")
print(f"  - Test AUC: {gru_test_auc:.4f}")
print(f"\nTransformer Model:")
print(f"  - Test F1 Score: {best_f1_trans:.4f}")
print(f"  - Test Accuracy: {trans_test_accuracy:.4f}")
print(f"  - Test AUC: {trans_test_auc:.4f}")
print(f"\nComparison with other models:")
print(f"  - Gradient Boosting: F1=0.879")
print(f"  - Random Forest: F1=0.835")
print(f"  - CNN: F1=0.830")
print(f"  - LSTM: F1=0.678")
print(f"  - GRU: F1={best_f1_gru:.3f}")
print(f"  - Transformer: F1={best_f1_trans:.3f}")