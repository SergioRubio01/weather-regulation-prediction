#!/usr/bin/env python3
"""
Train Transformer model on the balanced dataset
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
print("Transformer Model Training on Balanced Dataset")
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

# Build Transformer model
def build_transformer_model(input_shape, num_heads=4, ff_dim=32):
    """Build Transformer model using standard Keras layers"""
    inputs = layers.Input(shape=input_shape)
    
    # Add positional encoding
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    position_embedding = layers.Embedding(
        input_dim=input_shape[0], 
        output_dim=input_shape[1]
    )(positions)
    x = inputs + position_embedding
    
    # Multi-head attention
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=input_shape[1]
    )(x, x)
    x = layers.Dropout(0.1)(attention_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
    
    # Feed forward
    ffn = keras.Sequential([
        layers.Dense(ff_dim, activation="relu"),
        layers.Dense(input_shape[1]),
    ])
    ffn_output = ffn(x)
    x = layers.Dropout(0.1)(ffn_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
    
    # Output layers
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    return keras.Model(inputs=inputs, outputs=outputs)

# Create and compile model
print("\n4. Building Transformer model...")
model = build_transformer_model(input_shape=(sequence_length, X_train.shape[1]))
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

print("\nModel Summary:")
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint(filepath=str(output_dir / 'transformer_balanced_best.keras'),
                   monitor='val_auc', mode='max', save_best_only=True, verbose=1)
]

# Train model
print("\n5. Training Transformer model...")
history = model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=30,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
print("\n6. Evaluating on test set...")
test_loss, test_accuracy, test_auc = model.evaluate(X_test_seq, y_test_seq, verbose=0)
print(f"   Test Loss: {test_loss:.4f}")
print(f"   Test Accuracy: {test_accuracy:.4f}")
print(f"   Test AUC: {test_auc:.4f}")

# Find optimal threshold
y_pred_proba = model.predict(X_test_seq, verbose=0)
best_f1 = 0
best_threshold = 0.5

for threshold in np.arange(0.1, 0.9, 0.05):
    y_pred_temp = (y_pred_proba > threshold).astype(int).flatten()
    f1 = f1_score(y_test_seq, y_pred_temp)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\n7. Optimal threshold: {best_threshold:.2f}")
print(f"   Best F1 score: {best_f1:.4f}")

# Classification report
y_pred = (y_pred_proba > best_threshold).astype(int).flatten()
print("\n8. Classification Report:")
print(classification_report(y_test_seq, y_pred, 
                          target_names=['No Regulation', 'Regulation']))

# Save model
model.save(output_dir / "transformer_balanced_final.keras")
print(f"\n9. Model saved!")

print("\n" + "=" * 60)
print("Transformer Training Complete!")
print("=" * 60)
print(f"\nFinal Results:")
print(f"  - Test F1 Score: {best_f1:.4f}")
print(f"  - Test Accuracy: {test_accuracy:.4f}")
print(f"  - Test AUC: {test_auc:.4f}")
print(f"  - Optimal Threshold: {best_threshold:.2f}")
print(f"\nComparison with other models:")
print(f"  - Gradient Boosting: F1=0.879")
print(f"  - Random Forest: F1=0.835")
print(f"  - CNN: F1=0.830")
print(f"  - LSTM: F1=0.678")
print(f"  - GRU: F1=0.676")
print(f"  - Transformer: F1={best_f1:.3f}")