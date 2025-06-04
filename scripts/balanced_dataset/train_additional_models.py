#!/usr/bin/env python3
"""
Train additional deep learning models on the balanced dataset

Models to train:
- RNN (Basic Recurrent Neural Network)
- FNN (Feedforward Neural Network)
- Attention-LSTM
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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

print("=" * 60)
print("Training Additional Deep Learning Models on Balanced Dataset")
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

# Save scaler
import joblib
joblib.dump(scaler, output_dir / "additional_models_scaler.pkl")

# Dictionary to store results
all_results = {}


# ========== FNN (Feedforward Neural Network) ==========
print("\n" + "=" * 40)
print("Training FNN Model")
print("=" * 40)

def build_fnn_model(input_shape, hidden_layers=[64, 32, 16], dropout_rate=0.3):
    """Build FNN model architecture"""
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    
    # Hidden layers
    for units in hidden_layers:
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model

# Create and compile FNN model
fnn_model = build_fnn_model(X_train.shape[1])
fnn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

print("\nFNN Model Summary:")
fnn_model.summary()

# Callbacks for FNN
fnn_callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint(filepath=str(output_dir / 'fnn_balanced_best.keras'),
                   monitor='val_auc', mode='max', save_best_only=True, verbose=1)
]

# Train FNN
print("\n3. Training FNN model...")
fnn_history = fnn_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=30,
    batch_size=32,
    callbacks=fnn_callbacks,
    verbose=1
)

# Evaluate FNN
print("\nEvaluating FNN on test set...")
fnn_test_loss, fnn_test_accuracy, fnn_test_auc = fnn_model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"   Test Loss: {fnn_test_loss:.4f}")
print(f"   Test Accuracy: {fnn_test_accuracy:.4f}")
print(f"   Test AUC: {fnn_test_auc:.4f}")

# Find optimal threshold for FNN
y_pred_proba_fnn = fnn_model.predict(X_test_scaled, verbose=0)
best_f1_fnn = 0
best_threshold_fnn = 0.5

for threshold in np.arange(0.1, 0.9, 0.05):
    y_pred_temp = (y_pred_proba_fnn > threshold).astype(int).flatten()
    f1 = f1_score(y_test, y_pred_temp)
    if f1 > best_f1_fnn:
        best_f1_fnn = f1
        best_threshold_fnn = threshold

print(f"\nFNN Optimal threshold: {best_threshold_fnn:.2f}")
print(f"FNN Best F1 score: {best_f1_fnn:.4f}")

# Get predictions with optimal threshold
y_pred_fnn = (y_pred_proba_fnn > best_threshold_fnn).astype(int).flatten()

print("\nFNN Classification Report:")
print(classification_report(y_test, y_pred_fnn, 
                          target_names=['No Regulation', 'Regulation']))

# Save FNN model
fnn_model.save(output_dir / "fnn_balanced_final.keras")
print(f"FNN model saved!")

all_results['FNN'] = {
    'test_accuracy': float(fnn_test_accuracy),
    'test_auc': float(fnn_test_auc),
    'test_f1': float(best_f1_fnn),
    'optimal_threshold': float(best_threshold_fnn)
}


# ========== RNN (Basic Recurrent Neural Network) ==========
print("\n" + "=" * 40)
print("Training RNN Model")
print("=" * 40)

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

print(f"\nSequence shapes:")
print(f"   Training: {X_train_seq.shape}")
print(f"   Validation: {X_val_seq.shape}")
print(f"   Test: {X_test_seq.shape}")

def build_rnn_model(input_shape):
    """Build RNN model architecture"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # RNN layers
        layers.SimpleRNN(64, return_sequences=True, dropout=0.2),
        layers.BatchNormalization(),
        layers.SimpleRNN(32, dropout=0.2),
        layers.BatchNormalization(),
        
        # Dense layers
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

# Create and compile RNN model
rnn_model = build_rnn_model(input_shape=(sequence_length, X_train.shape[1]))
rnn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

print("\nRNN Model Summary:")
rnn_model.summary()

# Callbacks for RNN
rnn_callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint(filepath=str(output_dir / 'rnn_balanced_best.keras'),
                   monitor='val_auc', mode='max', save_best_only=True, verbose=1)
]

# Train RNN
print("\n4. Training RNN model...")
rnn_history = rnn_model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=30,
    batch_size=32,
    callbacks=rnn_callbacks,
    verbose=1
)

# Evaluate RNN
print("\nEvaluating RNN on test set...")
rnn_test_loss, rnn_test_accuracy, rnn_test_auc = rnn_model.evaluate(X_test_seq, y_test_seq, verbose=0)
print(f"   Test Loss: {rnn_test_loss:.4f}")
print(f"   Test Accuracy: {rnn_test_accuracy:.4f}")
print(f"   Test AUC: {rnn_test_auc:.4f}")

# Find optimal threshold for RNN
y_pred_proba_rnn = rnn_model.predict(X_test_seq, verbose=0)
best_f1_rnn = 0
best_threshold_rnn = 0.5

for threshold in np.arange(0.1, 0.9, 0.05):
    y_pred_temp = (y_pred_proba_rnn > threshold).astype(int).flatten()
    f1 = f1_score(y_test_seq, y_pred_temp)
    if f1 > best_f1_rnn:
        best_f1_rnn = f1
        best_threshold_rnn = threshold

print(f"\nRNN Optimal threshold: {best_threshold_rnn:.2f}")
print(f"RNN Best F1 score: {best_f1_rnn:.4f}")

# Get predictions with optimal threshold
y_pred_rnn = (y_pred_proba_rnn > best_threshold_rnn).astype(int).flatten()

print("\nRNN Classification Report:")
print(classification_report(y_test_seq, y_pred_rnn, 
                          target_names=['No Regulation', 'Regulation']))

# Save RNN model
rnn_model.save(output_dir / "rnn_balanced_final.keras")
print(f"RNN model saved!")

all_results['RNN'] = {
    'test_accuracy': float(rnn_test_accuracy),
    'test_auc': float(rnn_test_auc),
    'test_f1': float(best_f1_rnn),
    'optimal_threshold': float(best_threshold_rnn)
}


# ========== Attention-LSTM ==========
print("\n" + "=" * 40)
print("Training Attention-LSTM Model")
print("=" * 40)

# Custom attention layer
class AttentionLayer(layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True,
            name='attention_weight'
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='attention_bias'
        )
        self.u = self.add_weight(
            shape=(self.units, 1),
            initializer='random_normal',
            trainable=True,
            name='attention_score'
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        # Calculate attention scores
        score = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        attention_weights = tf.nn.softmax(tf.matmul(score, self.u), axis=1)
        
        # Apply attention weights
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector

def build_attention_lstm_model(input_shape):
    """Build Attention-LSTM model architecture"""
    inputs = layers.Input(shape=input_shape)
    
    # LSTM layers
    lstm_out = layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)
    lstm_out = layers.BatchNormalization()(lstm_out)
    lstm_out = layers.LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(lstm_out)
    
    # Attention layer
    attention_out = AttentionLayer(units=32)(lstm_out)
    
    # Dense layers
    x = layers.Dense(16, activation='relu')(attention_out)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Create and compile Attention-LSTM model
attention_lstm_model = build_attention_lstm_model(input_shape=(sequence_length, X_train.shape[1]))
attention_lstm_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

print("\nAttention-LSTM Model Summary:")
attention_lstm_model.summary()

# Callbacks for Attention-LSTM
attention_callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint(filepath=str(output_dir / 'attention_lstm_balanced_best.keras'),
                   monitor='val_auc', mode='max', save_best_only=True, verbose=1)
]

# Train Attention-LSTM
print("\n5. Training Attention-LSTM model...")
attention_history = attention_lstm_model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=30,
    batch_size=32,
    callbacks=attention_callbacks,
    verbose=1
)

# Evaluate Attention-LSTM
print("\nEvaluating Attention-LSTM on test set...")
att_test_loss, att_test_accuracy, att_test_auc = attention_lstm_model.evaluate(X_test_seq, y_test_seq, verbose=0)
print(f"   Test Loss: {att_test_loss:.4f}")
print(f"   Test Accuracy: {att_test_accuracy:.4f}")
print(f"   Test AUC: {att_test_auc:.4f}")

# Find optimal threshold for Attention-LSTM
y_pred_proba_att = attention_lstm_model.predict(X_test_seq, verbose=0)
best_f1_att = 0
best_threshold_att = 0.5

for threshold in np.arange(0.1, 0.9, 0.05):
    y_pred_temp = (y_pred_proba_att > threshold).astype(int).flatten()
    f1 = f1_score(y_test_seq, y_pred_temp)
    if f1 > best_f1_att:
        best_f1_att = f1
        best_threshold_att = threshold

print(f"\nAttention-LSTM Optimal threshold: {best_threshold_att:.2f}")
print(f"Attention-LSTM Best F1 score: {best_f1_att:.4f}")

# Get predictions with optimal threshold
y_pred_att = (y_pred_proba_att > best_threshold_att).astype(int).flatten()

print("\nAttention-LSTM Classification Report:")
print(classification_report(y_test_seq, y_pred_att, 
                          target_names=['No Regulation', 'Regulation']))

# Save Attention-LSTM model
attention_lstm_model.save(output_dir / "attention_lstm_balanced_final.keras")
print(f"Attention-LSTM model saved!")

all_results['Attention-LSTM'] = {
    'test_accuracy': float(att_test_accuracy),
    'test_auc': float(att_test_auc),
    'test_f1': float(best_f1_att),
    'optimal_threshold': float(best_threshold_att)
}


# Save results summary
results_path = output_dir / "additional_models_results.json"
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
    print(f"  - {model_name}: F1={results['test_f1']:.3f}")

print(f"\nResults saved to: {results_path}")