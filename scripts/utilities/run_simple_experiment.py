"""
Run a simple experiment with regulation data only
This script creates synthetic weather features to demonstrate the pipeline
"""

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Import our modules

print("=" * 60)
print("RUNNING SIMPLE EXPERIMENT WITH REGULATION DATA")
print("=" * 60)

# Load regulation data
print("\n1. Loading regulation data...")
reg_df = pd.read_csv("data/Regulations/Regulations_EGLL_converted.csv")
reg_df["start_time"] = pd.to_datetime(reg_df["start_time"])
reg_df["end_time"] = pd.to_datetime(reg_df["end_time"])

print(f"   Loaded {len(reg_df)} regulation records")
print(f"   Date range: {reg_df['start_time'].min()} to {reg_df['start_time'].max()}")
print(f"   Cancelled regulations: {reg_df['cancelled'].sum()}")
print(f"   Active regulations: {(~reg_df['cancelled']).sum()}")

# Create synthetic weather features for demonstration
print("\n2. Creating synthetic weather features...")
np.random.seed(42)
n_samples = 10000  # Number of time points

# Create time series
time_index = pd.date_range(
    start=reg_df["start_time"].min(), end=reg_df["start_time"].max(), periods=n_samples
)

# Generate synthetic weather features
features = pd.DataFrame(
    {
        "time": time_index,
        "temperature": np.random.normal(15, 5, n_samples),  # Temperature in Celsius
        "wind_speed": np.abs(np.random.normal(10, 5, n_samples)),  # Wind speed in knots
        "visibility": np.clip(
            np.random.normal(8000, 2000, n_samples), 0, 10000
        ),  # Visibility in meters
        "pressure": np.random.normal(1013, 10, n_samples),  # Pressure in hPa
        "humidity": np.clip(np.random.normal(70, 20, n_samples), 0, 100),  # Humidity percentage
        "ceiling": np.clip(
            np.random.normal(3000, 1000, n_samples), 0, 5000
        ),  # Cloud ceiling in feet
    }
)

# Create binary labels based on regulations
print("\n3. Creating labels based on regulation periods...")
features["has_regulation"] = 0

for _, reg in reg_df.iterrows():
    if not reg["cancelled"]:  # Only consider active regulations
        mask = (features["time"] >= reg["start_time"]) & (features["time"] < reg["end_time"])
        features.loc[mask, "has_regulation"] = 1

print(f"   Total samples: {len(features)}")
print(
    f"   Samples with regulation: {features['has_regulation'].sum()} ({features['has_regulation'].mean()*100:.1f}%)"
)

# Add derived features
print("\n4. Engineering features...")
features["temp_wind_interaction"] = features["temperature"] * features["wind_speed"]
features["low_visibility"] = (features["visibility"] < 5000).astype(int)
features["high_wind"] = (features["wind_speed"] > 20).astype(int)
features["hour"] = features["time"].dt.hour
features["day_of_week"] = features["time"].dt.dayofweek
features["month"] = features["time"].dt.month

# Prepare data for training
print("\n5. Preparing data for training...")
feature_cols = [
    "temperature",
    "wind_speed",
    "visibility",
    "pressure",
    "humidity",
    "ceiling",
    "temp_wind_interaction",
    "low_visibility",
    "high_wind",
    "hour",
    "day_of_week",
    "month",
]

X = features[feature_cols].values
y = features["has_regulation"].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")
print(f"   Training class distribution: {np.bincount(y_train)}")
print(f"   Test class distribution: {np.bincount(y_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
print("\n6. Training Random Forest model...")
# Use sklearn directly for simplicity
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1
)

# Train
start_time = datetime.now()
rf_model.fit(X_train_scaled, y_train)
training_time = (datetime.now() - start_time).total_seconds()
print(f"   Training completed in {training_time:.2f} seconds")

# Evaluate
print("\n7. Evaluating model...")
y_pred = rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n   Test Accuracy: {accuracy:.3f}")
print("\n   Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Regulation", "Regulation"]))

print("\n   Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"   True Negatives:  {cm[0,0]}")
print(f"   False Positives: {cm[0,1]}")
print(f"   False Negatives: {cm[1,0]}")
print(f"   True Positives:  {cm[1,1]}")

# Feature importance
print("\n8. Feature Importance:")
feature_importance = rf_model.feature_importances_
importance_df = pd.DataFrame(
    {"feature": feature_cols, "importance": feature_importance}
).sort_values("importance", ascending=False)

for _, row in importance_df.head(5).iterrows():
    print(f"   {row['feature']}: {row['importance']:.3f}")

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\nNOTE: This experiment used synthetic weather data for demonstration.")
print("For real experiments, you'll need actual METAR/TAF weather data.")
print("\nTo enable GPU support:")
print("1. Install CUDA 12.5")
print("2. Install cuDNN 9.5")
print("3. Set CUDA_PATH environment variable")
print("4. Restart your terminal")
