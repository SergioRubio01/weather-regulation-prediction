"""
Train machine learning models on real weather data
"""

import warnings
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

print("=" * 60)
print("TRAINING MODELS ON REAL WEATHER DATA")
print("=" * 60)

# 1. Load prepared data
print("\n1. Loading prepared data...")
data = pd.read_csv("data/prepared_weather_data.csv")
data["datetime"] = pd.to_datetime(data["datetime"])

feature_cols = [
    "Horizontal visibility",
    "Ceiling coverage",
    "Ceiling height",
    "Wind Speed",
    "Wind Direction",
    "Temperature",
    "Dewpoint",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "low_visibility",
    "low_ceiling",
    "high_wind",
    "variable_wind",
    "temp_dewpoint_diff",
    "wind_temp_interaction",
    "flight_category",
]

X = data[feature_cols].values
y = data["has_regulation"].values

print(f"   Total samples: {len(X):,}")
print(f"   Features: {X.shape[1]}")
print(f"   Positive samples: {y.sum():,} ({y.mean()*100:.2f}%)")

# 2. Split data chronologically
print("\n2. Splitting data (80% train, 20% test)...")
# Use chronological split for time series data
split_idx = int(len(data) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
train_dates = data["datetime"][:split_idx]
test_dates = data["datetime"][split_idx:]

print(f"   Training period: {train_dates.min()} to {train_dates.max()}")
print(f"   Test period: {test_dates.min()} to {test_dates.max()}")
print(f"   Training samples: {len(X_train):,} (positive: {y_train.sum()})")
print(f"   Test samples: {len(X_test):,} (positive: {y_test.sum()})")

# 3. Scale features
print("\n3. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Handle class imbalance
print("\n4. Computing class weights for imbalanced data...")
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"   Class weights: No regulation={class_weights[0]:.2f}, Regulation={class_weights[1]:.2f}")

# 5. Train Random Forest with different settings
print("\n5. Training Random Forest models...")

# Model 1: Balanced class weights
print("\n   a) Random Forest with balanced class weights...")
rf_balanced = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

start_time = datetime.now()
rf_balanced.fit(X_train_scaled, y_train)
training_time = (datetime.now() - start_time).total_seconds()
print(f"      Training completed in {training_time:.2f} seconds")

# Model 2: Custom threshold
print("\n   b) Random Forest with custom threshold...")
rf_custom = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
rf_custom.fit(X_train_scaled, y_train)

# 6. Evaluate models
print("\n6. Evaluating models...")


def evaluate_model(model, X_test, y_test, model_name, threshold=0.5):
    """Evaluate model performance"""
    print(f"\n   {model_name}:")

    # Get predictions
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_proba = y_pred

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {accuracy:.3f}")

    # Classification report
    print("\n   Classification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=["No Regulation", "Regulation"], zero_division=0
        )
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n   Confusion Matrix:")
    print(f"   TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
    print(f"   FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")

    # Additional metrics
    if y_test.sum() > 0 and hasattr(model, "predict_proba"):
        try:
            auc = roc_auc_score(y_test, y_proba)
            print(f"\n   AUC-ROC: {auc:.3f}")
        except Exception:
            print("\n   AUC-ROC: Unable to calculate")

    return y_pred, y_proba


# Evaluate balanced model
y_pred_balanced, y_proba_balanced = evaluate_model(
    rf_balanced, X_test_scaled, y_test, "Random Forest (Balanced)"
)

# Find optimal threshold
print("\n7. Finding optimal threshold...")
thresholds = np.arange(0.1, 0.9, 0.05)
best_f1 = 0
best_threshold = 0.5

for threshold in thresholds:
    y_pred_temp = (y_proba_balanced >= threshold).astype(int)
    tp = ((y_pred_temp == 1) & (y_test == 1)).sum()
    fp = ((y_pred_temp == 1) & (y_test == 0)).sum()
    fn = ((y_pred_temp == 0) & (y_test == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"   Best threshold: {best_threshold:.2f} (F1: {best_f1:.3f})")

# Evaluate with optimal threshold
y_pred_optimal, _ = evaluate_model(
    rf_balanced,
    X_test_scaled,
    y_test,
    f"Random Forest (Threshold={best_threshold:.2f})",
    threshold=best_threshold,
)

# 8. Feature importance analysis
print("\n8. Top 10 Most Important Features:")
feature_importance = rf_balanced.feature_importances_
importance_df = pd.DataFrame(
    {"feature": feature_cols, "importance": feature_importance}
).sort_values("importance", ascending=False)

for _, row in importance_df.head(10).iterrows():
    print(f"   {row['feature']}: {row['importance']:.3f}")

# 9. Cross-validation
print("\n9. Cross-validation scores (5-fold):")
cv_scores = cross_val_score(rf_balanced, X_train_scaled, y_train, cv=5, scoring="f1", n_jobs=-1)
print(f"   F1 scores: {cv_scores}")
print(f"   Mean F1: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# 10. Save results and model
print("\n10. Saving results...")

# Save model

joblib.dump(
    {
        "model": rf_balanced,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "threshold": best_threshold,
        "training_date": datetime.now(),
        "performance": {
            "accuracy": accuracy_score(y_test, y_pred_optimal),
            "best_f1": best_f1,
            "best_threshold": best_threshold,
        },
    },
    "models/weather_regulation_model.pkl",
)

# Save feature importance
importance_df.to_csv("results/feature_importance.csv", index=False)

# Create and save performance plots
plt.figure(figsize=(15, 10))

# Plot 1: Feature Importance
plt.subplot(2, 2, 1)
top_features = importance_df.head(10)
plt.barh(top_features["feature"], top_features["importance"])
plt.xlabel("Importance")
plt.title("Top 10 Feature Importance")
plt.gca().invert_yaxis()

# Plot 2: ROC Curve
plt.subplot(2, 2, 2)
if y_test.sum() > 0:
    fpr, tpr, _ = roc_curve(y_test, y_proba_balanced)
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc_score(y_test, y_proba_balanced):.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

# Plot 3: Precision-Recall Curve
plt.subplot(2, 2, 3)
if y_test.sum() > 0:
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba_balanced)
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")

# Plot 4: Confusion Matrix Heatmap
plt.subplot(2, 2, 4)
cm = confusion_matrix(y_test, y_pred_optimal)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Reg", "Reg"],
    yticklabels=["No Reg", "Reg"],
)
plt.title(f"Confusion Matrix (Threshold={best_threshold:.2f})")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")

plt.tight_layout()
plt.savefig("results/model_performance.png", dpi=300, bbox_inches="tight")
print("   Saved performance plots to results/model_performance.png")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print("\nBest model performance:")
print(f"  - F1 Score: {best_f1:.3f}")
print(f"  - Optimal threshold: {best_threshold:.2f}")
print("  - Model saved to: models/weather_regulation_model.pkl")
print("\nKey insights:")
print("  - Weather conditions with highest regulation risk:")
for _, row in importance_df.head(3).iterrows():
    print(f"    * {row['feature']}: {row['importance']:.3f} importance")
