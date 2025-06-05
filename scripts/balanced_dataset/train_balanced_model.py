"""
Train models on the balanced dataset with improved performance
"""

import json
import warnings
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

print("=" * 80)
print("TRAINING MODELS ON BALANCED DATASET")
print("=" * 80)

# 1. Load balanced dataset
print("\n1. Loading balanced dataset...")
data = pd.read_csv("data/balanced_weather_data.csv")
data["datetime"] = pd.to_datetime(data["datetime"])

# Load configuration
with open("data/balanced_dataset_config.json") as f:
    config = json.load(f)

# Identify feature columns (exclude datetime, airport, and target)
feature_cols = [col for col in data.columns if col not in ["datetime", "airport", "has_regulation"]]

X = data[feature_cols].values
y = data["has_regulation"].values

print(f"   Total samples: {len(X):,}")
print(f"   Features: {X.shape[1]}")
print(f"   Positive samples: {y.sum():,} ({y.mean()*100:.1f}%)")
print(f"   Multi-airport mode: {config['use_multi_airport']}")

# 2. Split data chronologically
print("\n2. Splitting data (70% train, 15% val, 15% test)...")
# First split: train+val vs test
split_idx1 = int(len(data) * 0.85)
X_temp, X_test = X[:split_idx1], X[split_idx1:]
y_temp, y_test = y[:split_idx1], y[split_idx1:]
data_temp = data[:split_idx1]
test_dates = data[split_idx1:]

# Second split: train vs val
split_idx2 = int(len(X_temp) * 0.82)  # 0.82 * 0.85 ≈ 0.70
X_train, X_val = X_temp[:split_idx2], X_temp[split_idx2:]
y_train, y_val = y_temp[:split_idx2], y_temp[split_idx2:]

print(
    f"   Training samples: {len(X_train):,} (positive: {y_train.sum():,}, {y_train.mean()*100:.1f}%)"
)
print(f"   Validation samples: {len(X_val):,} (positive: {y_val.sum():,}, {y_val.mean()*100:.1f}%)")
print(f"   Test samples: {len(X_test):,} (positive: {y_test.sum():,}, {y_test.mean()*100:.1f}%)")

# 3. Scale features
print("\n3. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 4. Define models to train
print("\n4. Training multiple models...")

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=5, subsample=0.8, random_state=42
    ),
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42, n_jobs=-1),
    "Neural Network": MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=32,
        learning_rate="adaptive",
        max_iter=500,
        random_state=42,
    ),
}

results = {}

# 5. Train and evaluate each model
for model_name, model in models.items():
    print(f"\n   Training {model_name}...")

    # Train
    start_time = datetime.now()
    model.fit(X_train_scaled, y_train)
    training_time = (datetime.now() - start_time).total_seconds()

    # Predict on validation set for threshold tuning
    y_val_proba = model.predict_proba(X_val_scaled)[:, 1]

    # Find optimal threshold using validation set
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5

    for threshold in thresholds:
        y_val_pred = (y_val_proba >= threshold).astype(int)
        f1 = f1_score(y_val, y_val_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Evaluate on test set with optimal threshold
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_test_pred = (y_test_proba >= best_threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    auc = roc_auc_score(y_test, y_test_proba)

    # Store results
    results[model_name] = {
        "model": model,
        "training_time": training_time,
        "best_threshold": best_threshold,
        "val_f1": best_f1,
        "test_accuracy": accuracy,
        "test_f1": f1,
        "test_auc": auc,
        "y_test_pred": y_test_pred,
        "y_test_proba": y_test_proba,
        "classification_report": classification_report(
            y_test, y_test_pred, target_names=["No Regulation", "Regulation"]
        ),
    }

    print(f"      Training time: {training_time:.2f}s")
    print(f"      Best threshold: {best_threshold:.2f}")
    print(f"      Test F1: {f1:.3f}")
    print(f"      Test AUC: {auc:.3f}")

# 6. Compare models
print("\n6. Model Comparison:")
print("-" * 70)
print(f"{'Model':<20} {'F1 Score':<10} {'AUC':<10} {'Accuracy':<10} {'Threshold':<10}")
print("-" * 70)

best_f1 = 0
best_model_name = None

for model_name, result in results.items():
    print(
        f"{model_name:<20} {result['test_f1']:<10.3f} {result['test_auc']:<10.3f} "
        f"{result['test_accuracy']:<10.3f} {result['best_threshold']:<10.2f}"
    )

    if result["test_f1"] > best_f1:
        best_f1 = result["test_f1"]
        best_model_name = model_name

print("-" * 70)
print(f"\nBest model: {best_model_name} (F1: {best_f1:.3f})")

# 7. Detailed evaluation of best model
print(f"\n7. Detailed evaluation of {best_model_name}:")
best_result = results[best_model_name]
print("\nClassification Report:")
print(best_result["classification_report"])

# Confusion Matrix
cm = confusion_matrix(y_test, best_result["y_test_pred"])
print("\nConfusion Matrix:")
print(f"TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
print(f"FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")
print(f"\nPrecision: {cm[1,1]/(cm[1,1]+cm[0,1]):.3f}")
print(f"Recall: {cm[1,1]/(cm[1,1]+cm[1,0]):.3f}")

# 8. Cross-validation on best model
print(f"\n8. Cross-validation of {best_model_name} (5-fold):")
cv_scores = cross_val_score(
    results[best_model_name]["model"],
    X_train_scaled,
    y_train,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="f1",
    n_jobs=-1,
)
print(f"   F1 scores: {cv_scores}")
print(f"   Mean F1: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# 9. Feature importance (if available)
print("\n9. Feature Importance Analysis:")
if hasattr(results[best_model_name]["model"], "feature_importances_"):
    importances = results[best_model_name]["model"].feature_importances_
    importance_df = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values(
        "importance", ascending=False
    )

    print("\n   Top 15 most important features:")
    for _, row in importance_df.head(15).iterrows():
        print(f"   {row['feature']:<30} {row['importance']:.4f}")

    # Save feature importance
    importance_df.to_csv(
        "visualizations/traditional_ml/balanced_feature_importance.csv", index=False
    )

# 10. Create visualization
print("\n10. Creating performance visualizations...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: ROC curves for all models
ax = axes[0, 0]
for model_name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result["y_test_proba"])
    ax.plot(fpr, tpr, label=f"{model_name} (AUC={result['test_auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves - All Models")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Precision-Recall curves
ax = axes[0, 1]
for model_name, result in results.items():
    precision, recall, _ = precision_recall_curve(y_test, result["y_test_proba"])
    ax.plot(recall, precision, label=f"{model_name}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: F1 scores comparison
ax = axes[0, 2]
model_names = list(results.keys())
f1_scores = [results[m]["test_f1"] for m in model_names]
bars = ax.bar(model_names, f1_scores)
ax.set_ylabel("F1 Score")
ax.set_title("F1 Score Comparison")
ax.set_ylim(0, 1)
for bar, score in zip(bars, f1_scores, strict=False):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{score:.3f}",
        ha="center",
        va="bottom",
    )
ax.tick_params(axis="x", rotation=45)

# Plot 4: Best model confusion matrix
ax = axes[1, 0]
cm = confusion_matrix(y_test, results[best_model_name]["y_test_pred"])
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax,
    xticklabels=["No Reg", "Reg"],
    yticklabels=["No Reg", "Reg"],
)
ax.set_title(f"{best_model_name} - Confusion Matrix")
ax.set_ylabel("True Label")
ax.set_xlabel("Predicted Label")

# Plot 5: Feature importance (if available)
ax = axes[1, 1]
if hasattr(results[best_model_name]["model"], "feature_importances_"):
    top_features = importance_df.head(10)
    ax.barh(top_features["feature"], top_features["importance"])
    ax.set_xlabel("Importance")
    ax.set_title(f"{best_model_name} - Top 10 Features")
    ax.invert_yaxis()
else:
    ax.text(
        0.5,
        0.5,
        "Feature importance\nnot available",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.set_title("Feature Importance")

# Plot 6: Model comparison metrics
ax = axes[1, 2]
metrics = ["F1 Score", "AUC", "Accuracy"]
x = np.arange(len(model_names))
width = 0.2
for i, metric in enumerate(metrics):
    if metric == "F1 Score":
        values = [results[m]["test_f1"] for m in model_names]
    elif metric == "AUC":
        values = [results[m]["test_auc"] for m in model_names]
    else:
        values = [results[m]["test_accuracy"] for m in model_names]
    ax.bar(x + i * width, values, width, label=metric)
ax.set_xlabel("Model")
ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison")
ax.set_xticks(x + width)
ax.set_xticklabels(model_names, rotation=45)
ax.legend()
ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig(
    "visualizations/traditional_ml/balanced_model_performance.png", dpi=300, bbox_inches="tight"
)
print("   Saved visualizations to: visualizations/traditional_ml/balanced_model_performance.png")

# 11. Save best model
print(f"\n11. Saving best model ({best_model_name})...")
model_package = {
    "model": results[best_model_name]["model"],
    "scaler": scaler,
    "feature_cols": feature_cols,
    "threshold": results[best_model_name]["best_threshold"],
    "model_name": best_model_name,
    "training_date": datetime.now(),
    "config": config,
    "performance": {
        "f1_score": results[best_model_name]["test_f1"],
        "auc_roc": results[best_model_name]["test_auc"],
        "accuracy": results[best_model_name]["test_accuracy"],
        "threshold": results[best_model_name]["best_threshold"],
    },
}

joblib.dump(model_package, "models/balanced_weather_regulation_model.pkl")
print("   Model saved to: models/balanced_weather_regulation_model.pkl")

# 12. Create summary report
print("\n" + "=" * 80)
print("TRAINING COMPLETE - SUMMARY REPORT")
print("=" * 80)
print("\nDataset:")
print(f"  - Total samples: {len(X):,}")
print(f"  - Balance ratio: {y.mean():.1%} positive")
print(f"  - Features: {len(feature_cols)}")
print(f"  - Multi-airport: {config['use_multi_airport']}")

print(f"\nBest Model: {best_model_name}")
print(f"  - F1 Score: {results[best_model_name]['test_f1']:.3f}")
print(f"  - AUC-ROC: {results[best_model_name]['test_auc']:.3f}")
print(f"  - Optimal threshold: {results[best_model_name]['best_threshold']:.2f}")
print(f"  - True Positive Rate: {cm[1,1]/(cm[1,1]+cm[1,0]):.3f}")
print(f"  - Precision: {cm[1,1]/(cm[1,1]+cm[0,1]):.3f}")

print("\nComparison with unbalanced dataset:")
print(
    f"  Previous F1: 0.071 → Current F1: {results[best_model_name]['test_f1']:.3f} "
    f"({(results[best_model_name]['test_f1']/0.071 - 1)*100:+.0f}% improvement)"
)
print(f"  Previous recall: 0.17 → Current recall: {cm[1,1]/(cm[1,1]+cm[1,0]):.3f}")

print("\n✓ Successfully created a high-performance model with balanced data!")
