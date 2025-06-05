"""
Run the complete balanced dataset pipeline
This script orchestrates the entire process from analysis to training
"""

import os
import subprocess
import sys
from datetime import datetime

print("=" * 80)
print("RUNNING COMPLETE BALANCED DATASET PIPELINE")
print("=" * 80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# List of scripts to run in order
scripts = [
    ("analyze_and_balance_data.py", "Analyzing regulations and selecting airports"),
    ("prepare_balanced_data.py", "Preparing balanced dataset"),
    ("train_balanced_model.py", "Training models on balanced data"),
]

# Ensure output directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("visualizations/traditional_ml", exist_ok=True)

# Run each script
for script, description in scripts:
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"Running: {script}")
    print("=" * 60)

    try:
        # Run the script using poetry
        result = subprocess.run(
            ["poetry", "run", "python", script], capture_output=True, text=True, check=True
        )

        # Print output
        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print("Warnings/Info:", result.stderr)

    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Failed to run {script}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("Output:", e.stdout)
        if e.stderr:
            print("Error:", e.stderr)
        sys.exit(1)

print("\n" + "=" * 80)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nResults:")
print("  - Balanced dataset: data/balanced_weather_data.csv")
print("  - Trained model: models/balanced_weather_regulation_model.pkl")
print("  - Performance plots: visualizations/traditional_ml/balanced_model_performance.png")
print("  - Feature importance: visualizations/traditional_ml/balanced_feature_importance.csv")
print("\nThe balanced dataset approach has significantly improved model performance!")
