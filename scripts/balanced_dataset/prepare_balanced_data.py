"""
Prepare a balanced dataset by intelligently sampling around regulation periods
This creates a dataset with approximately 50% positive samples for better model training
"""

import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

print("=" * 80)
print("PREPARING BALANCED WEATHER-REGULATION DATASET")
print("=" * 80)

# 1. Load configuration
print("\n1. Loading balanced dataset configuration...")
with open("data/balanced_dataset_config.json") as f:
    config = json.load(f)

print(f"   Selected airports: {config['selected_airports']}")
print(f"   Target balance ratio: {config['target_balance_ratio']}")
print(
    f"   Time window: {config['time_window_before_hours']}h before to {config['time_window_after_hours']}h after"
)

# 2. Load regulation windows
print("\n2. Loading regulation time windows...")
regulation_windows = pd.read_csv("data/regulation_time_windows.csv")
regulation_windows["start"] = pd.to_datetime(regulation_windows["start"])
regulation_windows["end"] = pd.to_datetime(regulation_windows["end"])
regulation_windows["regulation_start"] = pd.to_datetime(regulation_windows["regulation_start"])
regulation_windows["regulation_end"] = pd.to_datetime(regulation_windows["regulation_end"])

print(f"   Loaded {len(regulation_windows)} regulation windows")

# 3. Load METAR data for all selected airports
print("\n3. Loading METAR data for selected airports...")
all_metar_data = []

for airport in config["selected_airports"]:
    try:
        metar_file = f"data/METAR/METAR_{airport}_filtered.csv"
        metar_df = pd.read_csv(metar_file)
        metar_df["Complete date"] = pd.to_datetime(metar_df["Complete date"])
        metar_df["airport"] = airport
        all_metar_data.append(metar_df)
        print(f"   Loaded {len(metar_df):,} records for {airport}")
    except FileNotFoundError:
        print(f"   WARNING: No METAR data found for {airport}")
        continue

if not all_metar_data:
    print("   ERROR: No METAR data found for any selected airport!")
    exit(1)

# Combine all METAR data
combined_metar = pd.concat(all_metar_data, ignore_index=True)
combined_metar = combined_metar.rename(columns={"Complete date": "datetime"})
print(f"\n   Total METAR records: {len(combined_metar):,}")
print(f"   Date range: {combined_metar['datetime'].min()} to {combined_metar['datetime'].max()}")

# 4. Create labels and sample data intelligently
print("\n4. Creating balanced dataset...")

# First, mark all records as no regulation
combined_metar["has_regulation"] = 0
combined_metar["in_window"] = 0
combined_metar["time_to_regulation"] = np.nan

# Mark regulation periods and windows
for idx, window in regulation_windows.iterrows():
    if window["airport"] in combined_metar["airport"].values:
        # Mark actual regulation period
        reg_mask = (
            (combined_metar["airport"] == window["airport"])
            & (combined_metar["datetime"] >= window["regulation_start"])
            & (combined_metar["datetime"] < window["regulation_end"])
        )
        combined_metar.loc[reg_mask, "has_regulation"] = 1

        # Mark extended window
        window_mask = (
            (combined_metar["airport"] == window["airport"])
            & (combined_metar["datetime"] >= window["start"])
            & (combined_metar["datetime"] < window["end"])
        )
        combined_metar.loc[window_mask, "in_window"] = 1

        # Calculate time to regulation (useful feature)
        window_data = combined_metar[window_mask].copy()
        if len(window_data) > 0:
            time_diff = (
                window["regulation_start"] - window_data["datetime"]
            ).dt.total_seconds() / 3600
            combined_metar.loc[window_mask, "time_to_regulation"] = time_diff

    if idx % 50 == 0:
        print(f"   Processed {idx}/{len(regulation_windows)} windows...")

# 5. Balance the dataset
print("\n5. Balancing the dataset...")

# Separate data by class and window status
positive_samples = combined_metar[combined_metar["has_regulation"] == 1]
in_window_negative = combined_metar[
    (combined_metar["has_regulation"] == 0) & (combined_metar["in_window"] == 1)
]
outside_window_negative = combined_metar[
    (combined_metar["has_regulation"] == 0) & (combined_metar["in_window"] == 0)
]

print("\n   Class distribution:")
print(f"   - Positive samples (regulations): {len(positive_samples):,}")
print(f"   - Negative in window: {len(in_window_negative):,}")
print(f"   - Negative outside window: {len(outside_window_negative):,}")

# Calculate sampling strategy
target_total = len(positive_samples) * 2  # For 50% balance
n_negative_needed = target_total - len(positive_samples)

# Prioritize samples from within windows (more relevant)
n_from_window = min(len(in_window_negative), int(n_negative_needed * 0.7))
n_from_outside = n_negative_needed - n_from_window

print("\n   Sampling strategy:")
print(f"   - Keep all positive samples: {len(positive_samples):,}")
print(f"   - Sample from window negatives: {n_from_window:,}")
print(f"   - Sample from outside negatives: {n_from_outside:,}")

# Sample negative class
np.random.seed(42)
sampled_window_neg = (
    in_window_negative.sample(n=n_from_window, replace=False)
    if n_from_window > 0
    else pd.DataFrame()
)
sampled_outside_neg = (
    outside_window_negative.sample(n=n_from_outside, replace=False)
    if n_from_outside > 0
    else pd.DataFrame()
)

# Combine balanced dataset
balanced_data = pd.concat(
    [positive_samples, sampled_window_neg, sampled_outside_neg], ignore_index=True
)
balanced_data = balanced_data.sort_values(["airport", "datetime"]).reset_index(drop=True)

print("\n   Balanced dataset created:")
print(f"   - Total samples: {len(balanced_data):,}")
print(f"   - Positive ratio: {balanced_data['has_regulation'].mean():.2%}")

# 6. Feature engineering
print("\n6. Engineering features...")

# Clean data
balanced_data["variable_wind"] = (balanced_data["Wind Direction"] == "VRB").astype(int)
# Convert to numeric first, then fill NaN
balanced_data["Wind Direction"] = pd.to_numeric(
    balanced_data["Wind Direction"].replace("VRB", np.nan), errors="coerce"
)
# Calculate median on numeric values only
wind_median = balanced_data["Wind Direction"].median()
balanced_data["Wind Direction"].fillna(wind_median, inplace=True)

# Fill other missing values
for col in balanced_data.select_dtypes(include=[np.number]).columns:
    if balanced_data[col].isnull().any():
        balanced_data[col].fillna(balanced_data[col].median(), inplace=True)

# Time features
balanced_data["hour"] = balanced_data["datetime"].dt.hour
balanced_data["day_of_week"] = balanced_data["datetime"].dt.dayofweek
balanced_data["month"] = balanced_data["datetime"].dt.month
balanced_data["is_weekend"] = (balanced_data["day_of_week"] >= 5).astype(int)

# Weather condition features
balanced_data["low_visibility"] = (balanced_data["Horizontal visibility"] < 5000).astype(int)
balanced_data["very_low_visibility"] = (balanced_data["Horizontal visibility"] < 1000).astype(int)
balanced_data["low_ceiling"] = (balanced_data["Ceiling height"] < 1000).astype(int)
balanced_data["very_low_ceiling"] = (balanced_data["Ceiling height"] < 500).astype(int)
balanced_data["high_wind"] = (balanced_data["Wind Speed"] > 15).astype(int)
balanced_data["very_high_wind"] = (balanced_data["Wind Speed"] > 25).astype(int)

# Derived features
balanced_data["temp_dewpoint_diff"] = balanced_data["Temperature"] - balanced_data["Dewpoint"]
balanced_data["fog_risk"] = (
    (balanced_data["temp_dewpoint_diff"] < 3) & (balanced_data["Horizontal visibility"] < 5000)
).astype(int)
balanced_data["wind_temp_interaction"] = balanced_data["Wind Speed"] * balanced_data["Temperature"]


# Flight category
def get_flight_category(visibility, ceiling):
    if visibility >= 5000 and ceiling >= 3000:
        return 4  # VFR
    elif visibility >= 3000 and ceiling >= 1000:
        return 3  # MVFR
    elif visibility >= 1600 and ceiling >= 500:
        return 2  # IFR
    else:
        return 1  # LIFR


balanced_data["flight_category"] = balanced_data.apply(
    lambda row: get_flight_category(row["Horizontal visibility"], row["Ceiling height"]), axis=1
)

# Add multi-airport features if applicable
if config["use_multi_airport"]:
    # One-hot encode airports
    airport_dummies = pd.get_dummies(balanced_data["airport"], prefix="airport")
    balanced_data = pd.concat([balanced_data, airport_dummies], axis=1)
    print(f"   Added airport indicators for {len(config['selected_airports'])} airports")

# Time-based features for regulation prediction
if "time_to_regulation" in balanced_data.columns:
    balanced_data["hours_to_regulation"] = balanced_data["time_to_regulation"].fillna(999)
    balanced_data["regulation_imminent"] = (
        (balanced_data["hours_to_regulation"] >= 0) & (balanced_data["hours_to_regulation"] <= 2)
    ).astype(int)

# 7. Final feature selection
print("\n7. Selecting final features...")

base_features = [
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
    "very_low_visibility",
    "low_ceiling",
    "very_low_ceiling",
    "high_wind",
    "very_high_wind",
    "variable_wind",
    "temp_dewpoint_diff",
    "fog_risk",
    "wind_temp_interaction",
    "flight_category",
]

# Add time-based features if available
if "regulation_imminent" in balanced_data.columns:
    base_features.extend(["hours_to_regulation", "regulation_imminent"])

# Add airport features if multi-airport
if config["use_multi_airport"]:
    airport_features = [col for col in balanced_data.columns if col.startswith("airport_")]
    base_features.extend(airport_features)

print(f"   Total features: {len(base_features)}")

# 8. Save balanced dataset
print("\n8. Saving balanced dataset...")
output_cols = ["datetime", "airport"] + base_features + ["has_regulation"]
final_data = balanced_data[output_cols]

final_data.to_csv("data/balanced_weather_data.csv", index=False)
print("   Saved to: data/balanced_weather_data.csv")
print(f"   Shape: {final_data.shape}")

# 9. Generate summary statistics
print("\n9. Dataset Summary:")
print("-" * 50)
print(f"Total samples: {len(final_data):,}")
print(f"Date range: {final_data['datetime'].min()} to {final_data['datetime'].max()}")
print(
    f"Positive samples: {final_data['has_regulation'].sum():,} ({final_data['has_regulation'].mean():.1%})"
)
print(f"Features: {len(base_features)}")

if config["use_multi_airport"]:
    print("\nSamples by airport:")
    for airport in config["selected_airports"]:
        airport_data = final_data[final_data["airport"] == airport]
        print(
            f"  {airport}: {len(airport_data):,} samples, "
            f"{airport_data['has_regulation'].mean():.1%} positive"
        )

print("\nRegulation rate by conditions:")
print(
    f"  Low visibility: {final_data[final_data['low_visibility']==1]['has_regulation'].mean():.1%}"
)
print(
    f"  Very low visibility: {final_data[final_data['very_low_visibility']==1]['has_regulation'].mean():.1%}"
)
print(f"  Low ceiling: {final_data[final_data['low_ceiling']==1]['has_regulation'].mean():.1%}")
print(f"  High wind: {final_data[final_data['high_wind']==1]['has_regulation'].mean():.1%}")
print(f"  Fog risk: {final_data[final_data['fog_risk']==1]['has_regulation'].mean():.1%}")

print("\n" + "=" * 80)
print("BALANCED DATASET PREPARATION COMPLETE!")
print("=" * 80)
print("\nNext step: Run train_balanced_model.py to train on the balanced dataset")
print("\nKey improvements:")
print(f"  ✓ Balanced dataset with {final_data['has_regulation'].mean():.1%} positive samples")
print("  ✓ Focused sampling around regulation windows")
print("  ✓ Enhanced features including severe weather indicators")
if config["use_multi_airport"]:
    print("  ✓ Multi-airport analysis for better generalization")
