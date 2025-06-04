"""
Prepare real METAR weather data and merge with regulation data for training
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

print("=" * 60)
print("PREPARING REAL WEATHER DATA FOR TRAINING")
print("=" * 60)

# 1. Load METAR data
print("\n1. Loading METAR weather data...")
metar_df = pd.read_csv("data/METAR/METAR_EGLL_filtered.csv")
metar_df["Complete date"] = pd.to_datetime(metar_df["Complete date"])
metar_df = metar_df.rename(columns={"Complete date": "datetime"})

print(f"   Loaded {len(metar_df)} METAR records")
print(f"   Date range: {metar_df['datetime'].min()} to {metar_df['datetime'].max()}")
print("\n   Available weather features:")
for col in metar_df.columns[2:]:  # Skip index and datetime
    print(f"   - {col}")

# 2. Load regulation data
print("\n2. Loading regulation data...")
reg_df = pd.read_csv("data/Regulations/Regulations_EGLL_converted.csv")
reg_df["start_time"] = pd.to_datetime(reg_df["start_time"])
reg_df["end_time"] = pd.to_datetime(reg_df["end_time"])

# Filter only active regulations
active_regs = reg_df[~reg_df["cancelled"]].copy()
print(f"   Total regulations: {len(reg_df)}")
print(f"   Active regulations: {len(active_regs)}")
print(f"   Date range: {reg_df['start_time'].min()} to {reg_df['start_time'].max()}")

# 3. Create time-aligned features
print("\n3. Creating time-aligned dataset...")

# Round METAR times to nearest 30 minutes
metar_df["time_rounded"] = metar_df["datetime"].dt.round("30min")

# Create binary labels for regulations
print("   Creating regulation labels...")
metar_df["has_regulation"] = 0

# For each active regulation, mark the corresponding time periods
for idx, reg in active_regs.iterrows():
    # Find METAR records during regulation period
    mask = (metar_df["datetime"] >= reg["start_time"]) & (metar_df["datetime"] < reg["end_time"])
    metar_df.loc[mask, "has_regulation"] = 1

    if idx % 10 == 0:
        print(f"   Processed {idx}/{len(active_regs)} regulations...")

print(f"\n   Total samples: {len(metar_df)}")
print(
    f"   Samples with regulation: {metar_df['has_regulation'].sum()} ({metar_df['has_regulation'].mean()*100:.2f}%)"
)

# 4. Feature engineering
print("\n4. Engineering weather features...")

# Clean and prepare features
weather_features = metar_df.copy()

# Handle missing values and data cleaning
print("   Handling missing values and data cleaning...")

# Handle 'VRB' (variable) wind direction - create a separate feature
weather_features["variable_wind"] = (weather_features["Wind Direction"] == "VRB").astype(int)
print(f"   Found {weather_features['variable_wind'].sum()} records with variable wind direction")

# Convert wind direction to numeric, replacing VRB with NaN temporarily
weather_features["Wind Direction"] = pd.to_numeric(
    weather_features["Wind Direction"].replace("VRB", np.nan), errors="coerce"
)

# For variable winds, use the median wind direction (or you could use previous value)
# This preserves the information that wind was variable in the separate feature
weather_features["Wind Direction"].fillna(weather_features["Wind Direction"].median(), inplace=True)

# Fill missing values
for col in weather_features.select_dtypes(include=[np.number]).columns:
    if weather_features[col].isnull().any():
        weather_features[col].fillna(weather_features[col].median(), inplace=True)

# Add time-based features
weather_features["hour"] = weather_features["datetime"].dt.hour
weather_features["day_of_week"] = weather_features["datetime"].dt.dayofweek
weather_features["month"] = weather_features["datetime"].dt.month
weather_features["is_weekend"] = (weather_features["day_of_week"] >= 5).astype(int)

# Add weather condition features
weather_features["low_visibility"] = (weather_features["Horizontal visibility"] < 5000).astype(int)
weather_features["low_ceiling"] = (weather_features["Ceiling height"] < 1000).astype(int)
weather_features["high_wind"] = (weather_features["Wind Speed"] > 15).astype(int)

# Add interaction features
weather_features["temp_dewpoint_diff"] = (
    weather_features["Temperature"] - weather_features["Dewpoint"]
)
weather_features["wind_temp_interaction"] = (
    weather_features["Wind Speed"] * weather_features["Temperature"]
)


# Flight category based on visibility and ceiling
def get_flight_category(visibility, ceiling):
    """Determine flight category based on weather conditions"""
    if visibility >= 5000 and ceiling >= 3000:
        return 4  # VFR
    elif visibility >= 3000 and ceiling >= 1000:
        return 3  # MVFR
    elif visibility >= 1600 and ceiling >= 500:
        return 2  # IFR
    else:
        return 1  # LIFR


weather_features["flight_category"] = weather_features.apply(
    lambda row: get_flight_category(row["Horizontal visibility"], row["Ceiling height"]), axis=1
)

print("\n   Final feature set:")
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

for col in feature_cols:
    print(f"   - {col}")

# 5. Save prepared dataset
print("\n5. Saving prepared dataset...")
output_df = weather_features[["datetime"] + feature_cols + ["has_regulation"]]
output_df.to_csv("data/prepared_weather_data.csv", index=False)
print("   Saved to: data/prepared_weather_data.csv")
print(f"   Shape: {output_df.shape}")

# 6. Display data summary
print("\n6. Data Summary:")
print("-" * 40)
print(f"Total samples: {len(output_df):,}")
print(f"Date range: {output_df['datetime'].min()} to {output_df['datetime'].max()}")
print(
    f"Positive samples (with regulation): {output_df['has_regulation'].sum():,} ({output_df['has_regulation'].mean()*100:.2f}%)"
)
print(f"Negative samples (no regulation): {(~output_df['has_regulation'].astype(bool)).sum():,}")
print(f"Features: {len(feature_cols)}")

# Check class balance by weather conditions
print("\n7. Regulation rate by weather conditions:")
print("-" * 40)
print(
    f"Low visibility (<5000m): {output_df[output_df['low_visibility']==1]['has_regulation'].mean()*100:.1f}%"
)
print(
    f"Normal visibility: {output_df[output_df['low_visibility']==0]['has_regulation'].mean()*100:.1f}%"
)
print(
    f"Low ceiling (<1000ft): {output_df[output_df['low_ceiling']==1]['has_regulation'].mean()*100:.1f}%"
)
print(f"Normal ceiling: {output_df[output_df['low_ceiling']==0]['has_regulation'].mean()*100:.1f}%")
print(
    f"High wind (>15kt): {output_df[output_df['high_wind']==1]['has_regulation'].mean()*100:.1f}%"
)
print(f"Normal wind: {output_df[output_df['high_wind']==0]['has_regulation'].mean()*100:.1f}%")
print(
    f"Variable wind: {output_df[output_df['variable_wind']==1]['has_regulation'].mean()*100:.1f}%"
)
print(f"Steady wind: {output_df[output_df['variable_wind']==0]['has_regulation'].mean()*100:.1f}%")

print("\n" + "=" * 60)
print("DATA PREPARATION COMPLETE!")
print("=" * 60)
print("\nNext step: Run train_model_real_data.py to train models")
