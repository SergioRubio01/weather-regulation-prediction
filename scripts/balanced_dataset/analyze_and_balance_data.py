"""
Analyze regulations data and create a balanced dataset for better model training
This script:
1. Analyzes all regulations to find airports with highest regulation rates
2. Combines multiple airports for better data balance
3. Implements smart sampling strategies to achieve ~50% regulation rate
"""

import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

print("=" * 80)
print("ANALYZING REGULATIONS DATA FOR BALANCED DATASET CREATION")
print("=" * 80)

# 1. Load and analyze full regulations data
print("\n1. Loading and analyzing full regulations data...")
regulations_df = pd.read_csv("data/Regulations/List of Regulations.csv", encoding="utf-8-sig")
regulations_df["Regulation Start Time"] = pd.to_datetime(regulations_df["Regulation Start Time"])
regulations_df["Regulation End Date"] = pd.to_datetime(regulations_df["Regulation End Date"])

print(f"   Total regulations: {len(regulations_df):,}")
print(
    f"   Date range: {regulations_df['Regulation Start Time'].min()} to {regulations_df['Regulation Start Time'].max()}"
)

# Filter for aerodrome regulations only
aerodrome_regs = regulations_df[regulations_df["Protected Location Type"] == "Aerodrome"].copy()
print(f"   Aerodrome regulations: {len(aerodrome_regs):,}")

# Analyze by airport
print("\n2. Analyzing regulations by airport...")
airport_stats = []

for airport in aerodrome_regs["Protected Location Id"].unique():
    airport_data = aerodrome_regs[aerodrome_regs["Protected Location Id"] == airport]
    active_regs = airport_data[airport_data["Regulation Cancel Status"] != "Cancelled"]

    stats = {
        "Airport": airport,
        "Total_Regulations": len(airport_data),
        "Active_Regulations": len(active_regs),
        "Cancelled_Regulations": len(airport_data) - len(active_regs),
        "Cancel_Rate": (len(airport_data) - len(active_regs)) / len(airport_data) * 100,
        "Start_Date": airport_data["Regulation Start Time"].min(),
        "End_Date": airport_data["Regulation Start Time"].max(),
        "Avg_Duration_Min": (
            active_regs["Regulation Duration (min)"].mean() if len(active_regs) > 0 else 0
        ),
        "Total_Delay_Min": active_regs["ATFM Delay (min)"].sum() if len(active_regs) > 0 else 0,
    }
    airport_stats.append(stats)

airport_stats_df = pd.DataFrame(airport_stats).sort_values("Active_Regulations", ascending=False)

print("\n   Top 15 airports by active regulations:")
print(
    airport_stats_df[["Airport", "Active_Regulations", "Total_Regulations", "Cancel_Rate"]]
    .head(15)
    .to_string(index=False)
)

# 3. Select airports for balanced dataset
print("\n3. Selecting airports for balanced dataset...")


# Strategy: Select top airports with most active regulations
# Also check if we have METAR data for these airports
available_metar_airports = []

for file in os.listdir("data/METAR"):
    if file.startswith("METAR_") and file.endswith("_filtered.csv"):
        airport = file.replace("METAR_", "").replace("_filtered.csv", "")
        available_metar_airports.append(airport)

print(f"\n   Airports with METAR data available: {available_metar_airports}")

# Select airports that have both high regulations AND METAR data
selected_airports = []
for _, row in airport_stats_df.iterrows():
    if row["Airport"] in available_metar_airports and row["Active_Regulations"] > 20:
        selected_airports.append(row["Airport"])
        if len(selected_airports) >= 5:  # Limit to top 5 airports
            break

if not selected_airports:
    # If no overlap, just use EGLL
    selected_airports = ["EGLL"]
    print("\n   WARNING: No airports with both high regulations and METAR data. Using only EGLL.")
else:
    print(f"\n   Selected airports for analysis: {selected_airports}")

# 4. Analyze regulation patterns for selected airports
print("\n4. Analyzing regulation patterns for selected airports...")

selected_regs = aerodrome_regs[aerodrome_regs["Protected Location Id"].isin(selected_airports)]
active_selected_regs = selected_regs[selected_regs["Regulation Cancel Status"] != "Cancelled"]

print(f"   Total regulations for selected airports: {len(selected_regs):,}")
print(f"   Active regulations: {len(active_selected_regs):,}")
print(
    f"   Cancel rate: {(len(selected_regs) - len(active_selected_regs)) / len(selected_regs) * 100:.1f}%"
)

# Analyze regulation reasons
print("\n   Top regulation reasons:")
reason_counts = active_selected_regs["Regulation Reason Name"].value_counts().head(10)
for reason, count in reason_counts.items():
    print(f"   - {reason}: {count} ({count/len(active_selected_regs)*100:.1f}%)")

# Check if weather-related reasons are significant
weather_keywords = [
    "W - Weather",
    "Weather",
    "WX",
    "CB",
    "TS",
    "FOG",
    "WIND",
    "SNOW",
    "ICE",
    "T - Equipment (Terminal)",
    "I - Ind Action",
    "V - Environmental Issues",
]
weather_related = active_selected_regs[
    active_selected_regs["Regulation Reason Name"].str.contains(
        "|".join(weather_keywords), case=False, na=False
    )
]
print(
    f"\n   Weather/environment-related regulations: {len(weather_related)} ({len(weather_related)/len(active_selected_regs)*100:.1f}%)"
)

# 5. Save configuration for balanced preprocessing
print("\n5. Creating configuration for balanced dataset...")

config = {
    "selected_airports": selected_airports,
    "use_multi_airport": len(selected_airports) > 1,
    "target_balance_ratio": 0.5,  # Aim for 50% positive samples
    "sampling_strategy": "time_window",  # Focus on time windows around regulations
    "time_window_before_hours": 6,  # Hours before regulation to include
    "time_window_after_hours": 2,  # Hours after regulation to include
    "include_weather_reasons_only": False,  # Include all regulation types
    "min_regulation_duration_min": 30,  # Filter out very short regulations
    "regulation_stats": {
        "total_active_regulations": len(active_selected_regs),
        "avg_duration_hours": active_selected_regs["Regulation Duration (min)"].mean() / 60,
        "weather_related_pct": len(weather_related) / len(active_selected_regs) * 100,
    },
}

# Save configuration

with open("data/balanced_dataset_config.json", "w") as f:
    json.dump(config, f, indent=2, default=str)

print("\n   Configuration saved to: data/balanced_dataset_config.json")
print(f"   Selected airports: {config['selected_airports']}")
print(f"   Multi-airport mode: {config['use_multi_airport']}")
print(f"   Target balance ratio: {config['target_balance_ratio']}")

# 6. Create regulation time windows for efficient sampling
print("\n6. Creating regulation time windows for balanced sampling...")

regulation_windows = []
for _, reg in active_selected_regs.iterrows():
    if reg["Regulation Duration (min)"] >= config["min_regulation_duration_min"]:
        window = {
            "airport": reg["Protected Location Id"],
            "start": reg["Regulation Start Time"]
            - pd.Timedelta(hours=config["time_window_before_hours"]),
            "end": reg["Regulation End Date"]
            + pd.Timedelta(hours=config["time_window_after_hours"]),
            "regulation_start": reg["Regulation Start Time"],
            "regulation_end": reg["Regulation End Date"],
            "reason": reg["Regulation Reason Name"],
            "duration_hours": reg["Regulation Duration (min)"] / 60,
        }
        regulation_windows.append(window)

print(f"   Created {len(regulation_windows)} regulation time windows")
print(
    f"   Average window duration: {np.mean([w['duration_hours'] + config['time_window_before_hours'] + config['time_window_after_hours'] for w in regulation_windows]):.1f} hours"
)

# Save windows for use in preprocessing
pd.DataFrame(regulation_windows).to_csv("data/regulation_time_windows.csv", index=False)
print("   Saved regulation windows to: data/regulation_time_windows.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nNext step: Run prepare_balanced_data.py to create the balanced dataset")
print("\nSummary:")
print(f"  - Selected {len(selected_airports)} airport(s) for analysis")
print(f"  - Found {len(active_selected_regs)} active regulations")
print(f"  - Weather-related: {len(weather_related)/len(active_selected_regs)*100:.1f}%")
print("  - Strategy: Focus on time windows around regulations for balanced sampling")
