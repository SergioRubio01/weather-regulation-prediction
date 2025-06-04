# Balanced Dataset Procedure for Weather Regulation Prediction

This document details the complete procedure for creating a balanced dataset and training high-performance models for weather regulation prediction. This approach addresses the severe class imbalance issue (1.1% positive samples) and achieves exceptional results.

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution Overview](#solution-overview)
3. [Data Files Created](#data-files-created)
4. [Step-by-Step Procedure](#step-by-step-procedure)
5. [Key Changes Made](#key-changes-made)
6. [Results Summary](#results-summary)
7. [Commands Reference](#commands-reference)

## Problem Statement

The original dataset had severe class imbalance:

- Total samples: 51,455
- Positive samples (with regulation): 564 (1.1%)
- Negative samples: 50,891 (98.9%)

This led to poor model performance:

- F1 Score: 0.071
- Recall: 17% (missed 83% of regulations)
- Model predicted almost everything as "no regulation"

## Solution Overview

The solution implements an intelligent sampling strategy:

1. **Analyze all regulations** to find airports with most active regulations
2. **Create time windows** around regulation periods (6 hours before, 2 hours after)
3. **Balance the dataset** by focusing on these critical time windows
4. **Engineer advanced features** including severe weather indicators
5. **Train multiple models** and optimize classification thresholds

## Data Files Created

### Input Files Used

- `/data/Regulations/List of Regulations.csv` - Full regulations database
- `/data/Regulations/Regulations_filtered_EGLL.csv` - EGLL-specific regulations
- `/data/METAR/METAR_EGLL_filtered.csv` - Weather observations

### Generated Files

#### Configuration Files

- `data/balanced_dataset_config.json` - Configuration for balanced dataset creation
- `data/regulation_time_windows.csv` - Time windows around each regulation
- `configs/balanced_experiment.yaml` - Experiment configuration

#### Data Files

- `data/Regulations_EGLL_converted.csv` - Regulations in standardized format
- `data/prepared_weather_data.csv` - Original unbalanced dataset
- `data/balanced_weather_data.csv` - Final balanced dataset

#### Model and Results

- `models/balanced_weather_regulation_model.pkl` - Trained Gradient Boosting model
- `results/balanced_feature_importance.csv` - Feature importance rankings
- `results/balanced_model_performance.png` - Performance visualization plots

#### Scripts Created

- `analyze_and_balance_data.py` - Regulation analysis and configuration
- `prepare_balanced_data.py` - Balanced dataset creation
- `train_balanced_model.py` - Model training and evaluation
- `run_balanced_pipeline.py` - Automated pipeline runner

## Step-by-Step Procedure

### Step 1: Environment Setup

```bash
# Ensure you're in the project directory
cd /mnt/c/Users/Sergio/Dev/weather-regulation-prediction

# Create necessary directories
mkdir -p models results

# Activate Poetry environment
poetry env activate
```

### Step 2: Convert Regulation Data Format

First, convert the regulation data to the expected format:

```python
# convert_regulation_data.py
import pandas as pd

def convert_regulation_data(input_file, output_file):
    df = pd.read_csv(input_file)

    converted_df = pd.DataFrame({
        'airport': df['Protected Location Id'],
        'start_time': pd.to_datetime(df['Regulation Start Time']),
        'end_time': pd.to_datetime(df['Regulation End Date']),
        'reason': df['Regulation Reason Name'],
        'description': df['Regulation Description'],
        'duration_min': df['Regulation Duration (min)'],
        'delay_min': df['ATFM Delay (min)'],
        'regulated_traffic': df['Regulated Traffic'],
        'cancelled': df['Regulation Cancel Status'] == 'Cancelled'
    })

    converted_df.to_csv(output_file, index=False)
```

**Command:**

```bash
poetry run python convert_regulation_data.py
```

### Step 3: Analyze Regulations and Select Airports

The `analyze_and_balance_data.py` script:

- Loads full regulations database (178,439 total regulations)
- Filters for aerodrome regulations (42,399 regulations)
- Ranks airports by active (non-cancelled) regulations
- Selects airports with both high regulations AND available METAR data
- Creates time windows for intelligent sampling

**Command:**

```bash
poetry run python analyze_and_balance_data.py
```

**Key findings:**

- EGLL had 89 active regulations (679 total, 86.9% cancel rate)
- 62.9% of regulations were weather-related
- Created 87 time windows (average 12.6 hours each)

### Step 4: Prepare Balanced Dataset

The `prepare_balanced_data.py` script implements the balancing strategy:

#### Data Cleaning

- Handle 'VRB' (variable) wind direction as a separate binary feature
- Fill missing values with median

#### Intelligent Sampling

- Keep ALL positive samples (798 samples during regulations)
- Sample 70% of negatives from time windows around regulations (558 samples)
- Sample 30% from quiet periods for diversity (240 samples)
- Result: 1,596 total samples with perfect 50% balance

#### Enhanced Feature Engineering

```python
# Severe weather indicators
balanced_data['very_low_visibility'] = (balanced_data['Horizontal visibility'] < 1000).astype(int)
balanced_data['very_low_ceiling'] = (balanced_data['Ceiling height'] < 500).astype(int)
balanced_data['very_high_wind'] = (balanced_data['Wind Speed'] > 25).astype(int)

# Fog risk indicator
balanced_data['fog_risk'] = ((balanced_data['temp_dewpoint_diff'] < 3) &
                             (balanced_data['Horizontal visibility'] < 5000)).astype(int)

# Time-based features
balanced_data['hours_to_regulation'] = balanced_data['time_to_regulation'].fillna(999)
balanced_data['regulation_imminent'] = (
    (balanced_data['hours_to_regulation'] >= 0) &
    (balanced_data['hours_to_regulation'] <= 2)
).astype(int)
```

**Command:**

```bash
poetry run python prepare_balanced_data.py
```

### Step 5: Train Models on Balanced Dataset

The `train_balanced_model.py` script trains and compares 4 models:

1. Random Forest (300 trees)
2. Gradient Boosting
3. Logistic Regression
4. Neural Network (2 hidden layers)

#### Key Training Features

- 70/15/15 train/validation/test split
- Threshold optimization on validation set
- Cross-validation for stability assessment
- Comprehensive evaluation metrics

**Command:**

```bash
poetry run python train_balanced_model.py
```

## Key Changes Made

### 1. Data Preprocessing Changes

#### Original `prepare_real_data.py`

```python
# Simple binary labeling
metar_df['has_regulation'] = 0
for idx, reg in active_regs.iterrows():
    mask = (metar_df['datetime'] >= reg['start_time']) & \
           (metar_df['datetime'] < reg['end_time'])
    metar_df.loc[mask, 'has_regulation'] = 1
```

#### New Balanced Approach

```python
# Track regulation windows and time to regulation
combined_metar['has_regulation'] = 0
combined_metar['in_window'] = 0
combined_metar['time_to_regulation'] = np.nan

for idx, window in regulation_windows.iterrows():
    # Mark actual regulation period
    reg_mask = (
        (combined_metar['airport'] == window['airport']) &
        (combined_metar['datetime'] >= window['regulation_start']) &
        (combined_metar['datetime'] < window['regulation_end'])
    )
    combined_metar.loc[reg_mask, 'has_regulation'] = 1

    # Mark extended window and calculate time to regulation
    window_mask = (
        (combined_metar['airport'] == window['airport']) &
        (combined_metar['datetime'] >= window['start']) &
        (combined_metar['datetime'] < window['end'])
    )
    combined_metar.loc[window_mask, 'in_window'] = 1
```

### 2. Variable Wind Handling

Fixed the issue where 'VRB' (variable wind) was incorrectly converted to 0:

```python
# Create separate feature for variable wind
balanced_data['variable_wind'] = (balanced_data['Wind Direction'] == 'VRB').astype(int)

# Convert to numeric and fill with median
balanced_data['Wind Direction'] = pd.to_numeric(
    balanced_data['Wind Direction'].replace('VRB', np.nan),
    errors='coerce'
)
wind_median = balanced_data['Wind Direction'].median()
balanced_data['Wind Direction'].fillna(wind_median, inplace=True)
```

### 3. Feature Engineering Enhancements

Added 7 new features compared to the original approach:

- `very_low_visibility` - Visibility < 1000m
- `very_low_ceiling` - Ceiling < 500ft  
- `very_high_wind` - Wind speed > 25kt
- `variable_wind` - Wind direction is variable
- `fog_risk` - Low visibility + small temp-dewpoint difference
- `hours_to_regulation` - Time until next regulation
- `regulation_imminent` - Regulation within 2 hours

### 4. Model Training Improvements

- **Threshold Optimization**: Instead of using default 0.5, find optimal threshold on validation set
- **Multiple Models**: Compare 4 different algorithms
- **Better Metrics**: Focus on F1 score instead of accuracy for imbalanced data

## Results Summary

### Performance Comparison

| Metric | Original (Unbalanced) | Balanced Dataset | Improvement |
|--------|----------------------|------------------|-------------|
| **Dataset Size** | 51,455 | 1,596 | -96.9% |
| **Positive Ratio** | 1.1% | 50.0% | +4,445% |
| **F1 Score** | 0.071 | 0.879 | +1,138% |
| **Recall** | 17% | 100% | +488% |
| **Precision** | ~4% | 78.4% | +1,860% |
| **AUC-ROC** | 0.757 | 0.954 | +26% |

### Model Performance (Test Set)

| Model | F1 Score | AUC | Optimal Threshold |
|-------|----------|-----|-------------------|
| **Gradient Boosting** | **0.879** | **0.954** | 0.15 |
| Random Forest | 0.835 | 0.914 | 0.35 |
| Logistic Regression | 0.698 | 0.775 | 0.30 |
| Neural Network | 0.651 | 0.774 | 0.10 |

### Feature Importance (Top 5)

1. `hours_to_regulation` - 67.4%
2. `hour` - 8.3%
3. `Dewpoint` - 3.1%
4. `wind_temp_interaction` - 3.1%
5. `Wind Speed` - 2.6%

### Regulation Detection by Weather Conditions

| Condition | Regulation Rate |
|-----------|----------------|
| High wind (>15kt) | 73.2% |
| Very low visibility (<1000m) | 60.0% |
| Low ceiling (<1000ft) | 58.4% |
| Fog risk | 47.8% |

## Commands Reference

### Complete Pipeline Execution

Run all steps automatically:

```bash
poetry run python run_balanced_pipeline.py
```

### Individual Steps

```bash
# Step 1: Analyze regulations
poetry run python analyze_and_balance_data.py

# Step 2: Prepare balanced dataset  
poetry run python prepare_balanced_data.py

# Step 3: Train models
poetry run python train_balanced_model.py
```

### Testing the Model

```python
import joblib
import pandas as pd

# Load the model
model_package = joblib.load('models/balanced_weather_regulation_model.pkl')
model = model_package['model']
scaler = model_package['scaler']
feature_cols = model_package['feature_cols']
threshold = model_package['threshold']

# Prepare your data with the same features
# ... load and engineer features ...

# Scale features
X_scaled = scaler.transform(X)

# Predict probabilities
probabilities = model.predict_proba(X_scaled)[:, 1]

# Apply optimal threshold
predictions = (probabilities >= threshold).astype(int)
```

## Conclusions

The balanced dataset approach successfully transformed a poorly performing model (F1: 0.071) into a highly effective regulation prediction system (F1: 0.879) by:

1. **Intelligent Sampling**: Focusing on time windows around regulations
2. **Enhanced Features**: Adding severe weather indicators and time-based features
3. **Proper Evaluation**: Using appropriate metrics and threshold optimization
4. **Model Selection**: Testing multiple algorithms to find the best performer

The final model achieves 100% recall (catches all regulations) with 78.4% precision, making it suitable for operational use where missing a regulation is more costly than false alarms.
