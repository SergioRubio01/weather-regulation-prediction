# Scripts Directory

This directory contains various scripts organized by their purpose:

## Directory Structure

### `/balanced_dataset/`

Scripts specifically for working with balanced datasets:

- `analyze_and_balance_data.py` - Analyzes data distribution and creates balanced datasets
- `prepare_balanced_data.py` - Prepares balanced data for training
- `train_balanced_model.py` - Trains models on balanced datasets
- `run_balanced_pipeline.py` - Runs the complete balanced dataset pipeline

### `/utilities/`

General utility scripts:

- `check_cuda.py` - Checks CUDA availability and configuration
- `install_cuda_wsl.sh` - Installation script for CUDA on WSL
- `convert_regulation_data.py` - Converts regulation data formats
- `prepare_real_data.py` - Prepares real-world data for training
- `train_model_real_data.py` - Trains models on real data
- `run_simple_experiment.py` - Runs simple experiments for testing

### `/legacy/`

Legacy scripts from the original implementation (maintained for backward compatibility):

- `filter_TAFs.py` - Filters TAF weather forecast data
- `functions.py` - Legacy utility functions
- `model.py` - Original Dly_Classifier implementation
- `plots.py` - Legacy plotting functions

### `/examples/`

Example scripts demonstrating system usage:

- `example_training_pipeline.py` - Demonstrates the training pipeline
- `example_usage.py` - Comprehensive usage examples for both legacy and new systems

## Usage

Most scripts can be run directly from the project root:

```bash
# Run balanced dataset pipeline
python scripts/balanced_dataset/run_balanced_pipeline.py

# Check CUDA availability
python scripts/utilities/check_cuda.py

# Run example usage
python scripts/examples/example_usage.py
```

Note: Some scripts may require adjusting import paths if run from different directories.
