# Project Reorganization Summary

## Overview

The project has been reorganized to improve structure and maintainability while keeping only essential files in the root directory.

## Changes Made

### 1. Root Directory Cleanup

The following files remain in the root directory:

- **Essential files**: `LICENSE`, `README.md`, `pyproject.toml`, `poetry.lock`
- **Documentation**: `CLAUDE.md`, `REFACTORING_PLAN.md`, `BALANCED_DATASET_PROCEDURE.md`
- **Main entry point**: `run_experiments.py`
- **Core directories**: `configs/`, `data/`, `docs/`, `models/`, `notebooks/`, `results/`, `tests/`, `training/`, `visualization/`

### 2. New Directory Structure

#### `/scripts/` - All executable scripts organized by purpose

- **`/balanced_dataset/`** - Scripts for balanced dataset operations
  - `analyze_and_balance_data.py`
  - `prepare_balanced_data.py`
  - `train_balanced_model.py`
  - `run_balanced_pipeline.py`

- **`/utilities/`** - General utility scripts
  - `check_cuda.py`
  - `install_cuda_wsl.sh`
  - `install_cudnn.bat`
  - `convert_regulation_data.py`
  - `prepare_real_data.py`
  - `train_model_real_data.py`
  - `run_simple_experiment.py`

- **`/legacy/`** - Legacy scripts (backward compatibility)
  - `filter_TAFs.py`
  - `functions.py`
  - `model.py`
  - `plots.py`

- **`/examples/`** - Example usage scripts
  - `example_training_pipeline.py`
  - `example_usage.py`

#### `/src/` - Core source modules

- `config.py` - Configuration classes and validation
- `config_parser.py` - Configuration parsing and management
- `config_utils.py` - Configuration utilities and CLI tools
- `__init__.py` - Package initialization

### 3. Import Updates

All imports have been updated to reflect the new structure:

- Configuration imports now use `from src.config import ...`
- Legacy model imports use `from scripts.legacy.model import ...`
- Example scripts have updated path handling

### 4. Benefits

- **Cleaner root directory** - Only essential files remain
- **Better organization** - Scripts grouped by purpose
- **Clear separation** - Legacy vs modern code clearly separated
- **Maintainability** - Easier to navigate and understand project structure
- **Package structure** - `src/` is now a proper Python package

## Usage After Reorganization

```bash
# Run main experiments (unchanged)
python run_experiments.py --config configs/quick_test.yaml

# Run balanced dataset pipeline
python scripts/balanced_dataset/run_balanced_pipeline.py

# Check CUDA availability
python scripts/utilities/check_cuda.py

# Run examples
python scripts/examples/example_usage.py

# Use legacy scripts
python scripts/legacy/model.py
```

## Migration Notes

- All functionality remains unchanged
- Backward compatibility is maintained
- Import paths have been updated where necessary
- Documentation in scripts/README.md provides additional details
