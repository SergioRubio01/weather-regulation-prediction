#!/usr/bin/env python3
"""
Train deep learning models using the balanced dataset approach

This script uses the existing experiment runner infrastructure with the balanced dataset.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Now run experiments with the balanced deep learning config
os.system(f"""
cd {project_root} && poetry run python run_experiments.py \
    --config configs/balanced_deep_learning.yaml \
    --models lstm gru cnn transformer \
    --data-file data/balanced_weather_data.csv
""")