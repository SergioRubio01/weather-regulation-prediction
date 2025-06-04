"""
Models package for weather regulation prediction system.
"""

import warnings

from .base_model import BaseModel, ModelMetrics
from .random_forest import RandomForestModel

# Try to import TensorFlow-dependent models
TF_MODELS_AVAILABLE = True
try:
    from .attention_lstm import AttentionLSTMModel
    from .autoencoder import AutoencoderModel
    from .cnn import CNNModel
    from .fnn import FNNModel
    from .gru import GRUModel
    from .hybrid_models import CNNLSTMModel, CNNRNNModel
    from .lstm import LSTMModel
    from .rnn import RNNModel
    from .transformer import TransformerModel
    from .wavenet import WaveNetModel
except ImportError as e:
    TF_MODELS_AVAILABLE = False
    warnings.warn(
        f"TensorFlow models not available due to import error: {e}. "
        "Only RandomForest and Ensemble models will be available.",
        RuntimeWarning,
        stacklevel=2,
    )
    # Create dummy classes for TensorFlow models
    AttentionLSTMModel = None
    AutoencoderModel = None
    CNNModel = None
    FNNModel = None
    GRUModel = None
    CNNLSTMModel = None
    CNNRNNModel = None
    LSTMModel = None
    RNNModel = None
    TransformerModel = None
    WaveNetModel = None

# Import ensemble model last as it may depend on other models
try:
    from .ensemble import EnsembleModel
except ImportError as e:
    warnings.warn(
        f"EnsembleModel not available due to import error: {e}",
        RuntimeWarning,
        stacklevel=2,
    )
    EnsembleModel = None

__all__ = [
    "BaseModel",
    "ModelMetrics",
    "RandomForestModel",
    "LSTMModel",
    "CNNModel",
    "RNNModel",
    "FNNModel",
    "WaveNetModel",
    "GRUModel",
    "TransformerModel",
    "AttentionLSTMModel",
    "EnsembleModel",
    "AutoencoderModel",
    "CNNRNNModel",
    "CNNLSTMModel",
]
