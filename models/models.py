"""
Unified model interface for experiments.
"""

from .random_forest import get_model as get_rf
from .knn import get_model as get_knn
from .linear_model import get_model as get_linear
from .adaboost import get_model as get_adaboost
from .random_features import get_model as get_random_features
from .mlp import train_mlp

MODEL_GETTERS = {
    "Random Forest": get_rf,
    "KNN": get_knn,
    "Linear Model": get_linear,
    "AdaBoost": get_adaboost,
    "Random Features": get_random_features,
    "Neural Network": train_mlp,  # For MLP, use train_mlp directly
}


def get_model(name: str, params: dict = None):
    """Return a model instance or training function for the given name and parameters."""
    if name not in MODEL_GETTERS:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_GETTERS[name](params or {})


# For MLP, use train_mlp directly in experiment runner
