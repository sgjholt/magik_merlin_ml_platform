"""
ML Engine module for custom machine learning workflows.

This module provides a modern, sklearn-compatible ML engine with support for:
- XGBoost, CatBoost, and LightGBM models
- PyTorch Lightning for deep learning
- Automated model comparison and selection
- Hyperparameter optimization with Optuna
- MLflow integration for experiment tracking

The engine is designed to be lightweight, maintainable, and compatible
with Python 3.13+, avoiding dependency lock-in issues.
"""

from .automl import AutoMLPipeline
from .base import (
    BaseClassifier,
    BaseMLModel,
    BaseRegressor,
    ModelRegistry,
    model_registry,
)
from .classical import (
    CatBoostClassifier,
    CatBoostRegressor,
    LightGBMClassifier,
    LightGBMRegressor,
    XGBoostClassifier,
    XGBoostRegressor,
)

# Optional PyTorch Lightning imports (requires torch/lightning)
try:
    from .deep_learning import LightningClassifier, LightningRegressor

    _HAS_LIGHTNING = True
except ImportError:
    _HAS_LIGHTNING = False
    LightningClassifier = None  # type: ignore[misc, assignment]
    LightningRegressor = None  # type: ignore[misc, assignment]

__all__ = [
    "AutoMLPipeline",
    "BaseClassifier",
    "BaseMLModel",
    "BaseRegressor",
    "CatBoostClassifier",
    "CatBoostRegressor",
    "LightGBMClassifier",
    "LightGBMRegressor",
    "ModelRegistry",
    "XGBoostClassifier",
    "XGBoostRegressor",
    "model_registry",
]

# Add Lightning models to __all__ if available
if _HAS_LIGHTNING:
    __all__.extend(["LightningClassifier", "LightningRegressor"])
