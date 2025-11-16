"""
ML Engine module for custom machine learning workflows.

This module provides a modern, sklearn-compatible ML engine with support for:
- XGBoost, CatBoost, and LightGBM models
- PyTorch Lightning for deep learning (coming soon)
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
