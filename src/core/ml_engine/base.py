"""
Base classes for ML Engine with sklearn-compatible interfaces.

This module provides abstract base classes that all ML models should inherit from,
ensuring a consistent interface across different model types and frameworks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator as SklearnBaseEstimator
from sklearn.base import ClassifierMixin, RegressorMixin

from ..logging import get_logger


class BaseMLModel(SklearnBaseEstimator, ABC):
    """
    Abstract base class for all ML models in the platform.

    Provides sklearn-compatible interface with additional functionality
    for MLflow tracking, model serialization, and metadata management.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize base model with hyperparameters."""
        self.logger = get_logger(
            self.__class__.__name__,
            pipeline_stage="ml_engine",
            model_type=self.__class__.__name__,
        )
        self.model = None
        self.is_fitted = False
        self.feature_names_in_: list[str] | None = None
        self.n_features_in_: int | None = None

        # Store hyperparameters
        self.hyperparameters = kwargs

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> BaseMLModel:
        """
        Train the model on the given data.

        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional training arguments

        Returns:
            self: Fitted model instance
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Features to predict on

        Returns:
            Predictions as numpy array
        """
        pass

    def _validate_input(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """
        Validate and convert input to DataFrame.

        Args:
            X: Input features

        Returns:
            DataFrame with validated features
        """
        if isinstance(X, np.ndarray):
            if self.feature_names_in_ is not None:
                X = pd.DataFrame(X, columns=self.feature_names_in_)
            else:
                X = pd.DataFrame(X)
        elif not isinstance(X, pd.DataFrame):
            msg = f"Expected DataFrame or ndarray, got {type(X)}"
            raise TypeError(msg)

        return X

    def _store_feature_info(self, X: pd.DataFrame) -> None:
        """Store feature information from training data."""
        self.n_features_in_ = X.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        Get model hyperparameters (sklearn-compatible).

        Args:
            deep: If True, return params for nested objects

        Returns:
            Dictionary of hyperparameters
        """
        return self.hyperparameters.copy()

    def set_params(self, **params: Any) -> BaseMLModel:
        """
        Set model hyperparameters (sklearn-compatible).

        Args:
            **params: Hyperparameters to set

        Returns:
            self
        """
        for key, value in params.items():
            self.hyperparameters[key] = value
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def get_metadata(self) -> dict[str, Any]:
        """
        Get model metadata for tracking and serialization.

        Returns:
            Dictionary containing model metadata
        """
        return {
            "model_class": self.__class__.__name__,
            "is_fitted": self.is_fitted,
            "n_features": self.n_features_in_,
            "feature_names": self.feature_names_in_,
            "hyperparameters": self.hyperparameters,
        }


class BaseClassifier(BaseMLModel, ClassifierMixin):
    """Base class for classification models."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features to predict on

        Returns:
            Class probabilities as numpy array
        """
        pass

    def score(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        """
        Return accuracy score (sklearn-compatible).

        Args:
            X: Test features
            y: True labels
            sample_weight: Optional sample weights

        Returns:
            Accuracy score
        """
        from sklearn.metrics import accuracy_score

        y_pred = self.predict(X)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)


class BaseRegressor(BaseMLModel, RegressorMixin):
    """Base class for regression models."""

    def score(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        """
        Return R² score (sklearn-compatible).

        Args:
            X: Test features
            y: True targets
            sample_weight: Optional sample weights

        Returns:
            R² score
        """
        from sklearn.metrics import r2_score

        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)


class ModelRegistry:
    """
    Registry for managing available models in the ML engine.

    Provides centralized access to all supported model types and their
    configurations.
    """

    def __init__(self) -> None:
        """Initialize the model registry."""
        self._models: dict[str, type[BaseMLModel]] = {}
        self._categories: dict[str, list[str]] = {
            "classification": [],
            "regression": [],
            "deep_learning": [],
        }

    def register(
        self,
        name: str,
        model_class: type[BaseMLModel],
        category: str,
    ) -> None:
        """
        Register a new model in the registry.

        Args:
            name: Model name identifier
            model_class: Model class to register
            category: Model category (classification, regression, etc.)
        """
        self._models[name] = model_class
        if category in self._categories:
            self._categories[category].append(name)
        else:
            self._categories[category] = [name]

    def get_model(self, name: str) -> type[BaseMLModel]:
        """
        Get a registered model class by name.

        Args:
            name: Model name identifier

        Returns:
            Model class

        Raises:
            KeyError: If model not found
        """
        if name not in self._models:
            msg = f"Model '{name}' not found in registry. Available: {list(self._models.keys())}"
            raise KeyError(msg)
        return self._models[name]

    def list_models(self, category: str | None = None) -> list[str]:
        """
        List all registered models.

        Args:
            category: Optional category filter

        Returns:
            List of model names
        """
        if category is None:
            return list(self._models.keys())
        return self._categories.get(category, [])

    def get_categories(self) -> list[str]:
        """Get all model categories."""
        return list(self._categories.keys())


# Global model registry instance
model_registry = ModelRegistry()
