"""
Classical ML model wrappers for XGBoost, CatBoost, and LightGBM.

Provides sklearn-compatible wrappers around popular gradient boosting libraries
with standardized interfaces and MLflow integration.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import BaseClassifier, BaseRegressor, model_registry

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class XGBoostClassifier(BaseClassifier):
    """XGBoost classifier wrapper with sklearn-compatible interface."""

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize XGBoost classifier.

        Args:
            **kwargs: XGBoost hyperparameters
        """
        if not XGBOOST_AVAILABLE:
            msg = "XGBoost is not installed. Install with: uv add xgboost"
            raise ImportError(msg)

        super().__init__(**kwargs)

        # Set default parameters
        default_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "random_state": 42,
        }
        self.hyperparameters = {**default_params, **kwargs}

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> XGBoostClassifier:
        """Train XGBoost classifier."""
        X = self._validate_input(X)
        self._store_feature_info(X)

        # Determine if multiclass
        n_classes = len(np.unique(y))
        if n_classes > 2:
            self.hyperparameters["objective"] = "multi:softprob"
            self.hyperparameters["num_class"] = n_classes

        self.model = xgb.XGBClassifier(**self.hyperparameters)
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True

        self.logger.info(
            "XGBoost classifier trained successfully",
            extra={
                "n_samples": len(X),
                "n_features": X.shape[1],
                "n_classes": n_classes,
            },
        )

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Make predictions with XGBoost classifier."""
        if not self.is_fitted:
            msg = "Model must be fitted before making predictions"
            raise RuntimeError(msg)

        X = self._validate_input(X)
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            msg = "Model must be fitted before making predictions"
            raise RuntimeError(msg)

        X = self._validate_input(X)
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self.is_fitted:
            msg = "Model must be fitted first"
            raise RuntimeError(msg)

        importance = self.model.feature_importances_
        return pd.DataFrame(
            {
                "feature": self.feature_names_in_ or list(range(len(importance))),
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)


class XGBoostRegressor(BaseRegressor):
    """XGBoost regressor wrapper with sklearn-compatible interface."""

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize XGBoost regressor.

        Args:
            **kwargs: XGBoost hyperparameters
        """
        if not XGBOOST_AVAILABLE:
            msg = "XGBoost is not installed. Install with: uv add xgboost"
            raise ImportError(msg)

        super().__init__(**kwargs)

        # Set default parameters
        default_params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "random_state": 42,
        }
        self.hyperparameters = {**default_params, **kwargs}

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> XGBoostRegressor:
        """Train XGBoost regressor."""
        X = self._validate_input(X)
        self._store_feature_info(X)

        self.model = xgb.XGBRegressor(**self.hyperparameters)
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True

        self.logger.info(
            "XGBoost regressor trained successfully",
            extra={
                "n_samples": len(X),
                "n_features": X.shape[1],
            },
        )

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Make predictions with XGBoost regressor."""
        if not self.is_fitted:
            msg = "Model must be fitted before making predictions"
            raise RuntimeError(msg)

        X = self._validate_input(X)
        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self.is_fitted:
            msg = "Model must be fitted first"
            raise RuntimeError(msg)

        importance = self.model.feature_importances_
        return pd.DataFrame(
            {
                "feature": self.feature_names_in_ or list(range(len(importance))),
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)


class LightGBMClassifier(BaseClassifier):
    """LightGBM classifier wrapper with sklearn-compatible interface."""

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize LightGBM classifier.

        Args:
            **kwargs: LightGBM hyperparameters
        """
        if not LIGHTGBM_AVAILABLE:
            msg = "LightGBM is not installed. Install with: uv add lightgbm"
            raise ImportError(msg)

        super().__init__(**kwargs)

        # Set default parameters
        default_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "random_state": 42,
            "verbosity": -1,
        }
        self.hyperparameters = {**default_params, **kwargs}

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> LightGBMClassifier:
        """Train LightGBM classifier."""
        X = self._validate_input(X)
        self._store_feature_info(X)

        # Determine if multiclass
        n_classes = len(np.unique(y))
        if n_classes > 2:
            self.hyperparameters["objective"] = "multiclass"
            self.hyperparameters["num_class"] = n_classes
            self.hyperparameters["metric"] = "multi_logloss"

        self.model = lgb.LGBMClassifier(**self.hyperparameters)
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True

        self.logger.info(
            "LightGBM classifier trained successfully",
            extra={
                "n_samples": len(X),
                "n_features": X.shape[1],
                "n_classes": n_classes,
            },
        )

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Make predictions with LightGBM classifier."""
        if not self.is_fitted:
            msg = "Model must be fitted before making predictions"
            raise RuntimeError(msg)

        X = self._validate_input(X)
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            msg = "Model must be fitted before making predictions"
            raise RuntimeError(msg)

        X = self._validate_input(X)
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self.is_fitted:
            msg = "Model must be fitted first"
            raise RuntimeError(msg)

        importance = self.model.feature_importances_
        return pd.DataFrame(
            {
                "feature": self.feature_names_in_ or list(range(len(importance))),
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)


class LightGBMRegressor(BaseRegressor):
    """LightGBM regressor wrapper with sklearn-compatible interface."""

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize LightGBM regressor.

        Args:
            **kwargs: LightGBM hyperparameters
        """
        if not LIGHTGBM_AVAILABLE:
            msg = "LightGBM is not installed. Install with: uv add lightgbm"
            raise ImportError(msg)

        super().__init__(**kwargs)

        # Set default parameters
        default_params = {
            "objective": "regression",
            "metric": "rmse",
            "random_state": 42,
            "verbosity": -1,
        }
        self.hyperparameters = {**default_params, **kwargs}

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> LightGBMRegressor:
        """Train LightGBM regressor."""
        X = self._validate_input(X)
        self._store_feature_info(X)

        self.model = lgb.LGBMRegressor(**self.hyperparameters)
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True

        self.logger.info(
            "LightGBM regressor trained successfully",
            extra={
                "n_samples": len(X),
                "n_features": X.shape[1],
            },
        )

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Make predictions with LightGBM regressor."""
        if not self.is_fitted:
            msg = "Model must be fitted before making predictions"
            raise RuntimeError(msg)

        X = self._validate_input(X)
        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self.is_fitted:
            msg = "Model must be fitted first"
            raise RuntimeError(msg)

        importance = self.model.feature_importances_
        return pd.DataFrame(
            {
                "feature": self.feature_names_in_ or list(range(len(importance))),
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)


class CatBoostClassifier(BaseClassifier):
    """CatBoost classifier wrapper with sklearn-compatible interface."""

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize CatBoost classifier.

        Args:
            **kwargs: CatBoost hyperparameters
        """
        if not CATBOOST_AVAILABLE:
            msg = "CatBoost is not installed. Install with: uv add catboost"
            raise ImportError(msg)

        super().__init__(**kwargs)

        # Set default parameters
        default_params = {
            "loss_function": "Logloss",
            "random_state": 42,
            "verbose": False,
        }
        self.hyperparameters = {**default_params, **kwargs}

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> CatBoostClassifier:
        """Train CatBoost classifier."""
        X = self._validate_input(X)
        self._store_feature_info(X)

        # Determine if multiclass
        n_classes = len(np.unique(y))
        if n_classes > 2:
            self.hyperparameters["loss_function"] = "MultiClass"

        # Identify categorical features
        cat_features = []
        if isinstance(X, pd.DataFrame):
            cat_features = [
                i
                for i, dtype in enumerate(X.dtypes)
                if dtype == "object" or dtype.name == "category"
            ]

        self.model = cb.CatBoostClassifier(**self.hyperparameters)
        self.model.fit(X, y, cat_features=cat_features, **kwargs)
        self.is_fitted = True

        self.logger.info(
            "CatBoost classifier trained successfully",
            extra={
                "n_samples": len(X),
                "n_features": X.shape[1],
                "n_classes": n_classes,
                "n_cat_features": len(cat_features),
            },
        )

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Make predictions with CatBoost classifier."""
        if not self.is_fitted:
            msg = "Model must be fitted before making predictions"
            raise RuntimeError(msg)

        X = self._validate_input(X)
        return self.model.predict(X).flatten()

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            msg = "Model must be fitted before making predictions"
            raise RuntimeError(msg)

        X = self._validate_input(X)
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self.is_fitted:
            msg = "Model must be fitted first"
            raise RuntimeError(msg)

        importance = self.model.feature_importances_
        return pd.DataFrame(
            {
                "feature": self.feature_names_in_ or list(range(len(importance))),
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)


class CatBoostRegressor(BaseRegressor):
    """CatBoost regressor wrapper with sklearn-compatible interface."""

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize CatBoost regressor.

        Args:
            **kwargs: CatBoost hyperparameters
        """
        if not CATBOOST_AVAILABLE:
            msg = "CatBoost is not installed. Install with: uv add catboost"
            raise ImportError(msg)

        super().__init__(**kwargs)

        # Set default parameters
        default_params = {
            "loss_function": "RMSE",
            "random_state": 42,
            "verbose": False,
        }
        self.hyperparameters = {**default_params, **kwargs}

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> CatBoostRegressor:
        """Train CatBoost regressor."""
        X = self._validate_input(X)
        self._store_feature_info(X)

        # Identify categorical features
        cat_features = []
        if isinstance(X, pd.DataFrame):
            cat_features = [
                i
                for i, dtype in enumerate(X.dtypes)
                if dtype == "object" or dtype.name == "category"
            ]

        self.model = cb.CatBoostRegressor(**self.hyperparameters)
        self.model.fit(X, y, cat_features=cat_features, **kwargs)
        self.is_fitted = True

        self.logger.info(
            "CatBoost regressor trained successfully",
            extra={
                "n_samples": len(X),
                "n_features": X.shape[1],
                "n_cat_features": len(cat_features),
            },
        )

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Make predictions with CatBoost regressor."""
        if not self.is_fitted:
            msg = "Model must be fitted before making predictions"
            raise RuntimeError(msg)

        X = self._validate_input(X)
        return self.model.predict(X).flatten()

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self.is_fitted:
            msg = "Model must be fitted first"
            raise RuntimeError(msg)

        importance = self.model.feature_importances_
        return pd.DataFrame(
            {
                "feature": self.feature_names_in_ or list(range(len(importance))),
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)


# Register all classical models in the global registry
if XGBOOST_AVAILABLE:
    model_registry.register("xgboost_classifier", XGBoostClassifier, "classification")
    model_registry.register("xgboost_regressor", XGBoostRegressor, "regression")

if LIGHTGBM_AVAILABLE:
    model_registry.register("lightgbm_classifier", LightGBMClassifier, "classification")
    model_registry.register("lightgbm_regressor", LightGBMRegressor, "regression")

if CATBOOST_AVAILABLE:
    model_registry.register("catboost_classifier", CatBoostClassifier, "classification")
    model_registry.register("catboost_regressor", CatBoostRegressor, "regression")
