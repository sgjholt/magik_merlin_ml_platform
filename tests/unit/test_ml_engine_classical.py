"""
Tests for classical ML model wrappers (XGBoost, LightGBM, CatBoost).

Tests the sklearn-compatible wrappers around gradient boosting libraries.
"""

import numpy as np
import pandas as pd
import pytest

from src.core.ml_engine.classical import (
    CATBOOST_AVAILABLE,
    LIGHTGBM_AVAILABLE,
    XGBOOST_AVAILABLE,
)

# Import models conditionally based on availability
if XGBOOST_AVAILABLE:
    from src.core.ml_engine.classical import XGBoostClassifier, XGBoostRegressor

if LIGHTGBM_AVAILABLE:
    from src.core.ml_engine.classical import LightGBMClassifier, LightGBMRegressor

if CATBOOST_AVAILABLE:
    from src.core.ml_engine.classical import CatBoostClassifier, CatBoostRegressor


@pytest.fixture
def binary_classification_data():
    """Create binary classification dataset."""
    np.random.seed(42)
    n_samples = 200
    n_features = 10

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    # Create binary target with some pattern
    y = pd.Series((X["feature_0"] + X["feature_1"] > 0).astype(int), name="target")

    return X, y


@pytest.fixture
def multiclass_classification_data():
    """Create multiclass classification dataset."""
    np.random.seed(42)
    n_samples = 200
    n_features = 10

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    # Create 3-class target
    y = pd.Series(np.random.choice([0, 1, 2], size=n_samples), name="target")

    return X, y


@pytest.fixture
def regression_data():
    """Create regression dataset."""
    np.random.seed(42)
    n_samples = 200
    n_features = 10

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    # Create continuous target with some pattern
    y = pd.Series(
        X["feature_0"] * 2
        + X["feature_1"]
        - X["feature_2"]
        + np.random.randn(n_samples) * 0.1,
        name="target",
    )

    return X, y


@pytest.fixture
def categorical_data():
    """Create dataset with categorical features."""
    np.random.seed(42)
    n_samples = 200

    X = pd.DataFrame(
        {
            "num_feature_1": np.random.randn(n_samples),
            "num_feature_2": np.random.randn(n_samples),
            "cat_feature_1": np.random.choice(["A", "B", "C"], size=n_samples),
            "cat_feature_2": np.random.choice(["X", "Y"], size=n_samples),
        }
    )
    y = pd.Series(np.random.choice([0, 1], size=n_samples))

    return X, y


# XGBoost Tests
@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not installed")
class TestXGBoostClassifier:
    """Test XGBoost classifier wrapper."""

    def test_initialization(self):
        """Test XGBoost classifier initialization."""
        model = XGBoostClassifier(n_estimators=100, max_depth=5)

        assert model.hyperparameters["n_estimators"] == 100
        assert model.hyperparameters["max_depth"] == 5
        assert model.hyperparameters["random_state"] == 42  # default

    def test_binary_classification(self, binary_classification_data):
        """Test binary classification."""
        X, y = binary_classification_data
        model = XGBoostClassifier(n_estimators=50)

        # Fit
        model.fit(X, y)
        assert model.is_fitted is True

        # Predict
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert set(predictions).issubset({0, 1})

        # Predict proba
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

        # Score
        score = model.score(X, y)
        assert score > 0.5  # Should be better than random

    def test_multiclass_classification(self, multiclass_classification_data):
        """Test multiclass classification."""
        X, y = multiclass_classification_data
        model = XGBoostClassifier(n_estimators=50)

        model.fit(X, y)
        predictions = model.predict(X)

        assert set(predictions).issubset({0, 1, 2})

        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 3)

    def test_feature_importance(self, binary_classification_data):
        """Test feature importance extraction."""
        X, y = binary_classification_data
        model = XGBoostClassifier(n_estimators=50)
        model.fit(X, y)

        importance_df = model.get_feature_importance()

        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == X.shape[1]
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns
        # Should be sorted by importance
        assert (
            importance_df["importance"].values
            == sorted(importance_df["importance"].values, reverse=True)
        ).all()

    def test_get_set_params(self):
        """Test parameter getting and setting."""
        model = XGBoostClassifier(n_estimators=100)

        params = model.get_params()
        assert params["n_estimators"] == 100

        model.set_params(n_estimators=200, max_depth=10)
        params = model.get_params()
        assert params["n_estimators"] == 200
        assert params["max_depth"] == 10


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not installed")
class TestXGBoostRegressor:
    """Test XGBoost regressor wrapper."""

    def test_regression(self, regression_data):
        """Test regression."""
        X, y = regression_data
        model = XGBoostRegressor(n_estimators=50)

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == len(y)
        assert isinstance(predictions, np.ndarray)

        # Score (R²)
        score = model.score(X, y)
        assert score > 0.0  # Should explain some variance

    def test_feature_importance(self, regression_data):
        """Test feature importance for regressor."""
        X, y = regression_data
        model = XGBoostRegressor(n_estimators=50)
        model.fit(X, y)

        importance_df = model.get_feature_importance()
        assert len(importance_df) == X.shape[1]


# LightGBM Tests
@pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not installed")
class TestLightGBMClassifier:
    """Test LightGBM classifier wrapper."""

    def test_initialization(self):
        """Test LightGBM classifier initialization."""
        model = LightGBMClassifier(n_estimators=100)

        assert model.hyperparameters["n_estimators"] == 100
        assert model.hyperparameters["verbosity"] == -1  # default

    def test_binary_classification(self, binary_classification_data):
        """Test binary classification."""
        X, y = binary_classification_data
        model = LightGBMClassifier(n_estimators=50)

        model.fit(X, y)
        predictions = model.predict(X)

        assert set(predictions).issubset({0, 1})

        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)

        score = model.score(X, y)
        assert score > 0.5

    def test_multiclass_classification(self, multiclass_classification_data):
        """Test multiclass classification."""
        X, y = multiclass_classification_data
        model = LightGBMClassifier(n_estimators=50)

        model.fit(X, y)
        proba = model.predict_proba(X)

        assert proba.shape == (len(X), 3)

    def test_feature_importance(self, binary_classification_data):
        """Test feature importance extraction."""
        X, y = binary_classification_data
        model = LightGBMClassifier(n_estimators=50)
        model.fit(X, y)

        importance_df = model.get_feature_importance()
        assert len(importance_df) == X.shape[1]


@pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not installed")
class TestLightGBMRegressor:
    """Test LightGBM regressor wrapper."""

    def test_regression(self, regression_data):
        """Test regression."""
        X, y = regression_data
        model = LightGBMRegressor(n_estimators=50)

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == len(y)

        score = model.score(X, y)
        assert score > 0.0


# CatBoost Tests
@pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not installed")
class TestCatBoostClassifier:
    """Test CatBoost classifier wrapper."""

    def test_initialization(self):
        """Test CatBoost classifier initialization."""
        model = CatBoostClassifier(iterations=100)

        assert model.hyperparameters["iterations"] == 100
        assert model.hyperparameters["verbose"] is False

    def test_binary_classification(self, binary_classification_data):
        """Test binary classification."""
        X, y = binary_classification_data
        model = CatBoostClassifier(iterations=50)

        model.fit(X, y)
        predictions = model.predict(X)

        assert set(predictions).issubset({0, 1})

        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)

        score = model.score(X, y)
        assert score > 0.5

    def test_multiclass_classification(self, multiclass_classification_data):
        """Test multiclass classification."""
        X, y = multiclass_classification_data
        model = CatBoostClassifier(iterations=50)

        model.fit(X, y)
        proba = model.predict_proba(X)

        assert proba.shape == (len(X), 3)

    def test_categorical_features(self, categorical_data):
        """Test handling of categorical features."""
        X, y = categorical_data
        model = CatBoostClassifier(iterations=50)

        # CatBoost should automatically detect categorical features
        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == len(y)

    def test_feature_importance(self, binary_classification_data):
        """Test feature importance extraction."""
        X, y = binary_classification_data
        model = CatBoostClassifier(iterations=50)
        model.fit(X, y)

        importance_df = model.get_feature_importance()
        assert len(importance_df) == X.shape[1]


@pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not installed")
class TestCatBoostRegressor:
    """Test CatBoost regressor wrapper."""

    def test_regression(self, regression_data):
        """Test regression."""
        X, y = regression_data
        model = CatBoostRegressor(iterations=50)

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == len(y)

        score = model.score(X, y)
        assert score > 0.0

    def test_categorical_features_regression(self, categorical_data):
        """Test handling of categorical features in regression."""
        X, y = categorical_data
        # Convert y to continuous for regression
        y = y.astype(float) + np.random.randn(len(y)) * 0.1

        model = CatBoostRegressor(iterations=50)
        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == len(y)


# Integration tests across all models
class TestModelComparison:
    """Test that all models have consistent interfaces."""

    @pytest.mark.skipif(
        not (XGBOOST_AVAILABLE and LIGHTGBM_AVAILABLE and CATBOOST_AVAILABLE),
        reason="All libraries not installed",
    )
    def test_all_classifiers_consistent_interface(self, binary_classification_data):
        """Test that all classifiers have consistent interfaces."""
        X, y = binary_classification_data

        models = [
            XGBoostClassifier(n_estimators=20),
            LightGBMClassifier(n_estimators=20),
            CatBoostClassifier(iterations=20),
        ]

        for model in models:
            # All should support fit
            model.fit(X, y)

            # All should support predict
            predictions = model.predict(X)
            assert len(predictions) == len(y)

            # All should support predict_proba
            proba = model.predict_proba(X)
            assert proba.shape[0] == len(X)

            # All should support score
            score = model.score(X, y)
            assert 0 <= score <= 1

            # All should support get_params/set_params
            params = model.get_params()
            assert isinstance(params, dict)

            # All should support get_metadata
            metadata = model.get_metadata()
            assert metadata["is_fitted"] is True

            # All should support feature importance
            importance = model.get_feature_importance()
            assert len(importance) == X.shape[1]

    @pytest.mark.skipif(
        not (XGBOOST_AVAILABLE and LIGHTGBM_AVAILABLE and CATBOOST_AVAILABLE),
        reason="All libraries not installed",
    )
    def test_all_regressors_consistent_interface(self, regression_data):
        """Test that all regressors have consistent interfaces."""
        X, y = regression_data

        models = [
            XGBoostRegressor(n_estimators=20),
            LightGBMRegressor(n_estimators=20),
            CatBoostRegressor(iterations=20),
        ]

        for model in models:
            model.fit(X, y)
            predictions = model.predict(X)
            score = model.score(X, y)

            assert len(predictions) == len(y)
            assert isinstance(score, (int, float))
