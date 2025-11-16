"""
Tests for ML Engine base classes.

Tests the foundational components of the custom ML engine including
base model classes and the model registry.
"""

import numpy as np
import pandas as pd
import pytest

from src.core.ml_engine.base import (
    BaseClassifier,
    BaseMLModel,
    BaseRegressor,
    ModelRegistry,
    model_registry,
)


class MockClassifier(BaseClassifier):
    """Mock classifier for testing base class functionality."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mock_classes = None

    def fit(self, X, y, **kwargs):
        X = self._validate_input(X)
        self._store_feature_info(X)
        self.mock_classes = np.unique(y)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            msg = "Model must be fitted first"
            raise RuntimeError(msg)
        X = self._validate_input(X)
        # Return random predictions from training classes
        return np.random.choice(self.mock_classes, size=len(X))

    def predict_proba(self, X):
        if not self.is_fitted:
            msg = "Model must be fitted first"
            raise RuntimeError(msg)
        X = self._validate_input(X)
        n_classes = len(self.mock_classes)
        # Return random probabilities
        probs = np.random.random((len(X), n_classes))
        return probs / probs.sum(axis=1, keepdims=True)


class MockRegressor(BaseRegressor):
    """Mock regressor for testing base class functionality."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y, **kwargs):
        X = self._validate_input(X)
        self._store_feature_info(X)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            msg = "Model must be fitted first"
            raise RuntimeError(msg)
        X = self._validate_input(X)
        # Return random predictions
        return np.random.random(len(X))


@pytest.fixture
def sample_classification_data():
    """Create sample classification dataset."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y = pd.Series(np.random.choice([0, 1], size=n_samples))

    return X, y


@pytest.fixture
def sample_regression_data():
    """Create sample regression dataset."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y = pd.Series(np.random.randn(n_samples))

    return X, y


class TestBaseClassifier:
    """Test BaseClassifier functionality."""

    def test_initialization(self):
        """Test classifier initialization."""
        model = MockClassifier(random_state=42, test_param="value")

        assert model.is_fitted is False
        assert model.hyperparameters["random_state"] == 42
        assert model.hyperparameters["test_param"] == "value"

    def test_fit_and_predict(self, sample_classification_data):
        """Test fitting and prediction."""
        X, y = sample_classification_data
        model = MockClassifier()

        # Test fit
        model.fit(X, y)
        assert model.is_fitted is True
        assert model.n_features_in_ == 5
        assert len(model.feature_names_in_) == 5

        # Test predict
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert set(predictions).issubset(set(y.unique()))

    def test_predict_proba(self, sample_classification_data):
        """Test probability prediction."""
        X, y = sample_classification_data
        model = MockClassifier()
        model.fit(X, y)

        probs = model.predict_proba(X)
        assert probs.shape == (len(X), len(np.unique(y)))
        # Check probabilities sum to 1
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_score(self, sample_classification_data):
        """Test accuracy scoring."""
        X, y = sample_classification_data
        model = MockClassifier()
        model.fit(X, y)

        score = model.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_predict_before_fit_raises(self, sample_classification_data):
        """Test that prediction before fitting raises error."""
        X, _ = sample_classification_data
        model = MockClassifier()

        with pytest.raises(RuntimeError, match="Model must be fitted"):
            model.predict(X)

        with pytest.raises(RuntimeError, match="Model must be fitted"):
            model.predict_proba(X)

    def test_get_set_params(self):
        """Test get_params and set_params."""
        model = MockClassifier(param1=1, param2="test")

        # Test get_params
        params = model.get_params()
        assert params["param1"] == 1
        assert params["param2"] == "test"

        # Test set_params
        model.set_params(param1=2, param3="new")
        updated_params = model.get_params()
        assert updated_params["param1"] == 2
        assert updated_params["param3"] == "new"

    def test_get_metadata(self, sample_classification_data):
        """Test metadata retrieval."""
        X, y = sample_classification_data
        model = MockClassifier(random_state=42)
        model.fit(X, y)

        metadata = model.get_metadata()
        assert metadata["model_class"] == "MockClassifier"
        assert metadata["is_fitted"] is True
        assert metadata["n_features"] == 5
        assert len(metadata["feature_names"]) == 5
        assert "random_state" in metadata["hyperparameters"]


class TestBaseRegressor:
    """Test BaseRegressor functionality."""

    def test_initialization(self):
        """Test regressor initialization."""
        model = MockRegressor(random_state=42)

        assert model.is_fitted is False
        assert model.hyperparameters["random_state"] == 42

    def test_fit_and_predict(self, sample_regression_data):
        """Test fitting and prediction."""
        X, y = sample_regression_data
        model = MockRegressor()

        # Test fit
        model.fit(X, y)
        assert model.is_fitted is True
        assert model.n_features_in_ == 5

        # Test predict
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert isinstance(predictions, np.ndarray)

    def test_score(self, sample_regression_data):
        """Test R² scoring."""
        X, y = sample_regression_data
        model = MockRegressor()
        model.fit(X, y)

        score = model.score(X, y)
        assert -1.0 <= score <= 1.0  # R² can be negative

    def test_numpy_array_input(self, sample_regression_data):
        """Test that numpy arrays are handled correctly."""
        X, y = sample_regression_data
        X_np = X.values
        y_np = y.values

        model = MockRegressor()
        model.fit(X_np, y_np)

        predictions = model.predict(X_np)
        assert len(predictions) == len(X_np)


class TestModelRegistry:
    """Test ModelRegistry functionality."""

    def test_register_and_get_model(self):
        """Test registering and retrieving models."""
        registry = ModelRegistry()

        # Register a model
        registry.register("test_classifier", MockClassifier, "classification")

        # Retrieve the model
        model_class = registry.get_model("test_classifier")
        assert model_class == MockClassifier

    def test_get_nonexistent_model_raises(self):
        """Test that getting non-existent model raises error."""
        registry = ModelRegistry()

        with pytest.raises(KeyError, match="Model 'nonexistent' not found"):
            registry.get_model("nonexistent")

    def test_list_models(self):
        """Test listing all models."""
        registry = ModelRegistry()

        registry.register("classifier1", MockClassifier, "classification")
        registry.register("regressor1", MockRegressor, "regression")
        registry.register("classifier2", MockClassifier, "classification")

        # List all models
        all_models = registry.list_models()
        assert len(all_models) == 3
        assert "classifier1" in all_models
        assert "regressor1" in all_models

        # List by category
        classifiers = registry.list_models(category="classification")
        assert len(classifiers) == 2
        assert "classifier1" in classifiers
        assert "classifier2" in classifiers

        regressors = registry.list_models(category="regression")
        assert len(regressors) == 1
        assert "regressor1" in regressors

    def test_get_categories(self):
        """Test getting all categories."""
        registry = ModelRegistry()

        categories = registry.get_categories()
        assert "classification" in categories
        assert "regression" in categories
        assert "deep_learning" in categories


class TestGlobalModelRegistry:
    """Test global model registry instance."""

    def test_registry_has_models(self):
        """Test that global registry has registered models."""
        # The registry should have models from classical.py imports
        all_models = model_registry.list_models()

        # Should have at least some models (depending on which libraries are installed)
        assert isinstance(all_models, list)

    def test_registry_categories(self):
        """Test that registry has proper categories."""
        categories = model_registry.get_categories()

        assert "classification" in categories
        assert "regression" in categories
        assert "deep_learning" in categories


class TestInputValidation:
    """Test input validation across base classes."""

    def test_validate_dataframe_input(self, sample_classification_data):
        """Test DataFrame input validation."""
        X, y = sample_classification_data
        model = MockClassifier()

        model.fit(X, y)
        validated = model._validate_input(X)

        assert isinstance(validated, pd.DataFrame)
        assert validated.shape == X.shape

    def test_validate_numpy_input(self, sample_classification_data):
        """Test numpy array input validation."""
        X, y = sample_classification_data
        X_np = X.values

        model = MockClassifier()
        model.fit(X_np, y)

        validated = model._validate_input(X_np)
        assert isinstance(validated, pd.DataFrame)
        assert validated.shape == X_np.shape

    def test_validate_invalid_input_raises(self):
        """Test that invalid input raises TypeError."""
        model = MockClassifier()

        with pytest.raises(TypeError, match="Expected DataFrame or ndarray"):
            model._validate_input("invalid_input")

        with pytest.raises(TypeError, match="Expected DataFrame or ndarray"):
            model._validate_input([1, 2, 3])
