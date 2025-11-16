"""
Tests for AutoML pipeline.

Tests the automated model comparison and hyperparameter optimization
functionality of the ML engine.
"""

import numpy as np
import pandas as pd
import pytest

from src.core.ml_engine.automl import OPTUNA_AVAILABLE, AutoMLPipeline
from src.core.ml_engine.classical import (
    CATBOOST_AVAILABLE,
    LIGHTGBM_AVAILABLE,
    XGBOOST_AVAILABLE,
)


@pytest.fixture
def classification_dataset():
    """Create classification dataset for AutoML testing."""
    np.random.seed(42)
    n_samples = 150
    n_features = 8

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    # Create binary target with clear pattern
    y = pd.Series(
        (X["feature_0"] + X["feature_1"] - X["feature_2"] > 0).astype(int),
        name="target",
    )

    return X, y


@pytest.fixture
def regression_dataset():
    """Create regression dataset for AutoML testing."""
    np.random.seed(42)
    n_samples = 150
    n_features = 8

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    # Create continuous target
    y = pd.Series(
        X["feature_0"] * 2
        + X["feature_1"] * 1.5
        - X["feature_2"]
        + np.random.randn(n_samples) * 0.2,
        name="target",
    )

    return X, y


class TestAutoMLPipelineInitialization:
    """Test AutoML pipeline initialization."""

    def test_classification_initialization(self):
        """Test initialization for classification tasks."""
        pipeline = AutoMLPipeline(task_type="classification")

        assert pipeline.task_type == "classification"
        assert pipeline.random_state == 42
        assert isinstance(pipeline.models, dict)
        assert isinstance(pipeline.results, dict)
        assert pipeline.best_model is None

    def test_regression_initialization(self):
        """Test initialization for regression tasks."""
        pipeline = AutoMLPipeline(task_type="regression")

        assert pipeline.task_type == "regression"

    def test_custom_random_state(self):
        """Test custom random state."""
        pipeline = AutoMLPipeline(task_type="classification", random_state=123)

        assert pipeline.random_state == 123


@pytest.mark.skipif(
    not (XGBOOST_AVAILABLE or LIGHTGBM_AVAILABLE or CATBOOST_AVAILABLE),
    reason="At least one ML library required",
)
class TestAutoMLCompareModels:
    """Test model comparison functionality."""

    def test_compare_classification_models(self, classification_dataset):
        """Test comparing classification models."""
        X, y = classification_dataset
        pipeline = AutoMLPipeline(task_type="classification")

        results_df = pipeline.compare_models(X, y, cv=3, test_size=0.2)

        # Check results structure
        assert isinstance(results_df, pd.DataFrame)
        assert not results_df.empty
        assert "model" in results_df.columns
        assert "cv_mean" in results_df.columns
        assert "cv_std" in results_df.columns
        assert "test_score" in results_df.columns

        # Check that models were trained
        assert len(pipeline.models) > 0

        # Check that best model was selected
        assert pipeline.best_model is not None
        assert pipeline.best_model_name is not None

        # Check that results are sorted by test score
        test_scores = results_df["test_score"].values
        assert (test_scores == sorted(test_scores, reverse=True)).all()

    def test_compare_regression_models(self, regression_dataset):
        """Test comparing regression models."""
        X, y = regression_dataset
        pipeline = AutoMLPipeline(task_type="regression")

        results_df = pipeline.compare_models(X, y, cv=3, test_size=0.2)

        assert isinstance(results_df, pd.DataFrame)
        assert not results_df.empty
        assert pipeline.best_model is not None

    def test_compare_specific_models(self, classification_dataset):
        """Test comparing specific subset of models."""
        X, y = classification_dataset
        pipeline = AutoMLPipeline(task_type="classification")

        # Get available models
        from src.core.ml_engine.base import model_registry

        available_models = model_registry.list_models(category="classification")

        if len(available_models) >= 2:
            # Compare only first two models
            models_to_compare = available_models[:2]
            results_df = pipeline.compare_models(X, y, models=models_to_compare, cv=3)

            assert len(results_df) <= len(models_to_compare)

    def test_compare_models_cv_folds(self, classification_dataset):
        """Test different numbers of CV folds."""
        X, y = classification_dataset
        pipeline = AutoMLPipeline(task_type="classification")

        results_df = pipeline.compare_models(X, y, cv=5, test_size=0.2)

        # Check CV scores exist
        assert "cv_scores" in results_df.iloc[0]
        # Each model should have cv_folds number of scores
        assert len(results_df.iloc[0]["cv_scores"]) == 5

    def test_get_best_model(self, classification_dataset):
        """Test getting the best model."""
        X, y = classification_dataset
        pipeline = AutoMLPipeline(task_type="classification")

        pipeline.compare_models(X, y, cv=3)

        best_model = pipeline.get_best_model()
        assert best_model is not None
        assert best_model.is_fitted is True

    def test_get_best_model_before_training_raises(self):
        """Test that getting best model before training raises error."""
        pipeline = AutoMLPipeline(task_type="classification")

        with pytest.raises(RuntimeError, match="No models have been trained"):
            pipeline.get_best_model()

    def test_predict_with_best_model(self, classification_dataset):
        """Test making predictions with best model."""
        X, y = classification_dataset
        pipeline = AutoMLPipeline(task_type="classification")

        pipeline.compare_models(X, y, cv=3)

        predictions = pipeline.predict(X)
        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})

    def test_results_summary(self, classification_dataset):
        """Test getting results summary."""
        X, y = classification_dataset
        pipeline = AutoMLPipeline(task_type="classification")

        pipeline.compare_models(X, y, cv=3)

        summary = pipeline.get_results_summary()

        assert summary["task_type"] == "classification"
        assert summary["n_models_trained"] > 0
        assert summary["best_model"] is not None
        assert isinstance(summary["results"], dict)


@pytest.mark.skipif(
    not OPTUNA_AVAILABLE or not XGBOOST_AVAILABLE,
    reason="Optuna and XGBoost required for optimization tests",
)
class TestAutoMLHyperparameterOptimization:
    """Test hyperparameter optimization functionality."""

    def test_optimize_xgboost_classifier(self, classification_dataset):
        """Test optimizing XGBoost classifier."""
        X, y = classification_dataset
        pipeline = AutoMLPipeline(task_type="classification")

        result = pipeline.optimize_hyperparameters(
            X,
            y,
            model_name="xgboost_classifier",
            n_trials=10,  # Small number for testing
            cv=3,
        )

        # Check result structure
        assert "model" in result
        assert "best_params" in result
        assert "best_score" in result
        assert "study" in result

        # Check model was trained
        assert result["model"].is_fitted is True

        # Check parameters were found
        assert isinstance(result["best_params"], dict)
        assert len(result["best_params"]) > 0

        # Check score is reasonable
        assert 0.0 <= result["best_score"] <= 1.0

    def test_optimize_stores_results(self, classification_dataset):
        """Test that optimization stores results in pipeline."""
        X, y = classification_dataset
        pipeline = AutoMLPipeline(task_type="classification")

        pipeline.optimize_hyperparameters(
            X,
            y,
            model_name="xgboost_classifier",
            n_trials=5,
            cv=3,
        )

        # Check that optimized model is stored
        assert "xgboost_classifier_optimized" in pipeline.models
        assert "xgboost_classifier_optimized" in pipeline.results

    def test_optimization_param_spaces(self, classification_dataset):
        """Test that different models have appropriate param spaces."""
        _X, _y = classification_dataset
        pipeline = AutoMLPipeline(task_type="classification")

        # Test XGBoost param space
        class MockTrial:
            """Mock Optuna trial for testing."""

            def suggest_int(self, name, low, high):
                return (low + high) // 2

            def suggest_float(self, name, low, high):
                return (low + high) / 2

        trial = MockTrial()
        xgb_params = pipeline._get_param_space("xgboost_classifier", trial)

        assert "n_estimators" in xgb_params
        assert "max_depth" in xgb_params
        assert "learning_rate" in xgb_params


@pytest.mark.skipif(
    not LIGHTGBM_AVAILABLE,
    reason="LightGBM required",
)
class TestAutoMLWithLightGBM:
    """Test AutoML with LightGBM models."""

    def test_compare_lightgbm_models(self, classification_dataset):
        """Test comparing LightGBM models."""
        X, y = classification_dataset
        pipeline = AutoMLPipeline(task_type="classification")

        results_df = pipeline.compare_models(X, y, models=["lightgbm_classifier"], cv=3)

        assert not results_df.empty
        assert results_df.iloc[0]["model"] == "lightgbm_classifier"


@pytest.mark.skipif(
    not CATBOOST_AVAILABLE,
    reason="CatBoost required",
)
class TestAutoMLWithCatBoost:
    """Test AutoML with CatBoost models."""

    def test_compare_catboost_models(self, classification_dataset):
        """Test comparing CatBoost models."""
        X, y = classification_dataset
        pipeline = AutoMLPipeline(task_type="classification")

        results_df = pipeline.compare_models(X, y, models=["catboost_classifier"], cv=3)

        assert not results_df.empty
        assert results_df.iloc[0]["model"] == "catboost_classifier"


class TestAutoMLErrorHandling:
    """Test error handling in AutoML pipeline."""

    def test_predict_before_training_raises(self):
        """Test that prediction before training raises error."""
        pipeline = AutoMLPipeline(task_type="classification")

        X = pd.DataFrame(np.random.randn(10, 5))

        with pytest.raises(RuntimeError, match="No models have been trained"):
            pipeline.predict(X)

    @pytest.mark.skipif(OPTUNA_AVAILABLE, reason="Test for missing Optuna")
    def test_optimize_without_optuna_raises(self, classification_dataset):
        """Test that optimization without Optuna raises error."""
        X, y = classification_dataset
        pipeline = AutoMLPipeline(task_type="classification")

        with pytest.raises(ImportError, match="Optuna is not installed"):
            pipeline.optimize_hyperparameters(
                X, y, model_name="xgboost_classifier", n_trials=5
            )


class TestAutoMLIntegration:
    """Integration tests for AutoML pipeline."""

    @pytest.mark.skipif(
        not (XGBOOST_AVAILABLE and OPTUNA_AVAILABLE),
        reason="XGBoost and Optuna required",
    )
    def test_full_automl_workflow(self, classification_dataset):
        """Test complete AutoML workflow: compare then optimize."""
        X, y = classification_dataset
        pipeline = AutoMLPipeline(task_type="classification")

        # Step 1: Compare models
        comparison_results = pipeline.compare_models(X, y, cv=3)

        # Step 2: Optimize best model
        best_model_name = pipeline.best_model_name
        if best_model_name:
            optimization_results = pipeline.optimize_hyperparameters(
                X, y, model_name=best_model_name, n_trials=5, cv=3
            )

            # Step 3: Make predictions
            predictions = pipeline.predict(X, use_best=True)

            # Verify
            assert len(comparison_results) > 0
            assert optimization_results["model"].is_fitted
            assert len(predictions) == len(X)

    @pytest.mark.skipif(
        not (XGBOOST_AVAILABLE and LIGHTGBM_AVAILABLE),
        reason="Multiple ML libraries required",
    )
    def test_automl_with_multiple_libraries(self, regression_dataset):
        """Test AutoML with multiple different libraries."""
        X, y = regression_dataset
        pipeline = AutoMLPipeline(task_type="regression")

        results_df = pipeline.compare_models(
            X,
            y,
            models=["xgboost_regressor", "lightgbm_regressor"],
            cv=3,
        )

        # Should have results for both models
        assert len(results_df) == 2

        # Both should have reasonable scores
        assert all(
            results_df["test_score"] > -1.0
        )  # R² can be negative but not too bad
