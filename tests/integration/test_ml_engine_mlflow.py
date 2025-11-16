"""
Integration tests for ML Engine with MLflow tracking.

Tests the integration between the custom ML engine and MLflow
experiment tracking system.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.core.experiments.tracking import ExperimentTracker
from src.core.ml_engine.automl import AutoMLPipeline
from src.core.ml_engine.classical import XGBOOST_AVAILABLE

if XGBOOST_AVAILABLE:
    from src.core.ml_engine.classical import XGBoostClassifier


@pytest.fixture
def sample_data():
    """Create sample dataset for testing."""
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
def temp_mlflow_dir():
    """Create temporary directory for MLflow artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestMLEngineMLflowIntegration:
    """Test ML Engine integration with MLflow."""

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_automl_with_experiment_tracker(self, sample_data, temp_mlflow_dir):
        """Test AutoML pipeline with MLflow experiment tracker."""
        X, y = sample_data

        # Create a mock experiment tracker
        tracker = MagicMock(spec=ExperimentTracker)
        tracker.log_params = MagicMock()
        tracker.log_metrics = MagicMock()
        tracker.log_artifact = MagicMock()

        # Create AutoML pipeline with tracker
        pipeline = AutoMLPipeline(
            task_type="classification",
            experiment_tracker=tracker,
        )

        # Compare models
        results_df = pipeline.compare_models(X, y, models=["xgboost_classifier"], cv=3)

        # Verify tracking was called
        assert tracker.log_metrics.called
        # Should have logged metrics for the model
        call_args = tracker.log_metrics.call_args_list
        assert len(call_args) > 0

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_model_comparison_logs_metrics(self, sample_data):
        """Test that model comparison logs metrics to MLflow."""
        X, y = sample_data

        tracker = MagicMock(spec=ExperimentTracker)
        pipeline = AutoMLPipeline(
            task_type="classification",
            experiment_tracker=tracker,
        )

        pipeline.compare_models(X, y, models=["xgboost_classifier"], cv=3)

        # Verify metrics were logged
        assert tracker.log_metrics.called

        # Check that metrics contain model scores
        logged_metrics = tracker.log_metrics.call_args[0][0]
        assert any("xgboost" in str(key).lower() for key in logged_metrics)

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_hyperparameter_optimization_logs_params(self, sample_data):
        """Test that hyperparameter optimization logs params to MLflow."""
        X, y = sample_data

        tracker = MagicMock(spec=ExperimentTracker)
        pipeline = AutoMLPipeline(
            task_type="classification",
            experiment_tracker=tracker,
        )

        # Skip if Optuna not available
        try:
            from src.core.ml_engine.automl import OPTUNA_AVAILABLE

            if not OPTUNA_AVAILABLE:
                pytest.skip("Optuna not available")

            pipeline.optimize_hyperparameters(
                X,
                y,
                model_name="xgboost_classifier",
                n_trials=5,
                cv=3,
            )

            # Verify params were logged
            assert tracker.log_params.called

            # Check that params were logged
            logged_params = tracker.log_params.call_args[0][0]
            assert isinstance(logged_params, dict)
            assert len(logged_params) > 0

        except ImportError:
            pytest.skip("Optuna not installed")

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_model_metadata_tracked(self, sample_data):
        """Test that model metadata is properly tracked."""
        X, y = sample_data

        tracker = MagicMock(spec=ExperimentTracker)
        pipeline = AutoMLPipeline(
            task_type="classification",
            experiment_tracker=tracker,
        )

        pipeline.compare_models(X, y, models=["xgboost_classifier"], cv=3)

        # Get the trained model
        model = pipeline.best_model

        # Check model metadata
        metadata = model.get_metadata()

        assert metadata["is_fitted"] is True
        assert metadata["n_features"] == X.shape[1]
        assert "hyperparameters" in metadata


class TestBackwardCompatibilityWithMLflow:
    """Test backward compatibility layer with MLflow."""

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_automl_workflow_compatibility(self, sample_data):
        """Test AutoMLWorkflow (PyCaret replacement) with MLflow."""
        from src.core.ml.pycaret_integration import AutoMLWorkflow

        X, y = sample_data
        data = X.copy()
        data["target"] = y

        tracker = MagicMock(spec=ExperimentTracker)
        workflow = AutoMLWorkflow(experiment_tracker=tracker)

        # Run AutoML
        results = workflow.run_automl(
            data=data,
            task_type="classification",
            target="target",
            tune_hyperparameters=False,  # Skip optimization for speed
        )

        # Check results
        assert "final_model" in results
        assert results["final_model"] is not None
        assert results["final_model"].is_fitted

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_automl_workflow_predictions(self, sample_data):
        """Test predictions from AutoMLWorkflow."""
        from src.core.ml.pycaret_integration import AutoMLWorkflow

        X, y = sample_data
        data = X.copy()
        data["target"] = y

        workflow = AutoMLWorkflow()
        workflow.run_automl(
            data=data,
            task_type="classification",
            target="target",
            tune_hyperparameters=False,
        )

        # Get predictions
        predictions_df = workflow.get_predictions(data=data)

        assert predictions_df is not None
        assert "predictions" in predictions_df.columns
        assert len(predictions_df) == len(data)

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_automl_workflow_feature_importance(self, sample_data):
        """Test feature importance from AutoMLWorkflow."""
        from src.core.ml.pycaret_integration import AutoMLWorkflow

        X, y = sample_data
        data = X.copy()
        data["target"] = y

        workflow = AutoMLWorkflow()
        workflow.run_automl(
            data=data,
            task_type="classification",
            target="target",
            tune_hyperparameters=False,
        )

        # Get feature importance
        importance = workflow.get_model_interpretation()

        # Check if model supports feature importance
        if importance is not None:
            assert isinstance(importance, pd.DataFrame)
            assert "feature" in importance.columns
            assert "importance" in importance.columns


class TestMLflowExperimentLifecycle:
    """Test complete experiment lifecycle with MLflow."""

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_full_experiment_with_mlflow(self, sample_data, temp_mlflow_dir):
        """Test complete experiment lifecycle with MLflow tracking."""
        X, y = sample_data

        # Mock experiment tracker
        tracker = MagicMock(spec=ExperimentTracker)
        tracker.log_params = MagicMock()
        tracker.log_metrics = MagicMock()
        tracker.is_server_available = MagicMock(return_value=True)

        # Create and run AutoML pipeline
        pipeline = AutoMLPipeline(
            task_type="classification",
            experiment_tracker=tracker,
        )

        # Step 1: Compare models
        results = pipeline.compare_models(X, y, models=["xgboost_classifier"], cv=3)

        # Step 2: Make predictions
        predictions = pipeline.predict(X)

        # Verify complete lifecycle
        assert tracker.log_metrics.called
        assert len(results) > 0
        assert len(predictions) == len(X)
        assert pipeline.best_model is not None


class TestErrorHandling:
    """Test error handling in MLflow integration."""

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_graceful_handling_when_mlflow_unavailable(self, sample_data):
        """Test that ML engine works even when MLflow is unavailable."""
        X, y = sample_data

        # Create pipeline without tracker
        pipeline = AutoMLPipeline(
            task_type="classification",
            experiment_tracker=None,  # No MLflow
        )

        # Should still work
        results = pipeline.compare_models(X, y, models=["xgboost_classifier"], cv=3)

        assert len(results) > 0
        assert pipeline.best_model is not None

    def test_automl_workflow_without_mlflow(self, sample_data):
        """Test AutoMLWorkflow without MLflow tracker."""
        from src.core.ml.pycaret_integration import AutoMLWorkflow

        X, y = sample_data
        data = X.copy()
        data["target"] = y

        # Create workflow without tracker
        workflow = AutoMLWorkflow(experiment_tracker=None)

        # Skip if no models available
        try:
            results = workflow.run_automl(
                data=data,
                task_type="classification",
                target="target",
                tune_hyperparameters=False,
            )

            assert results["final_model"] is not None

        except ValueError as e:
            if "No models available" in str(e):
                pytest.skip("No ML models installed")
            else:
                raise
