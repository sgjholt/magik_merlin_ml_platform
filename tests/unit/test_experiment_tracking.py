"""
Unit tests for experiment tracking
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.core.experiments.tracking import MLFLOW_AVAILABLE, ExperimentTracker


class TestExperimentTracker:
    """Test ExperimentTracker functionality"""

    def test_initialization_without_mlflow(self):
        """Test initialization when MLflow is not available"""
        with patch("src.core.experiments.tracking.MLFLOW_AVAILABLE", False):
            tracker = ExperimentTracker()
            assert tracker.tracking_uri == "http://127.0.0.1:5000"
            assert tracker.experiment_name == "ml-platform-experiments"
            assert tracker.active_run is None

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_initialization_with_mlflow(self):
        """Test initialization when MLflow is available"""
        with patch("src.core.experiments.tracking.mlflow") as mock_mlflow:
            with patch.object(
                ExperimentTracker, "_check_mlflow_server", return_value=True
            ):
                tracker = ExperimentTracker()
                mock_mlflow.set_tracking_uri.assert_called_once_with(
                    "http://127.0.0.1:5000"
                )
                mock_mlflow.set_experiment.assert_called_once_with(
                    "ml-platform-experiments"
                )

    def test_custom_initialization(self):
        """Test initialization with custom parameters"""
        tracker = ExperimentTracker(
            tracking_uri="http://custom:5001", experiment_name="test-experiment"
        )
        assert tracker.tracking_uri == "http://custom:5001"
        assert tracker.experiment_name == "test-experiment"

    def test_start_run_without_mlflow(self):
        """Test start_run when MLflow is not available"""
        with patch("src.core.experiments.tracking.MLFLOW_AVAILABLE", False):
            tracker = ExperimentTracker()
            result = tracker.start_run("test-run")
            assert result is False
            assert tracker.active_run is None

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_start_run_with_mlflow(self):
        """Test start_run when MLflow is available"""
        with patch("src.core.experiments.tracking.mlflow") as mock_mlflow:
            mock_run = Mock()
            mock_mlflow.start_run.return_value = mock_run

            with patch.object(
                ExperimentTracker, "_check_mlflow_server", return_value=True
            ):
                tracker = ExperimentTracker()
                result = tracker.start_run("test-run")

                assert result is True
                assert tracker.active_run == mock_run
                mock_mlflow.start_run.assert_called_once_with(run_name="test-run")

    def test_end_run_without_active_run(self):
        """Test end_run without active run"""
        tracker = ExperimentTracker()
        # Should not raise an error
        tracker.end_run()

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_end_run_with_active_run(self):
        """Test end_run with active run"""
        with patch("src.core.experiments.tracking.mlflow") as mock_mlflow:
            tracker = ExperimentTracker()
            tracker.active_run = Mock()

            tracker.end_run()

            mock_mlflow.end_run.assert_called_once()
            assert tracker.active_run is None

    def test_log_params_without_mlflow(self):
        """Test log_params when MLflow is not available"""
        with patch("src.core.experiments.tracking.MLFLOW_AVAILABLE", False):
            tracker = ExperimentTracker()
            # Should not raise an error
            tracker.log_params({"test_param": "value"})

    def test_log_params_without_active_run(self):
        """Test log_params without active run"""
        tracker = ExperimentTracker()
        # Should not raise an error
        tracker.log_params({"test_param": "value"})

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_log_params_with_active_run(self):
        """Test log_params with active run"""
        with patch("src.core.experiments.tracking.mlflow") as mock_mlflow:
            tracker = ExperimentTracker()
            tracker.active_run = Mock()

            params = {"param1": "value1", "param2": "value2"}
            tracker.log_params(params)

            assert mock_mlflow.log_param.call_count == 2

    def test_log_metrics_without_mlflow(self):
        """Test log_metrics when MLflow is not available"""
        with patch("src.core.experiments.tracking.MLFLOW_AVAILABLE", False):
            tracker = ExperimentTracker()
            # Should not raise an error
            tracker.log_metrics({"accuracy": 0.85})

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_log_metrics_with_active_run(self):
        """Test log_metrics with active run"""
        with patch("src.core.experiments.tracking.mlflow") as mock_mlflow:
            tracker = ExperimentTracker()
            tracker.active_run = Mock()

            metrics = {"accuracy": 0.85, "precision": 0.82, "invalid": "not_numeric"}
            tracker.log_metrics(metrics)

            # Should only log numeric metrics
            assert mock_mlflow.log_metric.call_count == 2

    def test_log_artifact_without_mlflow(self):
        """Test log_artifact when MLflow is not available"""
        with patch("src.core.experiments.tracking.MLFLOW_AVAILABLE", False):
            tracker = ExperimentTracker()
            # Should not raise an error
            tracker.log_artifact("test_artifact", Mock())

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_log_artifact_with_active_run(self):
        """Test log_artifact with active run"""
        with patch("src.core.experiments.tracking.mlflow") as mock_mlflow:
            tracker = ExperimentTracker()
            tracker.active_run = Mock()

            tracker.log_artifact("test_artifact", Mock())

            mock_mlflow.log_param.assert_called_once_with(
                "artifact_test_artifact", "created"
            )

    def test_log_model_without_mlflow(self):
        """Test log_model when MLflow is not available"""
        with patch("src.core.experiments.tracking.MLFLOW_AVAILABLE", False):
            tracker = ExperimentTracker()
            # Should not raise an error
            tracker.log_model(Mock(), "test_model")

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_log_model_sklearn_compatible(self):
        """Test log_model with sklearn-compatible model"""
        with patch("src.core.experiments.tracking.mlflow") as mock_mlflow:
            tracker = ExperimentTracker()
            tracker.active_run = Mock()

            # Mock sklearn-like model
            model = Mock()
            model.fit = Mock()
            model.predict = Mock()

            tracker.log_model(model, "sklearn_model")

            mock_mlflow.sklearn.log_model.assert_called_once_with(
                model, "sklearn_model"
            )

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_log_model_non_sklearn(self):
        """Test log_model with non-sklearn model"""
        with patch("src.core.experiments.tracking.mlflow") as mock_mlflow:
            with patch.object(
                ExperimentTracker, "_check_mlflow_server", return_value=True
            ):
                tracker = ExperimentTracker()
                tracker.active_run = Mock()

                # Mock non-sklearn model (no fit/predict methods)
                model = Mock(spec=[])  # Empty spec means no methods

                tracker.log_model(model, "other_model")

                mock_mlflow.log_param.assert_called_once()

    def test_is_active_without_mlflow(self):
        """Test is_active when MLflow is not available"""
        with patch("src.core.experiments.tracking.MLFLOW_AVAILABLE", False):
            tracker = ExperimentTracker()
            assert tracker.is_active() is False

    def test_is_active_without_run(self):
        """Test is_active without active run"""
        tracker = ExperimentTracker()
        assert tracker.is_active() is False

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_is_active_with_run(self):
        """Test is_active with active run"""
        with patch("src.core.experiments.tracking.mlflow") as mock_mlflow:
            with patch.object(
                ExperimentTracker, "_check_mlflow_server", return_value=True
            ):
                tracker = ExperimentTracker()
                tracker.active_run = Mock()
                assert tracker.is_active() is True


class TestMLflowIntegrationFallbacks:
    """Test fallback behavior when MLflow is not available"""

    def test_import_safety(self):
        """Test that module can be imported even without MLflow"""
        # This test passes if the module imports successfully
        from src.core.experiments.tracking import ExperimentTracker

        assert ExperimentTracker is not None

    def test_mlflow_available_flag(self):
        """Test MLFLOW_AVAILABLE flag reflects actual availability"""
        from src.core.experiments.tracking import MLFLOW_AVAILABLE

        try:
            import mlflow

            assert MLFLOW_AVAILABLE is True
        except ImportError:
            assert MLFLOW_AVAILABLE is False
