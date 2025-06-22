"""
Experiment tracking integration with MLflow
"""

import logging
from typing import Any

try:
    import mlflow
    import mlflow.sklearn

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None


class ExperimentTracker:
    """MLflow-based experiment tracking"""

    def __init__(
        self, tracking_uri: str | None = None, experiment_name: str | None = None
    ):
        self.tracking_uri = tracking_uri or "http://localhost:5000"
        self.experiment_name = experiment_name or "ml-platform-experiments"
        self.active_run = None
        self.logger = logging.getLogger(__name__)

        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)
            except Exception as e:
                self.logger.warning(f"Failed to setup MLflow: {e}")
        else:
            self.logger.warning("MLflow not available - experiment tracking disabled")

    def start_run(self, run_name: str | None = None) -> bool:
        """Start a new MLflow run"""
        if not MLFLOW_AVAILABLE:
            return False

        try:
            self.active_run = mlflow.start_run(run_name=run_name)
            return True
        except Exception as e:
            self.logger.error(f"Failed to start MLflow run: {e}")
            return False

    def end_run(self) -> None:
        """End the current MLflow run"""
        if MLFLOW_AVAILABLE and self.active_run:
            try:
                mlflow.end_run()
                self.active_run = None
            except Exception as e:
                self.logger.error(f"Failed to end MLflow run: {e}")

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to MLflow"""
        if not MLFLOW_AVAILABLE or not self.active_run:
            return

        try:
            for key, value in params.items():
                mlflow.log_param(key, value)
        except Exception as e:
            self.logger.error(f"Failed to log parameters: {e}")

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """Log metrics to MLflow"""
        if not MLFLOW_AVAILABLE or not self.active_run:
            return

        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
        except Exception as e:
            self.logger.error(f"Failed to log metrics: {e}")

    def log_artifact(self, name: str, artifact: Any) -> None:
        """Log artifacts to MLflow"""
        if not MLFLOW_AVAILABLE or not self.active_run:
            return

        try:
            # For now, just log that an artifact was created
            mlflow.log_param(f"artifact_{name}", "created")
        except Exception as e:
            self.logger.error(f"Failed to log artifact: {e}")

    def log_model(self, model: Any, name: str = "model") -> None:
        """Log a model to MLflow"""
        if not MLFLOW_AVAILABLE or not self.active_run:
            return

        try:
            # Try to log as sklearn model first
            if hasattr(model, "fit") and hasattr(model, "predict"):
                mlflow.sklearn.log_model(model, name)
            else:
                mlflow.log_param(f"model_{name}", str(type(model)))
        except Exception as e:
            self.logger.error(f"Failed to log model: {e}")

    def is_active(self) -> bool:
        """Check if tracking is active"""
        return MLFLOW_AVAILABLE and self.active_run is not None
