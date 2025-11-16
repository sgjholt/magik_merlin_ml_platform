"""
Experiment tracking integration with MLflow
"""

import logging
from typing import Any

import requests

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
        # Import here to avoid circular imports
        from ...config.settings import settings

        self.tracking_uri = tracking_uri or settings.mlflow_tracking_uri
        self.experiment_name = experiment_name or settings.mlflow_experiment_name
        self.active_run = None
        self.logger = logging.getLogger(__name__)
        self._server_available = False

        if MLFLOW_AVAILABLE:
            self._check_and_setup_mlflow()
        else:
            self.logger.warning("MLflow not available - experiment tracking disabled")

    def _check_mlflow_server(self, timeout: int = 5) -> bool:
        """Check if MLflow server is accessible"""
        try:
            # Just check if the root endpoint returns the MLflow UI
            response = requests.get(self.tracking_uri, timeout=timeout)
            return response.status_code == 200 and "MLflow" in response.text
        except requests.exceptions.RequestException:
            return False

    def _check_and_setup_mlflow(self) -> None:
        """Check MLflow server availability and setup connection"""
        try:
            # First check if server is running
            if self._check_mlflow_server():
                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)
                self._server_available = True
                self.logger.info(f"MLflow tracking configured: {self.tracking_uri}")
            else:
                self.logger.warning(
                    f"MLflow server not accessible at {self.tracking_uri}. "
                    "Start it with: ./scripts/mlflow.sh start"
                )
                self._server_available = False
        except Exception as e:
            self.logger.warning(f"Failed to setup MLflow: {e}")
            self._server_available = False

    def start_run(self, run_name: str | None = None) -> bool:
        """Start a new MLflow run"""
        if not MLFLOW_AVAILABLE or not self._server_available:
            return False

        try:
            # Double-check server is still available
            if not self._check_mlflow_server():
                self.logger.warning("MLflow server not available when starting run")
                self._server_available = False
                return False

            self.active_run = mlflow.start_run(run_name=run_name)
            self.logger.info(f"Started MLflow run: {self.active_run.info.run_id}")
            return True
        except Exception as e:
            self.logger.exception(f"Failed to start MLflow run: {e}")
            self._server_available = False
            return False

    def end_run(self) -> None:
        """End the current MLflow run"""
        if MLFLOW_AVAILABLE and self.active_run:
            try:
                mlflow.end_run()
                self.active_run = None
            except Exception as e:
                self.logger.exception(f"Failed to end MLflow run: {e}")

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to MLflow"""
        if not MLFLOW_AVAILABLE or not self.active_run:
            return

        try:
            for key, value in params.items():
                mlflow.log_param(key, value)
        except Exception as e:
            self.logger.exception(f"Failed to log parameters: {e}")

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """Log metrics to MLflow"""
        if not MLFLOW_AVAILABLE or not self.active_run:
            return

        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
        except Exception as e:
            self.logger.exception(f"Failed to log metrics: {e}")

    def log_artifact(self, name: str, artifact: Any) -> None:
        """Log artifacts to MLflow"""
        if not MLFLOW_AVAILABLE or not self.active_run:
            return

        try:
            # For now, just log that an artifact was created
            mlflow.log_param(f"artifact_{name}", "created")
        except Exception as e:
            self.logger.exception(f"Failed to log artifact: {e}")

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
            self.logger.exception(f"Failed to log model: {e}")

    def is_active(self) -> bool:
        """Check if tracking is active"""
        return (
            MLFLOW_AVAILABLE and self._server_available and self.active_run is not None
        )

    def is_server_available(self) -> bool:
        """Check if MLflow server is available"""
        return MLFLOW_AVAILABLE and self._server_available

    def reconnect(self) -> bool:
        """Attempt to reconnect to MLflow server"""
        if MLFLOW_AVAILABLE:
            self._check_and_setup_mlflow()
            return self._server_available
        return False
