"""
Enhanced Experiment Management System

Provides comprehensive experiment lifecycle management with:
- Experiment lifecycle tracking (setup, running, completed, failed)
- Model versioning and comparison
- Artifact management (plots, data, models)
- Real-time progress tracking
- Experiment metadata and tagging
- Performance benchmarking
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd

from ..logging import get_logger, log_performance
from .tracking import ExperimentTracker


class ExperimentStatus:
    """Experiment status constants"""

    PENDING = "pending"
    SETUP = "setup"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Experiment:
    """Enhanced experiment representation with metadata and lifecycle management"""

    def __init__(
        self,
        name: str,
        task_type: str,
        dataset_info: Dict[str, Any],
        experiment_id: Optional[str] = None,
    ):
        self.id = experiment_id or str(uuid4())
        self.name = name
        self.task_type = task_type
        self.dataset_info = dataset_info
        self.status = ExperimentStatus.PENDING
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.duration: Optional[float] = None

        # Experiment configuration
        self.config: Dict[str, Any] = {}
        self.tags: Dict[str, str] = {}
        self.parameters: Dict[str, Any] = {}
        self.metrics: Dict[str, float] = {}

        # Model and artifact tracking
        self.models: List[Dict[str, Any]] = []
        self.artifacts: List[Dict[str, Any]] = []
        self.best_model: Optional[Dict[str, Any]] = None

        # Progress tracking
        self.progress: Dict[str, Any] = {
            "current_stage": "initialization",
            "stages_completed": [],
            "total_stages": 0,
            "progress_percentage": 0,
        }

        # Error tracking
        self.errors: List[Dict[str, Any]] = []

        self.logger = get_logger(__name__, experiment_id=self.id, task_type=task_type)

    def update_status(self, status: str, message: Optional[str] = None) -> None:
        """Update experiment status with optional message"""
        old_status = self.status
        self.status = status

        if status == ExperimentStatus.RUNNING and not self.started_at:
            self.started_at = datetime.now()
        elif status in [
            ExperimentStatus.COMPLETED,
            ExperimentStatus.FAILED,
            ExperimentStatus.CANCELLED,
        ]:
            if not self.completed_at:
                self.completed_at = datetime.now()
                if self.started_at:
                    self.duration = (
                        self.completed_at - self.started_at
                    ).total_seconds()

        self.logger.info(
            f"Experiment status changed: {old_status} -> {status}",
            extra={
                "old_status": old_status,
                "new_status": status,
                "message": message,
                "duration": self.duration,
            },
        )

    def add_model(self, model_info: Dict[str, Any]) -> None:
        """Add model information to experiment"""
        model_info["added_at"] = datetime.now().isoformat()
        model_info["model_id"] = str(uuid4())
        self.models.append(model_info)

        self.logger.info(
            "Model added to experiment",
            extra={
                "model_name": model_info.get("name"),
                "model_type": model_info.get("type"),
                "model_id": model_info["model_id"],
            },
        )

    def add_artifact(self, artifact_info: Dict[str, Any]) -> None:
        """Add artifact information to experiment"""
        artifact_info["added_at"] = datetime.now().isoformat()
        artifact_info["artifact_id"] = str(uuid4())
        self.artifacts.append(artifact_info)

        self.logger.info(
            "Artifact added to experiment",
            extra={
                "artifact_name": artifact_info.get("name"),
                "artifact_type": artifact_info.get("type"),
                "artifact_id": artifact_info["artifact_id"],
            },
        )

    def update_progress(
        self, stage: str, percentage: float, details: Optional[Dict] = None
    ) -> None:
        """Update experiment progress"""
        self.progress.update(
            {
                "current_stage": stage,
                "progress_percentage": percentage,
                "last_updated": datetime.now().isoformat(),
            }
        )

        if details:
            self.progress.update(details)

        self.logger.debug(
            "Progress updated",
            extra={"stage": stage, "percentage": percentage, "details": details},
        )

    def add_error(
        self, error: Exception, stage: str, context: Optional[Dict] = None
    ) -> None:
        """Add error information to experiment"""
        error_info = {
            "error_id": str(uuid4()),
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
        }

        self.errors.append(error_info)

        self.logger.error(
            f"Error in stage {stage}",
            extra={
                "stage": stage,
                "error_type": type(error).__name__,
                "context": context,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "task_type": self.task_type,
            "dataset_info": self.dataset_info,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "duration": self.duration,
            "config": self.config,
            "tags": self.tags,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "models": self.models,
            "artifacts": self.artifacts,
            "best_model": self.best_model,
            "progress": self.progress,
            "errors": self.errors,
        }


class ExperimentManager:
    """
    Comprehensive experiment management system

    Handles experiment lifecycle, model versioning, artifact management,
    and provides advanced tracking capabilities beyond basic MLflow integration.
    """

    def __init__(self, experiment_tracker: Optional[ExperimentTracker] = None):
        self.experiment_tracker = experiment_tracker
        self.experiments: Dict[str, Experiment] = {}
        self.active_experiment: Optional[Experiment] = None

        # Storage paths
        self.experiments_dir = Path("experiments")
        self.models_dir = self.experiments_dir / "models"
        self.artifacts_dir = self.experiments_dir / "artifacts"
        self.metadata_dir = self.experiments_dir / "metadata"

        # Create directories
        for dir_path in [
            self.experiments_dir,
            self.models_dir,
            self.artifacts_dir,
            self.metadata_dir,
        ]:
            dir_path.mkdir(exist_ok=True, parents=True)

        self.logger = get_logger(__name__, pipeline_stage="experiment_management")

        # Load existing experiments
        self._load_experiments()

    def create_experiment(
        self,
        name: str,
        task_type: str,
        dataset_info: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Experiment:
        """Create a new experiment with comprehensive metadata"""

        experiment = Experiment(name, task_type, dataset_info)

        if config:
            experiment.config.update(config)
        if tags:
            experiment.tags.update(tags)

        # Add system tags
        experiment.tags.update(
            {
                "created_by": "ml_platform",
                "platform_version": "0.1.0",
                "task_type": task_type,
            }
        )

        self.experiments[experiment.id] = experiment
        self._save_experiment(experiment)

        self.logger.info(
            "New experiment created",
            extra={
                "experiment_name": name,
                "task_type": task_type,
                "dataset_rows": dataset_info.get("rows"),
                "dataset_cols": dataset_info.get("columns"),
            },
        )

        return experiment

    def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment and begin MLflow tracking"""

        if experiment_id not in self.experiments:
            self.logger.error(f"Experiment {experiment_id} not found")
            return False

        experiment = self.experiments[experiment_id]

        if experiment.status != ExperimentStatus.PENDING:
            self.logger.warning(f"Experiment {experiment_id} is not in pending status")
            return False

        # Start MLflow run if available
        mlflow_success = False
        if self.experiment_tracker:
            mlflow_success = self.experiment_tracker.start_run(run_name=experiment.name)
            if mlflow_success:
                # Log experiment metadata to MLflow
                self.experiment_tracker.log_params(
                    {
                        "experiment_id": experiment.id,
                        "task_type": experiment.task_type,
                        "dataset_rows": experiment.dataset_info.get("rows", 0),
                        "dataset_columns": experiment.dataset_info.get("columns", 0),
                        **experiment.config,
                        **experiment.tags,
                    }
                )

        experiment.update_status(ExperimentStatus.RUNNING, "Experiment started")
        self.active_experiment = experiment
        self._save_experiment(experiment)

        self.logger.info(
            "Experiment started",
            extra={
                "mlflow_tracking": mlflow_success,
                "experiment_name": experiment.name,
            },
        )

        return True

    def complete_experiment(
        self, experiment_id: str, final_metrics: Optional[Dict[str, float]] = None
    ) -> bool:
        """Complete an experiment with final results"""

        if experiment_id not in self.experiments:
            return False

        experiment = self.experiments[experiment_id]

        if final_metrics:
            experiment.metrics.update(final_metrics)

            # Log final metrics to MLflow
            if self.experiment_tracker and self.experiment_tracker.is_active():
                self.experiment_tracker.log_metrics(final_metrics)

        experiment.update_status(
            ExperimentStatus.COMPLETED, "Experiment completed successfully"
        )

        # End MLflow run
        if self.experiment_tracker:
            self.experiment_tracker.end_run()

        self._save_experiment(experiment)

        if self.active_experiment and self.active_experiment.id == experiment_id:
            self.active_experiment = None

        self.logger.info(
            "Experiment completed",
            extra={
                "final_metrics": final_metrics,
                "duration": experiment.duration,
                "models_created": len(experiment.models),
            },
        )

        return True

    def fail_experiment(
        self, experiment_id: str, error: Exception, context: Optional[Dict] = None
    ) -> bool:
        """Mark experiment as failed with error details"""

        if experiment_id not in self.experiments:
            return False

        experiment = self.experiments[experiment_id]
        experiment.add_error(error, "experiment_failure", context)
        experiment.update_status(
            ExperimentStatus.FAILED, f"Experiment failed: {error!s}"
        )

        # End MLflow run
        if self.experiment_tracker:
            self.experiment_tracker.end_run()

        self._save_experiment(experiment)

        if self.active_experiment and self.active_experiment.id == experiment_id:
            self.active_experiment = None

        return True

    @log_performance
    def save_model(
        self,
        experiment_id: str,
        model: Any,
        model_name: str,
        model_info: Optional[Dict] = None,
    ) -> str:
        """Save model with comprehensive metadata"""

        if experiment_id not in self.experiments:
            msg = f"Experiment {experiment_id} not found"
            raise ValueError(msg)

        experiment = self.experiments[experiment_id]
        model_id = str(uuid4())

        # Save model to disk
        model_path = self.models_dir / f"{experiment_id}_{model_id}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Create model metadata
        model_metadata = {
            "model_id": model_id,
            "name": model_name,
            "type": type(model).__name__,
            "file_path": str(model_path),
            "file_size": model_path.stat().st_size,
            "experiment_id": experiment_id,
            **(model_info or {}),
        }

        experiment.add_model(model_metadata)

        # Log to MLflow if available
        if self.experiment_tracker and self.experiment_tracker.is_active():
            self.experiment_tracker.log_model(model, model_name)

        self._save_experiment(experiment)

        return model_id

    def save_artifact(
        self, experiment_id: str, artifact: Any, artifact_name: str, artifact_type: str
    ) -> str:
        """Save experiment artifact with metadata"""

        if experiment_id not in self.experiments:
            msg = f"Experiment {experiment_id} not found"
            raise ValueError(msg)

        experiment = self.experiments[experiment_id]
        artifact_id = str(uuid4())

        # Determine save strategy based on type
        if artifact_type == "dataframe":
            artifact_path = (
                self.artifacts_dir / f"{experiment_id}_{artifact_id}.parquet"
            )
            artifact.to_parquet(artifact_path)
        elif artifact_type == "plot":
            artifact_path = self.artifacts_dir / f"{experiment_id}_{artifact_id}.html"
            artifact.write_html(str(artifact_path))
        else:
            # Generic pickle save
            artifact_path = self.artifacts_dir / f"{experiment_id}_{artifact_id}.pkl"
            with open(artifact_path, "wb") as f:
                pickle.dump(artifact, f)

        # Create artifact metadata
        artifact_metadata = {
            "artifact_id": artifact_id,
            "name": artifact_name,
            "type": artifact_type,
            "file_path": str(artifact_path),
            "file_size": artifact_path.stat().st_size,
            "experiment_id": experiment_id,
        }

        experiment.add_artifact(artifact_metadata)
        self._save_experiment(experiment)

        return artifact_id

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID"""
        return self.experiments.get(experiment_id)

    def list_experiments(
        self, status: Optional[str] = None, task_type: Optional[str] = None
    ) -> List[Experiment]:
        """List experiments with optional filtering"""
        experiments = list(self.experiments.values())

        if status:
            experiments = [exp for exp in experiments if exp.status == status]
        if task_type:
            experiments = [exp for exp in experiments if exp.task_type == task_type]

        # Sort by creation date (newest first)
        experiments.sort(key=lambda x: x.created_at, reverse=True)

        return experiments

    def get_experiment_comparison(self, experiment_ids: List[str]) -> pd.DataFrame:
        """Get comparison table for multiple experiments"""

        comparison_data = []
        for exp_id in experiment_ids:
            if exp_id in self.experiments:
                exp = self.experiments[exp_id]
                row = {
                    "experiment_id": exp.id,
                    "name": exp.name,
                    "task_type": exp.task_type,
                    "status": exp.status,
                    "duration": exp.duration,
                    "models_count": len(exp.models),
                    "created_at": exp.created_at,
                    **exp.metrics,
                }
                comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def _save_experiment(self, experiment: Experiment) -> None:
        """Save experiment metadata to disk"""
        metadata_path = self.metadata_dir / f"{experiment.id}.json"
        with open(metadata_path, "w") as f:
            json.dump(experiment.to_dict(), f, indent=2, default=str)

    def _load_experiments(self) -> None:
        """Load existing experiments from disk"""
        if not self.metadata_dir.exists():
            return

        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file) as f:
                    data = json.load(f)

                experiment = Experiment(
                    data["name"], data["task_type"], data["dataset_info"], data["id"]
                )

                # Restore experiment state
                experiment.status = data["status"]
                experiment.created_at = datetime.fromisoformat(data["created_at"])
                if data["started_at"]:
                    experiment.started_at = datetime.fromisoformat(data["started_at"])
                if data["completed_at"]:
                    experiment.completed_at = datetime.fromisoformat(
                        data["completed_at"]
                    )
                experiment.duration = data["duration"]
                experiment.config = data["config"]
                experiment.tags = data["tags"]
                experiment.parameters = data["parameters"]
                experiment.metrics = data["metrics"]
                experiment.models = data["models"]
                experiment.artifacts = data["artifacts"]
                experiment.best_model = data["best_model"]
                experiment.progress = data["progress"]
                experiment.errors = data["errors"]

                self.experiments[experiment.id] = experiment

            except Exception as e:
                self.logger.warning(
                    f"Failed to load experiment from {metadata_file}: {e}"
                )

        self.logger.info(f"Loaded {len(self.experiments)} existing experiments")
