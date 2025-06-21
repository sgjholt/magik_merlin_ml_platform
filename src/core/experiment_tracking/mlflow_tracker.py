import logging
from typing import Any

import mlflow
import mlflow.pycaret
import mlflow.sklearn
import pandas as pd


class MLflowExperimentTracker:
    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "ml-platform",
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.current_run = None

        # Set up MLflow
        mlflow.set_tracking_uri(self.tracking_uri)

        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        except Exception:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            self.experiment_id = experiment.experiment_id if experiment else None

    def start_run(
        self, run_name: str | None = None, tags: dict[str, Any] | None = None
    ) -> str:
        """Start a new MLflow run"""
        self.current_run = mlflow.start_run(
            experiment_id=self.experiment_id, run_name=run_name, tags=tags
        )
        return self.current_run.info.run_id

    def end_run(self) -> None:
        """End the current MLflow run"""
        if self.current_run:
            mlflow.end_run()
            self.current_run = None

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to MLflow"""
        if self.current_run:
            mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to MLflow"""
        if self.current_run:
            for name, value in metrics.items():
                mlflow.log_metric(name, value, step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log an artifact to MLflow"""
        if self.current_run:
            mlflow.log_artifact(local_path, artifact_path)

    def log_model(self, model, model_name: str, **kwargs) -> None:
        """Log a model to MLflow"""
        if self.current_run:
            mlflow.sklearn.log_model(model, model_name, **kwargs)

    def log_pycaret_experiment(
        self, experiment, model_name: str = "pycaret_model"
    ) -> None:
        """Log PyCaret experiment to MLflow"""
        if self.current_run:
            # Log the PyCaret experiment
            mlflow.pycaret.log_model(experiment, model_name)

            # Log experiment configuration
            if hasattr(experiment, "get_config"):
                config = experiment.get_config()
                self.log_params(
                    {
                        "pycaret_setup": str(config.get("setup", {})),
                        "target_column": config.get("target", ""),
                        "session_id": config.get("session_id", ""),
                        "train_size": config.get("train_size", 0.7),
                    }
                )

    def log_dataframe(self, df: pd.DataFrame, name: str) -> None:
        """Log a DataFrame as an artifact"""
        if self.current_run:
            temp_path = f"/tmp/{name}.csv"
            df.to_csv(temp_path, index=False)
            self.log_artifact(temp_path, f"data/{name}.csv")

    def get_experiment_runs(self, max_results: int = 100) -> pd.DataFrame:
        """Get runs from the current experiment"""
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id], max_results=max_results
        )
        return runs

    def get_best_run(
        self, metric_name: str, ascending: bool = False
    ) -> dict[str, Any] | None:
        """Get the best run based on a specific metric"""
        runs = self.get_experiment_runs()
        if runs.empty:
            return None

        metric_col = f"metrics.{metric_name}"
        if metric_col not in runs.columns:
            return None

        best_run = runs.loc[
            runs[metric_col].idxmin() if ascending else runs[metric_col].idxmax()
        ]
        return best_run.to_dict()

    def load_model(self, run_id: str, model_name: str = "model"):
        """Load a model from MLflow"""
        model_uri = f"runs:/{run_id}/{model_name}"
        return mlflow.sklearn.load_model(model_uri)

    def register_model(
        self, run_id: str, model_name: str, registry_model_name: str
    ) -> None:
        """Register a model to MLflow Model Registry"""
        model_uri = f"runs:/{run_id}/{model_name}"
        mlflow.register_model(model_uri, registry_model_name)

    def transition_model_stage(self, model_name: str, version: int, stage: str) -> None:
        """Transition model to a different stage"""
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name, version=version, stage=stage
        )

    def get_registered_models(self) -> list[dict[str, Any]]:
        """Get all registered models"""
        client = mlflow.tracking.MlflowClient()
        models = client.search_registered_models()

        model_list = []
        for model in models:
            model_info = {
                "name": model.name,
                "creation_timestamp": model.creation_timestamp,
                "last_updated_timestamp": model.last_updated_timestamp,
                "description": model.description,
                "latest_versions": [],
            }

            for version in model.latest_versions:
                version_info = {
                    "version": version.version,
                    "stage": version.current_stage,
                    "run_id": version.run_id,
                    "creation_timestamp": version.creation_timestamp,
                }
                model_info["latest_versions"].append(version_info)

            model_list.append(model_info)

        return model_list

    def delete_run(self, run_id: str) -> None:
        """Delete a run"""
        client = mlflow.tracking.MlflowClient()
        client.delete_run(run_id)

    def search_runs(
        self, filter_string: str = "", max_results: int = 1000
    ) -> pd.DataFrame:
        """Search runs with filters"""
        return mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            max_results=max_results,
        )

    def compare_runs(self, run_ids: list[str]) -> pd.DataFrame:
        """Compare multiple runs"""
        all_runs = self.get_experiment_runs()
        filtered_runs = all_runs[all_runs["run_id"].isin(run_ids)]
        return filtered_runs

    def get_run_info(self, run_id: str) -> dict[str, Any] | None:
        """Get detailed information about a specific run"""
        client = mlflow.tracking.MlflowClient()
        try:
            run = client.get_run(run_id)
            return {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "params": dict(run.data.params),
                "metrics": dict(run.data.metrics),
                "tags": dict(run.data.tags),
            }
        except Exception as e:
            logging.exception(f"Error getting run info: {e!s}")
            return None
