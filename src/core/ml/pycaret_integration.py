"""
ML Engine Integration (replaces PyCaret)

This module provides backward compatibility with the old PyCaret interface
while using our new custom ML engine internally.

DEPRECATED: This module is maintained for backward compatibility only.
New code should use src.core.ml_engine directly.
"""

from typing import Any

import pandas as pd

from ..experiments.tracking import ExperimentTracker
from ..logging import get_logger
from ..ml_engine import AutoMLPipeline

# For backward compatibility
PYCARET_AVAILABLE = False


class AutoMLWorkflow:
    """
    Automated ML workflow using custom ML Engine (PyCaret replacement).

    This class provides a simplified interface compatible with the old PyCaret
    integration while using our modern custom ML engine internally.
    """

    def __init__(self, experiment_tracker: ExperimentTracker | None = None) -> None:
        """
        Initialize AutoML workflow.

        Args:
            experiment_tracker: Optional MLflow experiment tracker
        """
        self.experiment_tracker = experiment_tracker
        self.pipeline: AutoMLPipeline | None = None
        self.results: dict[str, Any] = {}
        self.logger = get_logger(__name__, pipeline_stage="automl")

    def run_automl(
        self,
        data: pd.DataFrame,
        task_type: str,
        target: str | None = None,
        test_data: pd.DataFrame | None = None,
        model_selection: str = "compare_all",
        tune_hyperparameters: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run complete AutoML workflow.

        Args:
            data: Training data
            task_type: Type of ML task (classification or regression)
            target: Target column name
            test_data: Optional test data
            model_selection: Model selection strategy
            tune_hyperparameters: Whether to optimize hyperparameters
            **kwargs: Additional arguments

        Returns:
            Dictionary with results including best model and evaluations
        """
        self.logger.info(
            f"Starting AutoML workflow for {task_type}",
            extra={
                "n_samples": len(data),
                "target": target,
                "tune_hyperparameters": tune_hyperparameters,
            },
        )

        # Prepare data
        if target is None:
            msg = "Target column must be specified"
            raise ValueError(msg)

        X = data.drop(columns=[target])
        y = data[target]

        # Initialize pipeline
        self.pipeline = AutoMLPipeline(
            task_type=task_type,
            experiment_tracker=self.experiment_tracker,
        )

        # Compare models
        comparison_results = self.pipeline.compare_models(X, y)

        self.logger.info(
            "Model comparison completed",
            extra={"n_models": len(comparison_results)},
        )

        # Optimize best model if requested
        optimized_model = None
        if tune_hyperparameters and self.pipeline.best_model_name:
            try:
                opt_result = self.pipeline.optimize_hyperparameters(
                    X,
                    y,
                    self.pipeline.best_model_name,
                )
                optimized_model = opt_result["model"]
            except Exception as e:
                self.logger.error(
                    "Hyperparameter optimization failed",
                    exc_info=True,
                    extra={"error_type": type(e).__name__},
                )

        # Store results
        self.results = {
            "pipeline": self.pipeline,
            "best_models": [self.pipeline.best_model],
            "comparison_results": comparison_results,
            "final_model": optimized_model or self.pipeline.best_model,
            "task_type": task_type,
            "target": target,
        }

        self.logger.info("AutoML workflow completed successfully")

        return self.results

    def get_predictions(
        self,
        data: pd.DataFrame | None = None,
    ) -> pd.DataFrame | None:
        """
        Get predictions from the final model.

        Args:
            data: Data to predict on (uses training data if None)

        Returns:
            DataFrame with predictions or None if no model trained
        """
        if not self.results.get("final_model") or not self.pipeline:
            return None

        try:
            if data is not None:
                # Remove target column if present
                target = self.results.get("target")
                if target and target in data.columns:
                    X = data.drop(columns=[target])
                else:
                    X = data
            else:
                msg = "Data must be provided for predictions"
                raise ValueError(msg)

            predictions = self.results["final_model"].predict(X)

            # Return as DataFrame
            return pd.DataFrame({"predictions": predictions})

        except Exception as e:
            self.logger.error(
                "Prediction failed",
                exc_info=True,
                extra={"error_type": type(e).__name__},
            )
            return None

    def get_model_interpretation(self, plot_type: str = "summary") -> Any:
        """
        Get model interpretation (feature importance).

        Args:
            plot_type: Type of interpretation plot (ignored for now)

        Returns:
            Feature importance DataFrame or None
        """
        if not self.results.get("final_model"):
            return None

        try:
            model = self.results["final_model"]
            if hasattr(model, "get_feature_importance"):
                return model.get_feature_importance()
            return None
        except Exception as e:
            self.logger.error(
                "Interpretation failed",
                exc_info=True,
                extra={"error_type": type(e).__name__},
            )
            return None
