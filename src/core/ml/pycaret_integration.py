"""
PyCaret integration for automated ML workflows
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

try:
    from pycaret.classification import ClassificationExperiment
    from pycaret.regression import RegressionExperiment
    from pycaret.clustering import ClusteringExperiment
    from pycaret.anomaly import AnomalyExperiment
    from pycaret.time_series import TSForecastingExperiment

    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False
    ClassificationExperiment = None
    RegressionExperiment = None
    ClusteringExperiment = None
    AnomalyExperiment = None
    TSForecastingExperiment = None

from ..experiments.tracking import ExperimentTracker


class PyCaretPipeline:
    """PyCaret ML pipeline wrapper"""

    def __init__(
        self, task_type: str, experiment_tracker: Optional[ExperimentTracker] = None
    ):
        if not PYCARET_AVAILABLE:
            raise ImportError(
                "PyCaret is not installed. Please install it with: uv add pycaret"
            )

        self.task_type = task_type.lower()
        self.experiment = None
        self.experiment_tracker = experiment_tracker
        self.is_setup = False
        self.best_model = None

        # Initialize appropriate experiment class
        if self.task_type == "classification":
            self.experiment = ClassificationExperiment()
        elif self.task_type == "regression":
            self.experiment = RegressionExperiment()
        elif self.task_type == "clustering":
            self.experiment = ClusteringExperiment()
        elif self.task_type == "anomaly":
            self.experiment = AnomalyExperiment()
        elif self.task_type == "time_series":
            self.experiment = TSForecastingExperiment()
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    def setup(
        self,
        data: pd.DataFrame,
        target: Optional[str] = None,
        test_data: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> None:
        """Setup PyCaret experiment"""
        try:
            setup_params = {"data": data, "session_id": 123, "silent": True, **kwargs}

            # Add target for supervised tasks
            if self.task_type in ["classification", "regression"] and target:
                setup_params["target"] = target

            # Add test data if provided
            if test_data is not None:
                setup_params["test_data"] = test_data

            self.experiment.setup(**setup_params)
            self.is_setup = True

            if self.experiment_tracker:
                self.experiment_tracker.log_params(
                    {
                        "task_type": self.task_type,
                        "data_shape": data.shape,
                        "target_column": target,
                        "test_data_provided": test_data is not None,
                    }
                )

        except Exception as e:
            raise RuntimeError(f"Failed to setup PyCaret experiment: {str(e)}")

    def compare_models(
        self,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        n_select: int = 1,
        **kwargs,
    ) -> pd.DataFrame:
        """Compare multiple models"""
        if not self.is_setup:
            raise RuntimeError("Must call setup() before comparing models")

        try:
            comparison_params = {"silent": True, "n_select": n_select, **kwargs}

            if include:
                comparison_params["include"] = include
            if exclude:
                comparison_params["exclude"] = exclude

            results = self.experiment.compare_models(**comparison_params)

            if self.experiment_tracker:
                self.experiment_tracker.log_artifact("model_comparison", results)

            return results

        except Exception as e:
            raise RuntimeError(f"Failed to compare models: {str(e)}")

    def create_model(self, model_name: str, **kwargs) -> Any:
        """Create and train a specific model"""
        if not self.is_setup:
            raise RuntimeError("Must call setup() before creating models")

        try:
            model = self.experiment.create_model(model_name, **kwargs)

            if self.experiment_tracker:
                self.experiment_tracker.log_params(
                    {"model_name": model_name, "model_params": kwargs}
                )

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to create model {model_name}: {str(e)}")

    def tune_hyperparameters(
        self, model: Any, n_iter: int = 10, optimize: str = "Accuracy", **kwargs
    ) -> Any:
        """Tune model hyperparameters"""
        if not self.is_setup:
            raise RuntimeError("Must call setup() before tuning")

        try:
            tuned_model = self.experiment.tune_model(
                model, n_iter=n_iter, optimize=optimize, silent=True, **kwargs
            )

            if self.experiment_tracker:
                self.experiment_tracker.log_params(
                    {"tuning_iterations": n_iter, "optimization_metric": optimize}
                )

            return tuned_model

        except Exception as e:
            raise RuntimeError(f"Failed to tune hyperparameters: {str(e)}")

    def evaluate_model(self, model: Any) -> Dict[str, Any]:
        """Evaluate model performance"""
        if not self.is_setup:
            raise RuntimeError("Must call setup() before evaluation")

        try:
            results = self.experiment.evaluate_model(model, silent=True)

            # Get model metrics
            metrics = self.experiment.pull()

            evaluation_results = {
                "model": model,
                "metrics": metrics.to_dict()
                if hasattr(metrics, "to_dict")
                else metrics,
                "plots": results,
            }

            if self.experiment_tracker:
                self.experiment_tracker.log_metrics(evaluation_results["metrics"])
                self.experiment_tracker.log_artifact("evaluation_plots", results)

            return evaluation_results

        except Exception as e:
            raise RuntimeError(f"Failed to evaluate model: {str(e)}")

    def finalize_model(self, model: Any) -> Any:
        """Finalize model (train on full dataset)"""
        if not self.is_setup:
            raise RuntimeError("Must call setup() before finalizing")

        try:
            final_model = self.experiment.finalize_model(model)
            self.best_model = final_model

            if self.experiment_tracker:
                self.experiment_tracker.log_artifact("final_model", final_model)

            return final_model

        except Exception as e:
            raise RuntimeError(f"Failed to finalize model: {str(e)}")

    def predict_model(
        self, model: Any, data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Make predictions with model"""
        if not self.is_setup:
            raise RuntimeError("Must call setup() before prediction")

        try:
            if data is not None:
                predictions = self.experiment.predict_model(model, data=data)
            else:
                predictions = self.experiment.predict_model(model)

            return predictions

        except Exception as e:
            raise RuntimeError(f"Failed to make predictions: {str(e)}")

    def interpret_model(self, model: Any, plot_type: str = "summary") -> Any:
        """Generate model interpretations"""
        if not self.is_setup:
            raise RuntimeError("Must call setup() before interpretation")

        try:
            interpretation = self.experiment.interpret_model(
                model, plot=plot_type, save=False
            )

            if self.experiment_tracker:
                self.experiment_tracker.log_artifact(
                    f"interpretation_{plot_type}", interpretation
                )

            return interpretation

        except Exception as e:
            raise RuntimeError(f"Failed to interpret model: {str(e)}")

    def get_available_models(self) -> List[str]:
        """Get list of available models for current task type"""
        if not self.is_setup:
            return []

        try:
            models = self.experiment.models()
            return list(models.index) if hasattr(models, "index") else []
        except Exception:
            # Return common models as fallback
            if self.task_type == "classification":
                return ["rf", "lr", "xgboost", "lightgbm", "dt", "nb", "svm"]
            elif self.task_type == "regression":
                return ["rf", "lr", "xgboost", "lightgbm", "dt", "ridge", "lasso"]
            elif self.task_type == "clustering":
                return ["kmeans", "ap", "meanshift", "sc", "hclust", "dbscan"]
            return []

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        try:
            if hasattr(self.experiment, "models") and callable(self.experiment.models):
                models_df = self.experiment.models()
                if model_name in models_df.index:
                    return models_df.loc[model_name].to_dict()
            return {"name": model_name, "available": True}
        except Exception:
            return {"name": model_name, "available": False}

    def save_experiment(self, path: str) -> None:
        """Save the experiment"""
        if not self.is_setup:
            raise RuntimeError("Must call setup() before saving")

        try:
            self.experiment.save_experiment(path)
        except Exception as e:
            raise RuntimeError(f"Failed to save experiment: {str(e)}")

    def load_experiment(self, path: str) -> None:
        """Load a saved experiment"""
        try:
            self.experiment.load_experiment(path)
            self.is_setup = True
        except Exception as e:
            raise RuntimeError(f"Failed to load experiment: {str(e)}")


class AutoMLWorkflow:
    """Automated ML workflow using PyCaret"""

    def __init__(self, experiment_tracker: Optional[ExperimentTracker] = None):
        if not PYCARET_AVAILABLE:
            raise ImportError(
                "PyCaret is not installed. Please install it with: uv add pycaret"
            )

        self.experiment_tracker = experiment_tracker
        self.pipeline = None
        self.results = {}

    def run_automl(
        self,
        data: pd.DataFrame,
        task_type: str,
        target: Optional[str] = None,
        test_data: Optional[pd.DataFrame] = None,
        model_selection: str = "compare_all",
        tune_hyperparameters: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run complete AutoML workflow"""

        self.pipeline = PyCaretPipeline(task_type, self.experiment_tracker)

        # Setup experiment
        self.pipeline.setup(data, target=target, test_data=test_data, **kwargs)

        # Model selection
        if model_selection == "compare_all":
            best_models = self.pipeline.compare_models(n_select=3)
        else:
            best_models = [self.pipeline.create_model(model_selection)]

        # Handle single model case
        if not isinstance(best_models, list):
            best_models = [best_models]

        # Tune hyperparameters if requested
        if tune_hyperparameters and best_models:
            tuned_models = []
            for model in best_models:
                try:
                    tuned = self.pipeline.tune_hyperparameters(model)
                    tuned_models.append(tuned)
                except Exception as e:
                    print(f"Failed to tune model: {e}")
                    tuned_models.append(model)
            best_models = tuned_models

        # Evaluate models
        evaluations = []
        for model in best_models:
            try:
                eval_result = self.pipeline.evaluate_model(model)
                evaluations.append(eval_result)
            except Exception as e:
                print(f"Failed to evaluate model: {e}")

        # Finalize best model
        final_model = None
        if best_models:
            try:
                final_model = self.pipeline.finalize_model(best_models[0])
            except Exception as e:
                print(f"Failed to finalize model: {e}")
                final_model = best_models[0]

        self.results = {
            "pipeline": self.pipeline,
            "best_models": best_models,
            "evaluations": evaluations,
            "final_model": final_model,
            "task_type": task_type,
            "target": target,
        }

        return self.results

    def get_predictions(
        self, data: Optional[pd.DataFrame] = None
    ) -> Optional[pd.DataFrame]:
        """Get predictions from the final model"""
        if not self.results.get("final_model") or not self.pipeline:
            return None

        try:
            return self.pipeline.predict_model(self.results["final_model"], data)
        except Exception as e:
            print(f"Failed to generate predictions: {e}")
            return None

    def get_model_interpretation(self, plot_type: str = "summary") -> Any:
        """Get model interpretation"""
        if not self.results.get("final_model") or not self.pipeline:
            return None

        try:
            return self.pipeline.interpret_model(self.results["final_model"], plot_type)
        except Exception as e:
            print(f"Failed to generate interpretation: {e}")
            return None
