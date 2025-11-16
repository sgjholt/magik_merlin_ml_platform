"""
AutoML pipeline for automated model selection and hyperparameter tuning.

Provides simple but powerful AutoML capabilities including:
- Automated model comparison
- Hyperparameter optimization with Optuna
- Cross-validation and model evaluation
- MLflow integration for tracking
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split

from ..experiments.tracking import ExperimentTracker
from ..logging import get_logger
from .base import BaseMLModel, model_registry

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class AutoMLPipeline:
    """
    Automated ML pipeline for model selection and optimization.

    Provides automated workflows for training, comparing, and optimizing
    machine learning models with minimal configuration.
    """

    def __init__(
        self,
        task_type: Literal["classification", "regression"],
        experiment_tracker: ExperimentTracker | None = None,
        random_state: int = 42,
    ) -> None:
        """
        Initialize AutoML pipeline.

        Args:
            task_type: Type of ML task (classification or regression)
            experiment_tracker: Optional MLflow experiment tracker
            random_state: Random seed for reproducibility
        """
        self.task_type = task_type
        self.experiment_tracker = experiment_tracker
        self.random_state = random_state
        self.logger = get_logger(
            __name__,
            pipeline_stage="automl",
            task_type=task_type,
        )

        self.models: dict[str, BaseMLModel] = {}
        self.results: dict[str, dict[str, Any]] = {}
        self.best_model: BaseMLModel | None = None
        self.best_model_name: str | None = None

    def compare_models(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        models: list[str] | None = None,
        cv: int = 5,
        test_size: float = 0.2,
    ) -> pd.DataFrame:
        """
        Compare multiple models using cross-validation.

        Args:
            X: Features
            y: Targets
            models: List of model names to compare (None = all available)
            cv: Number of cross-validation folds
            test_size: Test set size for final evaluation

        Returns:
            DataFrame with model comparison results
        """
        self.logger.info(
            "Starting model comparison",
            extra={
                "n_samples": len(X),
                "n_features": X.shape[1] if hasattr(X, "shape") else len(X[0]),
                "cv_folds": cv,
            },
        )

        # Get list of models to compare
        if models is None:
            models = model_registry.list_models(category=self.task_type)

        if not models:
            msg = f"No models available for task type: {self.task_type}"
            raise ValueError(msg)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
        )

        # Compare models
        comparison_results = []

        for model_name in models:
            try:
                self.logger.info(f"Training and evaluating: {model_name}")

                # Get model class and instantiate
                model_class = model_registry.get_model(model_name)
                model = model_class(random_state=self.random_state)

                # Cross-validation
                cv_scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring="accuracy"
                    if self.task_type == "classification"
                    else "r2",
                )

                # Train on full training set
                model.fit(X_train, y_train)

                # Evaluate on test set
                test_score = model.score(X_test, y_test)

                # Store results
                result = {
                    "model": model_name,
                    "cv_mean": cv_scores.mean(),
                    "cv_std": cv_scores.std(),
                    "test_score": test_score,
                    "cv_scores": cv_scores.tolist(),
                }

                comparison_results.append(result)
                self.models[model_name] = model
                self.results[model_name] = result

                # Log to MLflow if available
                if self.experiment_tracker:
                    self.experiment_tracker.log_metrics(
                        {
                            f"{model_name}_cv_mean": cv_scores.mean(),
                            f"{model_name}_cv_std": cv_scores.std(),
                            f"{model_name}_test_score": test_score,
                        }
                    )

                self.logger.info(
                    f"{model_name} completed",
                    extra={
                        "cv_mean": cv_scores.mean(),
                        "test_score": test_score,
                    },
                )

            except Exception as e:
                self.logger.exception(
                    f"Failed to train {model_name}",
                    extra={"error_type": type(e).__name__},
                )

        # Convert to DataFrame and sort by test score
        results_df = pd.DataFrame(comparison_results)
        if not results_df.empty:
            results_df = results_df.sort_values("test_score", ascending=False)

            # Store best model
            self.best_model_name = results_df.iloc[0]["model"]
            self.best_model = self.models[self.best_model_name]

            self.logger.info(
                "Model comparison completed",
                extra={
                    "best_model": self.best_model_name,
                    "best_score": results_df.iloc[0]["test_score"],
                },
            )

        return results_df

    def optimize_hyperparameters(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        model_name: str,
        n_trials: int = 50,
        cv: int = 5,
    ) -> dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.

        Args:
            X: Features
            y: Targets
            model_name: Name of model to optimize
            n_trials: Number of optimization trials
            cv: Number of cross-validation folds

        Returns:
            Dictionary with best parameters and score
        """
        if not OPTUNA_AVAILABLE:
            msg = "Optuna is not installed. Install with: uv add optuna"
            raise ImportError(msg)

        self.logger.info(
            f"Starting hyperparameter optimization for {model_name}",
            extra={"n_trials": n_trials},
        )

        model_class = model_registry.get_model(model_name)

        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function."""
            # Define hyperparameter search space based on model type
            params = self._get_param_space(model_name, trial)

            # Create and evaluate model
            model = model_class(**params)
            scores = cross_val_score(
                model,
                X,
                y,
                cv=cv,
                scoring="accuracy" if self.task_type == "classification" else "r2",
            )

            return scores.mean()

        # Run optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_score = study.best_value

        self.logger.info(
            "Hyperparameter optimization completed",
            extra={
                "best_score": best_score,
                "best_params": best_params,
            },
        )

        # Train final model with best parameters
        final_model = model_class(**best_params)
        final_model.fit(X, y)

        self.models[f"{model_name}_optimized"] = final_model
        self.results[f"{model_name}_optimized"] = {
            "model": f"{model_name}_optimized",
            "best_params": best_params,
            "best_score": best_score,
            "study": study,
        }

        # Log to MLflow if available
        if self.experiment_tracker:
            self.experiment_tracker.log_params(best_params)
            self.experiment_tracker.log_metrics(
                {f"{model_name}_optimized_score": best_score}
            )

        return {
            "model": final_model,
            "best_params": best_params,
            "best_score": best_score,
            "study": study,
        }

    def _get_param_space(
        self,
        model_name: str,
        trial: optuna.Trial,
    ) -> dict[str, Any]:
        """
        Define hyperparameter search space for each model type.

        Args:
            model_name: Name of the model
            trial: Optuna trial object

        Returns:
            Dictionary of hyperparameters to try
        """
        if "xgboost" in model_name:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": self.random_state,
            }
        if "lightgbm" in model_name:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": self.random_state,
                "verbosity": -1,
            }
        if "catboost" in model_name:
            return {
                "iterations": trial.suggest_int("iterations", 50, 300),
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
                "random_state": self.random_state,
                "verbose": False,
            }
        # Default parameter space
        return {"random_state": self.random_state}

    def get_best_model(self) -> BaseMLModel:
        """
        Get the best performing model.

        Returns:
            Best fitted model

        Raises:
            RuntimeError: If no models have been trained
        """
        if self.best_model is None:
            msg = "No models have been trained yet. Run compare_models() first."
            raise RuntimeError(msg)

        return self.best_model

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        use_best: bool = True,
    ) -> np.ndarray:
        """
        Make predictions using the best model.

        Args:
            X: Features to predict on
            use_best: Whether to use the best model (True) or last trained

        Returns:
            Predictions

        Raises:
            RuntimeError: If no models have been trained
        """
        if use_best:
            model = self.get_best_model()
        else:
            if not self.models:
                msg = "No models have been trained yet"
                raise RuntimeError(msg)
            model = list(self.models.values())[-1]

        return model.predict(X)

    def get_results_summary(self) -> dict[str, Any]:
        """
        Get summary of all results.

        Returns:
            Dictionary with results summary
        """
        return {
            "task_type": self.task_type,
            "n_models_trained": len(self.models),
            "best_model": self.best_model_name,
            "results": self.results,
        }
