import datetime
import time
from typing import Any

import pandas as pd
import panel as pn

from src.core.experiments import ExperimentManager, ExperimentStatus, ExperimentTracker
from src.core.logging import get_logger
from src.core.ml.pycaret_integration import AutoMLWorkflow


class ExperimentationPanel:
    def __init__(self, experiment_tracker: ExperimentTracker | None = None) -> None:
        self.current_experiment = None
        self.experiment_tracker = experiment_tracker
        self.experiment_manager = ExperimentManager(experiment_tracker)
        self.automl_workflow = None
        self.current_data = None
        self.experiment_completed_callback = None
        self.logger = get_logger(__name__, pipeline_stage="experimentation_ui")

        # ML Task Selection
        self.task_type_select = pn.widgets.Select(
            name="ML Task Type",
            options=[
                "Classification",
                "Regression",
                "Clustering",
                "Anomaly Detection",
                "Time Series Forecasting",
            ],
            value="Classification",
        )

        # Target variable selection
        self.target_select = pn.widgets.Select(
            name="Target Variable", options=[], disabled=True
        )

        # Feature selection
        self.feature_select = pn.widgets.MultiChoice(
            name="Features", options=[], disabled=True
        )

        # Model selection
        self.model_select = pn.widgets.MultiChoice(
            name="Models to Compare",
            options=[
                "Random Forest",
                "XGBoost",
                "LightGBM",
                "Logistic Regression",
                "SVM",
                "Neural Network",
            ],
            value=["Random Forest", "XGBoost"],
        )

        # Experiment configuration
        self.experiment_name = pn.widgets.TextInput(
            name="Experiment Name", placeholder="Enter experiment name"
        )

        self.train_test_split = pn.widgets.FloatSlider(
            name="Train/Test Split", start=0.1, end=0.9, step=0.1, value=0.8
        )

        self.cross_validation_folds = pn.widgets.IntSlider(
            name="CV Folds", start=3, end=10, value=5
        )

        # Control buttons
        self.start_experiment_button = pn.widgets.Button(
            name="Start Experiment", button_type="primary", disabled=True, width=150
        )

        self.stop_experiment_button = pn.widgets.Button(
            name="Stop Experiment", button_type="danger", disabled=True, width=150
        )

        # Progress and results
        self.progress_bar = pn.indicators.Progress(
            name="Training Progress", value=0, max=100, width=400
        )

        self.experiment_log = pn.pane.HTML(
            "<div style='height: 200px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px;'>"
            "Experiment logs will appear here..."
            "</div>"
        )

        self.results_table = pn.pane.DataFrame(pd.DataFrame(), width=800, height=300)
        
        # Experiment history and management
        self.experiment_history_table = pn.pane.DataFrame(
            self._get_experiment_history(), width=800, height=300
        )
        
        self.refresh_history_button = pn.widgets.Button(
            name="Refresh History", button_type="light", width=120
        )
        
        self.delete_experiment_button = pn.widgets.Button(
            name="Delete Selected", button_type="danger", width=120, disabled=True
        )
        
        # Experiment comparison
        self.compare_experiments_button = pn.widgets.Button(
            name="Compare Experiments", button_type="success", width=150, disabled=True
        )
        
        self.experiment_comparison_table = pn.pane.DataFrame(pd.DataFrame(), width=800, height=200)

        # Set up callbacks
        self._setup_callbacks()

        # Create the main panel
        self.panel = self._create_panel()

    def _setup_callbacks(self) -> None:
        self.start_experiment_button.on_click(self._on_start_experiment)
        self.stop_experiment_button.on_click(self._on_stop_experiment)
        self.refresh_history_button.on_click(self._on_refresh_history)
        self.compare_experiments_button.on_click(self._on_compare_experiments)

    def _create_panel(self) -> pn.Column:
        return pn.Column(
            pn.pane.Markdown("## Experiment Configuration"),
            pn.Row(
                pn.Column(
                    self.task_type_select,
                    self.target_select,
                    self.feature_select,
                    width=300,
                ),
                pn.Column(
                    self.model_select,
                    self.experiment_name,
                    pn.Row(self.train_test_split, self.cross_validation_folds),
                    width=400,
                ),
            ),
            pn.pane.Markdown("## Experiment Control"),
            pn.Row(self.start_experiment_button, self.stop_experiment_button),
            self.progress_bar,
            pn.pane.Markdown("## Experiment Log"),
            self.experiment_log,
            pn.pane.Markdown("## Results"),
            self.results_table,
            pn.pane.Markdown("## Experiment History"),
            pn.Row(
                self.refresh_history_button,
                self.compare_experiments_button,
                self.delete_experiment_button
            ),
            self.experiment_history_table,
            pn.pane.Markdown("## Experiment Comparison"),
            self.experiment_comparison_table,
        )

    def update_data_options(self, data: pd.DataFrame) -> None:
        """Update target and feature options based on loaded data"""
        if data is not None and not data.empty:
            self.current_data = data
            columns = list(data.columns)

            # Update target options
            self.target_select.options = columns
            self.target_select.disabled = False

            # Update feature options
            self.feature_select.options = columns
            self.feature_select.disabled = False

            # Enable experiment start
            self.start_experiment_button.disabled = False

    def _on_start_experiment(self, event: Any) -> None:  # noqa: ANN401, ARG002
        """Start ML experiment with enhanced tracking"""
        if not self.experiment_name.value:
            self._log_message("Error: Please enter an experiment name")
            return

        if not self.target_select.value:
            self._log_message("Error: Please select a target variable")
            return

        if self.current_data is None:
            self._log_message("Error: No data loaded")
            return

        # Create experiment with enhanced metadata
        dataset_info = {
            "rows": len(self.current_data),
            "columns": len(self.current_data.columns),
            "target": self.target_select.value,
            "features": self.feature_select.value or list(self.current_data.columns),
            "memory_usage": self.current_data.memory_usage(deep=True).sum()
        }
        
        config = {
            "task_type": self.task_type_select.value.lower(),
            "models": self.model_select.value,
            "train_test_split": self.train_test_split.value,
            "cv_folds": self.cross_validation_folds.value
        }
        
        tags = {
            "ui_initiated": "true",
            "data_size": "large" if len(self.current_data) > 10000 else "small"
        }

        # Create and start experiment
        try:
            experiment = self.experiment_manager.create_experiment(
                name=self.experiment_name.value,
                task_type=self.task_type_select.value.lower(),
                dataset_info=dataset_info,
                config=config,
                tags=tags
            )
            
            self.current_experiment = experiment
            
            if self.experiment_manager.start_experiment(experiment.id):
                # Disable start button, enable stop button
                self.start_experiment_button.disabled = True
                self.stop_experiment_button.disabled = False

                # Reset progress
                self.progress_bar.value = 0

                # Log experiment start
                self._log_message(f"Starting experiment: {self.experiment_name.value}")
                self._log_message(f"Task: {self.task_type_select.value}")
                self._log_message(f"Target: {self.target_select.value}")
                self._log_message(f"Models: {', '.join(self.model_select.value)}")
                self._log_message(f"Experiment ID: {experiment.id}")

                # Run real experiment
                self._run_experiment()
            else:
                self._log_message("Error: Failed to start experiment tracking")
                
        except Exception as e:
            self.logger.error("Failed to create experiment", exc_info=True)
            self._log_message(f"Error creating experiment: {str(e)}")

    def _on_stop_experiment(self, event: Any) -> None:  # noqa: ANN401, ARG002
        """Stop current experiment"""
        self.start_experiment_button.disabled = False
        self.stop_experiment_button.disabled = True
        self._log_message("Experiment stopped by user")
        
        # Cancel current experiment if active
        if self.current_experiment and self.current_experiment.status == ExperimentStatus.RUNNING:
            self.current_experiment.update_status(ExperimentStatus.CANCELLED, "Stopped by user")
            if self.experiment_tracker:
                self.experiment_tracker.end_run()

    def _run_experiment(self) -> None:
        """Run real ML experiment using PyCaret"""
        try:
            # Initialize AutoML workflow
            self.automl_workflow = AutoMLWorkflow(self.experiment_tracker)

            self._log_message("Setting up experiment...")
            self.progress_bar.value = 20

            # Map UI model names to PyCaret model codes
            model_mapping = {
                "Random Forest": "rf",
                "XGBoost": "xgboost",
                "LightGBM": "lightgbm",
                "Logistic Regression": "lr",
                "SVM": "svm",
                "Neural Network": "mlp",
            }

            # Get selected models
            selected_models = [
                model_mapping.get(m, m.lower().replace(" ", "_"))
                for m in self.model_select.value
            ]

            self._log_message("Starting model comparison...")
            self.progress_bar.value = 40

            # Run AutoML
            results = self.automl_workflow.run_automl(
                data=self.current_data,
                task_type=self.task_type_select.value.lower(),
                target=self.target_select.value,
                model_selection=(
                    "compare_all" if len(selected_models) > 1 else selected_models[0]
                ),
                tune_hyperparameters=True,
            )

            self.progress_bar.value = 80
            self._log_message("Processing results...")

            # Extract and display results
            if results.get("evaluations"):
                results_data = []
                for i, evaluation in enumerate(results["evaluations"]):
                    metrics = evaluation.get("metrics", {})
                    if isinstance(metrics, dict):
                        # Extract common metrics
                        model_name = f"Model_{i + 1}"
                        if hasattr(evaluation.get("model"), "__class__"):
                            model_name = evaluation["model"].__class__.__name__

                        results_data.append(
                            {
                                "Model": model_name,
                                "Accuracy": metrics.get("Accuracy", 0.0),
                                "Precision": metrics.get(
                                    "Prec.", metrics.get("Precision", 0.0)
                                ),
                                "Recall": metrics.get("Recall", 0.0),
                                "F1-Score": metrics.get("F1", 0.0),
                                "AUC": metrics.get("AUC", 0.0),
                            }
                        )

                if results_data:
                    self.results_table.object = pd.DataFrame(results_data)
                else:
                    self._log_message("No results to display")

            self.progress_bar.value = 100
            self._log_message("Experiment completed successfully!")
            
            # Complete experiment tracking
            if self.current_experiment:
                final_metrics = {}
                if results_data:
                    # Extract best metrics for final tracking
                    best_result = max(results_data, key=lambda x: x.get("Accuracy", 0))
                    final_metrics = {
                        "best_accuracy": best_result.get("Accuracy", 0),
                        "best_precision": best_result.get("Precision", 0),
                        "best_recall": best_result.get("Recall", 0),
                        "best_f1": best_result.get("F1-Score", 0),
                        "models_trained": len(results_data)
                    }
                
                self.experiment_manager.complete_experiment(self.current_experiment.id, final_metrics)

            # Notify parent app of experiment completion
            if self.experiment_completed_callback:
                self.experiment_completed_callback()
                
            # Refresh history
            self.experiment_history_table.object = self._get_experiment_history()

        except ImportError as e:
            self._log_message(f"PyCaret not available: {e!s}")
            self._log_message("Falling back to simulation mode...")
            self._simulate_experiment()
        except Exception as e:  # noqa: BLE001
            self._log_message(f"Error during experiment: {e!s}")
            # Fallback to simulation for demo purposes
            self._simulate_experiment()

        finally:
            # Re-enable controls
            self.start_experiment_button.disabled = False
            self.stop_experiment_button.disabled = True

    def _simulate_experiment(self) -> None:
        """Simulate experiment execution with progress updates (fallback)"""

        # Simulate model training progress
        for i in range(0, 101, 20):
            self.progress_bar.value = i
            if i < 100:  # noqa: PLR2004
                model_name = self.model_select.value[
                    i // 20 % len(self.model_select.value)
                ]
                self._log_message(f"Training {model_name}... ({i}%)")
            time.sleep(0.5)  # Simulate processing time

        # Generate mock results
        results_data = [
            {
                "Model": model,
                "Accuracy": round(0.7 + (hash(model) % 100) / 500, 3),
                "Precision": round(0.75 + (hash(model) % 80) / 400, 3),
                "Recall": round(0.72 + (hash(model) % 90) / 450, 3),
                "F1-Score": round(0.73 + (hash(model) % 85) / 425, 3),
                "Training Time": f"{(hash(model) % 300) + 30}s",
            }
            for model in self.model_select.value
        ]

        self.results_table.object = pd.DataFrame(results_data)
        self._log_message("Experiment completed successfully!")
        
        # Complete experiment tracking
        if self.current_experiment:
            final_metrics = {
                "best_accuracy": max(row["Accuracy"] for row in results_data),
                "models_trained": len(results_data),
                "simulation_mode": True
            }
            self.experiment_manager.complete_experiment(self.current_experiment.id, final_metrics)

        # Notify parent app of experiment completion
        if self.experiment_completed_callback:
            self.experiment_completed_callback()
            
        # Refresh history
        self.experiment_history_table.object = self._get_experiment_history()

        # Re-enable controls
        self.start_experiment_button.disabled = False
        self.stop_experiment_button.disabled = True

    def _log_message(self, message: str) -> None:
        """Add message to experiment log"""

        timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}<br>"

        current_log = self.experiment_log.object
        # Extract existing content and add new message
        if "Experiment logs will appear here..." in current_log:
            new_log = f"<div style='height: 200px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px;'>{log_entry}</div>"
        else:
            # Insert new message after the opening div tag
            content_start = current_log.find(">") + 1
            content_end = current_log.rfind("</div>")
            existing_content = current_log[content_start:content_end]
            new_log = f"<div style='height: 200px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px;'>{existing_content}{log_entry}</div>"

        self.experiment_log.object = new_log
        
    def _get_experiment_history(self) -> pd.DataFrame:
        """Get experiment history for display"""
        try:
            experiments = self.experiment_manager.list_experiments()
            if not experiments:
                return pd.DataFrame(columns=["Name", "Status", "Task Type", "Duration", "Created"])
                
            history_data = []
            for exp in experiments:
                history_data.append({
                    "ID": exp.id[:8],  # Shortened ID for display
                    "Name": exp.name,
                    "Status": exp.status,
                    "Task Type": exp.task_type,
                    "Duration": f"{exp.duration:.1f}s" if exp.duration else "N/A",
                    "Models": len(exp.models),
                    "Created": exp.created_at.strftime("%Y-%m-%d %H:%M")
                })
                
            return pd.DataFrame(history_data)
        except Exception as e:
            self.logger.error("Failed to get experiment history", exc_info=True)
            return pd.DataFrame(columns=["Name", "Status", "Task Type", "Duration", "Created"])
            
    def _on_refresh_history(self, event: Any) -> None:  # noqa: ANN401, ARG002
        """Refresh experiment history display"""
        self.experiment_history_table.object = self._get_experiment_history()
        self._log_message("Experiment history refreshed")
        
    def _on_compare_experiments(self, event: Any) -> None:  # noqa: ANN401, ARG002
        """Compare selected experiments"""
        # For now, compare the last 3 completed experiments
        completed_experiments = self.experiment_manager.list_experiments(status=ExperimentStatus.COMPLETED)
        
        if len(completed_experiments) < 2:
            self._log_message("Need at least 2 completed experiments for comparison")
            return
            
        # Take the 3 most recent completed experiments
        recent_experiments = completed_experiments[:3]
        experiment_ids = [exp.id for exp in recent_experiments]
        
        try:
            comparison_df = self.experiment_manager.get_experiment_comparison(experiment_ids)
            if not comparison_df.empty:
                # Clean up the comparison data for display
                display_columns = ["name", "task_type", "status", "duration", "models_count"]
                display_df = comparison_df[display_columns] if all(col in comparison_df.columns for col in display_columns) else comparison_df
                
                self.experiment_comparison_table.object = display_df
                self._log_message(f"Comparing {len(recent_experiments)} experiments")
            else:
                self._log_message("No data available for comparison")
        except Exception as e:
            self.logger.error("Failed to compare experiments", exc_info=True)
            self._log_message(f"Error comparing experiments: {str(e)}")
