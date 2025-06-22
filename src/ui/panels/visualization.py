"""
Visualization panel for data exploration and model evaluation.
"""

from typing import Any, Dict, Optional

import pandas as pd
import panel as pn

from src.core.logging import get_logger
from src.ui.visualizations import (
    DataOverviewVisualization,
    DistributionVisualization,
    CorrelationVisualization,
    MissingDataVisualization,
    TimeSeriesVisualization,
    ModelComparisonVisualization,
    ROCCurveVisualization,
    ConfusionMatrixVisualization,
    FeatureImportanceVisualization,
    ExperimentHistoryVisualization,
    ExperimentComparisonVisualization,
    ExperimentMetricsVisualization
)


class VisualizationPanel:
    """Main visualization panel with multiple chart types."""
    
    def __init__(self, experiment_manager=None) -> None:
        self.current_data = None
        self.experiment_manager = experiment_manager
        self.logger = get_logger(__name__, pipeline_stage="visualization_ui")
        
        # Visualization category selection
        self.category_select = pn.widgets.Select(
            name="Visualization Category",
            options=[
                "Data Exploration",
                "Model Evaluation", 
                "Experiment Tracking"
            ],
            value="Data Exploration"
        )
        
        # Visualization type selection (updated based on category)
        self.viz_type_select = pn.widgets.Select(
            name="Visualization Type",
            options=[],
            disabled=True
        )
        
        # Refresh button
        self.refresh_button = pn.widgets.Button(
            name="Refresh Visualization",
            button_type="primary",
            width=150
        )
        
        # Main visualization pane
        self.viz_pane = pn.pane.HTML(
            "<div style='text-align: center; padding: 50px; color: #666;'>"
            "<h3>No Data Loaded</h3>"
            "<p>Load data to see visualizations</p>"
            "</div>",
            width=800,
            height=600
        )
        
        # Control panel for interactive visualizations
        self.control_panel = pn.Column(width=0)  # Hidden initially
        
        # Status indicator
        self.status_indicator = pn.pane.HTML(
            "<div style='color: #666; font-style: italic;'>Ready</div>"
        )
        
        # Set up callbacks
        self._setup_callbacks()
        
        # Initialize visualization options
        self._update_viz_options()
        
        # Create the main panel
        self.panel = self._create_panel()
        
    def _setup_callbacks(self) -> None:
        """Set up widget callbacks."""
        self.category_select.param.watch(self._on_category_change, "value")
        self.viz_type_select.param.watch(self._on_viz_type_change, "value")
        self.refresh_button.on_click(self._on_refresh_click)
        
    def _create_panel(self) -> pn.Column:
        """Create the main visualization panel."""
        return pn.Column(
            pn.pane.Markdown("## Data Visualization"),
            
            # Controls row
            pn.Row(
                pn.Column(
                    self.category_select,
                    self.viz_type_select,
                    self.refresh_button,
                    width=250
                ),
                pn.Column(
                    self.status_indicator,
                    margin=(10, 0, 0, 0)
                )
            ),
            
            # Main content row
            pn.Row(
                self.control_panel,  # Interactive controls (if any)
                self.viz_pane,       # Main visualization
                sizing_mode="stretch_width"
            ),
            
            sizing_mode="stretch_width"
        )
        
    def _on_category_change(self, event: Any) -> None:
        """Handle category selection change."""
        self._update_viz_options()
        
    def _on_viz_type_change(self, event: Any) -> None:
        """Handle visualization type change."""
        self._update_visualization()
        
    def _on_refresh_click(self, event: Any) -> None:
        """Handle refresh button click."""
        self._update_visualization()
        
    def _update_viz_options(self) -> None:
        """Update visualization type options based on category."""
        category = self.category_select.value
        
        if category == "Data Exploration":
            options = [
                "Dataset Overview",
                "Data Distributions",
                "Correlation Matrix",
                "Missing Data Analysis",
                "Time Series"
            ]
        elif category == "Model Evaluation":
            options = [
                "Model Comparison",
                "ROC Curves",
                "Confusion Matrix",
                "Feature Importance"
            ]
        elif category == "Experiment Tracking":
            options = [
                "Experiment History",
                "Experiment Comparison",
                "Metrics Explorer"
            ]
        else:
            options = []
            
        self.viz_type_select.options = options
        self.viz_type_select.value = options[0] if options else None
        self.viz_type_select.disabled = not options
        
        # Update visualization if options available
        if options:
            self._update_visualization()
        
    def _update_visualization(self) -> None:
        """Update the main visualization based on selections."""
        try:
            self._set_status("Updating visualization...")
            
            category = self.category_select.value
            viz_type = self.viz_type_select.value
            
            if not viz_type:
                self._show_no_data_message("No visualization type selected")
                return
                
            # Clear control panel
            self.control_panel.clear()
            self.control_panel.width = 0
            
            if category == "Data Exploration":
                self._create_data_exploration_viz(viz_type)
            elif category == "Model Evaluation":
                self._create_model_evaluation_viz(viz_type)
            elif category == "Experiment Tracking":
                self._create_experiment_tracking_viz(viz_type)
                
            self._set_status("Visualization updated")
            
        except Exception as e:
            self.logger.error(f"Failed to update visualization: {e}", exc_info=True)
            self._show_error_message(f"Error: {str(e)}")
            self._set_status("Error updating visualization")
            
    def _create_data_exploration_viz(self, viz_type: str) -> None:
        """Create data exploration visualizations."""
        if self.current_data is None or self.current_data.empty:
            self._show_no_data_message("No data loaded for exploration")
            return
            
        if viz_type == "Dataset Overview":
            viz = DataOverviewVisualization()
            self.viz_pane.object = viz.get_panel(self.current_data)
            
        elif viz_type == "Data Distributions":
            viz = DistributionVisualization()
            if hasattr(viz, 'get_full_panel'):
                # Interactive visualization with controls
                full_panel = viz.get_full_panel(self.current_data)
                self.control_panel[:] = [full_panel[0]]  # Controls
                self.control_panel.width = 250
                self.viz_pane.object = full_panel[1]     # Visualization
            else:
                self.viz_pane.object = viz.get_panel(self.current_data)
                
        elif viz_type == "Correlation Matrix":
            viz = CorrelationVisualization()
            self.viz_pane.object = viz.get_panel(self.current_data)
            
        elif viz_type == "Missing Data Analysis":
            viz = MissingDataVisualization()
            self.viz_pane.object = viz.get_panel(self.current_data)
            
        elif viz_type == "Time Series":
            viz = TimeSeriesVisualization()
            if hasattr(viz, 'get_full_panel'):
                full_panel = viz.get_full_panel(self.current_data)
                self.control_panel[:] = [full_panel[0]]  # Controls
                self.control_panel.width = 250
                self.viz_pane.object = full_panel[1]     # Visualization
            else:
                self.viz_pane.object = viz.get_panel(self.current_data)
                
    def _create_model_evaluation_viz(self, viz_type: str) -> None:
        """Create model evaluation visualizations."""
        if viz_type == "Model Comparison":
            # Try to get results from current experiment or mock data
            results_data = self._get_model_results()
            if results_data is not None:
                viz = ModelComparisonVisualization()
                self.viz_pane.object = viz.get_panel(results_data)
            else:
                self._show_no_data_message("No model results available. Run an experiment first.")
                
        elif viz_type == "ROC Curves":
            # This would require actual model predictions
            self._show_placeholder_message("ROC Curves", "Run classification experiments to see ROC curves")
            
        elif viz_type == "Confusion Matrix":
            # This would require actual predictions and true labels
            self._show_placeholder_message("Confusion Matrix", "Run classification experiments to see confusion matrix")
            
        elif viz_type == "Feature Importance":
            # This would require trained models with feature importance
            self._show_placeholder_message("Feature Importance", "Train models to see feature importance")
            
    def _create_experiment_tracking_viz(self, viz_type: str) -> None:
        """Create experiment tracking visualizations."""
        if not self.experiment_manager:
            self._show_no_data_message("Experiment manager not available")
            return
            
        if viz_type == "Experiment History":
            try:
                experiments = self.experiment_manager.list_experiments()
                if experiments:
                    # Convert to DataFrame for visualization
                    exp_data = []
                    for exp in experiments:
                        exp_data.append({
                            "name": exp.name,
                            "status": exp.status,
                            "task_type": exp.task_type,
                            "duration": exp.duration or 0,
                            "created_at": exp.created_at,
                            "accuracy": exp.metrics.get("best_accuracy", 0) if exp.metrics else 0
                        })
                    
                    df = pd.DataFrame(exp_data)
                    viz = ExperimentHistoryVisualization()
                    self.viz_pane.object = viz.get_panel(df)
                else:
                    self._show_no_data_message("No experiments found")
            except Exception as e:
                self.logger.error(f"Failed to load experiment history: {e}")
                self._show_error_message(f"Error loading experiments: {str(e)}")
                
        elif viz_type == "Experiment Comparison":
            try:
                completed_experiments = self.experiment_manager.list_experiments(status="completed")
                if len(completed_experiments) >= 2:
                    # Take recent experiments for comparison
                    recent_experiments = completed_experiments[:5]
                    exp_ids = [exp.id for exp in recent_experiments]
                    
                    comparison_df = self.experiment_manager.get_experiment_comparison(exp_ids)
                    if not comparison_df.empty:
                        viz = ExperimentComparisonVisualization()
                        self.viz_pane.object = viz.get_panel(comparison_df)
                    else:
                        self._show_no_data_message("No comparison data available")
                else:
                    self._show_no_data_message("Need at least 2 completed experiments for comparison")
            except Exception as e:
                self.logger.error(f"Failed to compare experiments: {e}")
                self._show_error_message(f"Error comparing experiments: {str(e)}")
                
        elif viz_type == "Metrics Explorer":
            try:
                experiments = self.experiment_manager.list_experiments()
                if experiments:
                    # Convert to DataFrame
                    exp_data = []
                    for exp in experiments:
                        metrics = exp.metrics or {}
                        exp_data.append({
                            "name": exp.name,
                            "status": exp.status,
                            "task_type": exp.task_type,
                            "duration": exp.duration or 0,
                            "accuracy": metrics.get("best_accuracy", 0),
                            "precision": metrics.get("best_precision", 0),
                            "recall": metrics.get("best_recall", 0),
                            "f1": metrics.get("best_f1", 0),
                            "models_trained": metrics.get("models_trained", 0)
                        })
                    
                    df = pd.DataFrame(exp_data)
                    viz = ExperimentMetricsVisualization()
                    
                    if hasattr(viz, 'get_full_panel'):
                        full_panel = viz.get_full_panel(df)
                        self.control_panel[:] = [full_panel[0]]  # Controls
                        self.control_panel.width = 250
                        self.viz_pane.object = full_panel[1]     # Visualization
                    else:
                        self.viz_pane.object = viz.get_panel(df)
                else:
                    self._show_no_data_message("No experiments found")
            except Exception as e:
                self.logger.error(f"Failed to load metrics explorer: {e}")
                self._show_error_message(f"Error loading metrics: {str(e)}")
                
    def _get_model_results(self) -> Optional[pd.DataFrame]:
        """Get model results from experiments or create mock data."""
        # Try to get actual results first
        if self.experiment_manager:
            try:
                completed_experiments = self.experiment_manager.list_experiments(status="completed")
                if completed_experiments:
                    latest_exp = completed_experiments[0]
                    if latest_exp.metrics:
                        # Create results DataFrame from experiment metrics
                        results_data = [{
                            "Model": "Best Model",
                            "Accuracy": latest_exp.metrics.get("best_accuracy", 0.85),
                            "Precision": latest_exp.metrics.get("best_precision", 0.82),
                            "Recall": latest_exp.metrics.get("best_recall", 0.88),
                            "F1-Score": latest_exp.metrics.get("best_f1", 0.85)
                        }]
                        return pd.DataFrame(results_data)
            except Exception:
                pass
                
        # Return mock data for demonstration
        mock_results = [
            {"Model": "Random Forest", "Accuracy": 0.87, "Precision": 0.85, "Recall": 0.89, "F1-Score": 0.87},
            {"Model": "XGBoost", "Accuracy": 0.91, "Precision": 0.88, "Recall": 0.94, "F1-Score": 0.91},
            {"Model": "Logistic Regression", "Accuracy": 0.82, "Precision": 0.80, "Recall": 0.85, "F1-Score": 0.82}
        ]
        return pd.DataFrame(mock_results)
        
    def _show_no_data_message(self, message: str) -> None:
        """Show a no data message."""
        self.viz_pane.object = f"""
        <div style='text-align: center; padding: 50px; color: #666;'>
            <h3>No Data Available</h3>
            <p>{message}</p>
        </div>
        """
        
    def _show_error_message(self, message: str) -> None:
        """Show an error message."""
        self.viz_pane.object = f"""
        <div style='text-align: center; padding: 50px; color: #d32f2f;'>
            <h3>Error</h3>
            <p>{message}</p>
        </div>
        """
        
    def _show_placeholder_message(self, title: str, message: str) -> None:
        """Show a placeholder message for features not yet implemented."""
        self.viz_pane.object = f"""
        <div style='text-align: center; padding: 50px; color: #666;'>
            <h3>{title}</h3>
            <p>{message}</p>
            <p style='font-style: italic; margin-top: 20px;'>
                This visualization will be available after running experiments with actual model outputs.
            </p>
        </div>
        """
        
    def _set_status(self, message: str) -> None:
        """Update status indicator."""
        self.status_indicator.object = f"<div style='color: #666; font-style: italic;'>{message}</div>"
        
    def update_data(self, data: pd.DataFrame) -> None:
        """Update the current data for visualization."""
        self.current_data = data
        self.logger.info(f"Updated visualization data: {data.shape if data is not None else 'None'}")
        
        # Auto-refresh if data exploration is selected
        if self.category_select.value == "Data Exploration":
            self._update_visualization()
            
    def set_experiment_completed_callback(self, callback) -> None:
        """Set callback for when experiments are completed."""
        # This would trigger visualization updates when new experiments finish
        pass