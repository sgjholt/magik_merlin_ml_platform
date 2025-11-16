from typing import Any

import numpy as np
import pandas as pd
import panel as pn
import plotly.express as px
import plotly.graph_objects as go


class ModelEvaluationPanel:
    def __init__(self) -> None:
        # Model selection for comparison
        self.model_select = pn.widgets.MultiChoice(
            name="Select Models to Compare", options=[], disabled=True
        )

        # Metric selection
        self.metric_select = pn.widgets.Select(
            name="Primary Metric",
            options=["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"],
            value="Accuracy",
        )

        # Visualization type
        self.viz_type_select = pn.widgets.Select(
            name="Visualization Type",
            options=[
                "Performance Comparison",
                "Confusion Matrix",
                "ROC Curve",
                "Feature Importance",
                "Learning Curve",
            ],
            value="Performance Comparison",
        )

        # Control buttons
        self.refresh_button = pn.widgets.Button(
            name="Refresh Models", button_type="primary", width=150
        )

        self.compare_button = pn.widgets.Button(
            name="Compare Models", button_type="success", disabled=True, width=150
        )

        # Results display
        self.comparison_plot = pn.pane.Plotly(width=800, height=500)

        self.metrics_table = pn.pane.DataFrame(pd.DataFrame(), width=800, height=300)

        self.model_details = pn.pane.JSON({}, theme="light", width=400, height=300)

        # Feature importance plot
        self.feature_importance_plot = pn.pane.Plotly(width=600, height=400)

        # Set up callbacks
        self._setup_callbacks()

        # Create the main panel
        self.panel = self._create_panel()

        # Initialize with mock data
        self._initialize_mock_data()

    def _setup_callbacks(self) -> None:
        self.refresh_button.on_click(self._on_refresh_models)
        self.compare_button.on_click(self._on_compare_models)
        self.viz_type_select.param.watch(self._on_viz_type_change, "value")
        self.model_select.param.watch(self._on_model_selection_change, "value")

    def _create_panel(self) -> pn.Column:
        return pn.Column(
            pn.pane.Markdown("## Model Evaluation & Comparison"),
            pn.Row(
                pn.Column(
                    self.model_select,
                    self.metric_select,
                    self.viz_type_select,
                    pn.Row(self.refresh_button, self.compare_button),
                    width=400,
                ),
                pn.Column(
                    pn.pane.Markdown("### Model Details"), self.model_details, width=400
                ),
            ),
            pn.pane.Markdown("## Performance Visualization"),
            self.comparison_plot,
            pn.Row(
                pn.Column(
                    pn.pane.Markdown("### Detailed Metrics"),
                    self.metrics_table,
                    width=500,
                ),
                pn.Column(
                    pn.pane.Markdown("### Feature Importance"),
                    self.feature_importance_plot,
                    width=500,
                ),
            ),
        )

    def _initialize_mock_data(self) -> None:
        """Initialize with mock model data"""
        models = ["Random Forest", "XGBoost", "LightGBM", "Logistic Regression", "SVM"]
        self.model_select.options = models
        self.model_select.disabled = False

        # Mock metrics data
        self.mock_metrics = pd.DataFrame(
            {
                "Model": models,
                "Accuracy": [0.891, 0.887, 0.883, 0.875, 0.872],
                "Precision": [0.889, 0.885, 0.881, 0.873, 0.870],
                "Recall": [0.892, 0.888, 0.884, 0.876, 0.873],
                "F1-Score": [0.890, 0.886, 0.882, 0.874, 0.871],
                "AUC-ROC": [0.945, 0.942, 0.938, 0.931, 0.928],
                "Training Time (s)": [45, 62, 38, 12, 28],
            }
        )

        self.metrics_table.object = self.mock_metrics

    def _on_refresh_models(self, event: Any) -> None:
        """Refresh available models from MLflow"""
        # In real implementation, this would query MLflow registry
        self._initialize_mock_data()

    def _on_model_selection_change(self, event: Any) -> None:
        """Handle model selection change"""
        if event.new:
            self.compare_button.disabled = False

            # Update model details for first selected model
            model_name = event.new[0] if event.new else None
            if model_name:
                self._update_model_details(model_name)

    def _on_compare_models(self, event: Any) -> None:
        """Compare selected models"""
        if not self.model_select.value:
            return

        selected_models = self.model_select.value
        selected_data = self.mock_metrics[
            self.mock_metrics["Model"].isin(selected_models)
        ]

        # Update metrics table
        self.metrics_table.object = selected_data

        # Create visualization based on selected type
        self._create_visualization(selected_data)

    def _on_viz_type_change(self, event: Any) -> None:
        """Handle visualization type change"""
        if self.model_select.value:
            selected_data = self.mock_metrics[
                self.mock_metrics["Model"].isin(self.model_select.value)
            ]
            self._create_visualization(selected_data)

    def _create_visualization(self, data: pd.DataFrame) -> None:
        """Create visualization based on selected type"""
        viz_type = self.viz_type_select.value

        if viz_type == "Performance Comparison":
            self._create_performance_comparison(data)
        elif viz_type == "Confusion Matrix":
            self._create_confusion_matrix()
        elif viz_type == "ROC Curve":
            self._create_roc_curve()
        elif viz_type == "Learning Curve":
            self._create_learning_curve()

        # Always update feature importance
        self._create_feature_importance()

    def _create_performance_comparison(self, data: pd.DataFrame) -> None:
        """Create performance comparison bar chart"""
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]

        fig = go.Figure()

        for metric in metrics:
            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=data["Model"],
                    y=data[metric],
                    text=data[metric].round(3),
                    textposition="auto",
                )
            )

        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Score",
            barmode="group",
            height=500,
        )

        self.comparison_plot.object = fig

    def _create_confusion_matrix(self) -> None:
        """Create mock confusion matrix heatmap"""
        # Mock confusion matrix data
        confusion_matrix = np.array([[85, 12], [8, 95]])

        fig = px.imshow(
            confusion_matrix,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix - Best Model",
            labels=dict(x="Predicted", y="Actual", color="Count"),
        )

        fig.update_xaxes(tickvals=[0, 1], ticktext=["Class 0", "Class 1"])
        fig.update_yaxes(tickvals=[0, 1], ticktext=["Class 0", "Class 1"])

        self.comparison_plot.object = fig

    def _create_roc_curve(self) -> None:
        """Create mock ROC curve"""
        # Generate mock ROC curve data
        fpr = np.linspace(0, 1, 100)
        tpr_rf = 1 - np.exp(-5 * fpr) * (1 - fpr)
        tpr_xgb = 1 - np.exp(-4.5 * fpr) * (1 - fpr)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr_rf,
                mode="lines",
                name="Random Forest (AUC = 0.945)",
                line=dict(color="blue", width=2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr_xgb,
                mode="lines",
                name="XGBoost (AUC = 0.942)",
                line=dict(color="red", width=2),
            )
        )

        # Add diagonal line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Classifier",
                line=dict(color="gray", width=1, dash="dash"),
            )
        )

        fig.update_layout(
            title="ROC Curve Comparison",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=500,
        )

        self.comparison_plot.object = fig

    def _create_learning_curve(self) -> None:
        """Create mock learning curve"""
        training_sizes = np.array([50, 100, 200, 500, 1000, 2000])
        train_scores = 1 - np.exp(-training_sizes / 500) * 0.3
        val_scores = (
            train_scores - 0.05 - np.random.normal(0, 0.01, len(training_sizes))
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=training_sizes,
                y=train_scores,
                mode="lines+markers",
                name="Training Score",
                line=dict(color="blue"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=training_sizes,
                y=val_scores,
                mode="lines+markers",
                name="Validation Score",
                line=dict(color="red"),
            )
        )

        fig.update_layout(
            title="Learning Curve",
            xaxis_title="Training Set Size",
            yaxis_title="Accuracy Score",
            height=500,
        )

        self.comparison_plot.object = fig

    def _create_feature_importance(self) -> None:
        """Create mock feature importance plot"""
        features = ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]
        importance = [0.25, 0.22, 0.18, 0.15, 0.12]

        fig = go.Figure(
            go.Bar(x=importance, y=features, orientation="h", marker_color="lightblue")
        )

        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=400,
        )

        self.feature_importance_plot.object = fig

    def _update_model_details(self, model_name: str) -> None:
        """Update model details panel"""
        # Mock model details
        details = {
            "Model Name": model_name,
            "Algorithm": model_name,
            "Training Date": "2024-01-15",
            "Training Duration": "45 seconds",
            "Dataset Size": "10,000 samples",
            "Features": 20,
            "Hyperparameters": {
                "n_estimators": 100,
                "max_depth": 10,
                "learning_rate": 0.1,
            },
            "Cross Validation": "5-fold",
            "Best Score": 0.891,
        }

        self.model_details.object = details
