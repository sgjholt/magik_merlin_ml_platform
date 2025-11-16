"""
Model evaluation and performance visualizations.
"""

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .base import BaseVisualization, PlotTheme, create_empty_figure


class ModelComparisonVisualization(BaseVisualization):
    """Visualization for comparing multiple models."""

    def create_figure(self, data: pd.DataFrame, **kwargs: Any) -> go.Figure:
        """Create model comparison chart."""
        if data.empty:
            return create_empty_figure("Model Comparison", "No model results available")

        # Expected columns: Model, Accuracy, Precision, Recall, F1-Score, etc.
        required_cols = ["Model"]
        if not all(col in data.columns for col in required_cols):
            return create_empty_figure(
                "Model Comparison", "Required columns missing: Model"
            )

        # Get metric columns (exclude non-numeric)
        metric_cols = []
        for col in data.columns:
            if col != "Model" and pd.api.types.is_numeric_dtype(data[col]):
                metric_cols.append(col)

        if not metric_cols:
            return create_empty_figure("Model Comparison", "No numeric metrics found")

        # Create subplots for different views
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Model Performance Comparison",
                "Metric Rankings",
                "Performance Radar Chart",
                "Model Scores Table",
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "scatterpolar"}, {"type": "table"}],
            ],
        )

        # 1. Bar chart comparing all metrics
        for i, metric in enumerate(
            metric_cols[:5]
        ):  # Limit to 5 metrics for readability
            fig.add_trace(
                go.Bar(
                    x=data["Model"],
                    y=data[metric],
                    name=metric,
                    marker_color=PlotTheme.PRIMARY_COLORS[
                        i % len(PlotTheme.PRIMARY_COLORS)
                    ],
                ),
                row=1,
                col=1,
            )

        # 2. Ranking chart (1st metric for simplicity)
        if len(metric_cols) > 0:
            primary_metric = metric_cols[0]
            ranked_data = data.sort_values(primary_metric, ascending=False)

            fig.add_trace(
                go.Bar(
                    x=ranked_data["Model"],
                    y=ranked_data[primary_metric],
                    name=f"Ranked by {primary_metric}",
                    marker_color="lightblue",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

        # 3. Radar chart for top 3 models
        if len(data) >= 1 and len(metric_cols) >= 3:
            # Normalize metrics to 0-1 scale for radar chart
            normalized_data = data.copy()
            for col in metric_cols:
                if normalized_data[col].max() > 0:
                    normalized_data[col] = (
                        normalized_data[col] / normalized_data[col].max()
                    )

            # Take top 3 models by first metric
            top_models = normalized_data.nlargest(3, metric_cols[0])

            for i, (_, model_row) in enumerate(top_models.iterrows()):
                fig.add_trace(
                    go.Scatterpolar(
                        r=[model_row[col] for col in metric_cols],
                        theta=metric_cols,
                        fill="toself",
                        name=model_row["Model"],
                        line_color=PlotTheme.PRIMARY_COLORS[
                            i % len(PlotTheme.PRIMARY_COLORS)
                        ],
                    ),
                    row=2,
                    col=1,
                )

        # 4. Detailed scores table
        table_data = data.round(4) if len(data) > 0 else pd.DataFrame()
        fig.add_trace(
            go.Table(
                header={"values": list(table_data.columns)},
                cells={
                    "values": [table_data[col].values for col in table_data.columns]
                },
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title="Model Performance Analysis", height=800, showlegend=True
        )

        return PlotTheme.apply_theme(fig)


class ROCCurveVisualization(BaseVisualization):
    """ROC curve visualization for binary classification."""

    def create_figure(self, data: dict[str, Any], **kwargs: Any) -> go.Figure:
        """Create ROC curve plot."""
        # Expected data format: {"model_name": {"fpr": [...], "tpr": [...], "auc": 0.85}}
        if not data:
            return create_empty_figure("ROC Curves", "No ROC data available")

        fig = go.Figure()

        # Add diagonal line (random classifier)
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line={"dash": "dash", "color": "gray"},
                name="Random Classifier (AUC = 0.5)",
                showlegend=True,
            )
        )

        # Add ROC curves for each model
        for i, (model_name, roc_data) in enumerate(data.items()):
            if "fpr" not in roc_data or "tpr" not in roc_data:
                continue

            auc_score = roc_data.get("auc", 0.0)

            fig.add_trace(
                go.Scatter(
                    x=roc_data["fpr"],
                    y=roc_data["tpr"],
                    mode="lines",
                    name=f"{model_name} (AUC = {auc_score:.3f})",
                    line={
                        "width": 3,
                        "color": PlotTheme.PRIMARY_COLORS[
                            i % len(PlotTheme.PRIMARY_COLORS)
                        ],
                    },
                )
            )

        fig.update_layout(
            title="ROC Curves Comparison",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis={"range": [0, 1]},
            yaxis={"range": [0, 1]},
            width=600,
            height=600,
        )

        return PlotTheme.apply_theme(fig)


class PrecisionRecallVisualization(BaseVisualization):
    """Precision-Recall curve visualization."""

    def create_figure(self, data: dict[str, Any], **kwargs: Any) -> go.Figure:
        """Create Precision-Recall curve plot."""
        if not data:
            return create_empty_figure(
                "Precision-Recall Curves", "No PR data available"
            )

        fig = go.Figure()

        # Add baseline (random classifier based on class distribution)
        baseline = kwargs.get("baseline", 0.5)
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[baseline, baseline],
                mode="lines",
                line={"dash": "dash", "color": "gray"},
                name=f"Random Classifier (AP = {baseline:.3f})",
                showlegend=True,
            )
        )

        # Add PR curves for each model
        for i, (model_name, pr_data) in enumerate(data.items()):
            if "precision" not in pr_data or "recall" not in pr_data:
                continue

            ap_score = pr_data.get("average_precision", 0.0)

            fig.add_trace(
                go.Scatter(
                    x=pr_data["recall"],
                    y=pr_data["precision"],
                    mode="lines",
                    name=f"{model_name} (AP = {ap_score:.3f})",
                    line={
                        "width": 3,
                        "color": PlotTheme.PRIMARY_COLORS[
                            i % len(PlotTheme.PRIMARY_COLORS)
                        ],
                    },
                )
            )

        fig.update_layout(
            title="Precision-Recall Curves Comparison",
            xaxis_title="Recall",
            yaxis_title="Precision",
            xaxis={"range": [0, 1]},
            yaxis={"range": [0, 1]},
            width=600,
            height=600,
        )

        return PlotTheme.apply_theme(fig)


class ConfusionMatrixVisualization(BaseVisualization):
    """Confusion matrix heatmap."""

    def create_figure(
        self, data: np.ndarray | pd.DataFrame, **kwargs: Any
    ) -> go.Figure:
        """Create confusion matrix heatmap."""
        if data is None or (hasattr(data, "size") and data.size == 0):
            return create_empty_figure(
                "Confusion Matrix", "No confusion matrix data available"
            )

        # Convert to numpy array if needed
        if isinstance(data, pd.DataFrame):
            cm_data = data.values
            labels = data.index.tolist()
        else:
            cm_data = np.array(data)
            labels = kwargs.get(
                "labels", [f"Class {i}" for i in range(cm_data.shape[0])]
            )

        # Normalize if requested
        normalize = kwargs.get("normalize", False)
        if normalize:
            cm_data = cm_data.astype("float") / cm_data.sum(axis=1)[:, np.newaxis]

        # Create annotations
        annotations = []
        for i in range(cm_data.shape[0]):
            for j in range(cm_data.shape[1]):
                if normalize:
                    text = f"{cm_data[i, j]:.2%}"
                else:
                    text = f"{int(cm_data[i, j])}"
                annotations.append(
                    {
                        "x": j,
                        "y": i,
                        "text": text,
                        "showarrow": False,
                        "font": {
                            "color": "white"
                            if cm_data[i, j] > cm_data.max() / 2
                            else "black"
                        },
                    }
                )

        fig = go.Figure(
            data=go.Heatmap(
                z=cm_data,
                x=labels,
                y=labels,
                colorscale="Blues",
                colorbar={"title": "Proportion" if normalize else "Count"},
            )
        )

        fig.update_layout(
            title="Confusion Matrix" + (" (Normalized)" if normalize else ""),
            xaxis_title="Predicted",
            yaxis_title="Actual",
            annotations=annotations,
            width=500,
            height=500,
        )

        return PlotTheme.apply_theme(fig)


class FeatureImportanceVisualization(BaseVisualization):
    """Feature importance visualization."""

    def create_figure(self, data: pd.DataFrame, **kwargs: Any) -> go.Figure:
        """Create feature importance plot."""
        if data.empty:
            return create_empty_figure(
                "Feature Importance", "No feature importance data available"
            )

        # Expected columns: feature, importance
        if "feature" not in data.columns or "importance" not in data.columns:
            return create_empty_figure(
                "Feature Importance", "Required columns: feature, importance"
            )

        # Sort by importance
        sorted_data = data.sort_values("importance", ascending=True)

        # Limit to top N features
        max_features = kwargs.get("max_features", 20)
        if len(sorted_data) > max_features:
            sorted_data = sorted_data.tail(max_features)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                y=sorted_data["feature"],
                x=sorted_data["importance"],
                orientation="h",
                marker_color="lightgreen",
                text=sorted_data["importance"].round(3),
                textposition="outside",
            )
        )

        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=max(400, len(sorted_data) * 25),
        )

        return PlotTheme.apply_theme(fig)


class LearningCurveVisualization(BaseVisualization):
    """Learning curve visualization showing training vs validation performance."""

    def create_figure(self, data: dict[str, Any], **kwargs: Any) -> go.Figure:
        """Create learning curve plot."""
        if not data or "train_sizes" not in data:
            return create_empty_figure(
                "Learning Curves", "No learning curve data available"
            )

        train_sizes = data["train_sizes"]

        fig = go.Figure()

        # Training score
        if "train_scores_mean" in data:
            train_mean = data["train_scores_mean"]
            train_std = data.get("train_scores_std", np.zeros_like(train_mean))

            # Add confidence interval
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([train_sizes, train_sizes[::-1]]),
                    y=np.concatenate(
                        [train_mean + train_std, (train_mean - train_std)[::-1]]
                    ),
                    fill="toself",
                    fillcolor="rgba(0,100,80,0.2)",
                    line={"color": "transparent"},
                    name="Training ±1σ",
                    showlegend=False,
                )
            )

            # Add mean line
            fig.add_trace(
                go.Scatter(
                    x=train_sizes,
                    y=train_mean,
                    mode="lines+markers",
                    name="Training Score",
                    line={"color": "green", "width": 2},
                )
            )

        # Validation score
        if "validation_scores_mean" in data:
            val_mean = data["validation_scores_mean"]
            val_std = data.get("validation_scores_std", np.zeros_like(val_mean))

            # Add confidence interval
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([train_sizes, train_sizes[::-1]]),
                    y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]),
                    fill="toself",
                    fillcolor="rgba(255,0,0,0.2)",
                    line={"color": "transparent"},
                    name="Validation ±1σ",
                    showlegend=False,
                )
            )

            # Add mean line
            fig.add_trace(
                go.Scatter(
                    x=train_sizes,
                    y=val_mean,
                    mode="lines+markers",
                    name="Validation Score",
                    line={"color": "red", "width": 2},
                )
            )

        fig.update_layout(
            title="Learning Curves",
            xaxis_title="Training Set Size",
            yaxis_title="Score",
            hovermode="x unified",
        )

        return PlotTheme.apply_theme(fig)


class RegressionVisualization(BaseVisualization):
    """Visualizations for regression model evaluation."""

    def create_figure(self, data: dict[str, Any], **kwargs: Any) -> go.Figure:
        """Create regression evaluation plots."""
        if not data:
            return create_empty_figure(
                "Regression Analysis", "No regression data available"
            )

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Actual vs Predicted",
                "Residuals vs Predicted",
                "Residuals Distribution",
                "Residuals Q-Q Plot",
            ],
        )

        y_true = data.get("y_true", [])
        y_pred = data.get("y_pred", [])

        if not y_true or not y_pred:
            return create_empty_figure(
                "Regression Analysis", "Missing y_true or y_pred data"
            )

        residuals = np.array(y_true) - np.array(y_pred)

        # 1. Actual vs Predicted
        fig.add_trace(
            go.Scatter(
                x=y_true,
                y=y_pred,
                mode="markers",
                name="Predictions",
                marker={"color": "blue", "opacity": 0.6},
            ),
            row=1,
            col=1,
        )

        # Add perfect prediction line
        min_val, max_val = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Perfect Prediction",
                line={"color": "red", "dash": "dash"},
            ),
            row=1,
            col=1,
        )

        # 2. Residuals vs Predicted
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode="markers",
                name="Residuals",
                marker={"color": "green", "opacity": 0.6},
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Add zero line
        fig.add_trace(
            go.Scatter(
                x=[min(y_pred), max(y_pred)],
                y=[0, 0],
                mode="lines",
                line={"color": "red", "dash": "dash"},
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # 3. Residuals distribution
        fig.add_trace(
            go.Histogram(
                x=residuals, name="Residuals Distribution", nbinsx=30, showlegend=False
            ),
            row=2,
            col=1,
        )

        # 4. Q-Q plot (simplified)
        sorted_residuals = np.sort(residuals)
        n = len(sorted_residuals)
        theoretical_quantiles = np.linspace(-2, 2, n)

        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sorted_residuals,
                mode="markers",
                name="Q-Q Plot",
                marker={"color": "purple", "opacity": 0.6},
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        # Add ideal line for Q-Q plot
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=theoretical_quantiles * np.std(residuals),
                mode="lines",
                line={"color": "red", "dash": "dash"},
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        # Calculate metrics
        mse = np.mean(residuals**2)
        r2 = 1 - (np.sum(residuals**2) / np.sum((y_true - np.mean(y_true)) ** 2))

        fig.update_layout(
            title=f"Regression Analysis (R² = {r2:.3f}, MSE = {mse:.3f})", height=700
        )

        return PlotTheme.apply_theme(fig)
