"""
Experiment tracking and monitoring visualizations.
"""

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .base import (
    BaseVisualization,
    InteractiveVisualization,
    PlotTheme,
    create_empty_figure,
)


class ExperimentHistoryVisualization(BaseVisualization):
    """Visualization for experiment history and trends."""

    def create_figure(self, data: pd.DataFrame, **kwargs: Any) -> go.Figure:
        """Create experiment history visualization."""
        if data.empty:
            return create_empty_figure("Experiment History", "No experiments available")

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Experiments Over Time",
                "Success Rate by Task Type",
                "Performance Trends",
                "Experiment Duration Distribution"
            ]
        )

        # 1. Experiments over time
        if "created_at" in data.columns:
            # Convert to datetime if string
            if data["created_at"].dtype == "object":
                data["created_at"] = pd.to_datetime(data["created_at"], errors="coerce")

            # Group by date
            daily_counts = data.groupby(data["created_at"].dt.date).size()

            fig.add_trace(
                go.Scatter(
                    x=daily_counts.index,
                    y=daily_counts.values,
                    mode="lines+markers",
                    name="Experiments per Day",
                    line={"color": "blue"}
                ),
                row=1, col=1
            )

        # 2. Success rate by task type
        if "task_type" in data.columns and "status" in data.columns:
            success_rates = []
            task_types = data["task_type"].unique()

            for task_type in task_types:
                task_data = data[data["task_type"] == task_type]
                completed = len(task_data[task_data["status"] == "completed"])
                total = len(task_data)
                success_rate = completed / total if total > 0 else 0
                success_rates.append(success_rate)

            fig.add_trace(
                go.Bar(
                    x=task_types,
                    y=success_rates,
                    name="Success Rate",
                    marker_color="lightgreen",
                    showlegend=False
                ),
                row=1, col=2
            )

        # 3. Performance trends (if metrics available)
        if any(col in data.columns for col in ["accuracy", "best_accuracy", "score"]):
            metric_col = None
            for col in ["accuracy", "best_accuracy", "score"]:
                if col in data.columns:
                    metric_col = col
                    break

            if metric_col and "created_at" in data.columns:
                performance_data = data.dropna(subset=[metric_col, "created_at"])

                fig.add_trace(
                    go.Scatter(
                        x=performance_data["created_at"],
                        y=performance_data[metric_col],
                        mode="lines+markers",
                        name="Performance",
                        line={"color": "red"}
                    ),
                    row=2, col=1
                )

        # 4. Duration distribution
        if "duration" in data.columns:
            duration_data = data["duration"].dropna()

            fig.add_trace(
                go.Histogram(
                    x=duration_data,
                    name="Duration Distribution",
                    nbinsx=20,
                    marker_color="orange",
                    showlegend=False
                ),
                row=2, col=2
            )

        fig.update_layout(
            title="Experiment History Analysis",
            height=700,
            showlegend=True
        )

        return PlotTheme.apply_theme(fig)


class ExperimentComparisonVisualization(BaseVisualization):
    """Compare multiple experiments side by side."""

    def create_figure(self, data: pd.DataFrame, **kwargs: Any) -> go.Figure:
        """Create experiment comparison visualization."""
        if data.empty:
            return create_empty_figure("Experiment Comparison", "No experiments to compare")

        if len(data) < 2:
            return create_empty_figure("Experiment Comparison", "Need at least 2 experiments to compare")

        # Extract numeric metrics for comparison
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        # Common metric columns to prioritize
        priority_metrics = ["accuracy", "precision", "recall", "f1", "duration", "score"]

        # Select metrics to display
        display_metrics = []
        for metric in priority_metrics:
            matching_cols = [col for col in numeric_cols if metric.lower() in col.lower()]
            if matching_cols:
                display_metrics.append(matching_cols[0])

        # Add remaining numeric columns
        for col in numeric_cols:
            if col not in display_metrics and col not in ["id", "created_at"]:
                display_metrics.append(col)

        display_metrics = display_metrics[:6]  # Limit to 6 metrics

        if not display_metrics:
            return create_empty_figure("Experiment Comparison", "No numeric metrics found")

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Metric Comparison",
                "Normalized Radar Chart",
                "Ranking by Metrics",
                "Detailed Comparison Table"
            ],
            specs=[
                [{"type": "bar"}, {"type": "scatterpolar"}],
                [{"type": "bar"}, {"type": "table"}]
            ]
        )

        # 1. Metric comparison bar chart
        for i, metric in enumerate(display_metrics[:4]):  # Show top 4 metrics
            fig.add_trace(
                go.Bar(
                    x=data["name"] if "name" in data.columns else data.index,
                    y=data[metric],
                    name=metric.replace("_", " ").title(),
                    marker_color=PlotTheme.PRIMARY_COLORS[i % len(PlotTheme.PRIMARY_COLORS)]
                ),
                row=1, col=1
            )

        # 2. Normalized radar chart
        if len(display_metrics) >= 3:
            # Normalize metrics to 0-1 scale
            normalized_data = data[display_metrics].copy()
            for col in display_metrics:
                col_max = normalized_data[col].max()
                if col_max > 0:
                    normalized_data[col] = normalized_data[col] / col_max

            # Add radar traces for each experiment
            for i, (idx, row) in enumerate(normalized_data.iterrows()):
                exp_name = data.loc[idx, "name"] if "name" in data.columns else f"Exp {idx}"

                fig.add_trace(
                    go.Scatterpolar(
                        r=row.values,
                        theta=display_metrics,
                        fill="toself",
                        name=exp_name,
                        line_color=PlotTheme.PRIMARY_COLORS[i % len(PlotTheme.PRIMARY_COLORS)]
                    ),
                    row=1, col=2
                )

        # 3. Ranking chart (by first metric)
        if display_metrics:
            primary_metric = display_metrics[0]
            ranked_data = data.sort_values(primary_metric, ascending=False)

            fig.add_trace(
                go.Bar(
                    x=ranked_data["name"] if "name" in ranked_data.columns else ranked_data.index,
                    y=ranked_data[primary_metric],
                    name=f"Ranked by {primary_metric.replace('_', ' ').title()}",
                    marker_color="lightblue",
                    showlegend=False
                ),
                row=2, col=1
            )

        # 4. Detailed comparison table
        table_cols = ["name"] if "name" in data.columns else []
        table_cols.extend(display_metrics)
        table_data = data[table_cols].round(4)

        fig.add_trace(
            go.Table(
                header={"values": [col.replace("_", " ").title() for col in table_cols]},
                cells={"values": [table_data[col].values for col in table_cols]}
            ),
            row=2, col=2
        )

        fig.update_layout(
            title="Experiment Comparison Analysis",
            height=800,
            showlegend=True
        )

        return PlotTheme.apply_theme(fig)


class ExperimentMetricsVisualization(InteractiveVisualization):
    """Interactive visualization for exploring experiment metrics."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(title="Experiment Metrics", **kwargs)
        self.x_metric_select = None
        self.y_metric_select = None
        self.color_metric_select = None
        self.size_metric_select = None

    def create_figure(self, data: pd.DataFrame, **kwargs: Any) -> go.Figure:
        """Create interactive metrics scatter plot."""
        if data.empty:
            return create_empty_figure("Experiment Metrics", "No experiment data available")

        # Initialize controls
        if self.x_metric_select is None:
            self._init_controls(data)

        # Get selections
        x_metric = self.x_metric_select.value if self.x_metric_select else None
        y_metric = self.y_metric_select.value if self.y_metric_select else None
        color_metric = self.color_metric_select.value if self.color_metric_select else None
        size_metric = self.size_metric_select.value if self.size_metric_select else None

        if not x_metric or not y_metric:
            return create_empty_figure("Experiment Metrics", "Please select X and Y metrics")

        # Create scatter plot
        fig = go.Figure()

        # Prepare data
        plot_data = data.dropna(subset=[x_metric, y_metric])

        if plot_data.empty:
            return create_empty_figure("Experiment Metrics", f"No data for {x_metric} vs {y_metric}")

        # Set up marker properties
        marker_props = {
            "size": 10,
            "opacity": 0.7,
            "line": {"width": 1, "color": "white"}
        }

        # Color mapping
        if color_metric and color_metric != "None":
            if color_metric == "task_type" or color_metric == "status":
                # Categorical coloring
                unique_values = plot_data[color_metric].unique()
                color_map = {val: PlotTheme.PRIMARY_COLORS[i % len(PlotTheme.PRIMARY_COLORS)]
                           for i, val in enumerate(unique_values)}
                marker_props["color"] = [color_map[val] for val in plot_data[color_metric]]
            else:
                # Continuous coloring
                marker_props["color"] = plot_data[color_metric]
                marker_props["colorscale"] = "Viridis"
                marker_props["showscale"] = True
                marker_props["colorbar"] = {"title": color_metric.replace("_", " ").title()}

        # Size mapping
        if size_metric and size_metric != "None":
            sizes = plot_data[size_metric]
            # Normalize sizes to reasonable range
            min_size, max_size = 5, 20
            normalized_sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min()) if sizes.max() > sizes.min() else 0.5
            marker_props["size"] = min_size + normalized_sizes * (max_size - min_size)

        # Add hover text
        hover_text = []
        for _, row in plot_data.iterrows():
            text = f"<b>{row.get('name', 'Experiment')}</b><br>"
            text += f"{x_metric}: {row[x_metric]:.3f}<br>"
            text += f"{y_metric}: {row[y_metric]:.3f}<br>"
            if color_metric and color_metric != "None":
                text += f"{color_metric}: {row[color_metric]}<br>"
            if size_metric and size_metric != "None":
                text += f"{size_metric}: {row[size_metric]:.3f}"
            hover_text.append(text)

        fig.add_trace(go.Scatter(
            x=plot_data[x_metric],
            y=plot_data[y_metric],
            mode="markers",
            marker=marker_props,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            name="Experiments"
        ))

        fig.update_layout(
            title=f"Experiment Metrics: {x_metric.replace('_', ' ').title()} vs {y_metric.replace('_', ' ').title()}",
            xaxis_title=x_metric.replace("_", " ").title(),
            yaxis_title=y_metric.replace("_", " ").title()
        )

        return PlotTheme.apply_theme(fig)

    def _init_controls(self, data: pd.DataFrame) -> None:
        """Initialize control widgets."""
        import panel as pn

        # Get numeric columns for metrics
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = ["task_type", "status"] if all(col in data.columns for col in ["task_type", "status"]) else []

        if not numeric_cols:
            return

        # X metric
        self.x_metric_select = pn.widgets.Select(
            name="X Metric",
            options=numeric_cols,
            value=numeric_cols[0]
        )
        self.add_control("x_metric", self.x_metric_select)

        # Y metric
        self.y_metric_select = pn.widgets.Select(
            name="Y Metric",
            options=numeric_cols,
            value=numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
        )
        self.add_control("y_metric", self.y_metric_select)

        # Color metric
        color_options = ["None"] + numeric_cols + categorical_cols
        self.color_metric_select = pn.widgets.Select(
            name="Color By",
            options=color_options,
            value="None"
        )
        self.add_control("color_metric", self.color_metric_select)

        # Size metric
        size_options = ["None"] + numeric_cols
        self.size_metric_select = pn.widgets.Select(
            name="Size By",
            options=size_options,
            value="None"
        )
        self.add_control("size_metric", self.size_metric_select)


class TrainingProgressVisualization(BaseVisualization):
    """Real-time training progress visualization."""

    def create_figure(self, data: dict[str, Any], **kwargs: Any) -> go.Figure:
        """Create training progress plot."""
        if not data:
            return create_empty_figure("Training Progress", "No training data available")

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Loss Curves", "Metrics Curves"],
            shared_xaxes=True
        )

        # Training loss
        if "train_loss" in data:
            epochs = list(range(1, len(data["train_loss"]) + 1))
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=data["train_loss"],
                    mode="lines",
                    name="Training Loss",
                    line={"color": "blue"}
                ),
                row=1, col=1
            )

        # Validation loss
        if "val_loss" in data:
            epochs = list(range(1, len(data["val_loss"]) + 1))
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=data["val_loss"],
                    mode="lines",
                    name="Validation Loss",
                    line={"color": "red"}
                ),
                row=1, col=1
            )

        # Training metrics
        if "train_metrics" in data:
            for metric_name, values in data["train_metrics"].items():
                epochs = list(range(1, len(values) + 1))
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=values,
                        mode="lines",
                        name=f"Train {metric_name}",
                        line={"dash": "dot"}
                    ),
                    row=2, col=1
                )

        # Validation metrics
        if "val_metrics" in data:
            for metric_name, values in data["val_metrics"].items():
                epochs = list(range(1, len(values) + 1))
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=values,
                        mode="lines",
                        name=f"Val {metric_name}",
                        line={"dash": "solid"}
                    ),
                    row=2, col=1
                )

        fig.update_layout(
            title="Training Progress",
            height=600,
            hovermode="x unified"
        )

        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Metric Value", row=2, col=1)

        return PlotTheme.apply_theme(fig)
