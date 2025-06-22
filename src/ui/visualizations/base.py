"""
Base visualization components and utilities for the ML platform.
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
import panel as pn
import plotly.graph_objects as go

from src.core.logging import get_logger


class BaseVisualization(ABC):
    """Abstract base class for all visualization components."""

    def __init__(self, title: str = "", width: int = 800, height: int = 400) -> None:
        self.title = title
        self.width = width
        self.height = height
        self.logger = get_logger(__name__, pipeline_stage="visualization")
        self._figure = None
        self._panel = None

    @abstractmethod
    def create_figure(self, data: pd.DataFrame, **kwargs: Any) -> go.Figure:
        """Create the plotly figure."""

    def update_data(self, data: pd.DataFrame, **kwargs: Any) -> None:
        """Update the visualization with new data."""
        try:
            self._figure = self.create_figure(data, **kwargs)
            if self._panel is not None:
                self._panel.object = self._figure
        except Exception as e:
            self.logger.error(f"Failed to update visualization: {e}", exc_info=True)

    def get_panel(self, data: pd.DataFrame, **kwargs: Any) -> pn.pane.Plotly:
        """Get the Panel pane for this visualization."""
        if self._figure is None:
            self._figure = self.create_figure(data, **kwargs)

        self._panel = pn.pane.Plotly(
            self._figure,
            width=self.width,
            height=self.height,
            sizing_mode="stretch_width"
        )
        return self._panel

    def save_figure(self, filename: str, format: str = "png") -> None:
        """Save the figure to file."""
        if self._figure is not None:
            self._figure.write_image(filename, format=format)


class PlotTheme:
    """Standard plotting theme for consistent appearance."""

    # Color palette
    PRIMARY_COLORS = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    BACKGROUND_COLOR = "#fafafa"
    GRID_COLOR = "#e1e1e1"
    TEXT_COLOR = "#333333"

    # Font settings
    FONT_FAMILY = "Arial, sans-serif"
    TITLE_FONT_SIZE = 16
    AXIS_FONT_SIZE = 12
    LEGEND_FONT_SIZE = 11

    @classmethod
    def get_layout_template(cls) -> dict[str, Any]:
        """Get standard layout template."""
        return {
            "font": {
                "family": cls.FONT_FAMILY,
                "size": cls.AXIS_FONT_SIZE,
                "color": cls.TEXT_COLOR
            },
            "title": {
                "font": {"size": cls.TITLE_FONT_SIZE},
                "x": 0.5,
                "xanchor": "center"
            },
            "plot_bgcolor": cls.BACKGROUND_COLOR,
            "paper_bgcolor": "white",
            "xaxis": {
                "gridcolor": cls.GRID_COLOR,
                "showgrid": True,
                "linecolor": cls.GRID_COLOR
            },
            "yaxis": {
                "gridcolor": cls.GRID_COLOR,
                "showgrid": True,
                "linecolor": cls.GRID_COLOR
            },
            "legend": {
                "font": {"size": cls.LEGEND_FONT_SIZE},
                "bgcolor": "rgba(255,255,255,0.8)",
                "bordercolor": cls.GRID_COLOR,
                "borderwidth": 1
            }
        }

    @classmethod
    def apply_theme(cls, fig: go.Figure) -> go.Figure:
        """Apply standard theme to a figure."""
        fig.update_layout(**cls.get_layout_template())
        return fig


class InteractiveVisualization(BaseVisualization):
    """Base class for interactive visualizations with controls."""

    def __init__(self, title: str = "", width: int = 800, height: int = 400) -> None:
        super().__init__(title, width, height)
        self.controls = {}
        self.control_panel = None

    def add_control(self, name: str, widget: pn.widgets.Widget) -> None:
        """Add a control widget."""
        self.controls[name] = widget
        widget.param.watch(self._on_control_change, "value")

    def _on_control_change(self, event: Any) -> None:
        """Handle control change events."""
        if hasattr(self, "_current_data"):
            self.update_data(self._current_data)

    def get_control_panel(self) -> pn.Column:
        """Get the control panel for this visualization."""
        if self.control_panel is None:
            self.control_panel = pn.Column(
                pn.pane.Markdown(f"### {self.title} Controls"),
                *list(self.controls.values()),
                width=250
            )
        return self.control_panel

    def get_full_panel(self, data: pd.DataFrame, **kwargs: Any) -> pn.Row:
        """Get the full panel including controls and visualization."""
        self._current_data = data
        return pn.Row(
            self.get_control_panel(),
            self.get_panel(data, **kwargs),
            sizing_mode="stretch_width"
        )


def create_empty_figure(title: str = "No Data", message: str = "No data available") -> go.Figure:
    """Create an empty figure with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        xanchor="center", yanchor="middle",
        showarrow=False,
        font={"size": 16, "color": "#666666"}
    )
    fig.update_layout(
        title=title,
        xaxis={"visible": False},
        yaxis={"visible": False},
        **PlotTheme.get_layout_template()
    )
    return PlotTheme.apply_theme(fig)


def format_number(value: int | float, precision: int = 2) -> str:
    """Format number for display."""
    if isinstance(value, int) or value.is_integer():
        return f"{int(value):,}"
    return f"{value:,.{precision}f}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    return numerator / denominator if denominator != 0 else default
