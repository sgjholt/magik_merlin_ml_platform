"""
Data exploration and profiling visualizations.
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


class DataOverviewVisualization(BaseVisualization):
    """Overview visualization showing dataset statistics."""

    def create_figure(self, data: pd.DataFrame, **kwargs: Any) -> go.Figure:
        """Create dataset overview figure."""
        if data.empty:
            return create_empty_figure("Dataset Overview", "No data available")

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Data Types Distribution",
                "Missing Values by Column",
                "Numeric Columns Statistics",
                "Dataset Information"
            ],
            specs=[
                [{"type": "pie"}, {"type": "bar"}],
                [{"type": "table"}, {"type": "table"}]
            ]
        )

        # 1. Data types distribution
        dtype_counts = data.dtypes.value_counts()
        fig.add_trace(
            go.Pie(
                labels=[str(dtype) for dtype in dtype_counts.index],
                values=dtype_counts.values,
                name="Data Types"
            ),
            row=1, col=1
        )

        # 2. Missing values
        missing_data = data.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

        if not missing_data.empty:
            fig.add_trace(
                go.Bar(
                    x=missing_data.values,
                    y=missing_data.index,
                    orientation="h",
                    name="Missing Values"
                ),
                row=1, col=2
            )

        # 3. Numeric statistics
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats_data = data[numeric_cols].describe().round(2)
            fig.add_trace(
                go.Table(
                    header={"values": ["Statistic"] + list(stats_data.columns)},
                    cells={
                        "values": [stats_data.index] + [stats_data[col].values for col in stats_data.columns]
                    }
                ),
                row=2, col=1
            )

        # 4. Dataset info
        info_data = [
            ["Shape", f"{data.shape[0]:,} rows × {data.shape[1]:,} columns"],
            ["Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024**2:.1f} MB"],
            ["Numeric Columns", str(len(data.select_dtypes(include=[np.number]).columns))],
            ["Categorical Columns", str(len(data.select_dtypes(include=["object", "category"]).columns))],
            ["DateTime Columns", str(len(data.select_dtypes(include=["datetime"]).columns))],
            ["Total Missing", f"{data.isnull().sum().sum():,} ({data.isnull().sum().sum() / data.size * 100:.1f}%)"],
            ["Duplicate Rows", f"{data.duplicated().sum():,}"]
        ]

        fig.add_trace(
            go.Table(
                header={"values": ["Property", "Value"]},
                cells={"values": [[row[0] for row in info_data], [row[1] for row in info_data]]}
            ),
            row=2, col=2
        )

        fig.update_layout(
            title=f"Dataset Overview: {kwargs.get('dataset_name', 'Data')}",
            height=700,
            showlegend=False
        )

        return PlotTheme.apply_theme(fig)


class DistributionVisualization(InteractiveVisualization):
    """Interactive visualization for exploring data distributions."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(title="Data Distributions", **kwargs)

        # Add controls
        self.column_select = None
        self.plot_type_select = None
        self.bins_slider = None

    def create_figure(self, data: pd.DataFrame, **kwargs: Any) -> go.Figure:
        """Create distribution visualization."""
        if data.empty:
            return create_empty_figure("Data Distribution", "No data available")

        # Initialize controls if not done
        if self.column_select is None:
            self._init_controls(data)

        # Get current selections
        selected_column = self.column_select.value if self.column_select else data.columns[0]
        plot_type = self.plot_type_select.value if self.plot_type_select else "histogram"
        bins = self.bins_slider.value if self.bins_slider else 30

        if selected_column not in data.columns:
            return create_empty_figure("Distribution", f"Column '{selected_column}' not found")

        column_data = data[selected_column].dropna()

        if column_data.empty:
            return create_empty_figure("Distribution", f"No data in column '{selected_column}'")

        # Determine if column is numeric
        is_numeric = pd.api.types.is_numeric_dtype(column_data)

        fig = go.Figure()

        if is_numeric:
            if plot_type == "histogram":
                fig.add_trace(go.Histogram(
                    x=column_data,
                    nbinsx=bins,
                    name=selected_column,
                    opacity=0.7
                ))
            elif plot_type == "box":
                fig.add_trace(go.Box(
                    y=column_data,
                    name=selected_column,
                    boxpoints="outliers"
                ))
            elif plot_type == "violin":
                fig.add_trace(go.Violin(
                    y=column_data,
                    name=selected_column,
                    box_visible=True,
                    meanline_visible=True
                ))
        else:
            # Categorical data
            value_counts = column_data.value_counts().head(20)  # Top 20 categories

            if plot_type in ["histogram", "bar"]:
                fig.add_trace(go.Bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    name=selected_column
                ))
            elif plot_type == "pie":
                fig.add_trace(go.Pie(
                    labels=value_counts.index,
                    values=value_counts.values,
                    name=selected_column
                ))

        # Add statistics annotation
        if is_numeric:
            stats_text = (
                f"Count: {len(column_data):,}<br>"
                f"Mean: {column_data.mean():.2f}<br>"
                f"Std: {column_data.std():.2f}<br>"
                f"Min: {column_data.min():.2f}<br>"
                f"Max: {column_data.max():.2f}"
            )
        else:
            stats_text = (
                f"Count: {len(column_data):,}<br>"
                f"Unique: {column_data.nunique():,}<br>"
                f"Most Common: {column_data.mode().iloc[0] if not column_data.mode().empty else 'N/A'}"
            )

        fig.add_annotation(
            text=stats_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            xanchor="left", yanchor="top",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#cccccc",
            borderwidth=1
        )

        fig.update_layout(
            title=f"Distribution of {selected_column}",
            xaxis_title=selected_column if plot_type != "pie" else "",
            yaxis_title="Frequency" if plot_type == "histogram" else "Value"
        )

        return PlotTheme.apply_theme(fig)

    def _init_controls(self, data: pd.DataFrame) -> None:
        """Initialize control widgets."""
        import panel as pn

        # Column selection
        self.column_select = pn.widgets.Select(
            name="Column",
            options=list(data.columns),
            value=data.columns[0] if len(data.columns) > 0 else None
        )
        self.add_control("column", self.column_select)

        # Plot type selection
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            plot_options = ["histogram", "box", "violin"]
        else:
            plot_options = ["bar", "pie"]

        self.plot_type_select = pn.widgets.Select(
            name="Plot Type",
            options=plot_options,
            value=plot_options[0]
        )
        self.add_control("plot_type", self.plot_type_select)

        # Bins slider (for histograms)
        self.bins_slider = pn.widgets.IntSlider(
            name="Bins",
            start=10, end=100, value=30, step=5
        )
        self.add_control("bins", self.bins_slider)


class CorrelationVisualization(BaseVisualization):
    """Correlation matrix visualization."""

    def create_figure(self, data: pd.DataFrame, **kwargs: Any) -> go.Figure:
        """Create correlation matrix heatmap."""
        if data.empty:
            return create_empty_figure("Correlation Matrix", "No data available")

        # Select only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])

        if numeric_data.empty:
            return create_empty_figure("Correlation Matrix", "No numeric columns found")

        if numeric_data.shape[1] < 2:
            return create_empty_figure("Correlation Matrix", "Need at least 2 numeric columns")

        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="RdBu",
            zmid=0,
            colorbar={"title": "Correlation"},
            text=np.around(corr_matrix.values, decimals=2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>"
        ))

        fig.update_layout(
            title="Feature Correlation Matrix",
            xaxis={"title": "Features", "side": "bottom"},
            yaxis={"title": "Features"},
            width=max(400, len(corr_matrix) * 40),
            height=max(400, len(corr_matrix) * 40)
        )

        return PlotTheme.apply_theme(fig)


class MissingDataVisualization(BaseVisualization):
    """Visualization for missing data patterns."""

    def create_figure(self, data: pd.DataFrame, **kwargs: Any) -> go.Figure:
        """Create missing data pattern visualization."""
        if data.empty:
            return create_empty_figure("Missing Data Patterns", "No data available")

        # Calculate missing data
        missing_data = data.isnull()
        missing_counts = missing_data.sum()

        if missing_counts.sum() == 0:
            return create_empty_figure("Missing Data Patterns", "No missing data found")

        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Missing Data by Column", "Missing Data Patterns"],
            row_heights=[0.4, 0.6]
        )

        # 1. Missing data counts by column
        missing_cols = missing_counts[missing_counts > 0].sort_values(ascending=True)

        fig.add_trace(
            go.Bar(
                y=missing_cols.index,
                x=missing_cols.values,
                orientation="h",
                name="Missing Count",
                marker_color="lightcoral"
            ),
            row=1, col=1
        )

        # 2. Missing data pattern heatmap
        # Sample data if too large
        sample_size = min(1000, len(data))
        if len(data) > sample_size:
            data_sample = data.sample(n=sample_size, random_state=42)
        else:
            data_sample = data

        # Only show columns with missing data
        cols_with_missing = missing_counts[missing_counts > 0].index
        if len(cols_with_missing) > 0:
            missing_pattern = data_sample[cols_with_missing].isnull().astype(int)

            fig.add_trace(
                go.Heatmap(
                    z=missing_pattern.values,
                    x=missing_pattern.columns,
                    y=list(range(len(missing_pattern))),
                    colorscale=[[0, "lightblue"], [1, "darkred"]],
                    showscale=False,
                    hovertemplate="<b>%{x}</b><br>Row: %{y}<br>Missing: %{z}<extra></extra>"
                ),
                row=2, col=1
            )

        fig.update_layout(
            title="Missing Data Analysis",
            height=700
        )

        return PlotTheme.apply_theme(fig)


class TimeSeriesVisualization(InteractiveVisualization):
    """Time series data visualization."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(title="Time Series Analysis", **kwargs)
        self.date_column_select = None
        self.value_column_select = None
        self.aggregation_select = None

    def create_figure(self, data: pd.DataFrame, **kwargs: Any) -> go.Figure:
        """Create time series visualization."""
        if data.empty:
            return create_empty_figure("Time Series", "No data available")

        # Find datetime columns
        datetime_cols = data.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
        date_like_cols = []

        # Check for date-like string columns
        for col in data.select_dtypes(include=["object"]).columns:
            try:
                pd.to_datetime(data[col].dropna().head(100))
                date_like_cols.append(col)
            except (ValueError, TypeError):
                continue

        all_date_cols = datetime_cols + date_like_cols

        if not all_date_cols:
            return create_empty_figure("Time Series", "No datetime columns found")

        # Initialize controls
        if self.date_column_select is None:
            self._init_controls(data, all_date_cols)

        date_col = self.date_column_select.value if self.date_column_select else all_date_cols[0]
        value_col = self.value_column_select.value if self.value_column_select else None

        if date_col not in data.columns or value_col not in data.columns:
            return create_empty_figure("Time Series", "Selected columns not found")

        # Prepare data
        ts_data = data[[date_col, value_col]].copy()
        ts_data = ts_data.dropna()

        # Convert date column
        try:
            ts_data[date_col] = pd.to_datetime(ts_data[date_col])
        except (ValueError, TypeError):
            return create_empty_figure("Time Series", f"Cannot parse {date_col} as datetime")

        ts_data = ts_data.sort_values(date_col)

        # Create time series plot
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=ts_data[date_col],
            y=ts_data[value_col],
            mode="lines+markers",
            name=value_col,
            line={"width": 2},
            marker={"size": 4}
        ))

        # Add trend line if enough data points
        if len(ts_data) > 10:
            from scipy import stats
            x_numeric = (ts_data[date_col] - ts_data[date_col].min()).dt.days
            slope, intercept, r_value, _, _ = stats.linregress(x_numeric, ts_data[value_col])

            trend_line = intercept + slope * x_numeric
            fig.add_trace(go.Scatter(
                x=ts_data[date_col],
                y=trend_line,
                mode="lines",
                name=f"Trend (R²={r_value**2:.3f})",
                line={"dash": "dash", "color": "red"}
            ))

        fig.update_layout(
            title=f"Time Series: {value_col} over {date_col}",
            xaxis_title=date_col,
            yaxis_title=value_col,
            hovermode="x unified"
        )

        return PlotTheme.apply_theme(fig)

    def _init_controls(self, data: pd.DataFrame, date_cols: list[str]) -> None:
        """Initialize control widgets."""
        import panel as pn

        # Date column selection
        self.date_column_select = pn.widgets.Select(
            name="Date Column",
            options=date_cols,
            value=date_cols[0] if date_cols else None
        )
        self.add_control("date_column", self.date_column_select)

        # Value column selection (numeric only)
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        self.value_column_select = pn.widgets.Select(
            name="Value Column",
            options=numeric_cols,
            value=numeric_cols[0] if numeric_cols else None
        )
        self.add_control("value_column", self.value_column_select)
