"""
Visualization components for data analysis and model evaluation.
"""

from .base import (
    BaseVisualization,
    InteractiveVisualization,
    PlotTheme,
    create_empty_figure,
)
from .data_exploration import (
    CorrelationVisualization,
    DataOverviewVisualization,
    DistributionVisualization,
    MissingDataVisualization,
    TimeSeriesVisualization,
)
from .experiment_tracking import (
    ExperimentComparisonVisualization,
    ExperimentHistoryVisualization,
    ExperimentMetricsVisualization,
    TrainingProgressVisualization,
)
from .model_evaluation import (
    ConfusionMatrixVisualization,
    FeatureImportanceVisualization,
    LearningCurveVisualization,
    ModelComparisonVisualization,
    PrecisionRecallVisualization,
    RegressionVisualization,
    ROCCurveVisualization,
)

__all__ = [
    # Base classes
    "BaseVisualization",
    "InteractiveVisualization",
    "PlotTheme",
    "create_empty_figure",

    # Data exploration
    "DataOverviewVisualization",
    "DistributionVisualization",
    "CorrelationVisualization",
    "MissingDataVisualization",
    "TimeSeriesVisualization",

    # Model evaluation
    "ModelComparisonVisualization",
    "ROCCurveVisualization",
    "PrecisionRecallVisualization",
    "ConfusionMatrixVisualization",
    "FeatureImportanceVisualization",
    "LearningCurveVisualization",
    "RegressionVisualization",

    # Experiment tracking
    "ExperimentHistoryVisualization",
    "ExperimentComparisonVisualization",
    "ExperimentMetricsVisualization",
    "TrainingProgressVisualization"
]
