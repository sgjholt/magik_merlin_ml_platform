"""
Pipeline node system for building and executing ML workflows.

This module provides the base node classes and specific node implementations
for creating ML pipelines. Nodes are the building blocks of pipelines, each
performing a specific task in the workflow.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.core.logging import get_logger

logger = get_logger(__name__)


class NodeStatus(Enum):
    """Status of a pipeline node."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class NodeType(Enum):
    """Type of pipeline node."""

    DATA_LOADER = "data_loader"
    DATA_PREPROCESSOR = "data_preprocessor"
    FEATURE_ENGINEER = "feature_engineer"
    MODEL_TRAINER = "model_trainer"
    MODEL_EVALUATOR = "model_evaluator"
    MODEL_SAVER = "model_saver"
    CUSTOM = "custom"


@dataclass
class NodeOutput:
    """Output from a pipeline node."""

    data: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class NodeMetrics:
    """Metrics tracked during node execution."""

    execution_time: float = 0.0
    memory_usage: float = 0.0
    rows_processed: int = 0
    custom_metrics: dict[str, Any] = field(default_factory=dict)


class BaseNode(ABC):
    """
    Base class for all pipeline nodes.

    Nodes are the building blocks of pipelines. Each node performs a specific
    task and can pass data to downstream nodes.
    """

    def __init__(
        self,
        node_id: str,
        node_type: NodeType,
        name: str | None = None,
        description: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize a pipeline node.

        Args:
            node_id: Unique identifier for the node
            node_type: Type of the node
            name: Human-readable name for the node
            description: Description of what the node does
            config: Configuration parameters for the node
        """
        self.node_id = node_id
        self.node_type = node_type
        self.name = name or node_id
        self.description = description or ""
        self.config = config or {}
        self.status = NodeStatus.PENDING
        self.metrics = NodeMetrics()
        self.error: str | None = None
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def execute(self, inputs: dict[str, NodeOutput]) -> NodeOutput:
        """
        Execute the node's logic.

        Args:
            inputs: Dictionary mapping input names to NodeOutput objects

        Returns:
            NodeOutput containing the result of execution

        Raises:
            Exception: If execution fails
        """

    def run(self, inputs: dict[str, NodeOutput]) -> NodeOutput:
        """
        Run the node with error handling and metrics tracking.

        Args:
            inputs: Dictionary mapping input names to NodeOutput objects

        Returns:
            NodeOutput from the execution
        """
        import time

        self.status = NodeStatus.RUNNING
        start_time = time.time()

        try:
            self.logger.info(f"Executing node: {self.name} ({self.node_id})")
            output = self.execute(inputs)
            self.metrics.execution_time = time.time() - start_time
            self.status = NodeStatus.COMPLETED
            self.logger.info(
                f"Node {self.name} completed in {self.metrics.execution_time:.2f}s"
            )
            return output

        except Exception as e:
            self.metrics.execution_time = time.time() - start_time
            self.status = NodeStatus.FAILED
            self.error = str(e)
            self.logger.exception(f"Node {self.name} failed: {e}")
            raise

    def to_dict(self) -> dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "name": self.name,
            "description": self.description,
            "config": self.config,
            "status": self.status.value,
            "error": self.error,
        }


class DataLoaderNode(BaseNode):
    """Node for loading data from various sources."""

    def __init__(
        self,
        node_id: str,
        source_type: str = "file",
        source_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize data loader node.

        Args:
            node_id: Unique identifier for the node
            source_type: Type of data source ('file', 'database', etc.)
            source_path: Path to data source
            **kwargs: Additional configuration parameters
        """
        config = {"source_type": source_type, "source_path": source_path, **kwargs}
        super().__init__(
            node_id=node_id,
            node_type=NodeType.DATA_LOADER,
            name=kwargs.get("name", "Data Loader"),
            description=kwargs.get("description", "Load data from source"),
            config=config,
        )

    def execute(self, inputs: dict[str, NodeOutput]) -> NodeOutput:
        """Load data from the configured source."""
        source_type = self.config.get("source_type", "file")
        source_path = self.config.get("source_path")

        if source_type == "file":
            if not source_path:
                msg = "source_path required for file data source"
                raise ValueError(msg)

            path = Path(source_path)
            if not path.exists():
                msg = f"Data file not found: {source_path}"
                raise FileNotFoundError(msg)

            # Load based on file extension
            if path.suffix == ".csv":
                data = pd.read_csv(path)
            elif path.suffix == ".parquet":
                data = pd.read_parquet(path)
            elif path.suffix in (".xlsx", ".xls"):
                data = pd.read_excel(path)
            else:
                msg = f"Unsupported file format: {path.suffix}"
                raise ValueError(msg)

            self.metrics.rows_processed = len(data)
            return NodeOutput(
                data=data,
                metadata={
                    "source_type": source_type,
                    "source_path": str(source_path),
                    "rows": len(data),
                    "columns": list(data.columns),
                },
            )

        msg = f"Unsupported source type: {source_type}"
        raise ValueError(msg)


class DataPreprocessorNode(BaseNode):
    """Node for preprocessing data (handling missing values, encoding, etc.)."""

    def __init__(
        self,
        node_id: str,
        operations: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize data preprocessor node.

        Args:
            node_id: Unique identifier for the node
            operations: List of preprocessing operations to apply
            **kwargs: Additional configuration parameters
        """
        config = {"operations": operations or [], **kwargs}
        super().__init__(
            node_id=node_id,
            node_type=NodeType.DATA_PREPROCESSOR,
            name=kwargs.get("name", "Data Preprocessor"),
            description=kwargs.get("description", "Preprocess and clean data"),
            config=config,
        )

    def execute(self, inputs: dict[str, NodeOutput]) -> NodeOutput:
        """Preprocess the input data."""
        if "data" not in inputs:
            msg = "DataPreprocessorNode requires 'data' input"
            raise ValueError(msg)

        data = inputs["data"].data
        if not isinstance(data, pd.DataFrame):
            msg = f"Expected pandas DataFrame, got {type(data)}"
            raise TypeError(msg)

        # Create a copy to avoid modifying original
        processed_data = data.copy()
        operations_applied = []

        operations = self.config.get("operations", [])
        for operation in operations:
            if operation == "drop_missing":
                initial_rows = len(processed_data)
                processed_data = processed_data.dropna()
                rows_dropped = initial_rows - len(processed_data)
                operations_applied.append(f"drop_missing: {rows_dropped} rows")

            elif operation == "fill_missing_mean":
                numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
                processed_data[numeric_cols] = processed_data[numeric_cols].fillna(
                    processed_data[numeric_cols].mean()
                )
                operations_applied.append("fill_missing_mean: numeric columns")

            elif operation == "remove_duplicates":
                initial_rows = len(processed_data)
                processed_data = processed_data.drop_duplicates()
                rows_dropped = initial_rows - len(processed_data)
                operations_applied.append(f"remove_duplicates: {rows_dropped} rows")

        self.metrics.rows_processed = len(processed_data)
        return NodeOutput(
            data=processed_data,
            metadata={
                "operations_applied": operations_applied,
                "rows": len(processed_data),
                "columns": list(processed_data.columns),
            },
        )


class TrainTestSplitNode(BaseNode):
    """Node for splitting data into train and test sets."""

    def __init__(
        self,
        node_id: str,
        test_size: float = 0.2,
        random_state: int | None = 42,
        **kwargs: Any,
    ) -> None:
        """
        Initialize train-test split node.

        Args:
            node_id: Unique identifier for the node
            test_size: Proportion of data for test set
            random_state: Random seed for reproducibility
            **kwargs: Additional configuration parameters
        """
        config = {"test_size": test_size, "random_state": random_state, **kwargs}
        super().__init__(
            node_id=node_id,
            node_type=NodeType.DATA_PREPROCESSOR,
            name=kwargs.get("name", "Train-Test Split"),
            description=kwargs.get(
                "description", "Split data into train and test sets"
            ),
            config=config,
        )

    def execute(self, inputs: dict[str, NodeOutput]) -> NodeOutput:
        """Split the input data into train and test sets."""
        if "data" not in inputs:
            msg = "TrainTestSplitNode requires 'data' input"
            raise ValueError(msg)

        data = inputs["data"].data
        if not isinstance(data, pd.DataFrame):
            msg = f"Expected pandas DataFrame, got {type(data)}"
            raise TypeError(msg)

        target_column = self.config.get("target_column")
        if not target_column:
            msg = "target_column must be specified in config"
            raise ValueError(msg)

        if target_column not in data.columns:
            msg = f"Target column '{target_column}' not found in data"
            raise ValueError(msg)

        X = data.drop(columns=[target_column])
        y = data[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.get("test_size", 0.2),
            random_state=self.config.get("random_state", 42),
        )

        self.metrics.rows_processed = len(data)
        return NodeOutput(
            data={
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            },
            metadata={
                "train_size": len(X_train),
                "test_size": len(X_test),
                "target_column": target_column,
                "features": list(X.columns),
            },
        )


class FeatureScalerNode(BaseNode):
    """Node for scaling features using StandardScaler."""

    def __init__(self, node_id: str, **kwargs: Any) -> None:
        """
        Initialize feature scaler node.

        Args:
            node_id: Unique identifier for the node
            **kwargs: Additional configuration parameters
        """
        super().__init__(
            node_id=node_id,
            node_type=NodeType.FEATURE_ENGINEER,
            name=kwargs.get("name", "Feature Scaler"),
            description=kwargs.get(
                "description", "Scale features using StandardScaler"
            ),
            config=kwargs,
        )
        self.scaler = StandardScaler()

    def execute(self, inputs: dict[str, NodeOutput]) -> NodeOutput:
        """Scale the input features."""
        if "data" not in inputs:
            msg = "FeatureScalerNode requires 'data' input"
            raise ValueError(msg)

        data = inputs["data"].data

        # Handle train/test split data
        if isinstance(data, dict) and "X_train" in data:
            X_train = data["X_train"]
            X_test = data["X_test"]

            # Fit scaler on training data and transform both
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index,
            )
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index,
            )

            scaled_data = {
                **data,
                "X_train": X_train_scaled,
                "X_test": X_test_scaled,
            }

            self.metrics.rows_processed = len(X_train) + len(X_test)
            return NodeOutput(
                data=scaled_data,
                metadata={
                    "scaler": "StandardScaler",
                    "train_size": len(X_train_scaled),
                    "test_size": len(X_test_scaled),
                },
            )

        # Handle single DataFrame
        if isinstance(data, pd.DataFrame):
            scaled_array = self.scaler.fit_transform(data)
            scaled_data = pd.DataFrame(
                scaled_array, columns=data.columns, index=data.index
            )
            self.metrics.rows_processed = len(scaled_data)
            return NodeOutput(
                data=scaled_data,
                metadata={"scaler": "StandardScaler", "rows": len(scaled_data)},
            )

        msg = f"Unsupported data type: {type(data)}"
        raise TypeError(msg)


class ModelTrainerNode(BaseNode):
    """Node for training ML models."""

    def __init__(
        self,
        node_id: str,
        model_type: str = "xgboost",
        model_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize model trainer node.

        Args:
            node_id: Unique identifier for the node
            model_type: Type of model to train
            model_params: Parameters for the model
            **kwargs: Additional configuration parameters
        """
        config = {
            "model_type": model_type,
            "model_params": model_params or {},
            **kwargs,
        }
        super().__init__(
            node_id=node_id,
            node_type=NodeType.MODEL_TRAINER,
            name=kwargs.get("name", f"{model_type} Trainer"),
            description=kwargs.get("description", f"Train {model_type} model"),
            config=config,
        )
        self.model = None

    def execute(self, inputs: dict[str, NodeOutput]) -> NodeOutput:
        """Train the model on the input data."""
        if "data" not in inputs:
            msg = "ModelTrainerNode requires 'data' input"
            raise ValueError(msg)

        data = inputs["data"].data
        if not isinstance(data, dict) or "X_train" not in data or "y_train" not in data:
            msg = "ModelTrainerNode requires train/test split data"
            raise ValueError(msg)

        X_train = data["X_train"]
        y_train = data["y_train"]

        # Import and create model based on type
        model_type = self.config.get("model_type", "xgboost")
        model_params = self.config.get("model_params", {})

        from src.core.ml_engine import ModelRegistry

        # Get model class from registry
        registry = ModelRegistry()
        task_type = self.config.get("task_type", "classification")

        if model_type == "xgboost":
            if task_type == "classification":
                model_class = registry.get_model("xgboost_classifier")
            else:
                model_class = registry.get_model("xgboost_regressor")
        elif model_type == "lightgbm":
            if task_type == "classification":
                model_class = registry.get_model("lightgbm_classifier")
            else:
                model_class = registry.get_model("lightgbm_regressor")
        elif model_type == "catboost":
            if task_type == "classification":
                model_class = registry.get_model("catboost_classifier")
            else:
                model_class = registry.get_model("catboost_regressor")
        else:
            msg = f"Unsupported model type: {model_type}"
            raise ValueError(msg)

        # Create and train model
        self.model = model_class(**model_params)
        self.model.fit(X_train, y_train)

        # Get training metrics
        train_score = self.model.score(X_train, y_train)

        self.metrics.rows_processed = len(X_train)
        self.metrics.custom_metrics["train_score"] = train_score

        return NodeOutput(
            data={**data, "model": self.model},
            metadata={
                "model_type": model_type,
                "task_type": task_type,
                "train_score": train_score,
                "train_size": len(X_train),
            },
        )


class ModelEvaluatorNode(BaseNode):
    """Node for evaluating trained models."""

    def __init__(self, node_id: str, **kwargs: Any) -> None:
        """
        Initialize model evaluator node.

        Args:
            node_id: Unique identifier for the node
            **kwargs: Additional configuration parameters
        """
        super().__init__(
            node_id=node_id,
            node_type=NodeType.MODEL_EVALUATOR,
            name=kwargs.get("name", "Model Evaluator"),
            description=kwargs.get("description", "Evaluate model performance"),
            config=kwargs,
        )

    def execute(self, inputs: dict[str, NodeOutput]) -> NodeOutput:
        """Evaluate the model on test data."""
        if "data" not in inputs:
            msg = "ModelEvaluatorNode requires 'data' input"
            raise ValueError(msg)

        data = inputs["data"].data
        if (
            not isinstance(data, dict)
            or "model" not in data
            or "X_test" not in data
            or "y_test" not in data
        ):
            msg = "ModelEvaluatorNode requires trained model and test data"
            raise ValueError(msg)

        model = data["model"]
        X_test = data["X_test"]
        y_test = data["y_test"]

        # Evaluate model
        test_score = model.score(X_test, y_test)
        predictions = model.predict(X_test)

        # Get feature importance if available
        feature_importance = None
        if hasattr(model, "get_feature_importance"):
            feature_importance = model.get_feature_importance()

        self.metrics.rows_processed = len(X_test)
        self.metrics.custom_metrics["test_score"] = test_score

        return NodeOutput(
            data={
                **data,
                "predictions": predictions,
                "test_score": test_score,
                "feature_importance": feature_importance,
            },
            metadata={
                "test_score": test_score,
                "test_size": len(X_test),
                "has_feature_importance": feature_importance is not None,
            },
        )


class ModelSaverNode(BaseNode):
    """Node for saving trained models to disk."""

    def __init__(
        self,
        node_id: str,
        save_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize model saver node.

        Args:
            node_id: Unique identifier for the node
            save_path: Path where model should be saved
            **kwargs: Additional configuration parameters
        """
        config = {"save_path": save_path, **kwargs}
        super().__init__(
            node_id=node_id,
            node_type=NodeType.MODEL_SAVER,
            name=kwargs.get("name", "Model Saver"),
            description=kwargs.get("description", "Save trained model to disk"),
            config=config,
        )

    def execute(self, inputs: dict[str, NodeOutput]) -> NodeOutput:
        """Save the model to disk."""
        if "data" not in inputs:
            msg = "ModelSaverNode requires 'data' input"
            raise ValueError(msg)

        data = inputs["data"].data
        if not isinstance(data, dict) or "model" not in data:
            msg = "ModelSaverNode requires trained model in data"
            raise ValueError(msg)

        model = data["model"]
        save_path = self.config.get("save_path")

        if not save_path:
            msg = "save_path must be specified in config"
            raise ValueError(msg)

        # Create directory if it doesn't exist
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        with save_path_obj.open("wb") as f:
            pickle.dump(model, f)

        return NodeOutput(
            data=data,
            metadata={"save_path": str(save_path), "saved": True},
        )
