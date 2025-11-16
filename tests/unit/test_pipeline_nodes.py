"""
Tests for pipeline nodes.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.pipeline_orchestration.nodes import (
    DataLoaderNode,
    DataPreprocessorNode,
    FeatureScalerNode,
    ModelEvaluatorNode,
    ModelSaverNode,
    ModelTrainerNode,
    NodeOutput,
    NodeStatus,
    NodeType,
    TrainTestSplitNode,
)


class TestDataLoaderNode:
    """Tests for DataLoaderNode."""

    def test_load_csv(self, tmp_path):
        """Test loading CSV file."""
        # Create test CSV
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        csv_path = tmp_path / "test.csv"
        data.to_csv(csv_path, index=False)

        # Create node and execute
        node = DataLoaderNode(
            node_id="loader1",
            source_type="file",
            source_path=str(csv_path),
        )

        output = node.run({})

        assert node.status == NodeStatus.COMPLETED
        assert isinstance(output.data, pd.DataFrame)
        assert len(output.data) == 3
        assert list(output.data.columns) == ["a", "b"]

    def test_load_parquet(self, tmp_path):
        """Test loading Parquet file."""
        # Create test Parquet
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        parquet_path = tmp_path / "test.parquet"
        data.to_parquet(parquet_path, index=False)

        # Create node and execute
        node = DataLoaderNode(
            node_id="loader1",
            source_type="file",
            source_path=str(parquet_path),
        )

        output = node.run({})

        assert node.status == NodeStatus.COMPLETED
        assert isinstance(output.data, pd.DataFrame)
        assert len(output.data) == 3

    def test_file_not_found(self, tmp_path):
        """Test error handling for missing file."""
        node = DataLoaderNode(
            node_id="loader1",
            source_type="file",
            source_path=str(tmp_path / "missing.csv"),
        )

        with pytest.raises(FileNotFoundError):
            node.run({})

        assert node.status == NodeStatus.FAILED


class TestDataPreprocessorNode:
    """Tests for DataPreprocessorNode."""

    def test_drop_missing(self):
        """Test dropping missing values."""
        # Create data with missing values
        data = pd.DataFrame({"a": [1, 2, None, 4], "b": [5, None, 7, 8]})
        input_output = NodeOutput(data=data)

        node = DataPreprocessorNode(
            node_id="prep1",
            operations=["drop_missing"],
        )

        output = node.run({"data": input_output})

        assert node.status == NodeStatus.COMPLETED
        assert len(output.data) == 2  # Only 2 rows have no missing values

    def test_fill_missing_mean(self):
        """Test filling missing values with mean."""
        data = pd.DataFrame({"a": [1.0, 2.0, None, 4.0], "b": [5.0, None, 7.0, 8.0]})
        input_output = NodeOutput(data=data)

        node = DataPreprocessorNode(
            node_id="prep1",
            operations=["fill_missing_mean"],
        )

        output = node.run({"data": input_output})

        assert node.status == NodeStatus.COMPLETED
        assert not output.data.isnull().any().any()

    def test_remove_duplicates(self):
        """Test removing duplicate rows."""
        data = pd.DataFrame({"a": [1, 2, 1, 3], "b": [4, 5, 4, 6]})
        input_output = NodeOutput(data=data)

        node = DataPreprocessorNode(
            node_id="prep1",
            operations=["remove_duplicates"],
        )

        output = node.run({"data": input_output})

        assert node.status == NodeStatus.COMPLETED
        assert len(output.data) == 3  # One duplicate removed


class TestTrainTestSplitNode:
    """Tests for TrainTestSplitNode."""

    def test_train_test_split(self):
        """Test train-test split."""
        data = pd.DataFrame(
            {
                "feature1": np.random.rand(100),
                "feature2": np.random.rand(100),
                "target": np.random.randint(0, 2, 100),
            }
        )
        input_output = NodeOutput(data=data)

        node = TrainTestSplitNode(
            node_id="split1",
            test_size=0.2,
            target_column="target",
            random_state=42,
        )

        output = node.run({"data": input_output})

        assert node.status == NodeStatus.COMPLETED
        assert "X_train" in output.data
        assert "X_test" in output.data
        assert "y_train" in output.data
        assert "y_test" in output.data

        assert len(output.data["X_train"]) == 80
        assert len(output.data["X_test"]) == 20

    def test_missing_target_column(self):
        """Test error when target column is missing."""
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        input_output = NodeOutput(data=data)

        node = TrainTestSplitNode(
            node_id="split1",
            target_column="missing_column",
        )

        with pytest.raises(ValueError):
            node.run({"data": input_output})


class TestFeatureScalerNode:
    """Tests for FeatureScalerNode."""

    def test_scale_single_dataframe(self):
        """Test scaling a single DataFrame."""
        # Use larger sample size for more stable scaling statistics
        np.random.seed(42)
        data = pd.DataFrame({
            "a": np.random.randn(100) * 10 + 50,
            "b": np.random.randn(100) * 5 + 20,
        })
        input_output = NodeOutput(data=data)

        node = FeatureScalerNode(node_id="scaler1")
        output = node.run({"data": input_output})

        assert node.status == NodeStatus.COMPLETED
        assert isinstance(output.data, pd.DataFrame)

        # Check that mean is approximately 0 and std is approximately 1
        assert abs(output.data.mean().mean()) < 0.01
        assert abs(output.data.std().mean() - 1.0) < 0.01

    def test_scale_train_test_split(self):
        """Test scaling train and test sets."""
        X_train = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
        X_test = pd.DataFrame({"a": [5], "b": [50]})
        y_train = pd.Series([0, 1, 0, 1])
        y_test = pd.Series([0])

        input_data = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }
        input_output = NodeOutput(data=input_data)

        node = FeatureScalerNode(node_id="scaler1")
        output = node.run({"data": input_output})

        assert node.status == NodeStatus.COMPLETED
        assert "X_train" in output.data
        assert "X_test" in output.data
        assert isinstance(output.data["X_train"], pd.DataFrame)
        assert isinstance(output.data["X_test"], pd.DataFrame)


class TestModelTrainerNode:
    """Tests for ModelTrainerNode."""

    def test_train_xgboost_classifier(self):
        """Test training XGBoost classifier."""
        # Create simple dataset
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.rand(50, 3), columns=["a", "b", "c"])
        y_train = pd.Series(np.random.randint(0, 2, 50))
        X_test = pd.DataFrame(np.random.rand(10, 3), columns=["a", "b", "c"])
        y_test = pd.Series(np.random.randint(0, 2, 10))

        input_data = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }
        input_output = NodeOutput(data=input_data)

        node = ModelTrainerNode(
            node_id="trainer1",
            model_type="xgboost",
            task_type="classification",
            model_params={"n_estimators": 10, "max_depth": 3},
        )

        output = node.run({"data": input_output})

        assert node.status == NodeStatus.COMPLETED
        assert "model" in output.data
        assert hasattr(output.data["model"], "predict")
        assert output.metadata["train_score"] > 0


class TestModelEvaluatorNode:
    """Tests for ModelEvaluatorNode."""

    def test_evaluate_model(self):
        """Test model evaluation."""
        # Create simple dataset and train model
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.rand(50, 3), columns=["a", "b", "c"])
        y_train = pd.Series(np.random.randint(0, 2, 50))
        X_test = pd.DataFrame(np.random.rand(10, 3), columns=["a", "b", "c"])
        y_test = pd.Series(np.random.randint(0, 2, 10))

        # Train a model first
        from src.core.ml_engine import XGBoostClassifier

        model = XGBoostClassifier(n_estimators=10, max_depth=3)
        model.fit(X_train, y_train)

        input_data = {
            "model": model,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }
        input_output = NodeOutput(data=input_data)

        node = ModelEvaluatorNode(node_id="eval1")
        output = node.run({"data": input_output})

        assert node.status == NodeStatus.COMPLETED
        assert "predictions" in output.data
        assert "test_score" in output.data
        assert output.metadata["test_score"] > 0


class TestModelSaverNode:
    """Tests for ModelSaverNode."""

    def test_save_model(self, tmp_path):
        """Test saving model to disk."""
        # Create and train a simple model
        from src.core.ml_engine import XGBoostClassifier

        np.random.seed(42)
        X = pd.DataFrame(np.random.rand(50, 3), columns=["a", "b", "c"])
        y = pd.Series(np.random.randint(0, 2, 50))

        model = XGBoostClassifier(n_estimators=10)
        model.fit(X, y)

        # Save model
        save_path = tmp_path / "model.pkl"
        input_data = {"model": model}
        input_output = NodeOutput(data=input_data)

        node = ModelSaverNode(
            node_id="saver1",
            save_path=str(save_path),
        )

        output = node.run({"data": input_output})

        assert node.status == NodeStatus.COMPLETED
        assert save_path.exists()
        assert output.metadata["saved"] is True
