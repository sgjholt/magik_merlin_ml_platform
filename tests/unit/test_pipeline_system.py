"""
Tests for pipeline system (Pipeline, Executor, Scheduler, Storage).
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.pipeline_orchestration import (
    DataLoaderNode,
    DataPreprocessorNode,
    FeatureScalerNode,
    ModelEvaluatorNode,
    ModelTrainerNode,
    Pipeline,
    PipelineExecutor,
    PipelineScheduler,
    PipelineStatus,
    PipelineStorage,
    ScheduleConfig,
    TrainTestSplitNode,
)


class TestPipeline:
    """Tests for Pipeline class."""

    def test_create_pipeline(self):
        """Test creating a pipeline."""
        pipeline = Pipeline(
            pipeline_id="test_pipeline",
            name="Test Pipeline",
            description="A test pipeline",
        )

        assert pipeline.pipeline_id == "test_pipeline"
        assert pipeline.name == "Test Pipeline"
        assert pipeline.status == PipelineStatus.PENDING

    def test_add_node(self):
        """Test adding nodes to pipeline."""
        pipeline = Pipeline(
            pipeline_id="test_pipeline",
            name="Test Pipeline",
        )

        node1 = DataLoaderNode(node_id="loader", source_type="file")
        pipeline.add_node(node1)

        assert len(pipeline.nodes) == 1
        assert "loader" in pipeline.nodes

    def test_add_edge(self):
        """Test adding edges between nodes."""
        pipeline = Pipeline(
            pipeline_id="test_pipeline",
            name="Test Pipeline",
        )

        node1 = DataLoaderNode(node_id="loader", source_type="file")
        node2 = DataPreprocessorNode(node_id="prep", operations=[])

        pipeline.add_node(node1)
        pipeline.add_node(node2)
        pipeline.add_edge("loader", "prep")

        assert len(pipeline.edges) == 1
        assert pipeline.edges[0].source_node_id == "loader"
        assert pipeline.edges[0].target_node_id == "prep"

    def test_validate_pipeline(self):
        """Test pipeline validation."""
        pipeline = Pipeline(
            pipeline_id="test_pipeline",
            name="Test Pipeline",
        )

        # Empty pipeline is invalid
        is_valid, errors = pipeline.validate()
        assert not is_valid
        assert len(errors) > 0

        # Add a node
        node1 = DataLoaderNode(node_id="loader", source_type="file")
        pipeline.add_node(node1)

        # Single node pipeline is valid
        is_valid, errors = pipeline.validate()
        assert is_valid

    def test_cycle_detection(self):
        """Test that cycles are detected."""
        pipeline = Pipeline(
            pipeline_id="test_pipeline",
            name="Test Pipeline",
        )

        node1 = DataLoaderNode(node_id="node1", source_type="file")
        node2 = DataPreprocessorNode(node_id="node2", operations=[])
        node3 = DataPreprocessorNode(node_id="node3", operations=[])

        pipeline.add_node(node1)
        pipeline.add_node(node2)
        pipeline.add_node(node3)

        pipeline.add_edge("node1", "node2")
        pipeline.add_edge("node2", "node3")

        # Adding edge that creates cycle should raise error
        with pytest.raises(ValueError):
            pipeline.add_edge("node3", "node1")

    def test_compute_execution_order(self):
        """Test topological sort for execution order."""
        pipeline = Pipeline(
            pipeline_id="test_pipeline",
            name="Test Pipeline",
        )

        node1 = DataLoaderNode(node_id="node1", source_type="file")
        node2 = DataPreprocessorNode(node_id="node2", operations=[])
        node3 = DataPreprocessorNode(node_id="node3", operations=[])

        pipeline.add_node(node1)
        pipeline.add_node(node2)
        pipeline.add_node(node3)

        pipeline.add_edge("node1", "node2")
        pipeline.add_edge("node2", "node3")

        execution_order = pipeline.compute_execution_order()

        assert execution_order == ["node1", "node2", "node3"]

    def test_pipeline_to_dict(self):
        """Test converting pipeline to dictionary."""
        pipeline = Pipeline(
            pipeline_id="test_pipeline",
            name="Test Pipeline",
        )

        node1 = DataLoaderNode(node_id="loader", source_type="file")
        pipeline.add_node(node1)

        pipeline_dict = pipeline.to_dict()

        assert pipeline_dict["pipeline_id"] == "test_pipeline"
        assert pipeline_dict["name"] == "Test Pipeline"
        assert len(pipeline_dict["nodes"]) == 1


class TestPipelineExecutor:
    """Tests for PipelineExecutor."""

    def test_execute_simple_pipeline(self, tmp_path):
        """Test executing a simple pipeline."""
        # Create test data
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        csv_path = tmp_path / "test.csv"
        data.to_csv(csv_path, index=False)

        # Create pipeline
        pipeline = Pipeline(
            pipeline_id="test_pipeline",
            name="Test Pipeline",
        )

        # Add nodes
        loader = DataLoaderNode(
            node_id="loader",
            source_type="file",
            source_path=str(csv_path),
        )
        prep = DataPreprocessorNode(
            node_id="prep",
            operations=["remove_duplicates"],
        )

        pipeline.add_node(loader)
        pipeline.add_node(prep)
        pipeline.add_edge("loader", "prep")

        # Execute pipeline
        executor = PipelineExecutor()
        result = executor.execute(pipeline)

        assert result.status == PipelineStatus.COMPLETED
        assert result.duration > 0
        assert "loader" in result.node_outputs
        assert "prep" in result.node_outputs

    def test_execute_ml_pipeline(self, tmp_path):
        """Test executing a complete ML pipeline."""
        # Create test data
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "feature1": np.random.rand(50),
                "feature2": np.random.rand(50),
                "target": np.random.randint(0, 2, 50),
            }
        )
        csv_path = tmp_path / "train.csv"
        data.to_csv(csv_path, index=False)

        # Create pipeline
        pipeline = Pipeline(
            pipeline_id="ml_pipeline",
            name="ML Pipeline",
        )

        # Add nodes
        loader = DataLoaderNode(
            node_id="loader",
            source_type="file",
            source_path=str(csv_path),
        )

        splitter = TrainTestSplitNode(
            node_id="splitter",
            test_size=0.2,
            target_column="target",
            random_state=42,
        )

        scaler = FeatureScalerNode(node_id="scaler")

        trainer = ModelTrainerNode(
            node_id="trainer",
            model_type="xgboost",
            task_type="classification",
            model_params={"n_estimators": 10, "max_depth": 3},
        )

        evaluator = ModelEvaluatorNode(node_id="evaluator")

        # Add nodes to pipeline
        pipeline.add_node(loader)
        pipeline.add_node(splitter)
        pipeline.add_node(scaler)
        pipeline.add_node(trainer)
        pipeline.add_node(evaluator)

        # Add edges
        pipeline.add_edge("loader", "splitter")
        pipeline.add_edge("splitter", "scaler")
        pipeline.add_edge("scaler", "trainer")
        pipeline.add_edge("trainer", "evaluator")

        # Execute pipeline
        executor = PipelineExecutor()
        result = executor.execute(pipeline)

        assert result.status == PipelineStatus.COMPLETED
        assert "evaluator" in result.node_outputs
        assert "test_score" in result.node_outputs["evaluator"].data

    def test_pipeline_with_error(self, tmp_path):
        """Test pipeline execution with error."""
        pipeline = Pipeline(
            pipeline_id="error_pipeline",
            name="Error Pipeline",
        )

        # Add node that will fail (missing file)
        loader = DataLoaderNode(
            node_id="loader",
            source_type="file",
            source_path=str(tmp_path / "missing.csv"),
        )

        pipeline.add_node(loader)

        # Execute pipeline
        executor = PipelineExecutor()
        result = executor.execute(pipeline)

        assert result.status == PipelineStatus.FAILED
        assert result.error is not None


class TestPipelineStorage:
    """Tests for PipelineStorage."""

    def test_save_and_load_pipeline(self, tmp_path):
        """Test saving and loading pipeline."""
        storage = PipelineStorage(storage_dir=tmp_path / "pipelines")

        # Create pipeline
        pipeline = Pipeline(
            pipeline_id="test_pipeline",
            name="Test Pipeline",
            description="Test description",
        )

        node1 = DataLoaderNode(node_id="loader", source_type="file")
        pipeline.add_node(node1)

        # Save pipeline
        save_path = storage.save_pipeline(pipeline)
        assert save_path.exists()

        # Load pipeline
        loaded_pipeline = storage.load_pipeline("test_pipeline")
        assert loaded_pipeline.pipeline_id == "test_pipeline"
        assert loaded_pipeline.name == "Test Pipeline"

    def test_list_pipelines(self, tmp_path):
        """Test listing pipelines."""
        storage = PipelineStorage(storage_dir=tmp_path / "pipelines")

        # Create and save multiple pipelines
        for i in range(3):
            pipeline = Pipeline(
                pipeline_id=f"pipeline_{i}",
                name=f"Pipeline {i}",
            )
            storage.save_pipeline(pipeline)

        # List pipelines
        pipelines = storage.list_pipelines()
        assert len(pipelines) == 3

    def test_delete_pipeline(self, tmp_path):
        """Test deleting pipeline."""
        storage = PipelineStorage(storage_dir=tmp_path / "pipelines")

        # Create and save pipeline
        pipeline = Pipeline(
            pipeline_id="test_pipeline",
            name="Test Pipeline",
        )
        storage.save_pipeline(pipeline)

        # Delete pipeline
        result = storage.delete_pipeline("test_pipeline")
        assert result is True

        # Try to load deleted pipeline
        with pytest.raises(FileNotFoundError):
            storage.load_pipeline("test_pipeline")

    def test_save_execution_result(self, tmp_path):
        """Test saving execution result."""
        from datetime import datetime

        from src.core.pipeline_orchestration.executor import ExecutionResult

        storage = PipelineStorage(storage_dir=tmp_path / "pipelines")

        result = ExecutionResult(
            pipeline_id="test_pipeline",
            status=PipelineStatus.COMPLETED,
            start_time=datetime.now(),
            duration=10.5,
        )

        # Save result
        result_path = storage.save_execution_result(result)
        assert result_path.exists()

    def test_list_executions(self, tmp_path):
        """Test listing executions."""
        from datetime import datetime

        from src.core.pipeline_orchestration.executor import ExecutionResult

        storage = PipelineStorage(storage_dir=tmp_path / "pipelines")

        # Save multiple execution results
        for i in range(3):
            result = ExecutionResult(
                pipeline_id="test_pipeline",
                status=PipelineStatus.COMPLETED,
                start_time=datetime.now(),
            )
            storage.save_execution_result(result)

        # List executions
        executions = storage.list_executions("test_pipeline")
        assert len(executions) == 3


class TestPipelineScheduler:
    """Tests for PipelineScheduler."""

    def test_schedule_pipeline(self, tmp_path):
        """Test scheduling a pipeline."""
        from datetime import datetime, timedelta

        storage = PipelineStorage(storage_dir=tmp_path / "pipelines")
        executor = PipelineExecutor()
        scheduler = PipelineScheduler(executor)

        # Create simple pipeline
        pipeline = Pipeline(
            pipeline_id="scheduled_pipeline",
            name="Scheduled Pipeline",
        )

        # Schedule for future
        schedule = ScheduleConfig(
            schedule_type="once",
            start_time=datetime.now() + timedelta(hours=1),
            enabled=True,
        )

        scheduler.schedule_pipeline(pipeline, schedule)

        # Check that pipeline is scheduled
        scheduled = scheduler.get_schedule("scheduled_pipeline")
        assert scheduled is not None
        assert scheduled.pipeline_id == "scheduled_pipeline"

    def test_unschedule_pipeline(self):
        """Test unscheduling a pipeline."""
        executor = PipelineExecutor()
        scheduler = PipelineScheduler(executor)

        pipeline = Pipeline(
            pipeline_id="test_pipeline",
            name="Test Pipeline",
        )

        schedule = ScheduleConfig(schedule_type="interval", interval_seconds=3600)
        scheduler.schedule_pipeline(pipeline, schedule)

        # Unschedule
        result = scheduler.unschedule_pipeline("test_pipeline")
        assert result is True

        # Check that pipeline is no longer scheduled
        scheduled = scheduler.get_schedule("test_pipeline")
        assert scheduled is None

    def test_list_scheduled_pipelines(self):
        """Test listing scheduled pipelines."""
        executor = PipelineExecutor()
        scheduler = PipelineScheduler(executor)

        # Schedule multiple pipelines
        for i in range(3):
            pipeline = Pipeline(
                pipeline_id=f"pipeline_{i}",
                name=f"Pipeline {i}",
            )
            schedule = ScheduleConfig(schedule_type="interval", interval_seconds=3600)
            scheduler.schedule_pipeline(pipeline, schedule)

        # List scheduled pipelines
        scheduled = scheduler.list_scheduled_pipelines()
        assert len(scheduled) == 3
