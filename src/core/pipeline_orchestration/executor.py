"""
Pipeline execution engine.

This module provides the PipelineExecutor class for running pipelines and
tracking execution progress.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.core.logging import get_logger

from .nodes import NodeOutput, NodeStatus
from .pipeline import Pipeline, PipelineStatus

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    """Result of pipeline execution."""

    pipeline_id: str
    status: PipelineStatus
    start_time: datetime
    end_time: datetime | None = None
    duration: float = 0.0
    node_outputs: dict[str, NodeOutput] = field(default_factory=dict)
    node_errors: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "pipeline_id": self.pipeline_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "node_errors": self.node_errors,
            "error": self.error,
        }


class PipelineExecutor:
    """
    Execute pipelines and track progress.

    The executor runs pipelines according to their computed execution order,
    passing data between nodes and handling errors.
    """

    def __init__(self) -> None:
        """Initialize pipeline executor."""
        self.logger = get_logger(__name__)
        self.running_pipelines: dict[str, Pipeline] = {}
        self.execution_results: dict[str, ExecutionResult] = {}
        self._stop_flags: dict[str, bool] = {}

    def execute(
        self,
        pipeline: Pipeline,
        *,
        async_mode: bool = False,
        progress_callback: Callable[[str, str, float], None] | None = None,
    ) -> ExecutionResult:
        """
        Execute a pipeline.

        Args:
            pipeline: The pipeline to execute
            async_mode: If True, run in background thread
            progress_callback: Optional callback for progress updates
                               (pipeline_id, node_id, progress_pct)

        Returns:
            ExecutionResult containing execution details

        Raises:
            ValueError: If pipeline is invalid
        """
        # Validate pipeline
        is_valid, errors = pipeline.validate()
        if not is_valid:
            msg = f"Invalid pipeline: {', '.join(errors)}"
            raise ValueError(msg)

        # Compute execution order
        try:
            execution_order = pipeline.compute_execution_order()
        except ValueError as e:
            msg = f"Failed to compute execution order: {e}"
            raise ValueError(msg) from e

        if async_mode:
            # Run in background thread
            def run_async() -> None:
                self._execute_pipeline(pipeline, execution_order, progress_callback)

            thread = threading.Thread(target=run_async, daemon=True)
            thread.start()

            # Return placeholder result
            return ExecutionResult(
                pipeline_id=pipeline.pipeline_id,
                status=PipelineStatus.RUNNING,
                start_time=datetime.now(),
            )

        # Run synchronously
        return self._execute_pipeline(pipeline, execution_order, progress_callback)

    def _execute_pipeline(
        self,
        pipeline: Pipeline,
        execution_order: list[str],
        progress_callback: Callable[[str, str, float], None] | None = None,
    ) -> ExecutionResult:
        """
        Internal method to execute pipeline.

        Args:
            pipeline: Pipeline to execute
            execution_order: Ordered list of node IDs
            progress_callback: Optional progress callback

        Returns:
            ExecutionResult
        """
        start_time = datetime.now()
        pipeline.status = PipelineStatus.RUNNING
        self.running_pipelines[pipeline.pipeline_id] = pipeline
        self._stop_flags[pipeline.pipeline_id] = False

        result = ExecutionResult(
            pipeline_id=pipeline.pipeline_id,
            status=PipelineStatus.RUNNING,
            start_time=start_time,
        )

        self.logger.info(f"Starting pipeline execution: {pipeline.name}")

        # Store node outputs for passing between nodes
        node_outputs: dict[str, NodeOutput] = {}

        try:
            total_nodes = len(execution_order)
            for idx, node_id in enumerate(execution_order):
                # Check if execution should stop
                if self._stop_flags.get(pipeline.pipeline_id, False):
                    self.logger.warning(f"Pipeline {pipeline.pipeline_id} cancelled")
                    pipeline.status = PipelineStatus.CANCELLED
                    result.status = PipelineStatus.CANCELLED
                    break

                node = pipeline.nodes[node_id]

                # Report progress
                progress_pct = (idx / total_nodes) * 100
                if progress_callback:
                    progress_callback(pipeline.pipeline_id, node_id, progress_pct)

                # Collect inputs for this node
                node_inputs = {}
                input_edges = pipeline.get_node_inputs(node_id)

                for edge in input_edges:
                    if edge.source_node_id in node_outputs:
                        node_inputs[edge.input_name] = node_outputs[edge.source_node_id]

                # Execute node
                try:
                    self.logger.info(f"Executing node: {node.name} ({node_id})")
                    output = node.run(node_inputs)
                    node_outputs[node_id] = output
                    result.node_outputs[node_id] = output

                except Exception as e:
                    self.logger.exception(f"Node {node_id} failed: {e}")
                    node.status = NodeStatus.FAILED
                    result.node_errors[node_id] = str(e)
                    pipeline.status = PipelineStatus.FAILED
                    result.status = PipelineStatus.FAILED
                    result.error = f"Node {node_id} failed: {e}"
                    break

            # If no errors and not cancelled, mark as completed
            if result.status == PipelineStatus.RUNNING:
                pipeline.status = PipelineStatus.COMPLETED
                result.status = PipelineStatus.COMPLETED
                self.logger.info(f"Pipeline {pipeline.name} completed successfully")

        except Exception as e:
            self.logger.exception(f"Pipeline execution failed: {e}")
            pipeline.status = PipelineStatus.FAILED
            result.status = PipelineStatus.FAILED
            result.error = str(e)

        finally:
            end_time = datetime.now()
            result.end_time = end_time
            result.duration = (end_time - start_time).total_seconds()

            # Clean up
            if pipeline.pipeline_id in self.running_pipelines:
                del self.running_pipelines[pipeline.pipeline_id]
            if pipeline.pipeline_id in self._stop_flags:
                del self._stop_flags[pipeline.pipeline_id]

            self.execution_results[pipeline.pipeline_id] = result

            # Final progress callback
            if progress_callback:
                progress_callback(pipeline.pipeline_id, "COMPLETED", 100.0)

        return result

    def cancel_pipeline(self, pipeline_id: str) -> bool:
        """
        Cancel a running pipeline.

        Args:
            pipeline_id: ID of pipeline to cancel

        Returns:
            True if pipeline was cancelled, False if not running
        """
        if pipeline_id in self.running_pipelines:
            self._stop_flags[pipeline_id] = True
            self.logger.info(f"Cancelling pipeline: {pipeline_id}")
            return True

        self.logger.warning(f"Pipeline {pipeline_id} is not running")
        return False

    def get_result(self, pipeline_id: str) -> ExecutionResult | None:
        """
        Get execution result for a pipeline.

        Args:
            pipeline_id: ID of the pipeline

        Returns:
            ExecutionResult if available, None otherwise
        """
        return self.execution_results.get(pipeline_id)

    def is_running(self, pipeline_id: str) -> bool:
        """
        Check if a pipeline is currently running.

        Args:
            pipeline_id: ID of the pipeline

        Returns:
            True if pipeline is running
        """
        return pipeline_id in self.running_pipelines

    def clear_results(self, pipeline_id: str | None = None) -> None:
        """
        Clear execution results.

        Args:
            pipeline_id: ID of specific pipeline to clear, or None to clear all
        """
        if pipeline_id:
            if pipeline_id in self.execution_results:
                del self.execution_results[pipeline_id]
                self.logger.info(f"Cleared results for pipeline: {pipeline_id}")
        else:
            self.execution_results.clear()
            self.logger.info("Cleared all execution results")
