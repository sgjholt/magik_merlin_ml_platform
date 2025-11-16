"""
Pipeline definition and management system.

This module provides the Pipeline class for defining, validating, and managing
ML pipelines composed of interconnected nodes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from src.core.logging import get_logger

from .nodes import BaseNode

logger = get_logger(__name__)


class PipelineStatus(Enum):
    """Status of a pipeline."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineEdge:
    """
    Connection between two nodes in a pipeline.

    Represents data flow from source node to target node.
    """

    source_node_id: str
    target_node_id: str
    output_name: str = "data"  # Name of the output from source node
    input_name: str = "data"  # Name of the input to target node


@dataclass
class PipelineMetadata:
    """Metadata for a pipeline."""

    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)
    description: str = ""


class Pipeline:
    """
    ML Pipeline composed of interconnected nodes.

    A pipeline defines a directed acyclic graph (DAG) of nodes that process
    data sequentially or in parallel.
    """

    def __init__(
        self,
        pipeline_id: str,
        name: str,
        description: str = "",
        metadata: PipelineMetadata | None = None,
    ) -> None:
        """
        Initialize a pipeline.

        Args:
            pipeline_id: Unique identifier for the pipeline
            name: Human-readable name
            description: Description of the pipeline
            metadata: Pipeline metadata
        """
        self.pipeline_id = pipeline_id
        self.name = name
        self.description = description
        self.metadata = metadata or PipelineMetadata()
        self.nodes: dict[str, BaseNode] = {}
        self.edges: list[PipelineEdge] = []
        self.status = PipelineStatus.PENDING
        self.execution_order: list[str] = []
        self.logger = get_logger(f"{__name__}.{pipeline_id}")

    def add_node(self, node: BaseNode) -> None:
        """
        Add a node to the pipeline.

        Args:
            node: The node to add

        Raises:
            ValueError: If a node with the same ID already exists
        """
        if node.node_id in self.nodes:
            msg = f"Node with ID '{node.node_id}' already exists"
            raise ValueError(msg)

        self.nodes[node.node_id] = node
        self.logger.info(f"Added node: {node.name} ({node.node_id})")
        self.metadata.updated_at = datetime.now()

    def add_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        output_name: str = "data",
        input_name: str = "data",
    ) -> None:
        """
        Add an edge connecting two nodes.

        Args:
            source_node_id: ID of the source node
            target_node_id: ID of the target node
            output_name: Name of the output from source node
            input_name: Name of the input to target node

        Raises:
            ValueError: If either node doesn't exist or edge would create a cycle
        """
        if source_node_id not in self.nodes:
            msg = f"Source node '{source_node_id}' not found"
            raise ValueError(msg)

        if target_node_id not in self.nodes:
            msg = f"Target node '{target_node_id}' not found"
            raise ValueError(msg)

        # Check for cycles (simplified check)
        if self._would_create_cycle(source_node_id, target_node_id):
            msg = f"Adding edge from '{source_node_id}' to '{target_node_id}' would create a cycle"
            raise ValueError(msg)

        edge = PipelineEdge(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            output_name=output_name,
            input_name=input_name,
        )
        self.edges.append(edge)
        self.logger.info(f"Added edge: {source_node_id} -> {target_node_id}")
        self.metadata.updated_at = datetime.now()

    def _would_create_cycle(self, source_id: str, target_id: str) -> bool:
        """
        Check if adding an edge would create a cycle.

        Args:
            source_id: Source node ID
            target_id: Target node ID

        Returns:
            True if edge would create a cycle
        """
        # Build adjacency list
        adj_list: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
        for edge in self.edges:
            adj_list[edge.source_node_id].append(edge.target_node_id)

        # Add the proposed edge
        adj_list[source_id].append(target_id)

        # DFS to detect cycle
        visited = set()
        rec_stack = set()

        def has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            for neighbor in adj_list.get(node_id, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node_id)
            return False

        for node_id in self.nodes:
            if node_id not in visited:
                if has_cycle(node_id):
                    return True

        return False

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate the pipeline structure.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check if pipeline has nodes
        if not self.nodes:
            errors.append("Pipeline has no nodes")

        # Check for disconnected nodes
        connected_nodes = set()
        for edge in self.edges:
            connected_nodes.add(edge.source_node_id)
            connected_nodes.add(edge.target_node_id)

        # Allow single-node pipelines
        if len(self.nodes) > 1:
            disconnected = set(self.nodes.keys()) - connected_nodes
            if disconnected:
                errors.append(f"Disconnected nodes found: {disconnected}")

        # Check for cycles (should not happen if add_edge works correctly)
        if self._has_cycle():
            errors.append("Pipeline contains a cycle")

        # Validate each node's configuration
        for node in self.nodes.values():
            node_errors = self._validate_node(node)
            errors.extend(node_errors)

        is_valid = len(errors) == 0
        return is_valid, errors

    def _has_cycle(self) -> bool:
        """Check if the pipeline has cycles."""
        adj_list: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
        for edge in self.edges:
            adj_list[edge.source_node_id].append(edge.target_node_id)

        visited = set()
        rec_stack = set()

        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            for neighbor in adj_list.get(node_id, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node_id)
            return False

        for node_id in self.nodes:
            if node_id not in visited:
                if dfs(node_id):
                    return True

        return False

    def _validate_node(self, node: BaseNode) -> list[str]:
        """Validate a node's configuration."""
        errors = []

        # Add node-specific validation here
        # For now, just check that node has required attributes
        if not hasattr(node, "node_id") or not node.node_id:
            errors.append(f"Node {node.name} has no node_id")

        return errors

    def compute_execution_order(self) -> list[str]:
        """
        Compute topological sort of nodes for execution.

        Returns:
            List of node IDs in execution order

        Raises:
            ValueError: If pipeline is invalid or contains cycles
        """
        # Validate first
        is_valid, errors = self.validate()
        if not is_valid:
            msg = f"Invalid pipeline: {', '.join(errors)}"
            raise ValueError(msg)

        # Build adjacency list and in-degree map
        adj_list: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
        in_degree: dict[str, int] = dict.fromkeys(self.nodes, 0)

        for edge in self.edges:
            adj_list[edge.source_node_id].append(edge.target_node_id)
            in_degree[edge.target_node_id] += 1

        # Kahn's algorithm for topological sort
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        execution_order = []

        while queue:
            # Sort to ensure deterministic order
            queue.sort()
            node_id = queue.pop(0)
            execution_order.append(node_id)

            for neighbor in adj_list[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check if all nodes were processed
        if len(execution_order) != len(self.nodes):
            msg = "Pipeline contains a cycle (topological sort failed)"
            raise ValueError(msg)

        self.execution_order = execution_order
        self.logger.info(f"Execution order computed: {execution_order}")
        return execution_order

    def get_node_inputs(self, node_id: str) -> list[PipelineEdge]:
        """
        Get all edges that provide input to a node.

        Args:
            node_id: ID of the target node

        Returns:
            List of edges pointing to this node
        """
        return [edge for edge in self.edges if edge.target_node_id == node_id]

    def get_node_outputs(self, node_id: str) -> list[PipelineEdge]:
        """
        Get all edges that consume output from a node.

        Args:
            node_id: ID of the source node

        Returns:
            List of edges originating from this node
        """
        return [edge for edge in self.edges if edge.source_node_id == node_id]

    def to_dict(self) -> dict[str, Any]:
        """Convert pipeline to dictionary representation."""
        return {
            "pipeline_id": self.pipeline_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "edges": [
                {
                    "source": edge.source_node_id,
                    "target": edge.target_node_id,
                    "output_name": edge.output_name,
                    "input_name": edge.input_name,
                }
                for edge in self.edges
            ],
            "execution_order": self.execution_order,
            "metadata": {
                "created_at": self.metadata.created_at.isoformat(),
                "updated_at": self.metadata.updated_at.isoformat(),
                "created_by": self.metadata.created_by,
                "version": self.metadata.version,
                "tags": self.metadata.tags,
                "description": self.metadata.description,
            },
        }

    def save(self, path: Path | str) -> None:
        """
        Save pipeline to JSON file.

        Args:
            path: Path where pipeline should be saved
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with path_obj.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

        self.logger.info(f"Pipeline saved to {path}")

    @classmethod
    def load(cls, path: Path | str) -> Pipeline:
        """
        Load pipeline from JSON file.

        Args:
            path: Path to pipeline JSON file

        Returns:
            Loaded Pipeline instance
        """
        path_obj = Path(path)
        if not path_obj.exists():
            msg = f"Pipeline file not found: {path}"
            raise FileNotFoundError(msg)

        with path_obj.open() as f:
            data = json.load(f)

        # Note: This is a simplified loader that only loads the structure
        # Full reconstruction of nodes would require node factory
        pipeline = cls(
            pipeline_id=data["pipeline_id"],
            name=data["name"],
            description=data.get("description", ""),
        )

        # Load metadata
        metadata_dict = data.get("metadata", {})
        pipeline.metadata = PipelineMetadata(
            created_at=datetime.fromisoformat(
                metadata_dict.get("created_at", datetime.now().isoformat())
            ),
            updated_at=datetime.fromisoformat(
                metadata_dict.get("updated_at", datetime.now().isoformat())
            ),
            created_by=metadata_dict.get("created_by", "system"),
            version=metadata_dict.get("version", "1.0.0"),
            tags=metadata_dict.get("tags", []),
            description=metadata_dict.get("description", ""),
        )

        # Note: Nodes need to be reconstructed separately
        # This is a structure-only load
        logger.info(f"Pipeline loaded from {path}")
        return pipeline

    def __repr__(self) -> str:
        """String representation of pipeline."""
        return f"Pipeline(id={self.pipeline_id}, name={self.name}, nodes={len(self.nodes)}, edges={len(self.edges)})"
