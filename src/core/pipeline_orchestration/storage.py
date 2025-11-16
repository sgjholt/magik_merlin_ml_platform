"""
Pipeline storage and versioning system.

This module provides functionality for persisting pipelines, execution results,
and managing pipeline versions.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.logging import get_logger

from .executor import ExecutionResult
from .pipeline import Pipeline

logger = get_logger(__name__)


@dataclass
class PipelineVersion:
    """Pipeline version information."""

    version: str
    created_at: datetime
    created_by: str
    description: str
    file_path: Path


class PipelineStorage:
    """
    Manage pipeline storage, versioning, and execution history.

    Pipelines are stored as JSON files with associated execution history
    and version tracking.
    """

    def __init__(self, storage_dir: Path | str = "data/pipelines") -> None:
        """
        Initialize pipeline storage.

        Args:
            storage_dir: Directory for storing pipeline files
        """
        self.storage_dir = Path(storage_dir)
        self.pipelines_dir = self.storage_dir / "pipelines"
        self.executions_dir = self.storage_dir / "executions"
        self.versions_dir = self.storage_dir / "versions"

        # Create directories
        self.pipelines_dir.mkdir(parents=True, exist_ok=True)
        self.executions_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger(__name__)
        self.logger.info(f"Pipeline storage initialized at {self.storage_dir}")

    def save_pipeline(
        self,
        pipeline: Pipeline,
        *,
        create_version: bool = False,
    ) -> Path:
        """
        Save a pipeline to storage.

        Args:
            pipeline: Pipeline to save
            create_version: If True, create a new version

        Returns:
            Path to saved pipeline file
        """
        # Save current pipeline
        pipeline_path = self.pipelines_dir / f"{pipeline.pipeline_id}.json"

        # Create version if requested and pipeline already exists
        if create_version and pipeline_path.exists():
            self._create_version(pipeline)

        # Save pipeline
        pipeline.save(pipeline_path)
        self.logger.info(f"Saved pipeline {pipeline.name} to {pipeline_path}")

        # Update metadata
        self._update_metadata(pipeline)

        return pipeline_path

    def load_pipeline(self, pipeline_id: str) -> Pipeline:
        """
        Load a pipeline from storage.

        Args:
            pipeline_id: ID of the pipeline to load

        Returns:
            Loaded Pipeline instance

        Raises:
            FileNotFoundError: If pipeline not found
        """
        pipeline_path = self.pipelines_dir / f"{pipeline_id}.json"

        if not pipeline_path.exists():
            msg = f"Pipeline {pipeline_id} not found"
            raise FileNotFoundError(msg)

        pipeline = Pipeline.load(pipeline_path)
        self.logger.info(f"Loaded pipeline {pipeline.name} from {pipeline_path}")
        return pipeline

    def delete_pipeline(self, pipeline_id: str) -> bool:
        """
        Delete a pipeline from storage.

        Args:
            pipeline_id: ID of the pipeline to delete

        Returns:
            True if deleted, False if not found
        """
        pipeline_path = self.pipelines_dir / f"{pipeline_id}.json"

        if not pipeline_path.exists():
            self.logger.warning(f"Pipeline {pipeline_id} not found")
            return False

        # Delete pipeline file
        pipeline_path.unlink()

        # Delete versions
        version_dir = self.versions_dir / pipeline_id
        if version_dir.exists():
            shutil.rmtree(version_dir)

        # Delete executions
        execution_dir = self.executions_dir / pipeline_id
        if execution_dir.exists():
            shutil.rmtree(execution_dir)

        self.logger.info(f"Deleted pipeline {pipeline_id}")
        return True

    def list_pipelines(self) -> list[dict[str, Any]]:
        """
        List all stored pipelines.

        Returns:
            List of pipeline metadata dictionaries
        """
        pipelines = []

        for pipeline_file in self.pipelines_dir.glob("*.json"):
            try:
                with pipeline_file.open() as f:
                    data = json.load(f)

                pipelines.append(
                    {
                        "pipeline_id": data["pipeline_id"],
                        "name": data["name"],
                        "description": data.get("description", ""),
                        "status": data.get("status", "unknown"),
                        "nodes": len(data.get("nodes", {})),
                        "updated_at": data.get("metadata", {}).get("updated_at"),
                    }
                )
            except Exception as e:
                self.logger.error(f"Error loading pipeline {pipeline_file}: {e}")

        return pipelines

    def save_execution_result(
        self,
        result: ExecutionResult,
    ) -> Path:
        """
        Save pipeline execution result.

        Args:
            result: Execution result to save

        Returns:
            Path to saved result file
        """
        # Create execution directory for this pipeline
        exec_dir = self.executions_dir / result.pipeline_id
        exec_dir.mkdir(parents=True, exist_ok=True)

        # Generate execution file name with timestamp
        timestamp = result.start_time.strftime("%Y%m%d_%H%M%S")
        exec_file = exec_dir / f"exec_{timestamp}.json"

        # Save result
        with exec_file.open("w") as f:
            json.dump(result.to_dict(), f, indent=2)

        self.logger.info(f"Saved execution result to {exec_file}")
        return exec_file

    def load_execution_result(
        self,
        pipeline_id: str,
        execution_id: str,
    ) -> dict[str, Any]:
        """
        Load a specific execution result.

        Args:
            pipeline_id: ID of the pipeline
            execution_id: ID of the execution (timestamp)

        Returns:
            Execution result dictionary

        Raises:
            FileNotFoundError: If execution not found
        """
        exec_file = self.executions_dir / pipeline_id / f"exec_{execution_id}.json"

        if not exec_file.exists():
            msg = f"Execution {execution_id} not found for pipeline {pipeline_id}"
            raise FileNotFoundError(msg)

        with exec_file.open() as f:
            return json.load(f)

    def list_executions(
        self,
        pipeline_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        List execution results for a pipeline.

        Args:
            pipeline_id: ID of the pipeline
            limit: Maximum number of results to return

        Returns:
            List of execution result summaries
        """
        exec_dir = self.executions_dir / pipeline_id

        if not exec_dir.exists():
            return []

        executions = []
        exec_files = sorted(exec_dir.glob("exec_*.json"), reverse=True)

        if limit:
            exec_files = exec_files[:limit]

        for exec_file in exec_files:
            try:
                with exec_file.open() as f:
                    data = json.load(f)
                executions.append(data)
            except Exception as e:
                self.logger.error(f"Error loading execution {exec_file}: {e}")

        return executions

    def _create_version(self, pipeline: Pipeline) -> None:
        """
        Create a new version of the pipeline.

        Args:
            pipeline: Pipeline to version
        """
        version_dir = self.versions_dir / pipeline.pipeline_id
        version_dir.mkdir(parents=True, exist_ok=True)

        # Generate version number
        existing_versions = list(version_dir.glob("v*.json"))
        version_num = len(existing_versions) + 1
        version_str = f"v{version_num}"

        # Copy current pipeline to version
        current_path = self.pipelines_dir / f"{pipeline.pipeline_id}.json"
        version_path = version_dir / f"{version_str}.json"

        if current_path.exists():
            shutil.copy2(current_path, version_path)
            self.logger.info(
                f"Created version {version_str} of pipeline {pipeline.name}"
            )

    def list_versions(self, pipeline_id: str) -> list[PipelineVersion]:
        """
        List all versions of a pipeline.

        Args:
            pipeline_id: ID of the pipeline

        Returns:
            List of PipelineVersion objects
        """
        version_dir = self.versions_dir / pipeline_id

        if not version_dir.exists():
            return []

        versions = []
        for version_file in sorted(version_dir.glob("v*.json")):
            try:
                with version_file.open() as f:
                    data = json.load(f)

                version = PipelineVersion(
                    version=version_file.stem,
                    created_at=datetime.fromisoformat(
                        data.get("metadata", {}).get(
                            "updated_at", datetime.now().isoformat()
                        )
                    ),
                    created_by=data.get("metadata", {}).get("created_by", "unknown"),
                    description=data.get("description", ""),
                    file_path=version_file,
                )
                versions.append(version)
            except Exception as e:
                self.logger.error(f"Error loading version {version_file}: {e}")

        return versions

    def restore_version(
        self,
        pipeline_id: str,
        version: str,
    ) -> Pipeline:
        """
        Restore a pipeline to a specific version.

        Args:
            pipeline_id: ID of the pipeline
            version: Version to restore (e.g., "v1")

        Returns:
            Restored Pipeline instance

        Raises:
            FileNotFoundError: If version not found
        """
        version_file = self.versions_dir / pipeline_id / f"{version}.json"

        if not version_file.exists():
            msg = f"Version {version} not found for pipeline {pipeline_id}"
            raise FileNotFoundError(msg)

        # Create new version of current pipeline first
        try:
            current_pipeline = self.load_pipeline(pipeline_id)
            self._create_version(current_pipeline)
        except FileNotFoundError:
            pass

        # Copy version to current
        current_path = self.pipelines_dir / f"{pipeline_id}.json"
        shutil.copy2(version_file, current_path)

        # Load and return restored pipeline
        pipeline = Pipeline.load(current_path)
        self.logger.info(f"Restored pipeline {pipeline_id} to version {version}")
        return pipeline

    def _update_metadata(self, pipeline: Pipeline) -> None:
        """
        Update metadata file for quick access.

        Args:
            pipeline: Pipeline to update metadata for
        """
        metadata_file = self.storage_dir / "metadata.json"

        # Load existing metadata
        metadata = {}
        if metadata_file.exists():
            with metadata_file.open() as f:
                metadata = json.load(f)

        # Update pipeline entry
        metadata[pipeline.pipeline_id] = {
            "pipeline_id": pipeline.pipeline_id,
            "name": pipeline.name,
            "description": pipeline.description,
            "updated_at": datetime.now().isoformat(),
            "nodes": len(pipeline.nodes),
            "edges": len(pipeline.edges),
        }

        # Save metadata
        with metadata_file.open("w") as f:
            json.dump(metadata, f, indent=2)

    def get_storage_stats(self) -> dict[str, Any]:
        """
        Get statistics about pipeline storage.

        Returns:
            Dictionary with storage statistics
        """
        total_pipelines = len(list(self.pipelines_dir.glob("*.json")))

        total_executions = sum(
            len(list(exec_dir.glob("exec_*.json")))
            for exec_dir in self.executions_dir.iterdir()
            if exec_dir.is_dir()
        )

        total_versions = sum(
            len(list(version_dir.glob("v*.json")))
            for version_dir in self.versions_dir.iterdir()
            if version_dir.is_dir()
        )

        return {
            "total_pipelines": total_pipelines,
            "total_executions": total_executions,
            "total_versions": total_versions,
            "storage_dir": str(self.storage_dir),
        }
