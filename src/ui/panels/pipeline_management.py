"""
Pipeline Management Panel for creating and managing ML pipelines.

This panel provides a user-friendly interface for:
- Creating new pipelines
- Viewing and editing existing pipelines
- Executing pipelines
- Monitoring pipeline status
- Viewing pipeline execution history
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import panel as pn
import pandas as pd

from src.core.logging import get_logger
from src.core.pipeline_orchestration import (
    DataLoaderNode,
    DataPreprocessorNode,
    FeatureScalerNode,
    ModelEvaluatorNode,
    ModelSaverNode,
    ModelTrainerNode,
    Pipeline,
    PipelineExecutor,
    PipelineScheduler,
    PipelineStorage,
    ScheduleConfig,
    TrainTestSplitNode,
)

logger = get_logger(__name__)


class PipelineManagementPanel:
    """
    Panel for managing ML pipelines.

    Provides UI for creating, executing, and monitoring pipelines.
    """

    def __init__(self) -> None:
        """Initialize pipeline management panel."""
        self.logger = get_logger(__name__, pipeline_stage="ui_pipeline")

        # Initialize pipeline system components
        self.storage = PipelineStorage()
        self.executor = PipelineExecutor()
        self.scheduler = PipelineScheduler(self.executor)

        # UI state
        self.selected_pipeline: Pipeline | None = None
        self.execution_progress: dict[str, float] = {}

        # Build UI components
        self._build_ui()

        # Start scheduler
        self.scheduler.start()

    def _build_ui(self) -> None:
        """Build the panel UI."""
        # Header
        self.header = pn.pane.Markdown("## 🔄 Pipeline Management")

        # Pipeline list
        self.pipeline_table = pn.widgets.Tabulator(
            pd.DataFrame(columns=["ID", "Name", "Nodes", "Status", "Updated"]),
            page_size=10,
            sizing_mode="stretch_width",
            selectable=1,
        )
        self.pipeline_table.on_click(self._on_pipeline_selected)

        # Action buttons
        self.refresh_button = pn.widgets.Button(
            name="🔄 Refresh",
            button_type="primary",
            width=120,
        )
        self.refresh_button.on_click(self._on_refresh_pipelines)

        self.new_pipeline_button = pn.widgets.Button(
            name="➕ New Pipeline",
            button_type="success",
            width=120,
        )
        self.new_pipeline_button.on_click(self._on_new_pipeline)

        self.delete_pipeline_button = pn.widgets.Button(
            name="🗑️ Delete",
            button_type="danger",
            width=120,
            disabled=True,
        )
        self.delete_pipeline_button.on_click(self._on_delete_pipeline)

        # Pipeline builder section
        self.pipeline_name_input = pn.widgets.TextInput(
            name="Pipeline Name",
            placeholder="Enter pipeline name",
            width=300,
        )

        self.pipeline_desc_input = pn.widgets.TextAreaInput(
            name="Description",
            placeholder="Enter pipeline description",
            height=80,
            width=600,
        )

        # Node configuration
        self.node_type_select = pn.widgets.Select(
            name="Node Type",
            options=[
                "Data Loader",
                "Data Preprocessor",
                "Train-Test Split",
                "Feature Scaler",
                "Model Trainer",
                "Model Evaluator",
                "Model Saver",
            ],
            width=200,
        )

        self.add_node_button = pn.widgets.Button(
            name="➕ Add Node",
            button_type="success",
            width=120,
        )
        self.add_node_button.on_click(self._on_add_node)

        # Pipeline nodes display
        self.pipeline_nodes_text = pn.pane.Markdown(
            "**Pipeline Nodes:**\n\nNo nodes added yet."
        )

        # Execution section
        self.execute_button = pn.widgets.Button(
            name="▶️ Execute Pipeline",
            button_type="primary",
            width=150,
            disabled=True,
        )
        self.execute_button.on_click(self._on_execute_pipeline)

        self.schedule_button = pn.widgets.Button(
            name="📅 Schedule",
            button_type="primary",
            width=120,
            disabled=True,
        )
        self.schedule_button.on_click(self._on_schedule_pipeline)

        # Execution history
        self.execution_history_table = pn.widgets.Tabulator(
            pd.DataFrame(columns=["Time", "Status", "Duration"]),
            page_size=5,
            sizing_mode="stretch_width",
        )

        # Progress indicator
        self.progress_bar = pn.indicators.Progress(
            name="Execution Progress",
            value=0,
            width=400,
            visible=False,
        )

        self.status_text = pn.pane.Markdown("")

        # Build the complete panel layout
        self.panel = pn.Column(
            self.header,
            pn.pane.Markdown("---"),
            # Pipeline list section
            pn.pane.Markdown("### 📋 Existing Pipelines"),
            pn.Row(
                self.refresh_button,
                self.new_pipeline_button,
                self.delete_pipeline_button,
            ),
            self.pipeline_table,
            pn.pane.Markdown("---"),
            # Pipeline builder section
            pn.pane.Markdown("### ⚙️ Pipeline Builder"),
            pn.Row(
                self.pipeline_name_input,
                self.pipeline_desc_input,
            ),
            pn.pane.Markdown("#### Add Nodes"),
            pn.Row(
                self.node_type_select,
                self.add_node_button,
            ),
            self.pipeline_nodes_text,
            pn.pane.Markdown("---"),
            # Execution section
            pn.pane.Markdown("### ▶️ Execute Pipeline"),
            pn.Row(
                self.execute_button,
                self.schedule_button,
            ),
            self.progress_bar,
            self.status_text,
            pn.pane.Markdown("#### Execution History"),
            self.execution_history_table,
            sizing_mode="stretch_width",
        )

        # Load pipelines on init
        self._refresh_pipelines()

    def _refresh_pipelines(self) -> None:
        """Refresh the list of pipelines."""
        try:
            pipelines = self.storage.list_pipelines()

            if pipelines:
                df = pd.DataFrame(pipelines)
                df = df.rename(
                    columns={
                        "pipeline_id": "ID",
                        "name": "Name",
                        "nodes": "Nodes",
                        "status": "Status",
                        "updated_at": "Updated",
                    }
                )
                self.pipeline_table.value = df[
                    ["ID", "Name", "Nodes", "Status", "Updated"]
                ]
            else:
                self.pipeline_table.value = pd.DataFrame(
                    columns=["ID", "Name", "Nodes", "Status", "Updated"]
                )

            self.logger.info(f"Loaded {len(pipelines)} pipelines")
        except Exception as e:
            self.logger.exception(f"Error loading pipelines: {e}")
            self.status_text.object = f"**Error:** Failed to load pipelines: {e}"

    def _on_refresh_pipelines(self, event: Any) -> None:
        """Handle refresh button click."""
        self._refresh_pipelines()

    def _on_pipeline_selected(self, event: Any) -> None:
        """Handle pipeline selection."""
        if event.row is not None and not self.pipeline_table.value.empty:
            pipeline_id = self.pipeline_table.value.iloc[event.row]["ID"]

            try:
                self.selected_pipeline = self.storage.load_pipeline(pipeline_id)
                self.delete_pipeline_button.disabled = False
                self.execute_button.disabled = False
                self.schedule_button.disabled = False

                # Update pipeline builder with selected pipeline info
                self.pipeline_name_input.value = self.selected_pipeline.name
                self.pipeline_desc_input.value = self.selected_pipeline.description

                # Update nodes display
                nodes_text = "**Pipeline Nodes:**\n\n"
                for node_id, node in self.selected_pipeline.nodes.items():
                    nodes_text += f"- **{node.name}** ({node.node_type.value})\n"

                self.pipeline_nodes_text.object = nodes_text

                # Load execution history
                self._load_execution_history(pipeline_id)

                self.logger.info(f"Selected pipeline: {self.selected_pipeline.name}")
            except Exception as e:
                self.logger.exception(f"Error loading pipeline: {e}")
                self.status_text.object = f"**Error:** Failed to load pipeline: {e}"

    def _on_new_pipeline(self, event: Any) -> None:
        """Handle new pipeline button click."""
        # Clear current selection
        self.selected_pipeline = None
        self.pipeline_name_input.value = ""
        self.pipeline_desc_input.value = ""
        self.pipeline_nodes_text.object = "**Pipeline Nodes:**\n\nNo nodes added yet."
        self.delete_pipeline_button.disabled = True
        self.execute_button.disabled = True
        self.schedule_button.disabled = True

        # Create new pipeline
        pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.selected_pipeline = Pipeline(
            pipeline_id=pipeline_id,
            name="New Pipeline",
            description="",
        )

        self.status_text.object = (
            "**New pipeline created.** Add nodes and configure settings."
        )
        self.logger.info("Created new pipeline")

    def _on_add_node(self, event: Any) -> None:
        """Handle add node button click."""
        if self.selected_pipeline is None:
            self.status_text.object = "**Error:** Create or select a pipeline first"
            return

        node_type = self.node_type_select.value
        node_id = f"node_{len(self.selected_pipeline.nodes) + 1}"

        try:
            # Create node based on type
            if node_type == "Data Loader":
                node = DataLoaderNode(
                    node_id=node_id,
                    source_type="file",
                    source_path="/path/to/data.csv",
                )
            elif node_type == "Data Preprocessor":
                node = DataPreprocessorNode(
                    node_id=node_id,
                    operations=["drop_missing", "remove_duplicates"],
                )
            elif node_type == "Train-Test Split":
                node = TrainTestSplitNode(
                    node_id=node_id,
                    test_size=0.2,
                    target_column="target",
                )
            elif node_type == "Feature Scaler":
                node = FeatureScalerNode(node_id=node_id)
            elif node_type == "Model Trainer":
                node = ModelTrainerNode(
                    node_id=node_id,
                    model_type="xgboost",
                    task_type="classification",
                )
            elif node_type == "Model Evaluator":
                node = ModelEvaluatorNode(node_id=node_id)
            elif node_type == "Model Saver":
                node = ModelSaverNode(
                    node_id=node_id,
                    save_path="models/model.pkl",
                )
            else:
                self.status_text.object = f"**Error:** Unknown node type: {node_type}"
                return

            # Add node to pipeline
            self.selected_pipeline.add_node(node)

            # Auto-connect to previous node if exists
            if len(self.selected_pipeline.nodes) > 1:
                prev_node_id = f"node_{len(self.selected_pipeline.nodes) - 1}"
                self.selected_pipeline.add_edge(prev_node_id, node_id)

            # Update UI
            nodes_text = "**Pipeline Nodes:**\n\n"
            for nid, n in self.selected_pipeline.nodes.items():
                nodes_text += f"- **{n.name}** ({n.node_type.value})\n"

            self.pipeline_nodes_text.object = nodes_text
            self.status_text.object = f"**Added node:** {node.name}"

            # Enable execute button
            self.execute_button.disabled = False

            self.logger.info(f"Added node {node.name} to pipeline")

        except Exception as e:
            self.logger.exception(f"Error adding node: {e}")
            self.status_text.object = f"**Error:** Failed to add node: {e}"

    def _on_delete_pipeline(self, event: Any) -> None:
        """Handle delete pipeline button click."""
        if self.selected_pipeline is None:
            return

        try:
            pipeline_id = self.selected_pipeline.pipeline_id
            self.storage.delete_pipeline(pipeline_id)

            self.selected_pipeline = None
            self.delete_pipeline_button.disabled = True
            self.execute_button.disabled = True
            self.schedule_button.disabled = True

            self._refresh_pipelines()
            self.status_text.object = "**Pipeline deleted successfully**"
            self.logger.info(f"Deleted pipeline: {pipeline_id}")

        except Exception as e:
            self.logger.exception(f"Error deleting pipeline: {e}")
            self.status_text.object = f"**Error:** Failed to delete pipeline: {e}"

    def _on_execute_pipeline(self, event: Any) -> None:
        """Handle execute pipeline button click."""
        if self.selected_pipeline is None:
            return

        try:
            # Update pipeline name and description
            if self.pipeline_name_input.value:
                self.selected_pipeline.name = self.pipeline_name_input.value
            if self.pipeline_desc_input.value:
                self.selected_pipeline.description = self.pipeline_desc_input.value

            # Save pipeline before execution
            self.storage.save_pipeline(self.selected_pipeline)

            # Show progress bar
            self.progress_bar.visible = True
            self.progress_bar.value = 0

            # Define progress callback
            def on_progress(pipeline_id: str, node_id: str, progress: float) -> None:
                self.progress_bar.value = int(progress)
                self.status_text.object = f"**Executing:** {node_id} ({progress:.1f}%)"

            # Execute pipeline
            self.status_text.object = "**Executing pipeline...**"
            result = self.executor.execute(
                self.selected_pipeline,
                async_mode=True,
                progress_callback=on_progress,
            )

            self.status_text.object = (
                f"**Pipeline execution started.** Status: {result.status.value}"
            )
            self.logger.info(
                f"Started pipeline execution: {self.selected_pipeline.name}"
            )

            # Start monitoring thread
            import threading

            def monitor_execution() -> None:
                import time

                while self.executor.is_running(self.selected_pipeline.pipeline_id):
                    time.sleep(1)

                # Execution complete
                result = self.executor.get_result(self.selected_pipeline.pipeline_id)
                if result:
                    self.storage.save_execution_result(result)
                    self.progress_bar.visible = False
                    self.status_text.object = f"**Execution complete:** {result.status.value} (Duration: {result.duration:.2f}s)"
                    self._load_execution_history(self.selected_pipeline.pipeline_id)

            threading.Thread(target=monitor_execution, daemon=True).start()

        except Exception as e:
            self.logger.exception(f"Error executing pipeline: {e}")
            self.status_text.object = f"**Error:** Failed to execute pipeline: {e}"
            self.progress_bar.visible = False

    def _on_schedule_pipeline(self, event: Any) -> None:
        """Handle schedule pipeline button click."""
        if self.selected_pipeline is None:
            return

        # Simple daily schedule for now
        schedule = ScheduleConfig(
            schedule_type="cron",
            cron_expression="0 0 * * *",  # Daily at midnight
            enabled=True,
        )

        try:
            self.scheduler.schedule_pipeline(self.selected_pipeline, schedule)
            self.status_text.object = "**Pipeline scheduled:** Daily at midnight"
            self.logger.info(f"Scheduled pipeline: {self.selected_pipeline.name}")
        except Exception as e:
            self.logger.exception(f"Error scheduling pipeline: {e}")
            self.status_text.object = f"**Error:** Failed to schedule pipeline: {e}"

    def _load_execution_history(self, pipeline_id: str) -> None:
        """Load execution history for a pipeline."""
        try:
            executions = self.storage.list_executions(pipeline_id, limit=10)

            if executions:
                history_data = []
                for exec_data in executions:
                    history_data.append(
                        {
                            "Time": exec_data.get("start_time", ""),
                            "Status": exec_data.get("status", ""),
                            "Duration": f"{exec_data.get('duration', 0):.2f}s",
                        }
                    )

                self.execution_history_table.value = pd.DataFrame(history_data)
            else:
                self.execution_history_table.value = pd.DataFrame(
                    columns=["Time", "Status", "Duration"]
                )

        except Exception as e:
            self.logger.exception(f"Error loading execution history: {e}")
