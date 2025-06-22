import subprocess
import threading
import pandas as pd
import panel as pn

from ..core.experiments.tracking import ExperimentTracker
from ..core.logging import get_logger
from .panels.data_management import DataManagementPanel
from .panels.deployment import DeploymentPanel
from .panels.experimentation import ExperimentationPanel
from .panels.model_evaluation import ModelEvaluationPanel
from .panels.visualization import VisualizationPanel


class MLPlatformApp:
    def __init__(self) -> None:
        pn.extension("bokeh", "plotly", "tabulator")

        # Initialize experiment tracker
        self.experiment_tracker = ExperimentTracker()

        # Initialize panels
        self.data_panel = DataManagementPanel()
        self.experiment_panel = ExperimentationPanel(self.experiment_tracker)
        self.evaluation_panel = ModelEvaluationPanel()
        self.deployment_panel = DeploymentPanel()
        self.visualization_panel = VisualizationPanel(self.experiment_panel.experiment_manager)

        # Set up data flow between panels
        self.data_panel.data_updated_callback = self._on_data_updated
        self.experiment_panel.experiment_completed_callback = (
            self._on_experiment_completed
        )

        # Status indicators for sidebar
        self.data_status_indicator = pn.pane.HTML(
            "<span style='color: gray;'>●</span> No Data Loaded"
        )
        
        # MLflow status indicator
        self.mlflow_status_indicator = pn.pane.HTML(
            "<span style='color: orange;'>●</span> MLflow (Not Connected)"
        )
        
        # MLflow control buttons
        self.start_mlflow_button = pn.widgets.Button(
            name="🚀 Start MLflow", button_type="success", width=100
        )
        self.stop_mlflow_button = pn.widgets.Button(
            name="🛑 Stop MLflow", button_type="danger", width=100, disabled=True
        )
        self.mlflow_ui_button = pn.widgets.Button(
            name="📊 MLflow UI", button_type="primary", width=100, disabled=True
        )
        
        # Set up MLflow button callbacks
        self.start_mlflow_button.on_click(self._on_start_mlflow)
        self.stop_mlflow_button.on_click(self._on_stop_mlflow)
        self.mlflow_ui_button.on_click(self._on_open_mlflow_ui)
        
        # Logger
        self.logger = get_logger(__name__, pipeline_stage="ui_app")
        
        # Update initial MLflow status
        self._update_mlflow_status()

        # Session tracking
        self.session_stats = {
            "experiments_run": 0,
            "models_trained": 0,
            "data_sources_connected": 0,
            "deployments_active": 0,
        }

        # Create dynamic session info for overview card
        self.session_info_markdown = None

        # Create the main template
        self.template = pn.template.MaterialTemplate(
            title="ML Experimentation Platform",
            sidebar=[self._create_sidebar()],
            header_background="#2596be",
        )

        # Set up the main content area
        self._setup_main_content()

    def _create_sidebar(self) -> pn.Column:
        # Core status functionality at the top
        status_section = pn.Column(
            pn.pane.Markdown("## Platform Status"),
            pn.pane.HTML("<hr>"),
            pn.pane.Markdown("**Experiment Tracker:**"),
            self.mlflow_status_indicator,
            pn.Row(
                self.start_mlflow_button,
                self.stop_mlflow_button,
                sizing_mode="stretch_width"
            ),
            self.mlflow_ui_button,
            pn.pane.HTML("<hr>"),
            pn.pane.Markdown("**Data Sources:**"),
            self.data_status_indicator,
            pn.pane.HTML("<hr>"),
            pn.pane.Markdown("**Quick Actions:**"),
            pn.widgets.Button(name="🔄 Refresh Status", button_type="light", width=220),
            pn.widgets.Button(name="📚 Documentation", button_type="light", width=220),
        )

        # Create comprehensive platform overview in single collapsible card
        overview_content = pn.Column(
            pn.pane.Markdown("""
            ## 🚀 Getting Started
            
            **Step-by-step workflow:**
            
            1. **📊 Data Management** - Connect and load data
            2. **🔬 Experimentation** - Run ML experiments  
            3. **📈 Model Evaluation** - Compare results
            4. **🚀 Deployment** - Deploy best models
            
            ## 🔧 Key Features
            
            **Core capabilities:**
            
            • **PyCaret Integration** - Automated ML workflows
            • **MLflow Tracking** - Experiment management  
            • **Multiple Data Sources** - Files, cloud, databases
            • **Interactive Visualizations** - Rich charts
            • **Production Deployment** - Model serving
            """),
            self._create_session_info_markdown(),
            margin=(5, 5),
        )

        platform_overview = pn.Card(
            overview_content,
            title="📋 Platform Overview",
            collapsed=True,
            width=220,
            margin=(10, 0),
        )

        return pn.Column(
            status_section,
            pn.pane.HTML("<hr>"),
            platform_overview,
            width=250,
            scroll=True,
        )

    def _setup_main_content(self) -> None:
        # Create main tabs with core functionality only
        main_tabs = pn.Tabs(
            ("📊 Data Management", self.data_panel.panel),
            ("🔬 Experimentation", self.experiment_panel.panel),
            ("📈 Model Evaluation", self.evaluation_panel.panel),
            ("📉 Visualizations", self.visualization_panel.panel),
            ("🚀 Deployment", self.deployment_panel.panel),
            dynamic=True,
            sizing_mode="stretch_width",
        )

        self.template.main.append(main_tabs)

    def _create_session_info_markdown(self) -> pn.pane.Markdown:
        """Create session info markdown component for overview card"""
        if self.session_info_markdown is None:
            self.session_info_markdown = pn.pane.Markdown(self._get_session_info_text())
        return self.session_info_markdown

    def _get_session_info_text(self) -> str:
        """Generate session info text for the overview card"""
        return f"""
        ## 📊 Current Session
        
        **Session Statistics:**
        
        • **Active Experiments:** {self.session_stats["experiments_run"]}
        • **Models Trained:** {self.session_stats["models_trained"]}
        • **Data Sources:** {self.session_stats["data_sources_connected"]} connected
        • **Deployments:** {self.session_stats["deployments_active"]} active
        """

    def _update_session_stats(self) -> None:
        """Update the session statistics display"""
        if self.session_info_markdown:
            self.session_info_markdown.object = self._get_session_info_text()

    def _on_data_updated(self, data: pd.DataFrame | None) -> None:
        """Callback when data is updated in data management panel"""
        # Update experiment panel with new data
        self.experiment_panel.update_data_options(data)
        
        # Update visualization panel with new data
        self.visualization_panel.update_data(data)

        # Update status indicator
        if data is not None and not data.empty:
            self.data_status_indicator.object = f"<span style='color: green;'>●</span> Data Loaded ({data.shape[0]} rows, {data.shape[1]} cols)"
            # Update session stats
            self.session_stats["data_sources_connected"] = 1
        else:
            self.data_status_indicator.object = (
                "<span style='color: gray;'>●</span> No Data Loaded"
            )
            self.session_stats["data_sources_connected"] = 0

        # Update session display
        self._update_session_stats()

    def _on_experiment_completed(self) -> None:
        """Callback when an experiment is completed"""
        self.session_stats["experiments_run"] += 1
        # Assume each experiment trains multiple models
        self.session_stats["models_trained"] += (
            3  # Average number of models per experiment
        )
        self._update_session_stats()

    def serve(self, port: int = 5006, *, show: bool = True, autoreload: bool = True):  # type: ignore[misc]
        self.template.servable()
        return pn.serve(
            self.template,
            port=port,
            show=show,
            autoreload=autoreload,
            allow_websocket_origin=[f"localhost:{port}"],
        )
        
    def _update_mlflow_status(self) -> None:
        """Update MLflow status indicator and buttons"""
        is_available = self.experiment_tracker.is_server_available()
        
        if is_available:
            self.mlflow_status_indicator.object = (
                "<span style='color: green;'>●</span> MLflow Connected"
            )
            self.start_mlflow_button.disabled = True
            self.stop_mlflow_button.disabled = False
            self.mlflow_ui_button.disabled = False
        else:
            self.mlflow_status_indicator.object = (
                "<span style='color: orange;'>●</span> MLflow (Not Connected)"
            )
            self.start_mlflow_button.disabled = False
            self.stop_mlflow_button.disabled = True
            self.mlflow_ui_button.disabled = True
            
    def _on_start_mlflow(self, event) -> None:
        """Start MLflow server"""
        self.logger.info("Starting MLflow server from UI")
        
        def start_server():
            try:
                # Start MLflow server in background
                process = subprocess.Popen(
                    ["python", "scripts/start_mlflow.py", "start"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Wait a moment for startup
                import time
                time.sleep(3)
                
                # Check if it started successfully
                if self.experiment_tracker.reconnect():
                    self.logger.info("MLflow server started successfully")
                    self._update_mlflow_status()
                else:
                    self.logger.error("Failed to start MLflow server")
                    
            except Exception as e:
                self.logger.error(f"Error starting MLflow server: {e}")
        
        # Start in background thread to avoid blocking UI
        threading.Thread(target=start_server, daemon=True).start()
        
    def _on_stop_mlflow(self, event) -> None:
        """Stop MLflow server"""
        self.logger.info("Stopping MLflow server from UI")
        
        try:
            # Stop MLflow server
            subprocess.run(["./scripts/mlflow.sh", "stop"], capture_output=True)
            
            # Update status
            self._update_mlflow_status()
            self.logger.info("MLflow server stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping MLflow server: {e}")
            
    def _on_open_mlflow_ui(self, event) -> None:
        """Open MLflow UI in browser"""
        self.logger.info("Opening MLflow UI")
        
        try:
            import webbrowser
            webbrowser.open(self.experiment_tracker.tracking_uri)
        except Exception as e:
            self.logger.error(f"Failed to open MLflow UI: {e}")
