import panel as pn

from .panels.data_management import DataManagementPanel
from .panels.deployment import DeploymentPanel
from .panels.experimentation import ExperimentationPanel
from .panels.model_evaluation import ModelEvaluationPanel
from ..core.experiments.tracking import ExperimentTracker


class MLPlatformApp:
    def __init__(self):
        pn.extension("bokeh", "plotly", "tabulator")

        # Initialize experiment tracker
        self.experiment_tracker = ExperimentTracker()

        # Initialize panels
        self.data_panel = DataManagementPanel()
        self.experiment_panel = ExperimentationPanel(self.experiment_tracker)
        self.evaluation_panel = ModelEvaluationPanel()
        self.deployment_panel = DeploymentPanel()

        # Set up data flow between panels
        self.data_panel.data_updated_callback = self._on_data_updated
        self.experiment_panel.experiment_completed_callback = (
            self._on_experiment_completed
        )

        # Status indicators for sidebar
        self.data_status_indicator = pn.pane.HTML(
            "<span style='color: gray;'>●</span> No Data Loaded"
        )

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

    def _create_sidebar(self):
        # Core status functionality at the top
        status_section = pn.Column(
            pn.pane.Markdown("## Platform Status"),
            pn.pane.HTML("<hr>"),
            pn.pane.Markdown("**Experiment Tracker:**"),
            pn.pane.HTML(
                "<span style='color: orange;'>●</span> MLflow (Not Connected)"
                if not self.experiment_tracker.is_active()
                else "<span style='color: green;'>●</span> MLflow Connected"
            ),
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

    def _setup_main_content(self):
        # Create main tabs with core functionality only
        main_tabs = pn.Tabs(
            ("📊 Data Management", self.data_panel.panel),
            ("🔬 Experimentation", self.experiment_panel.panel),
            ("📈 Model Evaluation", self.evaluation_panel.panel),
            ("🚀 Deployment", self.deployment_panel.panel),
            dynamic=True,
            sizing_mode="stretch_width",
        )

        self.template.main.append(main_tabs)

    def _create_session_info_markdown(self):
        """Create session info markdown component for overview card"""
        if self.session_info_markdown is None:
            self.session_info_markdown = pn.pane.Markdown(self._get_session_info_text())
        return self.session_info_markdown

    def _get_session_info_text(self):
        """Generate session info text for the overview card"""
        return f"""
        ## 📊 Current Session
        
        **Session Statistics:**
        
        • **Active Experiments:** {self.session_stats["experiments_run"]}
        • **Models Trained:** {self.session_stats["models_trained"]}
        • **Data Sources:** {self.session_stats["data_sources_connected"]} connected
        • **Deployments:** {self.session_stats["deployments_active"]} active
        """

    def _update_session_stats(self):
        """Update the session statistics display"""
        if self.session_info_markdown:
            self.session_info_markdown.object = self._get_session_info_text()

    def _on_data_updated(self, data):
        """Callback when data is updated in data management panel"""
        # Update experiment panel with new data
        self.experiment_panel.update_data_options(data)

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

    def _on_experiment_completed(self):
        """Callback when an experiment is completed"""
        self.session_stats["experiments_run"] += 1
        # Assume each experiment trains multiple models
        self.session_stats["models_trained"] += (
            3  # Average number of models per experiment
        )
        self._update_session_stats()

    def serve(self, port: int = 5006, show: bool = True, autoreload: bool = True):
        self.template.servable()
        return pn.serve(
            self.template,
            port=port,
            show=show,
            autoreload=autoreload,
            allow_websocket_origin=[f"localhost:{port}"],
        )
