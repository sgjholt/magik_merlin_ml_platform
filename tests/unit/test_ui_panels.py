"""
Unit tests for UI panel components
"""

import sys
from pathlib import Path

import pandas as pd

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.ui.panels.data_management import DataManagementPanel
from src.ui.panels.deployment import DeploymentPanel
from src.ui.panels.experimentation import ExperimentationPanel
from src.ui.panels.model_evaluation import ModelEvaluationPanel


class TestDataManagementPanel:
    """Test DataManagementPanel functionality"""

    def test_panel_initialization(self):
        """Test panel creates correctly"""
        panel = DataManagementPanel()

        assert panel.current_datasource is None
        assert panel.current_data is None
        assert hasattr(panel, "datasource_type_select")
        assert hasattr(panel, "connect_button")
        assert hasattr(panel, "data_preview")

    def test_available_data_sources(self):
        """Test available data source options"""
        panel = DataManagementPanel()
        options = panel.datasource_type_select.options

        # Local Files should always be available
        assert "Local Files" in options

        # Cloud sources may or may not be available depending on dependencies
        assert len(options) >= 1

    def test_connection_params_extraction(self):
        """Test connection parameter extraction"""
        panel = DataManagementPanel()

        # Simulate connection inputs
        panel.connection_inputs.clear()
        import panel as pn

        panel.connection_inputs.extend(
            [
                pn.widgets.TextInput(name="Base Path", value="/test/path"),
                pn.widgets.TextInput(name="User Name", value="testuser"),
            ]
        )

        params = panel._get_connection_params()

        assert params["base_path"] == "/test/path"
        assert params["user_name"] == "testuser"


class TestExperimentationPanel:
    """Test ExperimentationPanel functionality"""

    def test_panel_initialization(self):
        """Test panel creates correctly"""
        panel = ExperimentationPanel()

        assert panel.current_experiment is None
        assert hasattr(panel, "task_type_select")
        assert hasattr(panel, "target_select")
        assert hasattr(panel, "model_select")
        assert hasattr(panel, "start_experiment_button")

    def test_default_task_options(self):
        """Test default ML task options"""
        panel = ExperimentationPanel()
        options = panel.task_type_select.options

        expected_tasks = [
            "Classification",
            "Regression",
            "Clustering",
            "Anomaly Detection",
            "Time Series Forecasting",
        ]

        for task in expected_tasks:
            assert task in options

    def test_model_options(self):
        """Test available model options"""
        panel = ExperimentationPanel()
        options = panel.model_select.options

        expected_models = ["Random Forest", "XGBoost", "LightGBM"]
        for model in expected_models:
            assert model in options

    def test_update_data_options(self):
        """Test updating data options"""
        panel = ExperimentationPanel()

        # Create sample data
        test_data = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "target": [0, 1, 0]}
        )

        # Update options
        panel.update_data_options(test_data)

        assert not panel.target_select.disabled
        assert not panel.feature_select.disabled
        assert not panel.start_experiment_button.disabled

        # Check that columns are available
        assert set(panel.target_select.options) == set(test_data.columns)
        assert set(panel.feature_select.options) == set(test_data.columns)


class TestModelEvaluationPanel:
    """Test ModelEvaluationPanel functionality"""

    def test_panel_initialization(self):
        """Test panel creates correctly"""
        panel = ModelEvaluationPanel()

        assert hasattr(panel, "model_select")
        assert hasattr(panel, "metric_select")
        assert hasattr(panel, "viz_type_select")
        assert hasattr(panel, "comparison_plot")

    def test_visualization_options(self):
        """Test visualization type options"""
        panel = ModelEvaluationPanel()
        options = panel.viz_type_select.options

        expected_viz = [
            "Performance Comparison",
            "Confusion Matrix",
            "ROC Curve",
            "Feature Importance",
            "Learning Curve",
        ]

        for viz in expected_viz:
            assert viz in options

    def test_metric_options(self):
        """Test available metric options"""
        panel = ModelEvaluationPanel()
        options = panel.metric_select.options

        expected_metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
        for metric in expected_metrics:
            assert metric in options


class TestDeploymentPanel:
    """Test DeploymentPanel functionality"""

    def test_panel_initialization(self):
        """Test panel creates correctly"""
        panel = DeploymentPanel()

        assert hasattr(panel, "model_select")
        assert hasattr(panel, "deployment_name")
        assert hasattr(panel, "deployment_env")
        assert hasattr(panel, "deploy_button")

    def test_environment_options(self):
        """Test deployment environment options"""
        panel = DeploymentPanel()
        options = panel.deployment_env.options

        expected_envs = ["Development", "Staging", "Production"]
        for env in expected_envs:
            assert env in options

    def test_deployment_type_options(self):
        """Test deployment type options"""
        panel = DeploymentPanel()
        options = panel.deployment_type.options

        expected_types = ["REST API", "Batch Processing", "Real-time Stream"]
        for dtype in expected_types:
            assert dtype in options

    def test_instance_type_options(self):
        """Test instance type options"""
        panel = DeploymentPanel()
        options = panel.instance_type.options

        expected_instances = ["t3.micro", "t3.small", "t3.medium", "t3.large"]
        for instance in expected_instances:
            assert instance in options
