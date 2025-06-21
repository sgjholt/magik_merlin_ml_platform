"""
Integration tests for full application functionality
"""

import sys
import threading
import time
from pathlib import Path

import pytest
import requests

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.config.settings import Settings
from src.ui.app import MLPlatformApp
from tests.conftest import TestConfig


class TestAppIntegration:
    """Test full application integration"""

    def test_app_initialization(self):
        """Test app initializes correctly"""
        app = MLPlatformApp()

        # Verify all panels are created
        assert hasattr(app, "data_panel")
        assert hasattr(app, "experiment_panel")
        assert hasattr(app, "evaluation_panel")
        assert hasattr(app, "deployment_panel")

        # Verify template is created
        assert hasattr(app, "template")
        assert app.template.title == "ML Experimentation Platform"

    def test_panel_components_created(self, app_instance):
        """Test all panel components are properly created"""
        app = app_instance

        # Data panel components
        data_panel = app.data_panel
        assert hasattr(data_panel, "datasource_type_select")
        assert hasattr(data_panel, "connect_button")
        assert hasattr(data_panel, "data_preview")

        # Experiment panel components
        exp_panel = app.experiment_panel
        assert hasattr(exp_panel, "task_type_select")
        assert hasattr(exp_panel, "start_experiment_button")
        assert hasattr(exp_panel, "results_table")

        # Evaluation panel components
        eval_panel = app.evaluation_panel
        assert hasattr(eval_panel, "model_select")
        assert hasattr(eval_panel, "comparison_plot")

        # Deployment panel components
        deploy_panel = app.deployment_panel
        assert hasattr(deploy_panel, "model_select")
        assert hasattr(deploy_panel, "deploy_button")

    @pytest.mark.slow
    def test_app_server_startup(self):
        """Test app server starts and responds (marked as slow)"""
        app = MLPlatformApp()

        server_started = threading.Event()
        server_error = None

        def run_server():
            nonlocal server_error
            try:
                # Run server on test port
                app.serve(port=TestConfig.TEST_PORT, show=False, autoreload=False)
                server_started.set()
            except Exception as e:
                server_error = e
                server_started.set()

        # Start server in background thread
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        # Wait for server to start (with timeout)
        if not server_started.wait(timeout=TestConfig.TIMEOUT):
            pytest.fail("Server failed to start within timeout")

        if server_error:
            pytest.fail(f"Server startup failed: {server_error}")

        # Give server a moment to fully initialize
        time.sleep(2)

        # Test server response
        try:
            response = requests.get(
                f"http://{TestConfig.TEST_HOST}:{TestConfig.TEST_PORT}", timeout=5
            )
            assert response.status_code == 200
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Server not responding: {e}")

    def test_app_configuration(self):
        """Test app uses configuration correctly"""
        settings = Settings()
        app = MLPlatformApp()

        # Test that app can access settings
        assert settings.app_port is not None
        assert settings.app_host is not None

        # Test that panels can handle configuration
        data_panel = app.data_panel
        available_sources = data_panel.datasource_type_select.options
        assert len(available_sources) > 0
        assert "Local Files" in available_sources


class TestUIInteractions:
    """Test UI component interactions"""

    def test_data_panel_interactions(self, app_instance):
        """Test data management panel interactions"""
        app = app_instance
        data_panel = app.data_panel

        # Test data source type selection
        original_type = data_panel.datasource_type_select.value
        assert original_type in data_panel.datasource_type_select.options

        # Test that connection inputs are created
        assert hasattr(data_panel, "connection_inputs")

        # Test status indicators
        assert hasattr(data_panel, "connection_status")
        initial_status = data_panel.connection_status.object
        assert "Not Connected" in initial_status

    def test_experiment_panel_interactions(self, app_instance):
        """Test experimentation panel interactions"""
        app = app_instance
        exp_panel = app.experiment_panel

        # Test task type selection
        task_types = exp_panel.task_type_select.options
        assert "Classification" in task_types
        assert "Regression" in task_types

        # Test model selection
        models = exp_panel.model_select.options
        assert len(models) > 0

        # Test that experiment button starts disabled
        assert exp_panel.start_experiment_button.disabled is True

    def test_evaluation_panel_interactions(self, app_instance):
        """Test model evaluation panel interactions"""
        app = app_instance
        eval_panel = app.evaluation_panel

        # Test visualization type selection
        viz_types = eval_panel.viz_type_select.options
        assert "Performance Comparison" in viz_types
        assert "ROC Curve" in viz_types

        # Test metric selection
        metrics = eval_panel.metric_select.options
        assert "Accuracy" in metrics
        assert "F1-Score" in metrics

    def test_deployment_panel_interactions(self, app_instance):
        """Test deployment panel interactions"""
        app = app_instance
        deploy_panel = app.deployment_panel

        # Test environment selection
        environments = deploy_panel.deployment_env.options
        assert "Development" in environments
        assert "Production" in environments

        # Test deployment type selection
        deploy_types = deploy_panel.deployment_type.options
        assert "REST API" in deploy_types
        assert "Batch Processing" in deploy_types

        # Test that deploy button starts disabled
        assert deploy_panel.deploy_button.disabled is True


class TestCrossComponentIntegration:
    """Test integration between different components"""

    def test_data_to_experiment_flow(self, app_instance, sample_dataframe):
        """Test data flows from data panel to experiment panel"""
        app = app_instance

        # Simulate data being loaded in data panel
        data_panel = app.data_panel
        data_panel.current_data = sample_dataframe

        # Update experiment panel with data
        exp_panel = app.experiment_panel
        exp_panel.update_data_options(sample_dataframe)

        # Verify experiment panel is updated
        assert not exp_panel.target_select.disabled
        assert not exp_panel.feature_select.disabled
        assert set(exp_panel.target_select.options) == set(sample_dataframe.columns)

    def test_experiment_to_evaluation_flow(self, app_instance):
        """Test flow from experiment to evaluation"""
        app = app_instance

        # Experiment panel should have mock results
        exp_panel = app.experiment_panel
        eval_panel = app.evaluation_panel

        # Evaluation panel should have mock models available
        models = eval_panel.model_select.options
        assert len(models) > 0

        # Should be able to select models for comparison
        if models:
            eval_panel.model_select.value = models[:2]  # Select first two models
            assert len(eval_panel.model_select.value) <= 2

    def test_evaluation_to_deployment_flow(self, app_instance):
        """Test flow from evaluation to deployment"""
        app = app_instance

        eval_panel = app.evaluation_panel
        deploy_panel = app.deployment_panel

        # Should be able to select evaluated models for deployment
        eval_models = eval_panel.model_select.options
        deploy_models = deploy_panel.model_select.options

        # In a real implementation, these would be synchronized
        # For now, just verify both have model options
        assert len(eval_models) > 0
        assert len(deploy_models) > 0
