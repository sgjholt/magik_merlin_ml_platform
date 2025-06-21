"""
Unit tests for application layout and structure
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.ui.app import MLPlatformApp


class TestMLPlatformApp:
    """Test MLPlatformApp layout and functionality"""

    def test_app_initialization(self):
        """Test that the app initializes correctly"""
        app = MLPlatformApp()
        
        # Check core components exist
        assert app.data_panel is not None
        assert app.experiment_panel is not None
        assert app.evaluation_panel is not None
        assert app.deployment_panel is not None
        assert app.experiment_tracker is not None
        
        # Check template is created
        assert app.template is not None
        assert app.template.title == "ML Experimentation Platform"

    def test_data_status_indicator_initialization(self):
        """Test that data status indicator is properly initialized"""
        app = MLPlatformApp()
        
        assert app.data_status_indicator is not None
        assert "No Data Loaded" in app.data_status_indicator.object

    def test_data_updated_callback(self):
        """Test that data updated callback works correctly"""
        import pandas as pd
        
        app = MLPlatformApp()
        
        # Test with sample data
        sample_data = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0]
        })
        
        # Trigger callback
        app._on_data_updated(sample_data)
        
        # Check that status indicator was updated
        assert "Data Loaded" in app.data_status_indicator.object
        assert "3 rows, 3 cols" in app.data_status_indicator.object

    def test_data_updated_callback_with_none(self):
        """Test callback with None data"""
        app = MLPlatformApp()
        
        # Trigger callback with None
        app._on_data_updated(None)
        
        # Check that status indicator shows no data
        assert "No Data Loaded" in app.data_status_indicator.object

    def test_template_main_content(self):
        """Test that main content contains tabs"""
        app = MLPlatformApp()
        
        # Check that main content has been populated
        assert len(app.template.main) > 0
        
        # The main content should contain tabs
        main_content = app.template.main[0]
        assert hasattr(main_content, 'objects')  # Should be a Tabs object

    def test_session_stats_tracking(self):
        """Test that session statistics are tracked correctly"""
        app = MLPlatformApp()
        
        # Check initial stats
        assert app.session_stats["experiments_run"] == 0
        assert app.session_stats["models_trained"] == 0
        assert app.session_stats["data_sources_connected"] == 0
        
        # Test experiment completion callback
        app._on_experiment_completed()
        assert app.session_stats["experiments_run"] == 1
        assert app.session_stats["models_trained"] == 3

    def test_collapsible_overview_card_exists(self):
        """Test that the single collapsible overview card is created"""
        app = MLPlatformApp()
        
        # Check that session stats content exists
        assert app.session_info_markdown is not None
        assert "Session Statistics" in app.session_info_markdown.object

    def test_sidebar_content(self):
        """Test that sidebar contains status information"""
        app = MLPlatformApp()
        
        # Check that sidebar exists
        assert len(app.template.sidebar) > 0
        
        sidebar_content = app.template.sidebar[0]
        assert hasattr(sidebar_content, 'objects')  # Should be a Column object