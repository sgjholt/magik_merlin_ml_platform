"""
Integration tests for complete data workflows
"""

import sys
import time
from pathlib import Path

import pandas as pd
import pytest

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.core.data_sources import LocalFileDataSource
from src.core.data_sources.base import DataSourceConfig
from src.ui.panels.data_management import DataManagementPanel


class TestCompleteDataWorkflow:
    """Test complete data workflow from source to UI"""

    def test_end_to_end_data_loading(self, test_data_dir):
        """Test complete data loading workflow"""
        # Step 1: Create data source
        config = DataSourceConfig(
            name="integration_test",
            source_type="local_files",
            connection_params={"base_path": test_data_dir},
        )

        datasource = LocalFileDataSource(config)

        # Step 2: Test connection
        assert datasource.test_connection() is True

        # Step 3: List available files
        files = datasource.list_tables()
        assert len(files) > 0
        csv_files = [f for f in files if f.endswith(".csv")]
        assert len(csv_files) > 0

        # Step 4: Load data
        csv_file = csv_files[0]
        df = datasource.load_data(csv_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

        # Step 5: Generate profile
        profile = datasource.get_data_profile(df)
        assert profile["shape"] == df.shape
        assert len(profile["columns"]) == len(df.columns)

        # Step 6: Get preview
        preview = datasource.get_data_preview(csv_file, limit=10)
        assert len(preview) <= 10
        assert list(preview.columns) == list(df.columns)

    def test_data_source_ui_integration(self, test_data_dir):
        """Test data source integration with UI panel"""
        # Create UI panel
        panel = DataManagementPanel()

        # Verify panel is initialized
        assert panel.current_datasource is None
        assert panel.current_data is None

        # Simulate connection configuration
        panel.datasource_type_select.value = "Local Files"

        # Create mock connection inputs
        import panel as pn

        panel.connection_inputs.clear()
        panel.connection_inputs.append(
            pn.widgets.TextInput(name="Base Path", value=test_data_dir)
        )

        # Test connection parameter extraction
        params = panel._get_connection_params()
        assert params["base_path"] == test_data_dir

        # Verify available data sources
        available_sources = panel.datasource_type_select.options
        assert "Local Files" in available_sources

    def test_multiple_file_format_workflow(self, test_data_dir):
        """Test workflow with multiple file formats"""
        config = DataSourceConfig(
            name="multi_format_test",
            source_type="local_files",
            connection_params={"base_path": test_data_dir},
        )

        datasource = LocalFileDataSource(config)
        files = datasource.list_tables()

        # Test loading different formats
        formats_tested = set()

        for file in files:
            file_ext = Path(file).suffix.lower()
            if file_ext in [".csv", ".parquet", ".json"]:
                df = datasource.load_data(file)
                assert isinstance(df, pd.DataFrame)
                assert len(df) > 0
                formats_tested.add(file_ext)

        # Ensure we tested multiple formats
        assert len(formats_tested) >= 2

    def test_error_handling_workflow(self, test_data_dir):
        """Test error handling in complete workflow"""
        config = DataSourceConfig(
            name="error_test",
            source_type="local_files",
            connection_params={"base_path": test_data_dir},
        )

        datasource = LocalFileDataSource(config)

        # Test 1: Nonexistent file
        with pytest.raises(FileNotFoundError):
            datasource.load_data("nonexistent_file.csv")

        # Test 2: Unsupported format
        # Create a dummy file with unsupported extension
        dummy_file = Path(test_data_dir) / "test_file.xyz"
        dummy_file.write_text("dummy content")

        with pytest.raises(ValueError, match="Unsupported file format"):
            datasource.load_data("test_file.xyz")

        # Test 3: Invalid base path
        bad_config = DataSourceConfig(
            name="bad_test",
            source_type="local_files",
            connection_params={"base_path": "/nonexistent/path"},
        )

        bad_datasource = LocalFileDataSource(bad_config)
        assert bad_datasource.test_connection() is False

    def test_caching_workflow(self, test_data_dir):
        """Test caching behavior in workflow"""
        config = DataSourceConfig(
            name="cache_test",
            source_type="local_files",
            connection_params={"base_path": test_data_dir},
            cache_enabled=True,
        )

        datasource = LocalFileDataSource(config)
        files = datasource.list_tables()
        csv_files = [f for f in files if f.endswith(".csv")]

        if csv_files:
            csv_file = csv_files[0]

            # First load - should cache
            start_time = time.time()
            df1 = datasource.load_data(csv_file)
            first_load_time = time.time() - start_time

            # Verify cache is populated
            assert len(datasource._cache) > 0

            # Second load - should use cache (faster)
            start_time = time.time()
            df2 = datasource.load_data(csv_file)
            second_load_time = time.time() - start_time

            # Data should be identical
            assert df1.equals(df2)

            # Second load should be faster (though this might not always be reliable)
            # Just verify that caching mechanism is working
            assert len(datasource._cache) > 0

    def test_data_profiling_workflow(self, test_data_dir):
        """Test data profiling workflow"""
        config = DataSourceConfig(
            name="profile_test",
            source_type="local_files",
            connection_params={"base_path": test_data_dir},
        )

        datasource = LocalFileDataSource(config)
        files = datasource.list_tables()
        csv_files = [f for f in files if f.endswith(".csv")]

        if csv_files:
            csv_file = csv_files[0]
            df = datasource.load_data(csv_file)

            # Test schema extraction
            schema = datasource.get_schema(csv_file)
            assert "columns" in schema
            assert "dtypes" in schema
            assert "shape" in schema

            # Test data profiling
            profile = datasource.get_data_profile(df)
            assert profile["shape"] == df.shape
            assert profile["columns"] == list(df.columns)
            assert "dtypes" in profile
            assert "null_counts" in profile
            assert "memory_usage" in profile

            # Verify null counts make sense
            for col in df.columns:
                expected_nulls = df[col].isnull().sum()
                assert profile["null_counts"][col] == expected_nulls
