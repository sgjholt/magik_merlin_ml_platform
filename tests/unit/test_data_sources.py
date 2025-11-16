"""
Unit tests for data source functionality
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.core.data_sources.base import DataSourceConfig
from src.core.data_sources.local_file import LocalFileDataSource


class TestDataSourceConfig:
    """Test DataSourceConfig model"""

    def test_config_creation(self):
        """Test basic config creation"""
        config = DataSourceConfig(
            name="test", source_type="local", connection_params={"path": "/tmp"}
        )

        assert config.name == "test"
        assert config.source_type == "local"
        assert config.connection_params["path"] == "/tmp"
        assert config.cache_enabled is True
        assert config.cache_ttl == 3600

    def test_config_with_custom_cache(self):
        """Test config with custom cache settings"""
        config = DataSourceConfig(
            name="test",
            source_type="local",
            connection_params={},
            cache_enabled=False,
            cache_ttl=1800,
        )

        assert config.cache_enabled is False
        assert config.cache_ttl == 1800


class TestLocalFileDataSource:
    """Test LocalFileDataSource functionality"""

    def test_initialization(self, data_source_config):
        """Test data source initialization"""
        datasource = LocalFileDataSource(data_source_config)
        assert datasource.config == data_source_config
        assert datasource._connection is None
        assert datasource._cache == {}

    def test_supported_formats(self):
        """Test supported file formats"""
        expected_formats = {".csv", ".parquet", ".json", ".xlsx", ".xls"}
        actual_formats = set(LocalFileDataSource.SUPPORTED_FORMATS.keys())
        assert actual_formats == expected_formats

    def test_connect(self, test_data_dir):
        """Test connection to valid directory"""
        config = DataSourceConfig(
            name="test",
            source_type="local",
            connection_params={"base_path": test_data_dir},
        )

        datasource = LocalFileDataSource(config)
        assert datasource.connect() is True
        assert datasource.test_connection() is True

    def test_connect_invalid_directory(self):
        """Test connection to invalid directory"""
        config = DataSourceConfig(
            name="test",
            source_type="local",
            connection_params={"base_path": "/nonexistent/path"},
        )

        datasource = LocalFileDataSource(config)
        assert datasource.test_connection() is False

    def test_list_tables(self, test_data_dir):
        """Test listing available files"""
        config = DataSourceConfig(
            name="test",
            source_type="local",
            connection_params={"base_path": test_data_dir},
        )

        datasource = LocalFileDataSource(config)
        files = datasource.list_tables()

        # Should find the test files we created
        assert len(files) >= 3
        assert any(f.endswith(".csv") for f in files)
        assert any(f.endswith(".parquet") for f in files)
        assert any(f.endswith(".json") for f in files)

    def test_load_csv_data(self, test_data_dir):
        """Test loading CSV data"""
        config = DataSourceConfig(
            name="test",
            source_type="local",
            connection_params={"base_path": test_data_dir},
        )

        datasource = LocalFileDataSource(config)
        df = datasource.load_data("sample.csv")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100  # From our test data
        assert "feature_1" in df.columns
        assert "target" in df.columns

    def test_load_parquet_data(self, test_data_dir):
        """Test loading Parquet data"""
        config = DataSourceConfig(
            name="test",
            source_type="local",
            connection_params={"base_path": test_data_dir},
        )

        datasource = LocalFileDataSource(config)
        df = datasource.load_data("sample.parquet")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50  # From our test data
        assert "id" in df.columns
        assert "value" in df.columns

    def test_load_json_data(self, test_data_dir):
        """Test loading JSON data"""
        config = DataSourceConfig(
            name="test",
            source_type="local",
            connection_params={"base_path": test_data_dir},
        )

        datasource = LocalFileDataSource(config)
        df = datasource.load_data("sample.json")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 20  # From our test data
        assert "user_id" in df.columns
        assert "score" in df.columns

    def test_load_nonexistent_file(self, test_data_dir):
        """Test loading nonexistent file raises error"""
        config = DataSourceConfig(
            name="test",
            source_type="local",
            connection_params={"base_path": test_data_dir},
        )

        datasource = LocalFileDataSource(config)

        with pytest.raises(FileNotFoundError):
            datasource.load_data("nonexistent.csv")

    def test_load_unsupported_format(self, test_data_dir):
        """Test loading unsupported format raises error"""
        config = DataSourceConfig(
            name="test",
            source_type="local",
            connection_params={"base_path": test_data_dir},
        )

        datasource = LocalFileDataSource(config)

        # Create a dummy file with unsupported extension
        dummy_file = Path(test_data_dir) / "file.xyz"
        dummy_file.write_text("dummy content")

        with pytest.raises(ValueError, match="Unsupported file format"):
            datasource.load_data("file.xyz")

    def test_get_schema(self, test_data_dir):
        """Test getting file schema"""
        config = DataSourceConfig(
            name="test",
            source_type="local",
            connection_params={"base_path": test_data_dir},
        )

        datasource = LocalFileDataSource(config)
        schema = datasource.get_schema("sample.csv")

        assert "columns" in schema
        assert "dtypes" in schema
        assert "shape" in schema
        assert len(schema["columns"]) > 0

    def test_cache_functionality(self, test_data_dir):
        """Test data caching"""
        config = DataSourceConfig(
            name="test",
            source_type="local",
            connection_params={"base_path": test_data_dir},
            cache_enabled=True,
        )

        datasource = LocalFileDataSource(config)

        # First load should cache
        df1 = datasource.load_data("sample.csv")
        assert len(datasource._cache) == 1

        # Second load should use cache
        df2 = datasource.load_data("sample.csv")
        assert df1.equals(df2)

    def test_cache_disabled(self, test_data_dir):
        """Test with caching disabled"""
        config = DataSourceConfig(
            name="test",
            source_type="local",
            connection_params={"base_path": test_data_dir},
            cache_enabled=False,
        )

        datasource = LocalFileDataSource(config)
        datasource.load_data("sample.csv")

        # Cache should remain empty
        assert len(datasource._cache) == 0

    def test_data_profile(self, test_data_dir):
        """Test data profiling functionality"""
        config = DataSourceConfig(
            name="test",
            source_type="local",
            connection_params={"base_path": test_data_dir},
        )

        datasource = LocalFileDataSource(config)
        df = datasource.load_data("sample.csv")
        profile = datasource.get_data_profile(df)

        # Check that all expected keys are present
        assert "shape" in profile
        assert "schema" in profile
        assert "null_counts" in profile
        assert "memory_usage" in profile

        # Verify shape structure (dict with rows and columns)
        assert profile["shape"]["rows"] == df.shape[0]
        assert profile["shape"]["columns"] == df.shape[1]

        # Verify schema contains all columns
        assert set(profile["schema"].keys()) == set(df.columns)

    def test_save_data(self, test_data_dir, sample_dataframe):
        """Test saving data functionality"""
        config = DataSourceConfig(
            name="test",
            source_type="local",
            connection_params={"base_path": test_data_dir},
        )

        datasource = LocalFileDataSource(config)

        # Save as CSV
        datasource.save_data(sample_dataframe, "test_output.csv")

        # Verify file was created and can be loaded
        saved_df = datasource.load_data("test_output.csv")
        assert len(saved_df) == len(sample_dataframe)
        assert list(saved_df.columns) == list(sample_dataframe.columns)
