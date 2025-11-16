"""
Unit tests for configuration management
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.config.settings import Settings


class TestSettings:
    """Test application settings"""

    def test_default_settings(self):
        """Test default settings values"""
        settings = Settings()

        assert settings.mlflow_tracking_uri == "http://127.0.0.1:5000"
        assert settings.mlflow_experiment_name == "ml-platform-experiments"
        assert settings.aws_default_region == "us-east-1"
        assert settings.redis_host == "localhost"
        assert settings.redis_port == 6379
        assert settings.redis_db == 0
        assert settings.app_host == "127.0.0.1"
        assert settings.app_port == 5006
        assert settings.debug is True

    def test_optional_fields(self):
        """Test optional configuration fields"""
        settings = Settings()

        # These should be None by default
        assert settings.snowflake_user is None
        assert settings.snowflake_password is None
        assert settings.aws_access_key_id is None
        assert settings.aws_secret_access_key is None

    def test_environment_override(self, monkeypatch):
        """Test environment variable override"""
        # Set environment variables
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://custom:5001")
        monkeypatch.setenv("APP_PORT", "8080")
        monkeypatch.setenv("DEBUG", "false")

        settings = Settings()

        assert settings.mlflow_tracking_uri == "http://custom:5001"
        assert settings.app_port == 8080
        assert settings.debug is False

    def test_case_insensitive_env_vars(self, monkeypatch):
        """Test case-insensitive environment variables"""
        monkeypatch.setenv("app_host", "127.0.0.1")
        monkeypatch.setenv("REDIS_PORT", "6380")

        settings = Settings()

        assert settings.app_host == "127.0.0.1"
        assert settings.redis_port == 6380
