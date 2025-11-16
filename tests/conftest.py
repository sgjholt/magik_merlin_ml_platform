"""
Pytest configuration and shared fixtures
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


# Check for optional ML dependencies
def _check_import(module_name: str) -> bool:
    """Check if a module can be imported."""
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True


# Dependency availability flags
HAS_XGBOOST = _check_import("xgboost")
HAS_LIGHTGBM = _check_import("lightgbm")
HAS_CATBOOST = _check_import("catboost")
HAS_TORCH = _check_import("torch")
HAS_LIGHTNING = _check_import("lightning")

# Pytest markers for optional dependencies
requires_xgboost = pytest.mark.skipif(
    not HAS_XGBOOST, reason="XGBoost not installed (install with: uv sync --extra ml)"
)
requires_lightgbm = pytest.mark.skipif(
    not HAS_LIGHTGBM, reason="LightGBM not installed (install with: uv sync --extra ml)"
)
requires_catboost = pytest.mark.skipif(
    not HAS_CATBOOST, reason="CatBoost not installed (install with: uv sync --extra ml)"
)
requires_pytorch = pytest.mark.skipif(
    not HAS_TORCH or not HAS_LIGHTNING,
    reason="PyTorch/Lightning not installed (install with: uv sync --extra ml)",
)
requires_ml = pytest.mark.skipif(
    not (HAS_XGBOOST or HAS_LIGHTGBM or HAS_CATBOOST),
    reason="No ML libraries installed (install with: uv sync --extra ml)",
)


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory with test data files"""
    temp_dir = tempfile.mkdtemp()
    test_data_path = Path(temp_dir) / "test_data"
    test_data_path.mkdir(exist_ok=True)

    # Create sample CSV data
    np.random.seed(42)
    n_samples = 100

    csv_data = pd.DataFrame(
        {
            "feature_1": np.random.normal(0, 1, n_samples),
            "feature_2": np.random.normal(2, 1.5, n_samples),
            "feature_3": np.random.uniform(0, 10, n_samples),
            "category": np.random.choice(["A", "B", "C"], n_samples),
            "target": np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        }
    )
    csv_data.to_csv(test_data_path / "sample.csv", index=False)

    # Create sample Parquet data
    parquet_data = pd.DataFrame(
        {
            "id": range(1, 51),
            "value": np.random.randn(50),
            "label": np.random.choice(["X", "Y"], 50),
        }
    )
    parquet_data.to_parquet(test_data_path / "sample.parquet", index=False)

    # Create sample JSON data
    json_data = pd.DataFrame(
        {
            "user_id": [f"user_{i}" for i in range(1, 21)],
            "score": np.random.uniform(0, 100, 20),
            "active": np.random.choice([True, False], 20),
        }
    )
    json_data.to_json(test_data_path / "sample.json", orient="records")

    yield str(test_data_path)

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing"""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feature_1": np.random.normal(0, 1, 50),
            "feature_2": np.random.normal(2, 1.5, 50),
            "target": np.random.choice([0, 1], 50, p=[0.7, 0.3]),
        }
    )


@pytest.fixture
def data_source_config():
    """Create a basic data source configuration"""
    from src.core.data_sources.base import DataSourceConfig

    return DataSourceConfig(
        name="test_source",
        source_type="local_files",
        connection_params={"base_path": "./test_data"},
    )


@pytest.fixture(scope="session")
def app_instance():
    """Create an app instance for testing (without serving)"""
    from src.ui.app import MLPlatformApp

    return MLPlatformApp()


class TestConfig:
    """Test configuration constants"""

    TEST_PORT = 5007
    TEST_HOST = "localhost"
    TIMEOUT = 10
