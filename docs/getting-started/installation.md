# Installation

This guide will help you install and set up the Magik Merlin ML Platform.

## Prerequisites

- **Python 3.13+** - The platform uses modern Python features
- **uv** package manager (recommended) or pip
- **Git** for cloning the repository
- Optional: **Docker** for containerized deployment

## System Requirements

- **OS**: Linux, macOS, or Windows (WSL2 recommended)
- **RAM**: Minimum 4GB, 8GB+ recommended for ML workloads
- **Disk**: ~2GB for dependencies, more for data and models

## Installation Methods

### Method 1: Using uv (Recommended)

uv is a fast Python package manager that handles dependencies efficiently.

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/sgjholt/magik_merlin_ml_platform.git
cd magik_merlin_ml_platform

# Install core dependencies
uv sync

# Install with ML engine support (XGBoost, LightGBM, CatBoost, PyTorch)
uv sync --extra ml

# Install development dependencies (for contributing)
uv sync --extra dev --extra ml
```

### Method 2: Using pip

```bash
# Clone the repository
git clone https://github.com/sgjholt/magik_merlin_ml_platform.git
cd magik_merlin_ml_platform

# Create virtual environment
python3.13 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e ".[ml]"

# For development
pip install -e ".[dev,ml]"
```

## Platform Setup

After installation, run the setup script to configure the platform:

```bash
# Using the run.sh script (recommended)
./run.sh setup

# Or using Python directly
python setup_platform.py setup
```

This will:
- Create required directories (`experiments/`, `mlruns/`, etc.)
- Validate configuration
- Set up MLflow tracking server
- Initialize the database

## Verify Installation

Check that everything is installed correctly:

```bash
# Check Python version
python --version  # Should be 3.13+

# Check platform status
./run.sh status

# Run health checks
./run.sh health
```

Expected output:
```
✅ Python 3.13+ detected
✅ Dependencies installed
✅ MLflow server accessible
✅ Platform ready
```

## Optional Dependencies

### Cloud Integration

For cloud data sources (Snowflake, AWS S3, Azure):

```bash
uv sync --extra cloud --extra ml
```

### Development Tools

For contributing to the project:

```bash
uv sync --extra dev --extra ml
```

Includes:
- ruff (linting and formatting)
- pytest (testing)
- mypy (type checking)
- mkdocs (documentation)
- pre-commit (git hooks)

## Troubleshooting

### Common Issues

#### Python Version Error

```
Error: Python 3.13+ required
```

**Solution**: Install Python 3.13 or later:
- **macOS**: `brew install python@3.13`
- **Ubuntu**: Build from source or use deadsnakes PPA
- **Windows**: Download from [python.org](https://www.python.org/downloads/)

#### uv Not Found

```
command not found: uv
```

**Solution**: Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or using pip
pip install uv
```

#### ML Library Installation Fails

```
ERROR: Failed to build xgboost/lightgbm/catboost
```

**Solution**: Install system dependencies:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake
```

**macOS:**
```bash
brew install cmake libomp
```

#### MLflow Connection Error

```
Error: Cannot connect to MLflow server
```

**Solution**: Start MLflow server:
```bash
./run.sh mlflow start
# Or
./scripts/mlflow.sh start
```

### Getting Help

If you encounter issues:

1. Check the [troubleshooting guide](../user-guide/overview.md#troubleshooting)
2. Search [existing issues](https://github.com/sgjholt/magik_merlin_ml_platform/issues)
3. Open a [new issue](https://github.com/sgjholt/magik_merlin_ml_platform/issues/new) with:
   - Python version (`python --version`)
   - OS and version
   - Full error message
   - Steps to reproduce

## Next Steps

- [Quick Start Guide](quick-start.md) - Run your first experiment
- [Configuration](configuration.md) - Configure data sources and MLflow
- [User Guide](../user-guide/overview.md) - Learn about all features
