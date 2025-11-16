# Dependency Management Guide

## 📦 Modern Python Packaging

The ML Platform now uses modern Python packaging standards with **pyproject.toml** for all dependency management.

## 🗂️ Dependency Organization

### Single Source of Truth: `pyproject.toml`

All dependencies are now managed in `pyproject.toml` with organized groups:

```toml
[project]
dependencies = [
    # Core dependencies (always installed)
    "numpy>=1.21.0",
    "pandas>=1.5.0", 
    "panel>=1.0.0",
    # ... other core deps
]

[project.optional-dependencies]
dev = [
    # Development tools
    "pytest>=7.0.0",
    "black>=22.0.0",
    # ... other dev tools
]

cloud = [
    # Cloud services
    "mlflow>=2.0.0",
    "snowflake-connector-python>=3.0.0",
    # ... other cloud deps
]

ml = [
    # ML features (Phase 2)
    "pycaret>=3.0.0"
]
```

## 🚀 Installation Commands

### Core Installation (Minimal)
```bash
make install           # Installs only essential dependencies
# OR: uv pip install -e .
```

### Development Installation
```bash
make install-dev       # Core + development tools
# OR: uv pip install -e ".[dev]"
```

### Cloud Services Installation  
```bash
make install-cloud     # Core + cloud services (MLflow, Snowflake, AWS)
# OR: uv pip install -e ".[cloud]"
```

### ML Features Installation (Phase 2)
```bash
make install-ml        # Core + ML libraries (PyCaret)
# OR: uv pip install -e ".[ml]"
```

### Complete Installation
```bash
make install-all       # Everything
# OR: uv pip install -e ".[dev,cloud,ml]"
```

## 🧹 Cleanup Summary

### ❌ Removed Files
- `requirements.txt` (outdated)
- `requirements-core.txt` (consolidated)
- `requirements-dev.txt` (consolidated)
- `requirements-optional.txt` (consolidated)

### ✅ Benefits of New Approach

1. **Single Source**: All dependencies in `pyproject.toml`
2. **Modular**: Install only what you need
3. **Editable Install**: Package installed in development mode
4. **Standard**: Follows modern Python packaging standards
5. **Maintainable**: No duplicate dependency specifications

## 🛠️ Developer Workflow

### For New Developers
```bash
# 1. Clone repository
git clone <repo-url>
cd ml_platform

# 2. Create virtual environment
uv venv
source .venv/bin/activate

# 3. Install dependencies based on needs
make install-dev       # For development
# OR make install-all  # For everything
```

### For Different Use Cases

| Use Case | Installation Command | Includes |
|----------|---------------------|----------|
| **Basic Usage** | `make install` | Core ML platform functionality |
| **Development** | `make install-dev` | Core + testing, linting, formatting |
| **Cloud Integration** | `make install-cloud` | Core + MLflow, Snowflake, AWS |
| **ML Workflows** | `make install-ml` | Core + PyCaret (Phase 2) |
| **Everything** | `make install-all` | All optional dependencies |

## 📋 Dependency Groups Explained

### Core Dependencies (Always Installed)
- **Data Processing**: pandas, numpy, pyarrow
- **Web Framework**: panel, bokeh, plotly, holoviews
- **Configuration**: pydantic, python-dotenv
- **Utilities**: click, requests

### Development Dependencies (`[dev]`)
- **Testing**: pytest, pytest-cov, pytest-mock
- **Code Quality**: black, flake8, mypy, isort
- **Git Hooks**: pre-commit
- **Documentation**: sphinx, sphinx-rtd-theme

### Cloud Dependencies (`[cloud]`)
- **Experiment Tracking**: mlflow
- **Data Sources**: snowflake-connector-python, boto3
- **Databases**: sqlalchemy, redis
- **File Formats**: openpyxl
- **Deployment**: uvicorn

### ML Dependencies (`[ml]`)
- **Automated ML**: pycaret (Phase 2)

## 🔄 Migration Benefits

### Before (Multiple Files)
```
requirements.txt           # Outdated, everything mixed
requirements-core.txt      # Core dependencies
requirements-dev.txt       # Development tools
requirements-optional.txt  # Cloud services
```

### After (Single pyproject.toml)
```toml
[project]
dependencies = [...]       # Core only

[project.optional-dependencies]
dev = [...]               # Development tools
cloud = [...]             # Cloud services  
ml = [...]                # ML features
```

## ✅ Verification

The new dependency management is working correctly:
- ✅ All 35 unit tests passing
- ✅ Editable installation working
- ✅ Import functionality verified
- ✅ Makefile commands updated
- ✅ Documentation updated

## 🎯 Next Steps

For Phase 2 development:
```bash
# Install ML dependencies when ready
make install-ml

# This will add PyCaret and enable advanced ML workflows
```

This modern dependency management provides a clean, maintainable foundation for continued development.