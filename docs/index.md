# Magik Merlin ML Platform

A modern, comprehensive machine learning experimentation platform built with Panel UI and a custom ML engine. Designed for Python 3.13+ with MLflow integration and support for multiple data sources.

## Overview

The Magik Merlin ML Platform provides a complete environment for ML experimentation, from data ingestion to model deployment. It features a custom-built ML engine that avoids dependency lock-in while maintaining full sklearn compatibility.

### Key Features

- 📊 **Data Management**: Connect to multiple data sources (Local files, Snowflake, AWS S3)
- 🔬 **Custom ML Engine**: XGBoost, LightGBM, and CatBoost with sklearn-compatible interfaces
- 🤖 **AutoML Pipeline**: Automated model comparison and hyperparameter optimization with Optuna
- 📈 **Experiment Tracking**: Complete lifecycle management with MLflow integration
- 🎯 **Model Evaluation**: Comprehensive comparison with feature importance analysis
- 🎨 **Interactive UI**: Panel-based web interface with real-time updates
- 🚀 **Deployment Ready**: Production deployment and monitoring (Phase 3+)

## Why Choose This Platform?

### Modern Python Support
- **Python 3.13+** with latest language features
- No version lock-in (unlike PyCaret which only supports 3.9-3.11)
- Future-proof architecture

### Custom ML Engine
We built a custom ML engine instead of using existing frameworks to avoid:
- **Dependency constraints** - Install only what you need
- **Version lock-in** - Compatible with latest Python versions
- **Framework limitations** - Full control over ML workflows

While maintaining:
- ✅ **Sklearn compatibility** - Works with all sklearn tools
- ✅ **MLflow integration** - Automatic experiment tracking
- ✅ **Easy extensibility** - Add new models easily

### Production-Ready
- Comprehensive test suite (92 tests, >90% coverage)
- Strict code quality standards (ruff + mypy)
- Semantic versioning with automated releases
- Complete documentation

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/sgjholt/magik_merlin_ml_platform.git
cd magik_merlin_ml_platform

# Install with ML support
uv sync --extra ml

# Setup the platform
./run.sh setup
```

### Run the Platform

```bash
# Start development server
./run.sh dev

# Platform available at http://localhost:5006
```

See [Quick Start Guide](getting-started/quick-start.md) for detailed instructions.

## Documentation

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Getting Started__

    ---

    Installation, configuration, and your first experiment

    [:octicons-arrow-right-24: Get started](getting-started/installation.md)

-   :material-book-open-variant:{ .lg .middle } __User Guide__

    ---

    Comprehensive guides for all platform features

    [:octicons-arrow-right-24: User Guide](user-guide/overview.md)

-   :material-code-braces:{ .lg .middle } __Development__

    ---

    Contributing, architecture, and code quality standards

    [:octicons-arrow-right-24: Development](development/contributing.md)

-   :material-api:{ .lg .middle } __API Reference__

    ---

    Detailed API documentation for all modules

    [:octicons-arrow-right-24: API Docs](api/ml-engine.md)

</div>

## Project Status

### Phase 2: Core ML Functionality ✅ Complete

- [x] Custom ML Engine (XGBoost, LightGBM, CatBoost)
- [x] AutoML Pipeline with Optuna
- [x] 92 comprehensive tests (>90% coverage)
- [x] Complete documentation
- [x] MLflow integration

See [Roadmap](roadmap.md) for upcoming features.

## Requirements

- **Python 3.13+** (leverages modern language features)
- **uv** package manager (recommended) or pip
- Optional: MLflow server for experiment tracking
- ML Libraries: XGBoost, LightGBM, CatBoost

## Community

- **GitHub**: [sgjholt/magik_merlin_ml_platform](https://github.com/sgjholt/magik_merlin_ml_platform)
- **Issues**: Report bugs or request features
- **Discussions**: Ask questions and share ideas

## License

This project is licensed under the MIT License - see the LICENSE file for details.
