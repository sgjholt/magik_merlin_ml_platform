# Magik Merlin ML Platform

A modern Python ML experimentation platform with a custom ML engine, Panel UI, and MLflow integration. Built for Python 3.13+ with no dependency lock-in.

[![CI](https://github.com/sgjholt/magik_merlin_ml_platform/workflows/CI/badge.svg)](https://github.com/sgjholt/magik_merlin_ml_platform/actions)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🔬 **Custom ML Engine** - XGBoost, LightGBM, CatBoost with sklearn compatibility
- 🤖 **AutoML** - Automated model comparison and hyperparameter optimization
- 📊 **Interactive UI** - Panel-based web interface
- 📈 **Experiment Tracking** - MLflow integration
- 🚀 **Modern Python** - Python 3.13+ with latest features

## Quick Start

```bash
# Install with ML support
uv sync --extra ml

# Run the platform
./run.sh dev

# Platform available at http://localhost:5006
```

## Documentation

📚 **[Read the full documentation](https://sgjholt.github.io/magik_merlin_ml_platform/)** (coming soon)

Or build locally:

```bash
# Install docs dependencies
uv sync --extra dev

# Serve documentation locally
mkdocs serve

# View at http://localhost:8000
```

### Quick Links

- [Installation Guide](docs/getting-started/installation.md)
- [Quick Start Tutorial](docs/getting-started/quick-start.md)
- [ML Engine Guide](docs/user-guide/ml-engine.md)
- [Development Guide](docs/development/code-quality.md)
- [API Reference](docs/api/ml-engine.md)
- [Roadmap](docs/roadmap.md)

## Why This Platform?

### No Dependency Lock-In

We built a **custom ML engine** instead of using PyCaret because:
- ❌ PyCaret 3.3 only supports Python 3.9-3.11
- ✅ Our engine supports Python 3.13+ with modern features
- ✅ Install only what you need (XGBoost, LightGBM, or CatBoost)
- ✅ Full control over ML workflows

### Sklearn Compatible

```python
from src.core.ml_engine import AutoMLPipeline

# Works like any sklearn model
pipeline = AutoMLPipeline(task_type="classification")
results = pipeline.compare_models(X, y, cv=5)
best_model = pipeline.get_best_model()
predictions = best_model.predict(X_new)
```

### Production Ready

- ✅ 92 comprehensive tests (>90% coverage)
- ✅ Strict code quality (ruff + mypy)
- ✅ Automatic CI/CD with GitHub Actions
- ✅ Semantic versioning

## Requirements

- **Python 3.13+** - Modern language features
- **uv** - Fast package manager (or pip)
- Optional: MLflow server for tracking

## Installation

### Using uv (Recommended)

```bash
# Clone repository
git clone https://github.com/sgjholt/magik_merlin_ml_platform.git
cd magik_merlin_ml_platform

# Install with ML engine
uv sync --extra ml

# Setup platform
./run.sh setup
```

### Using pip

```bash
# Create virtual environment
python3.13 -m venv .venv
source .venv/bin/activate

# Install
pip install -e ".[ml]"
```

See [Installation Guide](docs/getting-started/installation.md) for details.

## Development

### Running Tests

```bash
./run.sh test              # All tests
./run.sh test-fast         # Unit tests only
./run.sh coverage          # With coverage report
```

### Code Quality

```bash
./run.sh lint              # Check code quality
./run.sh format            # Auto-format code
```

### Building Documentation

```bash
# Serve locally
mkdocs serve

# Build static site
mkdocs build
```

See [Development Guide](docs/development/code-quality.md) for detailed standards.

## Project Structure

```
ml_platform/
├── src/
│   ├── core/
│   │   ├── ml_engine/          # Custom ML Engine ✅
│   │   ├── data_sources/       # Data connectors
│   │   └── experiments/        # MLflow integration
│   └── ui/                     # Panel-based UI
├── tests/                      # 92 tests, >90% coverage
├── docs/                       # MkDocs documentation
├── examples/                   # Demo scripts & notebooks
└── pyproject.toml              # Project configuration
```

## Roadmap

### Phase 2: Core ML Functionality ✅ Complete

- [x] Custom ML Engine (XGBoost, LightGBM, CatBoost)
- [x] AutoML Pipeline with Optuna
- [x] 92 comprehensive tests
- [x] Complete documentation
- [x] MLflow integration

### Phase 3: Pipeline System (Planned)

- [ ] Visual pipeline builder
- [ ] Pipeline orchestration
- [ ] Deep learning integration (PyTorch Lightning)
- [ ] Advanced visualizations

See [full roadmap](docs/roadmap.md) for details.

## Contributing

We welcome contributions! See [Contributing Guide](docs/development/contributing.md).

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## Community

- **GitHub**: [Issues](https://github.com/sgjholt/magik_merlin_ml_platform/issues) | [Discussions](https://github.com/sgjholt/magik_merlin_ml_platform/discussions)
- **Documentation**: [Full docs](https://sgjholt.github.io/magik_merlin_ml_platform/)

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Made with ❤️ using Python 3.13+, Panel, and modern ML libraries**
