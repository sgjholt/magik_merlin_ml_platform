# ML Experimentation Platform

A comprehensive Panel-based machine learning experimentation platform with a custom ML engine, MLflow integration, and support for multiple data sources. Built for Python 3.13+ with modern gradient boosting libraries (XGBoost, LightGBM, CatBoost) and deep learning frameworks.

## 🚀 Features

- 📊 **Data Management**: Connect to multiple data sources (Local files, Snowflake, AWS S3)
- 🔬 **ML Experimentation**: Custom ML engine with XGBoost, LightGBM, and CatBoost
- 🤖 **AutoML Pipeline**: Automated model comparison and hyperparameter optimization (Optuna)
- 📈 **Experiment Tracking**: Complete experiment lifecycle management with MLflow
- 🎯 **Model Evaluation**: Comprehensive model comparison with feature importance
- 🚀 **Model Deployment**: Deploy and monitor models in production (Phase 3)

## 📋 Requirements

- **Python 3.13+** (leverages modern language features)
- Virtual environment (recommended: `uv`)
- Optional: MLflow server for experiment tracking
- ML Libraries: XGBoost, LightGBM, CatBoost (install with `--extra ml`)

## ⚡ Quick Start

### 1. Environment Setup
```bash
# Install core dependencies
uv sync

# Install with ML engine support (XGBoost, LightGBM, CatBoost, PyTorch)
uv sync --extra ml

# Install development dependencies
uv sync --extra dev --extra ml

# Setup platform (creates directories, starts MLflow)
python setup_platform.py setup
```

### 2. Start the Platform
```bash
# Start ML Platform (includes MLflow controls in UI)
python main.py

# Platform will be available at: http://localhost:5006
```

### 3. MLflow Server Management

**From UI (Recommended):**
- Use the **🚀 Start MLflow** / **🛑 Stop MLflow** buttons in the sidebar
- Click **📊 MLflow UI** to open the tracking interface
- Status indicator shows real-time connection status

**From Command Line:**
```bash
./scripts/mlflow.sh start     # Start MLflow server
./scripts/mlflow.sh status    # Check server status
./scripts/mlflow.sh stop      # Stop server  
./scripts/mlflow.sh ui        # Open web interface
./scripts/mlflow.sh restart   # Restart server
```

## 📁 Project Structure

```
ml_platform/
├── src/                         # Source code
│   ├── core/                    # Core ML functionality
│   │   ├── data_sources/        # Data connectors (Local, Snowflake, AWS)
│   │   ├── experiment_tracking/ # MLflow integration
│   │   ├── ml_engine/          # Custom ML Engine (XGBoost, LightGBM, CatBoost)
│   │   └── pipeline_orchestration/ # Workflow management (Phase 3)
│   ├── ui/                      # User interface
│   │   ├── panels/             # Main UI panels
│   │   ├── components/         # Reusable components
│   │   └── visualizations/     # Charts and plots
│   ├── config/                 # Configuration management
│   └── utils/                  # Utility functions
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests (fast)
│   └── integration/            # Integration tests
├── data/                       # Data storage
├── docs/                       # Documentation
├── main.py                     # Application entry point
└── pyproject.toml              # Dependencies and project config
```

## 🎯 Usage Guide

### 📊 Data Management
1. **Connect**: Choose data source (Local Files, Snowflake, AWS S3)
2. **Configure**: Set connection parameters
3. **Load**: Select and load your dataset
4. **Explore**: Preview data and view profiling statistics

### 🔬 ML Experimentation (Phase 2 ✅ Complete)
1. **Setup**: Choose ML task type (classification/regression) and target variable
2. **Compare**: Automatically compare XGBoost, LightGBM, and CatBoost models
3. **Optimize**: Hyperparameter tuning with Optuna (configurable trials and CV folds)
4. **Track**: All experiments logged to MLflow with parameters and metrics

### 📈 Model Evaluation (Phase 2 ✅ Complete)
1. **Compare**: Cross-validation scores and test performance for all models
2. **Visualize**: Feature importance charts from gradient boosting models
3. **Analyze**: Detailed performance metrics (accuracy, R², etc.)
4. **Select**: Best model automatically identified and ready for deployment

### 🚀 Deployment (Coming in Phase 3)
1. **Choose**: Select trained model for deployment
2. **Configure**: Set deployment environment and settings
3. **Deploy**: Launch model endpoint
4. **Monitor**: Track performance and usage

## ⚙️ Installation Options

### Core Installation (Recommended)
```bash
make install  # Essential dependencies only
```

### Development Installation
```bash
make install-dev  # Includes testing and code quality tools
```

### Cloud Services Installation
```bash
make install-cloud  # Adds Snowflake, AWS, MLflow support
```

### ML Features Installation (Phase 2)
```bash
make install-ml  # Adds PyCaret for automated ML workflows
```

### Full Installation
```bash
make install-all  # Everything including all optional dependencies
```

## 🧪 Testing

```bash
# Run all tests
make test

# Quick unit tests only
make test-fast

# Integration tests
make test-integration

# Coverage report
make test-coverage
```

## 🛠️ Development

### Code Quality
```bash
make format  # Format code with black
make lint    # Check code quality
```

### Environment Setup
```bash
make setup-dev  # Install dev tools and git hooks
```

## Architecture

The platform follows a modular architecture:

- **Data Layer**: Extensible data source connectors
- **ML Engine**: PyCaret integration for automated ML
- **Tracking Layer**: MLflow for experiment management
- **UI Layer**: Panel-based interactive dashboard
- **Orchestration**: Custom pipeline execution engine

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.