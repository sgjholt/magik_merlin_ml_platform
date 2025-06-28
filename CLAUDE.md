# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands for Development

### Quick Start Commands
```bash
# Complete platform setup (recommended first step)
./run.sh setup
# OR
make setup

# Start development server
./run.sh dev
# OR  
make dev

# Start with demo data
./run.sh demo
```

### Common Development Commands
```bash
# Code quality
./run.sh lint              # Run ruff linting with fixes
./run.sh format            # Format code with ruff
./run.sh test              # Run all tests
./run.sh test-fast         # Unit tests only
./run.sh coverage          # Generate coverage report

# MLflow management
./run.sh mlflow start      # Start MLflow tracking server
./run.sh mlflow stop       # Stop MLflow server
./run.sh mlflow status     # Check server status
./run.sh mlflow ui         # Open MLflow web interface

# Build and release
./run.sh build             # Build distribution packages
./run.sh version           # Show current version
./run.sh release           # Create semantic release (auto-increment version)

# Utilities
./run.sh status            # Show platform status
./run.sh health            # Run health checks
./run.sh clean             # Clean build artifacts
```

### Alternative Make Commands
All `./run.sh` commands have equivalent `make` commands:
```bash
make dev PORT=8080         # Start dev server on custom port
make test                  # Run tests
make mlflow-start          # Start MLflow
make release               # Create new release
```

### Running Single Tests
```bash
# Run specific test file
uv run pytest tests/unit/test_config.py -v

# Run specific test function
uv run pytest tests/unit/test_config.py::test_settings_load -v

# Run tests matching pattern
uv run pytest -k "test_data" -v
```

## Architecture Overview

### Core Architecture Pattern
The platform follows a **modular Panel-based architecture** with clear separation of concerns:

**UI Layer (Panel-based)** → **Core ML Services** → **Data Sources** → **MLflow Tracking**

### Key Architectural Components

#### 1. **Application Entry Point** (`main.py`)
- Initializes logging system first via `setup_logging()`
- Creates `MLPlatformApp` instance which orchestrates all panels
- Uses settings-based configuration for host/port/debug mode
- Serves Panel application with auto-reload in debug mode

#### 2. **Panel-Based UI Architecture** (`src/ui/`)
- **Main App** (`app.py`): Orchestrates 5 main panels in tabbed interface
- **Panels** (`panels/`): Self-contained UI modules for each major function
  - DataManagementPanel: Data source connections and loading
  - ExperimentationPanel: ML experiment configuration and execution  
  - ModelEvaluationPanel: Model comparison and metrics
  - VisualizationPanel: Interactive charts and data exploration
  - DeploymentPanel: Model deployment and monitoring
- **Visualizations** (`visualizations/`): Plotly-based interactive charts
  - Base classes for consistent theming and interactivity
  - Data exploration, model evaluation, and experiment tracking charts
  - Automatic integration with experiment manager for real-time updates

#### 3. **Data Flow Architecture**
- **Data Sources** → **Panel State Management** → **Experiment Tracking**
- Central data flow managed through callbacks:
  - `data_updated_callback`: Propagates data changes to experiment and visualization panels
  - `experiment_completed_callback`: Updates session statistics and experiment history
- Real-time UI updates via Panel's reactive programming model

#### 4. **Experiment Management System** (`src/core/experiments/`)
- **ExperimentManager**: Handles experiment lifecycle (create → start → complete)
- **ExperimentTracker**: MLflow integration for tracking runs, parameters, metrics
- **Persistent Storage**: JSON metadata files + MLflow backend for experiment history
- **Status Management**: Tracks experiment states (pending → running → completed/failed/cancelled)

#### 5. **MLflow Integration Architecture**
- **Dual Management**: UI controls + command-line scripts for MLflow server
- **Health Checking**: Automatic reconnection and status monitoring
- **Script-Based Server Management**: `scripts/start_mlflow.py` and `scripts/mlflow.sh`
- **Threaded Operations**: Non-blocking server start/stop from UI

#### 6. **Data Source Abstraction** (`src/core/data_sources/`)
- **Base Class Pattern**: All connectors inherit from `BaseDataSource`
- **Pluggable Architecture**: Easy to add new data sources (currently: Local, Snowflake, AWS S3)
- **Consistent Interface**: Uniform `connect()` and `load_data()` methods across all sources

#### 7. **Configuration Management** (`src/config/`)
- **Pydantic Settings**: Type-safe configuration with environment variable support
- **Centralized Config**: Single `settings.py` for all configuration (MLflow, data sources, app settings)
- **Environment Overrides**: `.env` file support for local development

#### 8. **Logging Architecture** (`src/core/logging/`)
- **Structured Logging**: JSON format in production, colored console in development
- **Pipeline Context**: Automatic injection of pipeline stage and experiment context
- **Performance Monitoring**: Built-in decorators for ML operation timing
- **Log Levels**: Environment-based log level configuration

### Key Integration Patterns

#### Panel-to-Panel Communication
Panels communicate via callbacks rather than direct coupling:
```python
# In MLPlatformApp
self.data_panel.data_updated_callback = self._on_data_updated
self.experiment_panel.experiment_completed_callback = self._on_experiment_completed
```

#### Experiment Lifecycle Integration
Experiments integrate across multiple panels:
1. **Data Panel**: Loads data → triggers `data_updated_callback`
2. **Experiment Panel**: Receives data → enables experiment creation
3. **Visualization Panel**: Receives data → enables data exploration charts
4. **Completion**: Experiment ends → updates history → enables comparison

#### MLflow Server State Management
The platform maintains MLflow server state across UI and backend:
- **UI Buttons**: Enable/disable based on server availability
- **Health Checking**: Periodic status updates via `experiment_tracker.is_server_available()`
- **Automatic Reconnection**: Handles server restarts gracefully

### Semantic Versioning Integration
- **Conventional Commits**: `feat:`, `fix:`, `BREAKING CHANGE:` trigger automatic version bumps
- **Automated Releases**: `./run.sh release` or `make release` creates new versions
- **CHANGELOG Generation**: Automatic changelog updates with commit details

### Testing Architecture
- **Unit Tests** (`tests/unit/`): Fast tests for individual components
- **Integration Tests** (`tests/integration/`): End-to-end workflow testing
- **Fixtures** (`tests/fixtures/`): Shared test data and mocks
- **Coverage**: HTML reports generated in `htmlcov/`

### Development Workflow Patterns
1. **Feature Development**: Create feature branch → implement → test → commit with conventional format
2. **Automatic Versioning**: Semantic-release detects commit types and bumps version automatically
3. **Code Quality**: Ruff enforces consistent formatting and linting rules
4. **Panel Development**: Extend existing panels or create new ones following the base patterns

### Important Implementation Details
- **Path Management**: Uses `REPO_ROOT = Path(__file__).parent.parent.parent` for reliable script paths
- **Thread Safety**: MLflow operations run in background threads to prevent UI blocking
- **Error Handling**: Comprehensive logging with structured error context
- **Resource Management**: Automatic cleanup of MLflow processes and temporary files