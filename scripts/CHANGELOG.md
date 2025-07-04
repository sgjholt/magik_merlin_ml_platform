# CHANGELOG


## v0.3.1 (2025-06-22)

### Bug Fixes

- Fixed server stopping routine by finding PIDs connected to running apps on address:port directly
  and killing them.
  ([`7ce9ee7`](https://github.com/sgjholt/magik_merlin_ml_platform/commit/7ce9ee7fa53b079ae14a39e636b59a80f3adce6c))


## v0.3.0 (2025-06-22)

### Bug Fixes

- Update changelog configuration for semantic release
  ([`5a525ea`](https://github.com/sgjholt/magik_merlin_ml_platform/commit/5a525ea8e25002dc330feacc4d6de4fbbc9e5037))

### Documentation

- Update README with automatic versioning information
  ([`cf8e8ed`](https://github.com/sgjholt/magik_merlin_ml_platform/commit/cf8e8ede1c5b696cd249186f8b47d109713d04e4))

Add explanation of how semantic versioning now works automatically with GitHub.

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>

### Features

- Add comprehensive visualization system with interactive charts
  ([`03ac692`](https://github.com/sgjholt/magik_merlin_ml_platform/commit/03ac6921602a0ce264965955ca5c4a5a66d3c5f7))

- Create modular visualization framework with base classes and themes - Implement data exploration
  visualizations (dataset overview, distributions, correlation matrix, missing data analysis, time
  series) - Add model evaluation visualizations (model comparison, ROC curves, confusion matrix,
  feature importance, learning curves, regression analysis) - Build experiment tracking
  visualizations (experiment history, comparison, metrics explorer, training progress) - Integrate
  visualization panel into main application with category-based navigation - Support interactive
  controls for dynamic chart customization - Apply consistent PlotTheme across all visualizations -
  Include comprehensive error handling and empty state management

The visualization system provides data scientists with powerful tools for exploring datasets,
  evaluating model performance, and tracking experiment progress through interactive Plotly-based
  charts.

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>


## v0.2.0 (2025-06-22)

### Bug Fixes

- Implement comprehensive MLflow server setup and configuration
  ([`30be635`](https://github.com/sgjholt/magik_merlin_ml_platform/commit/30be6352dc417f048acc323f0fb89ce579dc424c))

- Add MLflow server startup scripts with proper configuration management - Create setup_platform.py
  for automated environment initialization - Implement robust MLflow health checking with fallback
  endpoints - Fix ExperimentTracker to use settings for tracking URI configuration - Add MLflow
  management scripts (start, stop, status, ui commands) - Configure proper MLflow backend store and
  artifact storage - Update settings to use 127.0.0.1 for better local development - Test MLflow
  integration with experiment runs and parameter logging - Create comprehensive platform setup and
  validation scripts

MLflow server now starts properly and integrates seamlessly with experiment tracking.

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>

### Chores

- Prepare repository for GitHub with proper .gitignore
  ([`0517526`](https://github.com/sgjholt/magik_merlin_ml_platform/commit/05175269d9c25065896a225855761d9d5b1138ee))

- Update .gitignore to exclude MLflow data, experiments, and logs - Remove generated files and
  directories from tracking - Clean up egg-info and temporary artifacts - Prepare for semantic
  versioning with GitHub remote

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>

### Features

- Add MLflow server controls to Panel UI sidebar
  ([`f2d8cce`](https://github.com/sgjholt/magik_merlin_ml_platform/commit/f2d8cce80b1c1c3bae86a6065637a841d1e7dea3))

- Add Start/Stop MLflow buttons with real-time status indicators - Implement MLflow UI button to
  open tracking interface in browser - Add threaded MLflow server startup to prevent UI blocking -
  Update status indicators dynamically based on server availability - Include comprehensive MLflow
  management in README - Test programmatic server start/stop functionality for UI integration

Users can now manage MLflow server directly from the platform interface without command line.

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>

- Implement comprehensive experiment tracking capabilities
  ([`663ed53`](https://github.com/sgjholt/magik_merlin_ml_platform/commit/663ed53267f2b5ef0354eb63587a970306e03c2c))

- Add ExperimentManager for enhanced experiment lifecycle management - Support for experiment
  metadata, progress tracking, and error handling - Model versioning and artifact management with
  file storage - Real-time experiment history and comparison features in UI - Integration with
  MLflow for advanced experiment tracking - Persistent experiment storage with JSON metadata files -
  Enhanced experimentation panel with history table and comparison tools - Performance monitoring
  and structured experiment logging

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>

- Implement comprehensive logging system with Python standard library
  ([`6a20b0d`](https://github.com/sgjholt/magik_merlin_ml_platform/commit/6a20b0d635c4b4ca5e9252806636dc8528bbe7cd))

- Add enterprise-grade logging configuration with environment-based setup - Support for structured
  JSON logging in production, colored console in development - ML-specific log formatters with
  experiment tracking context - Performance monitoring decorator for ML operations - Comprehensive
  error logging with stack traces and context - File rotation and retention policies for different
  log types - Replace print statements with proper logging in data sources and ML modules -
  Integration with MLflow and monitoring systems

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>


## v0.1.0 (2025-06-22)

### Features

- Initial ML platform implementation with Panel UI and automated versioning
  ([`71d8be1`](https://github.com/sgjholt/magik_merlin_ml_platform/commit/71d8be132365dbf57526a09eba9c2f70f2f325ca))

- Complete Panel-based ML experimentation platform - Data source connectors for local files, AWS S3,
  Snowflake - PyCaret integration for automated ML workflows - MLflow experiment tracking -
  Interactive UI with data management, experimentation, evaluation, and deployment panels -
  Comprehensive test suite with unit and integration tests - Automated versioning with
  python-semantic-release - Type annotations and Ruff code quality tools

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
