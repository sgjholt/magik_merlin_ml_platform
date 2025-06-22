#!/bin/bash

# ML Platform Runner Script
# Simplifies common development and operational tasks

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLATFORM_NAME="ML Experimentation Platform"
DEFAULT_PORT=5006
MLFLOW_PORT=5000

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        "INFO")  echo -e "${GREEN}[INFO]${NC}  [$timestamp] $message" ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC}  [$timestamp] $message" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} [$timestamp] $message" ;;
        "DEBUG") echo -e "${BLUE}[DEBUG]${NC} [$timestamp] $message" ;;
        *)       echo -e "${CYAN}[$level]${NC} [$timestamp] $message" ;;
    esac
}

# Help function
show_help() {
    cat << EOF
${PURPLE}$PLATFORM_NAME${NC} - Command Runner

${YELLOW}USAGE:${NC}
    ./run.sh <command> [options]

${YELLOW}SETUP COMMANDS:${NC}
    ${GREEN}install${NC}              Install dependencies
    ${GREEN}install-dev${NC}          Install with development dependencies
    ${GREEN}install-ml${NC}           Install with ML dependencies (PyCaret, etc.)
    ${GREEN}install-cloud${NC}        Install with cloud dependencies
    ${GREEN}install-all${NC}          Install all dependencies
    ${GREEN}setup${NC}                Complete platform setup (dirs, MLflow, etc.)

${YELLOW}DEVELOPMENT COMMANDS:${NC}
    ${GREEN}dev${NC}                  Start development server
    ${GREEN}start${NC}                Start production server
    ${GREEN}mlflow start${NC}         Start MLflow tracking server
    ${GREEN}mlflow stop${NC}          Stop MLflow tracking server
    ${GREEN}mlflow status${NC}        Check MLflow server status
    ${GREEN}mlflow ui${NC}            Open MLflow web interface

${YELLOW}CODE QUALITY:${NC}
    ${GREEN}lint${NC}                 Run code linting with ruff
    ${GREEN}format${NC}               Format code with ruff
    ${GREEN}test${NC}                 Run all tests
    ${GREEN}test-fast${NC}            Run unit tests only
    ${GREEN}test-integration${NC}     Run integration tests only
    ${GREEN}coverage${NC}             Generate test coverage report

${YELLOW}BUILD & RELEASE:${NC}
    ${GREEN}build${NC}                Build distribution packages
    ${GREEN}version${NC}              Show current version
    ${GREEN}release${NC}              Create new semantic release
    ${GREEN}changelog${NC}            View recent changelog entries

${YELLOW}UTILITIES:${NC}
    ${GREEN}clean${NC}                Clean build artifacts and caches
    ${GREEN}status${NC}               Show platform status
    ${GREEN}logs${NC}                 Show application logs
    ${GREEN}health${NC}               Run health checks
    ${GREEN}demo${NC}                 Run with demo data

${YELLOW}OPTIONS:${NC}
    ${GREEN}--port PORT${NC}          Specify port (default: $DEFAULT_PORT)
    ${GREEN}--host HOST${NC}          Specify host (default: localhost)
    ${GREEN}--debug${NC}              Enable debug mode
    ${GREEN}--help${NC}               Show this help message

${YELLOW}EXAMPLES:${NC}
    ./run.sh setup                   # Complete platform setup
    ./run.sh dev --port 8080        # Start dev server on port 8080
    ./run.sh mlflow start           # Start MLflow server
    ./run.sh test                   # Run all tests
    ./run.sh release                # Create new release

EOF
}

# Check if uv is installed
check_uv() {
    if ! command -v uv &> /dev/null; then
        log "ERROR" "uv is not installed. Please install it first:"
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
}

# Parse command line arguments
parse_args() {
    PORT=$DEFAULT_PORT
    HOST="localhost"
    DEBUG=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --port)
                PORT="$2"
                shift 2
                ;;
            --host)
                HOST="$2"
                shift 2
                ;;
            --debug)
                DEBUG=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                break
                ;;
        esac
    done
}

# Installation commands
cmd_install() {
    log "INFO" "Installing core dependencies..."
    check_uv
    uv sync
    log "INFO" "Core dependencies installed successfully"
}

cmd_install_dev() {
    log "INFO" "Installing development dependencies..."
    check_uv
    uv sync --extra dev
    log "INFO" "Development dependencies installed successfully"
}

cmd_install_ml() {
    log "INFO" "Installing ML dependencies..."
    check_uv
    uv sync --extra ml
    log "INFO" "ML dependencies installed successfully"
}

cmd_install_cloud() {
    log "INFO" "Installing cloud dependencies..."
    check_uv
    uv sync --extra cloud
    log "INFO" "Cloud dependencies installed successfully"
}

cmd_install_all() {
    log "INFO" "Installing all dependencies..."
    check_uv
    uv sync --extra dev --extra ml --extra cloud
    log "INFO" "All dependencies installed successfully"
}

cmd_setup() {
    log "INFO" "Setting up ML Platform..."
    check_uv
    
    # Install dependencies
    cmd_install_dev
    
    # Create necessary directories
    log "INFO" "Creating platform directories..."
    mkdir -p data/{raw,processed,models}
    mkdir -p logs
    mkdir -p experiments/{metadata,models,artifacts}
    mkdir -p mlruns
    mkdir -p mlartifacts
    
    # Setup platform
    if [ -f "setup_platform.py" ]; then
        log "INFO" "Running platform setup..."
        uv run python setup_platform.py setup
    fi
    
    log "INFO" "Platform setup completed successfully!"
    log "INFO" "You can now run: ./run.sh dev"
}

# Development commands
cmd_dev() {
    log "INFO" "Starting development server on $HOST:$PORT..."
    check_uv
    
    export PANEL_DEV=true
    if [ "$DEBUG" = true ]; then
        export LOG_LEVEL=DEBUG
    fi
    
    uv run panel serve main.py --port "$PORT" --host "$HOST" --show --autoreload
}

cmd_start() {
    log "INFO" "Starting production server on $HOST:$PORT..."
    check_uv
    
    export PANEL_DEV=false
    uv run panel serve main.py --port "$PORT" --host "$HOST"
}

# MLflow commands
cmd_mlflow() {
    local action=$1
    case $action in
        "start")
            log "INFO" "Starting MLflow tracking server..."
            if pgrep -f "mlflow server" > /dev/null; then
                log "WARN" "MLflow server is already running"
                return 0
            fi
            
            if [ -f "scripts/mlflow.sh" ]; then
                ./scripts/mlflow.sh start --host "$HOST" --port "$MLFLOW_PORT"
            else
                uv run mlflow server --host "$HOST" --port "$MLFLOW_PORT" \
                    --backend-store-uri sqlite:///mlflow.db \
                    --default-artifact-root ./mlartifacts &
            fi
            log "INFO" "MLflow server started on http://$HOST:$MLFLOW_PORT"
            ;;
        "stop")
            log "INFO" "Stopping MLflow tracking server..."
            pkill -f "mlflow server" || log "WARN" "No MLflow server processes found"
            log "INFO" "MLflow server stopped"
            ;;
        "status")
            if pgrep -f "mlflow server" > /dev/null; then
                log "INFO" "MLflow server is running"
                curl -s "http://$HOST:$MLFLOW_PORT/health" > /dev/null && \
                    log "INFO" "MLflow server is healthy" || \
                    log "WARN" "MLflow server is not responding"
            else
                log "INFO" "MLflow server is not running"
            fi
            ;;
        "ui")
            log "INFO" "Opening MLflow UI..."
            if command -v open &> /dev/null; then
                open "http://$HOST:$MLFLOW_PORT"
            elif command -v xdg-open &> /dev/null; then
                xdg-open "http://$HOST:$MLFLOW_PORT"
            else
                log "INFO" "Please open http://$HOST:$MLFLOW_PORT in your browser"
            fi
            ;;
        *)
            log "ERROR" "Unknown MLflow command: $action"
            log "INFO" "Available: start, stop, status, ui"
            exit 1
            ;;
    esac
}

# Code quality commands
cmd_lint() {
    log "INFO" "Running code linting..."
    check_uv
    uv run ruff check src/ tests/ --fix
    log "INFO" "Linting completed"
}

cmd_format() {
    log "INFO" "Formatting code..."
    check_uv
    uv run ruff format src/ tests/
    log "INFO" "Code formatting completed"
}

cmd_test() {
    log "INFO" "Running all tests..."
    check_uv
    uv run pytest tests/ -v
    log "INFO" "Tests completed"
}

cmd_test_fast() {
    log "INFO" "Running unit tests..."
    check_uv
    uv run pytest tests/unit/ -v
    log "INFO" "Unit tests completed"
}

cmd_test_integration() {
    log "INFO" "Running integration tests..."
    check_uv
    uv run pytest tests/integration/ -v
    log "INFO" "Integration tests completed"
}

cmd_coverage() {
    log "INFO" "Generating test coverage report..."
    check_uv
    uv run pytest tests/ --cov=src --cov-report=html --cov-report=term
    log "INFO" "Coverage report generated in htmlcov/"
}

# Build & release commands
cmd_build() {
    log "INFO" "Building distribution packages..."
    check_uv
    uv build
    log "INFO" "Build completed. Packages in dist/"
}

cmd_version() {
    if [ -f "src/__init__.py" ]; then
        local version=$(grep "__version__" src/__init__.py | cut -d'"' -f2)
        log "INFO" "Current version: $version"
    else
        log "ERROR" "Cannot find version information"
        exit 1
    fi
}

cmd_release() {
    log "INFO" "Creating semantic release..."
    check_uv
    
    # Run tests first
    log "INFO" "Running tests before release..."
    cmd_test
    
    # Create release
    uv run semantic-release version
    log "INFO" "Release created successfully"
}

cmd_changelog() {
    log "INFO" "Recent changelog entries:"
    if [ -f "CHANGELOG.md" ]; then
        head -50 CHANGELOG.md
    else
        log "WARN" "CHANGELOG.md not found"
    fi
}

# Utility commands
cmd_clean() {
    log "INFO" "Cleaning build artifacts and caches..."
    
    # Python caches
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    
    # Build artifacts
    rm -rf build/ dist/ *.egg-info/ .eggs/
    
    # Test artifacts
    rm -rf .pytest_cache/ .coverage htmlcov/
    
    # Ruff cache
    rm -rf .ruff_cache/
    
    log "INFO" "Cleanup completed"
}

cmd_status() {
    log "INFO" "Platform Status:"
    echo
    
    # Version
    cmd_version
    
    # Dependencies
    echo -e "${YELLOW}Dependencies:${NC}"
    if command -v uv &> /dev/null; then
        echo "  ✓ uv installed"
    else
        echo "  ✗ uv not installed"
    fi
    
    # MLflow status
    echo -e "${YELLOW}MLflow:${NC}"
    if pgrep -f "mlflow server" > /dev/null; then
        echo "  ✓ MLflow server running"
    else
        echo "  ✗ MLflow server not running"
    fi
    
    # Git status
    echo -e "${YELLOW}Git:${NC}"
    if git rev-parse --git-dir > /dev/null 2>&1; then
        local branch=$(git branch --show-current)
        local status=$(git status --porcelain | wc -l | tr -d ' ')
        echo "  ✓ Git repository (branch: $branch, uncommitted: $status)"
    else
        echo "  ✗ Not a git repository"
    fi
}

cmd_logs() {
    log "INFO" "Application logs:"
    if [ -d "logs" ]; then
        find logs -name "*.log" -exec tail -f {} +
    else
        log "WARN" "No logs directory found"
    fi
}

cmd_health() {
    log "INFO" "Running health checks..."
    
    # Check Python environment
    log "INFO" "Checking Python environment..."
    uv run python -c "import sys; print(f'Python {sys.version}')"
    
    # Check imports
    log "INFO" "Checking core imports..."
    uv run python -c "
import src.ui.app
import src.core.experiments
import src.ui.visualizations
print('✓ All core modules import successfully')
"
    
    # Check MLflow connection
    log "INFO" "Checking MLflow connection..."
    if pgrep -f "mlflow server" > /dev/null; then
        curl -s "http://$HOST:$MLFLOW_PORT/health" > /dev/null && \
            log "INFO" "✓ MLflow server is healthy" || \
            log "WARN" "✗ MLflow server is not responding"
    else
        log "INFO" "MLflow server is not running"
    fi
    
    log "INFO" "Health checks completed"
}

cmd_demo() {
    log "INFO" "Starting platform with demo data..."
    
    # Create demo data if it doesn't exist
    if [ ! -f "data/demo_dataset.csv" ]; then
        log "INFO" "Creating demo dataset..."
        uv run python -c "
import pandas as pd
import numpy as np

# Create a simple demo dataset
np.random.seed(42)
n_samples = 1000

data = {
    'feature_1': np.random.normal(0, 1, n_samples),
    'feature_2': np.random.exponential(2, n_samples),
    'feature_3': np.random.choice(['A', 'B', 'C'], n_samples),
    'target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
}

df = pd.DataFrame(data)
df.to_csv('data/demo_dataset.csv', index=False)
print('Demo dataset created: data/demo_dataset.csv')
"
    fi
    
    export DEMO_MODE=true
    cmd_dev
}

# Main command dispatcher
main() {
    cd "$SCRIPT_DIR"
    
    if [ $# -eq 0 ]; then
        show_help
        exit 1
    fi
    
    # Parse global options first
    parse_args "$@"
    
    # Get remaining arguments after parsing options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --port|--host|--debug)
                # Skip already parsed options
                shift 2 2>/dev/null || shift
                ;;
            *)
                break
                ;;
        esac
    done
    
    local command=$1
    shift || true
    
    case $command in
        "install")        cmd_install ;;
        "install-dev")    cmd_install_dev ;;
        "install-ml")     cmd_install_ml ;;
        "install-cloud")  cmd_install_cloud ;;
        "install-all")    cmd_install_all ;;
        "setup")          cmd_setup ;;
        "dev")            cmd_dev ;;
        "start")          cmd_start ;;
        "mlflow")         cmd_mlflow "$@" ;;
        "lint")           cmd_lint ;;
        "format")         cmd_format ;;
        "test")           cmd_test ;;
        "test-fast")      cmd_test_fast ;;
        "test-integration") cmd_test_integration ;;
        "coverage")       cmd_coverage ;;
        "build")          cmd_build ;;
        "version")        cmd_version ;;
        "release")        cmd_release ;;
        "changelog")      cmd_changelog ;;
        "clean")          cmd_clean ;;
        "status")         cmd_status ;;
        "logs")           cmd_logs ;;
        "health")         cmd_health ;;
        "demo")           cmd_demo ;;
        "help"|"--help")  show_help ;;
        *)
            log "ERROR" "Unknown command: $command"
            echo
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"