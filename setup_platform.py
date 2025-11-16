#!/usr/bin/env python3
"""
ML Platform Setup Script

This script sets up the ML Platform environment including:
- MLflow server initialization
- Directory structure creation
- Basic configuration validation
- Dependency checks
"""

import subprocess
import sys
import time
from pathlib import Path

import click

from src.config.settings import settings
from src.core.logging import get_logger, setup_logging


def check_dependencies() -> bool:
    """Check if required dependencies are installed"""
    required_packages = ["mlflow", "panel", "pandas", "requests"]
    missing = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("Install with: uv sync")
        return False

    print("✅ All dependencies available")
    return True


def create_directories() -> None:
    """Create necessary directories for the platform"""
    directories = [
        "logs",
        "mlflow_data",
        "mlflow_data/artifacts",
        "experiments",
        "experiments/metadata",
        "experiments/models",
        "experiments/artifacts",
        "data",
        "notebooks",
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(exist_ok=True, parents=True)
        print(f"📁 Created directory: {dir_path}")


def setup_mlflow() -> bool:
    """Set up MLflow server"""
    print("🚀 Setting up MLflow server...")

    # Check if MLflow server is already running
    try:
        result = subprocess.run(
            [sys.executable, "scripts/start_mlflow.py", "status"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0 and "running" in result.stdout.lower():
            print("✅ MLflow server already running")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Start MLflow server in background
    try:
        print("Starting MLflow server...")
        process = subprocess.Popen(
            [sys.executable, "scripts/start_mlflow.py", "start", "--host", "localhost"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Give it time to start
        time.sleep(5)

        # Check if it started successfully
        if process.poll() is None:  # Still running
            print("✅ MLflow server started successfully")
            print(f"🌐 Web UI available at: {settings.mlflow_tracking_uri}")
            return True
        print("❌ Failed to start MLflow server")
        return False

    except Exception as e:
        print(f"❌ Error starting MLflow: {e}")
        return False


@click.command()
@click.option("--skip-mlflow", is_flag=True, help="Skip MLflow server setup")
@click.option("--reset", is_flag=True, help="Reset and recreate all directories")
def setup_platform(skip_mlflow: bool, reset: bool) -> None:
    """Set up the ML Platform environment"""

    print("🔧 Setting up ML Platform...")
    print("=" * 50)

    # Initialize logging
    setup_logging()
    logger = get_logger(__name__, pipeline_stage="platform_setup")

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Create directories
    if reset:
        print("🔄 Resetting platform directories...")
        import shutil

        for dir_name in ["mlflow_data", "experiments", "logs"]:
            if Path(dir_name).exists():
                shutil.rmtree(dir_name)
                print(f"🗑️  Removed {dir_name}")

    create_directories()

    # Setup MLflow
    if not skip_mlflow:
        if not setup_mlflow():
            print("\n⚠️  MLflow setup failed. You can:")
            print("   1. Run manually: ./scripts/mlflow.sh start")
            print("   2. Skip MLflow with: python setup_platform.py --skip-mlflow")
            print("   3. Check troubleshooting in README")
    else:
        print("⏭️  Skipping MLflow setup")

    # Validate configuration
    print("\n🔍 Validating configuration...")
    logger.info("Platform setup validation")

    from src.core.experiments import ExperimentTracker

    tracker = ExperimentTracker()
    if tracker.is_server_available():
        print("✅ MLflow tracking available")
    else:
        print("⚠️  MLflow tracking not available (can work in offline mode)")

    print("\n" + "=" * 50)
    print("🎉 ML Platform setup complete!")
    print("\nNext steps:")
    print("1. Start the platform: python main.py")
    print("2. Open web interface at: http://localhost:5006")
    print("3. Check MLflow UI at:", settings.mlflow_tracking_uri)

    if not skip_mlflow:
        print("\n💡 MLflow Commands:")
        print("   ./scripts/mlflow.sh status   # Check server status")
        print("   ./scripts/mlflow.sh ui       # Open MLflow web UI")
        print("   ./scripts/mlflow.sh stop     # Stop MLflow server")


@click.command()
def validate_setup() -> None:
    """Validate the current platform setup"""

    setup_logging()
    logger = get_logger(__name__, pipeline_stage="platform_validation")

    print("🔍 Validating ML Platform setup...")

    # Check directories
    required_dirs = ["logs", "mlflow_data", "experiments"]
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ Directory exists: {dir_path}")
        else:
            print(f"❌ Missing directory: {dir_path}")

    # Check MLflow
    from src.core.experiments import ExperimentTracker

    tracker = ExperimentTracker()
    if tracker.is_server_available():
        print("✅ MLflow server accessible")
    else:
        print("❌ MLflow server not accessible")
        print("   Start with: ./scripts/mlflow.sh start")

    # Test basic functionality
    try:
        from src.ui.app import MLPlatformApp

        app = MLPlatformApp()
        print("✅ ML Platform app initializes successfully")
    except Exception as e:
        print(f"❌ ML Platform app initialization failed: {e}")

    print("\n✅ Validation complete")


@click.group()
def cli():
    """ML Platform Setup and Validation"""


cli.add_command(setup_platform, name="setup")
cli.add_command(validate_setup, name="validate")


if __name__ == "__main__":
    cli()
