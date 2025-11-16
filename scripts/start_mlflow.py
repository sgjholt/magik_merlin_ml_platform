#!/usr/bin/env python3
"""
MLflow Server Startup Script

This script starts a local MLflow tracking server with proper configuration
for the ML Platform. It sets up the server with a local file store backend
and optional database backend for production use.
"""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import click
import requests

# Add project root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, REPO_ROOT.as_posix)

# Import settings directly to avoid circular imports
DEFAULT_MLFLOW_URI = "http://localhost:5000"
DEFAULT_EXPERIMENT_NAME = "ml-platform-experiments"


def check_mlflow_server(host: str, port: int, timeout: int = 30) -> bool:
    """Check if MLflow server is running and accessible"""
    url = f"http://{host}:{port}/health"

    for _ in range(timeout):
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:  # noqa: PLR2004
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)

    return False


def get_mlflow_port() -> int:
    """Extract port from MLflow tracking URI"""
    uri = DEFAULT_MLFLOW_URI
    if "localhost" in uri or "127.0.0.1" in uri:
        try:
            return int(uri.split(":")[-1])
        except (ValueError, IndexError):
            return 5000
    return 5000


def setup_mlflow_directories() -> tuple[Path, Path]:
    """Set up MLflow storage directories"""
    project_root = Path(__file__).parent.parent
    mlflow_dir = project_root / "mlflow_data"
    artifacts_dir = mlflow_dir / "artifacts"

    # Create directories
    mlflow_dir.mkdir(exist_ok=True)
    artifacts_dir.mkdir(exist_ok=True)

    return mlflow_dir, artifacts_dir


@click.command()
@click.option("--host", default="127.0.0.1", help="Host to bind MLflow server")
@click.option("--port", default=None, type=int, help="Port for MLflow server")
@click.option(
    "--backend-store-uri", default=None, help="Backend store URI (file or database)"
)
@click.option("--artifact-root", default=None, help="Artifact storage root")
@click.option("--no-serve-artifacts", is_flag=True, help="Disable artifact serving")
@click.option("--dev", is_flag=True, help="Development mode with auto-reload")
def start_mlflow_server(
    host: str,
    port: int | None,
    backend_store_uri: str | None,
    artifact_root: str | None,
    no_serve_artifacts: bool,
    dev: bool,
) -> None:
    """Start MLflow tracking server with proper configuration"""

    # Simple logging for script
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    # Use settings or defaults
    if port is None:
        port = get_mlflow_port()

    logger.info("Starting MLflow server on http://%s:%s", host, port)

    # Set up storage directories
    mlflow_dir, artifacts_dir = setup_mlflow_directories()

    if backend_store_uri is None:
        backend_store_uri = f"file://{mlflow_dir}/mlruns"

    if artifact_root is None:
        artifact_root = str(artifacts_dir)

    # Check if server is already running
    if check_mlflow_server(host, port, timeout=3):
        # If server is already running, log a warning and exit
        logger.warning("MLflow server already running on http://%s:%s", host, port)
        return

    # Build MLflow command
    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--host",
        host,
        "--port",
        str(port),
        "--backend-store-uri",
        backend_store_uri,
        "--default-artifact-root",
        artifact_root,
    ]

    if not no_serve_artifacts:
        cmd.append("--serve-artifacts")

    if dev:
        cmd.append("--dev")

    logger.info("MLflow command: %s", " ".join(cmd))
    logger.info("Backend store: %s", backend_store_uri)
    logger.info("Artifact root: %s", artifact_root)

    try:
        # Start MLflow server
        logger.info("Starting MLflow server...")
        env = os.environ.copy()
        env["MLFLOW_TRACKING_URI"] = f"http://{host}:{port}"

        process = subprocess.Popen(cmd, env=env)  # noqa: S603

        # Wait for server to start
        logger.info("Waiting for MLflow server to start...")
        if check_mlflow_server(host, port, timeout=30):
            logger.info("✅ MLflow server started successfully!")
            logger.info("🌐 Web UI: http://%s:%s", host, port)
            logger.info("📊 Tracking URI: http://%s:%s", host, port)
            logger.info("📁 Backend: %s", backend_store_uri)
            logger.info("📦 Artifacts: %s", artifact_root)
            logger.info("Press Ctrl+C to stop the server")

            # Keep the server running
            try:
                process.wait()
            except KeyboardInterrupt:
                logger.info("Stopping MLflow server...")
                process.terminate()
                process.wait()
                logger.info("MLflow server stopped")
        else:
            logger.error("❌ Failed to start MLflow server")
            process.terminate()
            return

    except FileNotFoundError:
        logger.exception("❌ MLflow not found. Install it with: uv add mlflow")
        return
    except Exception as e:
        msg = f"❌ Error starting MLflow server: {e!s}"
        logger.exception(msg)
        return


@click.command()
def stop_mlflow_server() -> None:
    """Stop MLflow server"""
    # Simple logging for script
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Stopping MLflow server...")
    mlflow_script_path = REPO_ROOT / "scripts" / "mlflow.sh"
    if not mlflow_script_path.exists():
        logger.error("❌ MLflow script not found: %s", mlflow_script_path)
        return
    logger.debug("Using MLflow script: %s", mlflow_script_path)
    cmd = [mlflow_script_path.as_posix(), "stop"]
    try:
        # Execute the stop command
        result = subprocess.run(  # noqa: S603
            cmd, check=True, capture_output=True, text=True
        )
        if result.returncode == 0:
            logger.info("✅ MLflow server stopped successfully")
        else:
            logger.error("❌ Failed to stop MLflow server: %s", result.stderr.strip())
    except subprocess.CalledProcessError as e:
        msg = f"❌ Error stopping MLflow server: {e!s}"
        logger.exception(msg)


@click.command()
@click.option("--host", default="127.0.0.1", help="MLflow server host")
@click.option("--port", default=None, type=int, help="MLflow server port")
def check_mlflow_status(host: str, port: int | None) -> None:
    """Check MLflow server status"""
    # Simple logging for script
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    # Use settings or defaults
    if port is None:
        port = get_mlflow_port()
    # Log the check
    logger.info("Checking MLflow server at http://%s:%s", host, port)

    if check_mlflow_server(host, port, timeout=5):
        try:
            # Check server info
            response = requests.get(
                f"http://{host}:{port}/api/2.0/mlflow/experiments/list", timeout=5
            )
            if response.status_code == 200:  # noqa: PLR2004
                experiments = response.json().get("experiments", [])
                logger.info("✅ MLflow server is running")
                logger.info("📊 Found %d experiments", len(experiments))
                logger.info("🌐 Web UI: http://%s:%s", host, port)
            else:
                msg = (
                    f"⚠️  MLflow server responding but API error: {response.status_code}"
                )
                logger.warning(msg)
        except Exception as e:
            msg = f"⚠️  Error checking MLflow API: {e!s}"
            logger.exception(msg)
    else:
        logger.error("❌ MLflow server not accessible at http://%s:%s", host, port)
        logger.info("💡 Start it with: python scripts/start_mlflow.py")


@click.group()
def cli() -> None:
    """MLflow Server Management Commands"""
    pass  # noqa: PIE790


cli.add_command(start_mlflow_server, name="start")
cli.add_command(stop_mlflow_server, name="stop")
cli.add_command(check_mlflow_status, name="status")


if __name__ == "__main__":
    cli()
