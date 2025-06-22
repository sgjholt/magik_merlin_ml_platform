#!/usr/bin/env python3
"""
ML Platform Main Application Entry Point
"""
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.config.settings import settings
from src.ui.app import MLPlatformApp


def main() -> None:
    """Main application entry point"""
    print("Starting ML Experimentation Platform...")
    print(f"MLflow Tracking URI: {settings.mlflow_tracking_uri}")
    print(f"Application will be available at: http://{settings.app_host}:{settings.app_port}")

    # Create and run the application
    app = MLPlatformApp()
    app.serve(
        port=settings.app_port,
        show=True,
        autoreload=settings.debug
    )


if __name__ == "__main__":
    main()
