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
from src.core.logging import get_logger, setup_logging
from src.ui.app import MLPlatformApp


def main() -> None:
    """Main application entry point"""
    # Initialize logging system first
    setup_logging()

    # Get logger with application context
    logger = get_logger(__name__, pipeline_stage="startup")

    logger.info("Starting ML Experimentation Platform")
    logger.info(
        "Configuration loaded",
        extra={
            "mlflow_uri": settings.mlflow_tracking_uri,
            "host": settings.app_host,
            "port": settings.app_port,
            "environment": settings.environment,
            "debug_mode": settings.debug,
        },
    )

    try:
        # Create and run the application
        app = MLPlatformApp()
        logger.info(
            f"Application will be available at: http://{settings.app_host}:{settings.app_port}"
        )

        app.serve(port=settings.app_port, show=True, autoreload=settings.debug)
    except Exception:
        logger.critical("Failed to start application", exc_info=True)
        raise


if __name__ == "__main__":
    main()
