"""
Comprehensive logging configuration for ML Platform.

This module provides enterprise-grade logging capabilities with:
- Environment-based configuration
- Structured JSON logging for production
- Performance-optimized for ML workloads
- Integration with MLflow and monitoring systems
"""

import json
import logging
import logging.config
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from ...config.settings import settings


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging in production environments."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON with ML-specific fields."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
            
        # Add extra fields for ML operations
        extra_fields = {
            "experiment_id", "run_id", "model_name", "data_source", 
            "pipeline_stage", "performance_metrics", "user_id", "session_id"
        }
        
        for field in extra_fields:
            if hasattr(record, field):
                log_entry[field] = getattr(record, field)
                
        return json.dumps(log_entry, default=str)


class ColoredConsoleFormatter(logging.Formatter):
    """Colored console formatter for development environments."""
    
    # Color codes for different log levels
    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green  
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[35m",   # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with colors and ML-specific context."""
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET
        
        # Format with color and additional context
        formatted = f"{color}[{record.levelname:8}]{reset} "
        formatted += f"{record.name:20} | "
        formatted += f"{record.getMessage()}"
        
        # Add ML context if available
        if hasattr(record, "experiment_id"):
            formatted += f" [exp:{record.experiment_id}]"
        if hasattr(record, "model_name"):
            formatted += f" [model:{record.model_name}]"
        if hasattr(record, "pipeline_stage"):
            formatted += f" [stage:{record.pipeline_stage}]"
            
        return formatted


def setup_logging() -> None:
    """
    Configure comprehensive logging for the ML platform.
    
    Sets up:
    - Environment-appropriate formatters (JSON for prod, colored for dev)
    - File rotation and retention
    - Performance-optimized loggers
    - ML-specific log categories
    """
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Determine environment and log level
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    
    # Configure logging based on environment
    if settings.environment == "production":
        _setup_production_logging(log_level, log_dir)
    else:
        _setup_development_logging(log_level, log_dir)
    
    # Configure third-party library logging
    _configure_third_party_loggers()
    
    # Log startup information
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging system initialized",
        extra={
            "environment": settings.environment,
            "log_level": settings.log_level,
            "log_directory": str(log_dir),
        }
    )


def _setup_production_logging(log_level: int, log_dir: Path) -> None:
    """Configure production logging with JSON format and file rotation."""
    
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": "src.core.logging.config.JSONFormatter",
            },
            "simple": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "json",
                "level": log_level,
            },
            "file_app": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": log_dir / "ml_platform.log",
                "maxBytes": 50 * 1024 * 1024,  # 50MB
                "backupCount": 10,
                "formatter": "json",
                "level": log_level,
            },
            "file_experiments": {
                "class": "logging.handlers.RotatingFileHandler", 
                "filename": log_dir / "experiments.log",
                "maxBytes": 100 * 1024 * 1024,  # 100MB
                "backupCount": 20,
                "formatter": "json",
                "level": logging.INFO,
            },
            "file_data": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": log_dir / "data_pipeline.log", 
                "maxBytes": 50 * 1024 * 1024,  # 50MB
                "backupCount": 10,
                "formatter": "json",
                "level": logging.INFO,
            },
            "file_errors": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": log_dir / "errors.log",
                "maxBytes": 25 * 1024 * 1024,  # 25MB
                "backupCount": 15,
                "formatter": "json",
                "level": logging.ERROR,
            }
        },
        "loggers": {
            "src": {
                "handlers": ["console", "file_app"],
                "level": log_level,
                "propagate": False,
            },
            "src.core.experiments": {
                "handlers": ["file_experiments"],
                "level": logging.INFO,
                "propagate": True,
            },
            "src.core.data_sources": {
                "handlers": ["file_data"],
                "level": logging.INFO,
                "propagate": True,
            },
            "src.core.ml": {
                "handlers": ["file_experiments"],
                "level": logging.INFO,
                "propagate": True,
            },
        },
        "root": {
            "handlers": ["console", "file_errors"],
            "level": logging.WARNING,
        }
    }
    
    logging.config.dictConfig(config)


def _setup_development_logging(log_level: int, log_dir: Path) -> None:
    """Configure development logging with colored console output."""
    
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "colored": {
                "()": "src.core.logging.config.ColoredConsoleFormatter",
            },
            "file": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "colored",
                "level": log_level,
            },
            "file_debug": {
                "class": "logging.FileHandler",
                "filename": log_dir / "debug.log",
                "mode": "w",  # Overwrite each run in development
                "formatter": "file",
                "level": logging.DEBUG,
            }
        },
        "loggers": {
            "src": {
                "handlers": ["console", "file_debug"],
                "level": log_level,
                "propagate": False,
            },
        },
        "root": {
            "handlers": ["console"],
            "level": logging.INFO,
        }
    }
    
    logging.config.dictConfig(config)


def _configure_third_party_loggers() -> None:
    """Configure logging levels for third-party libraries."""
    
    # Reduce noise from verbose libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("snowflake").setLevel(logging.WARNING)
    
    # Keep MLflow logging visible but not overwhelming
    logging.getLogger("mlflow").setLevel(logging.INFO)
    
    # Panel and Bokeh can be noisy in development
    if settings.environment != "production":
        logging.getLogger("bokeh").setLevel(logging.WARNING)
        logging.getLogger("panel").setLevel(logging.INFO)


def get_logger(name: str, **context: Any) -> logging.LoggerAdapter:
    """
    Get a logger with ML-specific context.
    
    Args:
        name: Logger name (typically __name__)
        **context: Additional context fields (experiment_id, model_name, etc.)
        
    Returns:
        LoggerAdapter with context information
        
    Example:
        logger = get_logger(__name__, experiment_id="exp_123", model_name="rf_model")
        logger.info("Training started", extra={"epoch": 1, "batch_size": 32})
    """
    base_logger = logging.getLogger(name)
    return logging.LoggerAdapter(base_logger, context)


def log_performance(func):
    """
    Decorator to log function performance metrics.
    
    Useful for ML operations like data loading, model training, etc.
    """
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        logger.debug(
            f"Starting {func.__name__}",
            extra={
                "function": func.__name__,
                "pipeline_stage": "start",
                "args_count": len(args),
                "kwargs_count": len(kwargs)
            }
        )
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            logger.info(
                f"Completed {func.__name__}",
                extra={
                    "function": func.__name__,
                    "pipeline_stage": "complete",
                    "duration_seconds": round(duration, 3),
                    "performance_metrics": {"execution_time": duration}
                }
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            logger.error(
                f"Failed {func.__name__}: {e}",
                extra={
                    "function": func.__name__, 
                    "pipeline_stage": "error",
                    "duration_seconds": round(duration, 3),
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            raise
            
    return wrapper