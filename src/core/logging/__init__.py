"""
ML Platform Logging System

Provides enterprise-grade logging with:
- Structured JSON logging for production
- Colored console output for development  
- ML-specific context and formatters
- Performance monitoring decorators
"""

from .config import get_logger, log_performance, setup_logging

__all__ = ["get_logger", "log_performance", "setup_logging"]