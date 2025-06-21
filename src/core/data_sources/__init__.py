from .base import DataSource
from .local_file import LocalFileDataSource

__all__ = ["DataSource", "LocalFileDataSource"]

# Optional imports for cloud connectors
try:
    from .snowflake_connector import SnowflakeDataSource

    __all__.append("SnowflakeDataSource")
except ImportError:
    SnowflakeDataSource = None

try:
    from .aws_connector import AWSDataSource

    __all__.append("AWSDataSource")
except ImportError:
    AWSDataSource = None
