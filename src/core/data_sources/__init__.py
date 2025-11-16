from .base import DataSource, DataSourceConfig
from .local_file import LocalFileDataSource

__all__ = ["DataSource", "DataSourceConfig", "LocalFileDataSource"]

# Optional imports for cloud connectors
try:
    from .snowflake_connector import SnowflakeDataSource

    __all__.append("SnowflakeDataSource")
except ImportError:
    SnowflakeDataSource = None

try:
    from .aws_connector import AWSS3DataSource

    __all__.append("AWSS3DataSource")
    # Backward compatibility
    AWSDataSource = AWSS3DataSource
    __all__.append("AWSDataSource")
except ImportError:
    AWSS3DataSource = None
    AWSDataSource = None

# Optional imports for database connectors (Phase 4)
try:
    from .databases import (
        MongoDBDataSource,
        MySQLDataSource,
        PostgreSQLDataSource,
        SQLAlchemyDataSource,
    )

    __all__.extend(
        [
            "MongoDBDataSource",
            "MySQLDataSource",
            "PostgreSQLDataSource",
            "SQLAlchemyDataSource",
        ]
    )
except ImportError:
    PostgreSQLDataSource = None
    MySQLDataSource = None
    MongoDBDataSource = None
    SQLAlchemyDataSource = None

# Optional imports for streaming connectors (Phase 4)
try:
    from .streaming import KafkaDataSource, StreamProcessor, WebSocketDataSource

    __all__.extend(
        [
            "KafkaDataSource",
            "StreamProcessor",
            "WebSocketDataSource",
        ]
    )
except ImportError:
    KafkaDataSource = None
    WebSocketDataSource = None
    StreamProcessor = None

# Optional imports for cloud storage connectors (Phase 4)
try:
    from .cloud_storage import (
        AzureBlobDataSource,
        BigQueryDataSource,
        GCSDataSource,
    )

    __all__.extend(
        [
            "AzureBlobDataSource",
            "BigQueryDataSource",
            "GCSDataSource",
        ]
    )
except ImportError:
    GCSDataSource = None
    AzureBlobDataSource = None
    BigQueryDataSource = None
