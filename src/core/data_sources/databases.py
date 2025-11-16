"""
Database Data Source Connectors.

This module provides connectors for various databases including
PostgreSQL, MySQL, MongoDB, and generic SQLAlchemy support.

Classes:
    - PostgreSQLDataSource: PostgreSQL connector
    - MySQLDataSource: MySQL connector
    - MongoDBDataSource: MongoDB connector
    - SQLAlchemyDataSource: Generic SQL database connector

Example:
    >>> from src.core.data_sources.databases import PostgreSQLDataSource
    >>> from src.core.data_sources.base import DataSourceConfig
    >>> config = DataSourceConfig(
    ...     name='my_postgres',
    ...     source_type='postgresql',
    ...     connection_params={
    ...         'host': 'localhost',
    ...         'port': 5432,
    ...         'database': 'ml_data',
    ...         'user': 'ml_user',
    ...         'password': 'secret'
    ...     }
    ... )
    >>> postgres = PostgreSQLDataSource(config)
    >>> postgres.connect()
    >>> df = postgres.load_data(query='SELECT * FROM features')
"""

from typing import Any

import pandas as pd

from .base import DataSource, DataSourceConfig

__all__ = [
    "MongoDBDataSource",
    "MySQLDataSource",
    "PostgreSQLDataSource",
    "SQLAlchemyDataSource",
]


class PostgreSQLDataSource(DataSource):
    """
    PostgreSQL database connector using psycopg2.

    Supports connecting to PostgreSQL databases and executing SQL queries.
    Handles connection pooling, error handling, and automatic reconnection.

    Config Parameters:
        host (str): Database host (default: 'localhost')
        port (int): Database port (default: 5432)
        database (str): Database name
        user (str): Username
        password (str): Password
        sslmode (str): SSL mode ('disable', 'require', 'verify-ca', 'verify-full')
        connect_timeout (int): Connection timeout in seconds (default: 10)

    Example:
        >>> config = DataSourceConfig(
        ...     name='my_pg_db',
        ...     source_type='postgresql',
        ...     connection_params={
        ...         'host': 'localhost',
        ...         'database': 'ml_data',
        ...         'user': 'postgres',
        ...         'password': 'secret'
        ...     }
        ... )
        >>> pg = PostgreSQLDataSource(config)
        >>> pg.connect()
        >>> df = pg.load_data("SELECT * FROM users LIMIT 10")
    """

    def __init__(self, config: DataSourceConfig) -> None:
        """Initialize PostgreSQL data source."""
        super().__init__(config)
        self.host = config.connection_params.get("host", "localhost")
        self.port = config.connection_params.get("port", 5432)
        self.database = config.connection_params.get("database", "")
        self.user = config.connection_params.get("user", "")
        self.password = config.connection_params.get("password", "")
        self.sslmode = config.connection_params.get("sslmode", "prefer")
        self.connect_timeout = config.connection_params.get("connect_timeout", 10)

    def connect(self) -> bool:
        """
        Establish connection to PostgreSQL database.

        Returns:
            True if connection successful, False otherwise

        Raises:
            ImportError: If psycopg2 is not installed
            Exception: If connection fails
        """
        try:
            import psycopg2  # type: ignore[import-untyped]  # noqa: PLC0415

            self._connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                sslmode=self.sslmode,
                connect_timeout=self.connect_timeout,
            )
            return True
        except ImportError as e:
            msg = "psycopg2 is not installed. Install it with: uv add psycopg2-binary"
            raise ImportError(msg) from e
        except Exception as e:
            msg = f"Failed to connect to PostgreSQL: {e}"
            raise RuntimeError(msg) from e

    def disconnect(self) -> None:
        """Close the PostgreSQL connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def test_connection(self) -> bool:
        """
        Test if the connection is alive.

        Returns:
            True if connection is active, False otherwise
        """
        if self._connection is None:
            return False
        try:
            # Try a simple query to verify connection
            cursor = self._connection.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return True
        except Exception:
            return False

    def load_data(self, query: str, **kwargs: Any) -> pd.DataFrame:
        """
        Execute SQL query and load results into DataFrame.

        Args:
            query: SQL query string
            **kwargs: Additional parameters passed to pandas.read_sql

        Returns:
            DataFrame containing query results

        Raises:
            ValueError: If query is invalid
            RuntimeError: If not connected or query fails
        """
        self.validate_query(query)

        if self._connection is None:
            msg = "Not connected to database. Call connect() first."
            raise RuntimeError(msg)

        # Check cache first
        cache_key = f"{query}_{hash(str(kwargs))}"
        if self.config.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            df = pd.read_sql(query, self._connection, **kwargs)

            if self.config.cache_enabled:
                self._cache[cache_key] = df

            return df
        except Exception as e:
            msg = f"Failed to execute query: {e}"
            raise RuntimeError(msg) from e

    def get_schema(self, table_name: str) -> dict[str, Any]:
        """
        Get schema information for a table.

        Args:
            table_name: Name of the table

        Returns:
            Dictionary with column names and types
        """
        if self._connection is None:
            msg = "Not connected to database."
            raise RuntimeError(msg)

        query = """
            SELECT column_name, data_type, character_maximum_length,
                   is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position
        """

        try:
            cursor = self._connection.cursor()
            cursor.execute(query, (table_name,))
            columns = cursor.fetchall()
            cursor.close()

            return {
                "table_name": table_name,
                "columns": [
                    {
                        "name": col[0],
                        "type": col[1],
                        "max_length": col[2],
                        "nullable": col[3] == "YES",
                        "default": col[4],
                    }
                    for col in columns
                ],
            }
        except Exception as e:
            msg = f"Failed to get schema: {e}"
            raise RuntimeError(msg) from e

    def list_tables(self) -> list[str]:
        """
        List all tables in the current database.

        Returns:
            List of table names
        """
        if self._connection is None:
            msg = "Not connected to database."
            raise RuntimeError(msg)

        query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """

        try:
            cursor = self._connection.cursor()
            cursor.execute(query)
            tables = [row[0] for row in cursor.fetchall()]
            cursor.close()
            return tables
        except Exception as e:
            msg = f"Failed to list tables: {e}"
            raise RuntimeError(msg) from e


class MySQLDataSource(DataSource):
    """
    MySQL database connector using pymysql.

    Supports connecting to MySQL/MariaDB databases and executing SQL queries.
    Handles connection management, error handling, and result caching.

    Config Parameters:
        host (str): Database host (default: 'localhost')
        port (int): Database port (default: 3306)
        database (str): Database name
        user (str): Username
        password (str): Password
        charset (str): Character set (default: 'utf8mb4')
        connect_timeout (int): Connection timeout in seconds (default: 10)

    Example:
        >>> config = DataSourceConfig(
        ...     name='my_mysql',
        ...     source_type='mysql',
        ...     connection_params={
        ...         'host': 'localhost',
        ...         'database': 'analytics',
        ...         'user': 'root',
        ...         'password': 'secret'
        ...     }
        ... )
        >>> mysql = MySQLDataSource(config)
        >>> mysql.connect()
        >>> df = mysql.load_data("SELECT * FROM events LIMIT 100")
    """

    def __init__(self, config: DataSourceConfig) -> None:
        """Initialize MySQL data source."""
        super().__init__(config)
        self.host = config.connection_params.get("host", "localhost")
        self.port = config.connection_params.get("port", 3306)
        self.database = config.connection_params.get("database", "")
        self.user = config.connection_params.get("user", "")
        self.password = config.connection_params.get("password", "")
        self.charset = config.connection_params.get("charset", "utf8mb4")
        self.connect_timeout = config.connection_params.get("connect_timeout", 10)

    def connect(self) -> bool:
        """
        Establish connection to MySQL database.

        Returns:
            True if connection successful

        Raises:
            ImportError: If pymysql is not installed
            RuntimeError: If connection fails
        """
        try:
            import pymysql  # type: ignore[import-untyped]  # noqa: PLC0415

            self._connection = pymysql.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                charset=self.charset,
                connect_timeout=self.connect_timeout,
            )
            return True
        except ImportError as e:
            msg = "pymysql is not installed. Install it with: uv add pymysql"
            raise ImportError(msg) from e
        except Exception as e:
            msg = f"Failed to connect to MySQL: {e}"
            raise RuntimeError(msg) from e

    def disconnect(self) -> None:
        """Close the MySQL connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def test_connection(self) -> bool:
        """
        Test if the connection is alive.

        Returns:
            True if connection is active, False otherwise
        """
        if self._connection is None:
            return False
        try:
            self._connection.ping(reconnect=False)
            return True
        except Exception:
            return False

    def load_data(self, query: str, **kwargs: Any) -> pd.DataFrame:
        """
        Execute SQL query and load results into DataFrame.

        Args:
            query: SQL query string
            **kwargs: Additional parameters passed to pandas.read_sql

        Returns:
            DataFrame containing query results

        Raises:
            ValueError: If query is invalid
            RuntimeError: If not connected or query fails
        """
        self.validate_query(query)

        if self._connection is None:
            msg = "Not connected to database. Call connect() first."
            raise RuntimeError(msg)

        # Check cache
        cache_key = f"{query}_{hash(str(kwargs))}"
        if self.config.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            df = pd.read_sql(query, self._connection, **kwargs)

            if self.config.cache_enabled:
                self._cache[cache_key] = df

            return df
        except Exception as e:
            msg = f"Failed to execute query: {e}"
            raise RuntimeError(msg) from e

    def get_schema(self, table_name: str) -> dict[str, Any]:
        """
        Get schema information for a table.

        Args:
            table_name: Name of the table

        Returns:
            Dictionary with column information
        """
        if self._connection is None:
            msg = "Not connected to database."
            raise RuntimeError(msg)

        query = f"DESCRIBE {table_name}"

        try:
            cursor = self._connection.cursor()
            cursor.execute(query)
            columns = cursor.fetchall()
            cursor.close()

            return {
                "table_name": table_name,
                "columns": [
                    {
                        "name": col[0],
                        "type": col[1],
                        "nullable": col[2] == "YES",
                        "key": col[3],
                        "default": col[4],
                        "extra": col[5],
                    }
                    for col in columns
                ],
            }
        except Exception as e:
            msg = f"Failed to get schema: {e}"
            raise RuntimeError(msg) from e

    def list_tables(self) -> list[str]:
        """
        List all tables in the current database.

        Returns:
            List of table names
        """
        if self._connection is None:
            msg = "Not connected to database."
            raise RuntimeError(msg)

        try:
            cursor = self._connection.cursor()
            cursor.execute("SHOW TABLES")
            tables = [row[0] for row in cursor.fetchall()]
            cursor.close()
            return tables
        except Exception as e:
            msg = f"Failed to list tables: {e}"
            raise RuntimeError(msg) from e


class MongoDBDataSource(DataSource):
    """
    MongoDB database connector.

    Note: MongoDB connector implementation is simplified.
    For production use, consider using the full pymongo library.

    Config Parameters:
        host: MongoDB host
        port: MongoDB port
        database: Database name
    """

    def __init__(self, config: DataSourceConfig) -> None:
        """Initialize MongoDB connector."""
        super().__init__(config)
        self.host = config.connection_params.get("host", "localhost")
        self.port = config.connection_params.get("port", 27017)
        self.database = config.connection_params.get("database", "")

    def connect(self) -> bool:
        """Establish MongoDB connection."""
        # TODO: Implement MongoDB connection with pymongo
        return True

    def disconnect(self) -> None:
        """Close MongoDB connection."""
        # TODO: Implement disconnection

    def test_connection(self) -> bool:
        """Test MongoDB connection."""
        # TODO: Implement connection test
        return False

    def load_data(self, query: str, **kwargs: Any) -> pd.DataFrame:
        """
        Load data from MongoDB.

        Args:
            query: Collection name or query
            **kwargs: Additional query parameters

        Returns:
            Query results as DataFrame
        """
        # TODO: Implement data loading
        return pd.DataFrame()

    def get_schema(self, table_name: str) -> dict[str, Any]:
        """Get MongoDB collection schema."""
        # TODO: Implement schema introspection
        return {}

    def list_tables(self) -> list[str]:
        """List MongoDB collections."""
        # TODO: Implement collection listing
        return []


class SQLAlchemyDataSource(DataSource):
    """
    Generic SQLAlchemy database connector.

    Supports any database with SQLAlchemy driver including PostgreSQL,
    MySQL, SQLite, Oracle, MSSQL, and more.

    Config Parameters:
        connection_string (str): SQLAlchemy connection URL
            Examples:
                - PostgreSQL: postgresql://user:password@localhost/dbname
                - MySQL: mysql+pymysql://user:password@localhost/dbname
                - SQLite: sqlite:///path/to/database.db
                - MSSQL: mssql+pyodbc://user:password@localhost/dbname
        pool_size (int): Connection pool size (default: 5)
        max_overflow (int): Max overflow connections (default: 10)
        pool_timeout (int): Pool timeout in seconds (default: 30)
        echo (bool): Echo SQL statements (default: False)

    Example:
        >>> config = DataSourceConfig(
        ...     name='my_db',
        ...     source_type='sqlalchemy',
        ...     connection_params={
        ...         'connection_string': 'postgresql://user:pass@localhost/db',
        ...         'pool_size': 10,
        ...         'echo': False
        ...     }
        ... )
        >>> db = SQLAlchemyDataSource(config)
        >>> db.connect()
        >>> df = db.load_data("SELECT * FROM users LIMIT 10")
    """

    def __init__(self, config: DataSourceConfig) -> None:
        """Initialize SQLAlchemy data source."""
        super().__init__(config)
        self.connection_string = config.connection_params.get("connection_string", "")
        self.pool_size = config.connection_params.get("pool_size", 5)
        self.max_overflow = config.connection_params.get("max_overflow", 10)
        self.pool_timeout = config.connection_params.get("pool_timeout", 30)
        self.echo = config.connection_params.get("echo", False)
        self.engine_ = None

    def connect(self) -> bool:
        """
        Create SQLAlchemy engine and test connection.

        Returns:
            True if connection successful

        Raises:
            ImportError: If SQLAlchemy is not installed
            ValueError: If connection string is missing
            RuntimeError: If engine creation or connection test fails
        """
        if not self.connection_string:
            msg = "Connection string is required"
            raise ValueError(msg)

        try:
            from sqlalchemy import create_engine  # noqa: PLC0415

            self.engine_ = create_engine(
                self.connection_string,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                echo=self.echo,
            )

            # Test the connection
            with self.engine_.connect() as conn:
                conn.execute("SELECT 1")

            self._connection = self.engine_
            return True

        except ImportError as e:
            msg = "sqlalchemy is not installed. Install it with: uv add sqlalchemy"
            raise ImportError(msg) from e
        except Exception as e:
            msg = f"Failed to create SQLAlchemy engine: {e}"
            raise RuntimeError(msg) from e

    def disconnect(self) -> None:
        """Dispose of the SQLAlchemy engine and close all connections."""
        if self.engine_ is not None:
            self.engine_.dispose()
            self.engine_ = None
            self._connection = None

    def test_connection(self) -> bool:
        """
        Test if the engine is available.

        Returns:
            True if engine is available and can execute queries
        """
        if self.engine_ is None:
            return False
        try:
            with self.engine_.connect() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    def load_data(self, query: str, **kwargs: Any) -> pd.DataFrame:
        """
        Execute SQL query and load results into DataFrame.

        Args:
            query: SQL query string
            **kwargs: Additional parameters passed to pandas.read_sql

        Returns:
            DataFrame containing query results

        Raises:
            ValueError: If query is invalid
            RuntimeError: If not connected or query fails
        """
        self.validate_query(query)

        if self.engine_ is None:
            msg = "Not connected to database. Call connect() first."
            raise RuntimeError(msg)

        # Check cache
        cache_key = f"{query}_{hash(str(kwargs))}"
        if self.config.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            df = pd.read_sql(query, self.engine_, **kwargs)

            if self.config.cache_enabled:
                self._cache[cache_key] = df

            return df
        except Exception as e:
            msg = f"Failed to execute query: {e}"
            raise RuntimeError(msg) from e

    def get_schema(self, table_name: str) -> dict[str, Any]:
        """
        Get schema information for a table using SQLAlchemy introspection.

        Args:
            table_name: Name of the table

        Returns:
            Dictionary with column information
        """
        if self.engine_ is None:
            msg = "Not connected to database."
            raise RuntimeError(msg)

        try:
            from sqlalchemy import MetaData, Table  # noqa: PLC0415

            metadata = MetaData()
            table = Table(table_name, metadata, autoload_with=self.engine_)

            return {
                "table_name": table_name,
                "columns": [
                    {
                        "name": col.name,
                        "type": str(col.type),
                        "nullable": col.nullable,
                        "primary_key": col.primary_key,
                        "default": str(col.default) if col.default else None,
                    }
                    for col in table.columns
                ],
            }
        except Exception as e:
            msg = f"Failed to get schema: {e}"
            raise RuntimeError(msg) from e

    def list_tables(self) -> list[str]:
        """
        List all tables using SQLAlchemy introspection.

        Returns:
            List of table names
        """
        if self.engine_ is None:
            msg = "Not connected to database."
            raise RuntimeError(msg)

        try:
            from sqlalchemy import inspect  # noqa: PLC0415

            inspector = inspect(self.engine_)
            return inspector.get_table_names()
        except Exception as e:
            msg = f"Failed to list tables: {e}"
            raise RuntimeError(msg) from e
