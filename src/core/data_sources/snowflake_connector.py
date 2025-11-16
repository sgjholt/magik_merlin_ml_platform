"""
Snowflake Data Warehouse Connector.

This module provides a production-ready connector for Snowflake,
supporting secure connections, query execution, schema introspection,
and data loading with comprehensive error handling.

Example:
    >>> from src.core.data_sources.snowflake_connector import SnowflakeDataSource
    >>> from src.core.data_sources.base import DataSourceConfig
    >>> config = DataSourceConfig(
    ...     name='snowflake_warehouse',
    ...     source_type='snowflake',
    ...     connection_params={
    ...         'account': 'xy12345.us-east-1',
    ...         'user': 'ml_user',
    ...         'password': 'secret',
    ...         'warehouse': 'COMPUTE_WH',
    ...         'database': 'ML_DB',
    ...         'schema': 'PUBLIC',
    ...         'role': 'ML_ROLE'
    ...     }
    ... )
    >>> sf = SnowflakeDataSource(config)
    >>> sf.connect()
    >>> df = sf.load_data("SELECT * FROM customers LIMIT 1000")
"""

from typing import Any

import pandas as pd

from .base import DataSource, DataSourceConfig

__all__ = ["SnowflakeDataSource"]


class SnowflakeDataSource(DataSource):
    """
    Snowflake Data Warehouse connector.

    Provides secure connection to Snowflake with support for:
    - Multiple authentication methods (password, key-pair, OAuth)
    - Role and warehouse management
    - Query execution with result caching
    - Schema introspection
    - Bulk data loading with write_pandas
    - Session parameters configuration

    Config Parameters:
        account (str): Snowflake account identifier (e.g., 'xy12345.us-east-1')
        user (str): Username
        password (str): Password (if using password auth)
        authenticator (str): Authentication method ('snowflake', 'externalbrowser', 'oauth')
        warehouse (str): Virtual warehouse name
        database (str): Database name
        schema (str): Schema name
        role (str): Role name
        private_key (str): Private key for key-pair auth (optional)
        session_parameters (dict): Additional session parameters

    Example:
        >>> config = DataSourceConfig(
        ...     name='my_snowflake',
        ...     source_type='snowflake',
        ...     connection_params={
        ...         'account': 'xy12345',
        ...         'user': 'john_doe',
        ...         'password': 'secure_pass',
        ...         'warehouse': 'ANALYTICS_WH',
        ...         'database': 'PROD_DB',
        ...         'schema': 'ML_SCHEMA',
        ...         'role': 'DATA_SCIENTIST'
        ...     }
        ... )
        >>> sf = SnowflakeDataSource(config)
        >>> sf.connect()
        >>> df = sf.load_data("SELECT * FROM features WHERE created_at > '2024-01-01'")
    """

    def __init__(self, config: DataSourceConfig) -> None:
        """Initialize Snowflake data source."""
        super().__init__(config)
        self.account = config.connection_params.get("account", "")
        self.user = config.connection_params.get("user", "")
        self.password = config.connection_params.get("password", "")
        self.authenticator = config.connection_params.get("authenticator", "snowflake")
        self.warehouse = config.connection_params.get("warehouse")
        self.database = config.connection_params.get("database")
        self.schema = config.connection_params.get("schema")
        self.role = config.connection_params.get("role")
        self.private_key = config.connection_params.get("private_key")
        self.session_parameters = config.connection_params.get("session_parameters", {})

    def connect(self) -> bool:
        """
        Establish connection to Snowflake.

        Returns:
            True if connection successful

        Raises:
            ImportError: If snowflake-connector-python is not installed
            RuntimeError: If connection fails
        """
        try:
            import snowflake.connector  # type: ignore[import-untyped]  # noqa: PLC0415

            # Build connection parameters
            conn_params = {
                "account": self.account,
                "user": self.user,
                "authenticator": self.authenticator,
            }

            # Add password if using password auth
            if self.authenticator == "snowflake" and self.password:
                conn_params["password"] = self.password

            # Add private key if using key-pair auth
            if self.private_key:
                conn_params["private_key"] = self.private_key

            # Add optional parameters
            if self.warehouse:
                conn_params["warehouse"] = self.warehouse
            if self.database:
                conn_params["database"] = self.database
            if self.schema:
                conn_params["schema"] = self.schema
            if self.role:
                conn_params["role"] = self.role

            # Add session parameters
            if self.session_parameters:
                conn_params["session_parameters"] = self.session_parameters

            self._connection = snowflake.connector.connect(**conn_params)
            return True

        except ImportError as e:
            msg = (
                "snowflake-connector-python is not installed. "
                "Install it with: uv add snowflake-connector-python"
            )
            raise ImportError(msg) from e
        except Exception as e:
            msg = f"Failed to connect to Snowflake: {e}"
            raise RuntimeError(msg) from e

    def disconnect(self) -> None:
        """Close Snowflake connection and release resources."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def test_connection(self) -> bool:
        """
        Test if connection is alive and can execute queries.

        Returns:
            True if connection is active
        """
        if self._connection is None:
            return False

        try:
            cursor = self._connection.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
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
            msg = "Not connected to Snowflake. Call connect() first."
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
        Get schema information for a Snowflake table.

        Args:
            table_name: Name of the table (can be database.schema.table)

        Returns:
            Dictionary with column information
        """
        if self._connection is None:
            msg = "Not connected to Snowflake."
            raise RuntimeError(msg)

        query = f"DESCRIBE TABLE {table_name}"

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
                        "kind": col[2],
                        "nullable": col[3] == "Y",
                        "default": col[4],
                        "primary_key": col[5] == "Y",
                        "unique_key": col[6] == "Y",
                        "comment": col[8] if len(col) > 8 else None,
                    }
                    for col in columns
                ],
            }
        except Exception as e:
            msg = f"Failed to get schema for {table_name}: {e}"
            raise RuntimeError(msg) from e

    def list_tables(self) -> list[str]:
        """
        List all tables in the current database and schema.

        Returns:
            List of table names
        """
        if self._connection is None:
            msg = "Not connected to Snowflake."
            raise RuntimeError(msg)

        try:
            cursor = self._connection.cursor()
            cursor.execute("SHOW TABLES")
            tables = [row[1] for row in cursor.fetchall()]  # name column is at index 1
            cursor.close()
            return tables
        except Exception as e:
            msg = f"Failed to list tables: {e}"
            raise RuntimeError(msg) from e

    def execute_query(self, query: str) -> None:
        """
        Execute a SQL query without returning results (DDL, DML).

        Args:
            query: SQL query to execute

        Raises:
            RuntimeError: If not connected or query fails
        """
        if self._connection is None:
            msg = "Not connected to Snowflake. Call connect() first."
            raise RuntimeError(msg)

        try:
            cursor = self._connection.cursor()
            cursor.execute(query)
            self._connection.commit()
            cursor.close()
        except Exception as e:
            msg = f"Failed to execute query: {e}"
            raise RuntimeError(msg) from e

    def write_data(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "replace",
        auto_create_table: bool = True,
    ) -> bool:
        """
        Write DataFrame to Snowflake table using write_pandas.

        Args:
            df: DataFrame to write
            table_name: Target table name
            if_exists: What to do if table exists ('replace', 'append', 'fail')
            auto_create_table: Whether to auto-create the table

        Returns:
            True if write successful

        Raises:
            RuntimeError: If not connected or write fails
        """
        if self._connection is None:
            msg = "Not connected to Snowflake. Call connect() first."
            raise RuntimeError(msg)

        try:
            from snowflake.connector.pandas_tools import (  # type: ignore[import-untyped]  # noqa: PLC0415
                write_pandas,
            )

            success, nchunks, nrows, _ = write_pandas(
                conn=self._connection,
                df=df,
                table_name=table_name,
                auto_create_table=auto_create_table,
                overwrite=(if_exists == "replace"),
            )
            return success
        except ImportError as e:
            msg = "snowflake.connector.pandas_tools not available"
            raise ImportError(msg) from e
        except Exception as e:
            msg = f"Failed to write data to {table_name}: {e}"
            raise RuntimeError(msg) from e

    def set_warehouse(self, warehouse: str) -> None:
        """
        Change the active warehouse for the session.

        Args:
            warehouse: Warehouse name
        """
        self.execute_query(f"USE WAREHOUSE {warehouse}")
        self.warehouse = warehouse

    def set_database(self, database: str) -> None:
        """
        Change the active database for the session.

        Args:
            database: Database name
        """
        self.execute_query(f"USE DATABASE {database}")
        self.database = database

    def set_schema(self, schema: str) -> None:
        """
        Change the active schema for the session.

        Args:
            schema: Schema name
        """
        self.execute_query(f"USE SCHEMA {schema}")
        self.schema = schema
