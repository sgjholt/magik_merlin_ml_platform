from typing import Any

import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

from .base import DataSource, DataSourceConfig


class SnowflakeDataSource(DataSource):
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.connection_params = config.connection_params

    def connect(self) -> bool:
        try:
            self._connection = snowflake.connector.connect(
                user=self.connection_params["user"],
                password=self.connection_params["password"],
                account=self.connection_params["account"],
                warehouse=self.connection_params.get("warehouse"),
                database=self.connection_params.get("database"),
                schema=self.connection_params.get("schema"),
            )
            return True
        except Exception as e:
            print(f"Failed to connect to Snowflake: {e!s}")
            return False

    def disconnect(self) -> None:
        if self._connection:
            self._connection.close()
            self._connection = None

    def test_connection(self) -> bool:
        if not self._connection:
            return self.connect()

        try:
            cursor = self._connection.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            return True
        except Exception:
            return False

    def load_data(self, query: str, **kwargs) -> pd.DataFrame:
        if not self._connection:
            if not self.connect():
                raise ConnectionError("Failed to connect to Snowflake")

        cache_key = f"snowflake_{hash(query)}"
        if self.config.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            df = pd.read_sql(query, self._connection)
            if self.config.cache_enabled:
                self._cache[cache_key] = df
            return df
        except Exception as e:
            raise RuntimeError(f"Error executing query: {e!s}")

    def get_schema(self, table_name: str) -> dict[str, Any]:
        query = f"DESCRIBE TABLE {table_name}"
        schema_df = self.load_data(query)
        return {
            "columns": schema_df["name"].tolist(),
            "types": dict(zip(schema_df["name"], schema_df["type"], strict=False)),
            "nullable": dict(
                zip(
                    schema_df["name"],
                    schema_df["null?"].map({"Y": True, "N": False}),
                    strict=False,
                )
            ),
        }

    def list_tables(self) -> list[str]:
        query = "SHOW TABLES"
        tables_df = self.load_data(query)
        return tables_df["name"].tolist()

    def execute_query(self, query: str) -> None:
        if not self._connection:
            if not self.connect():
                raise ConnectionError("Failed to connect to Snowflake")

        cursor = self._connection.cursor()
        try:
            cursor.execute(query)
            self._connection.commit()
        finally:
            cursor.close()

    def write_data(
        self, df: pd.DataFrame, table_name: str, if_exists: str = "replace"
    ) -> bool:
        if not self._connection:
            if not self.connect():
                raise ConnectionError("Failed to connect to Snowflake")

        try:
            success, nchunks, nrows, _ = write_pandas(
                self._connection,
                df,
                table_name,
                auto_create_table=True,
                overwrite=(if_exists == "replace"),
            )
            return success
        except Exception as e:
            raise RuntimeError(f"Error writing data to Snowflake: {e!s}")
