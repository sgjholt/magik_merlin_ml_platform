import json
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from pydantic import BaseModel


class DataSourceConfig(BaseModel):
    name: str
    source_type: str
    connection_params: dict[str, Any]
    cache_enabled: bool = True
    cache_ttl: int = 3600


class DataSource(ABC):
    def __init__(self, config: DataSourceConfig) -> None:
        self.config = config
        self._connection = None
        self._cache = {}

    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def disconnect(self) -> None:
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        pass

    @abstractmethod
    def load_data(self, query: str, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_schema(self, table_name: str) -> dict[str, Any]:
        pass

    @abstractmethod
    def list_tables(self) -> list[str]:
        pass

    def validate_query(self, query: str) -> bool:
        """Basic validation for SQL queries. Override in subclasses for specific checks."""
        if not isinstance(query, str) or not query.strip():
            msg = "Query must be a non-empty string."
            raise ValueError(msg)
        return True

    def get_data_preview(self, query: str, limit: int = 100) -> pd.DataFrame:
        return self.load_data(query, nrows=limit)

    def get_data_profile(self, df: pd.DataFrame) -> dict[str, Any]:
        return {
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "schema": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "null_counts": json.loads(df.isna().sum().to_json()),
            "memory_usage": {
                "total": self._memory_usage_to_best_unit(
                    df.memory_usage(deep=True).sum()
                ),
                "per_column": {
                    col: self._memory_usage_to_best_unit(mem)
                    for col, mem in df.memory_usage(deep=True).to_dict().items()
                },
            },
        }

    def _memory_usage_to_best_unit(self, memory_bytes: int, precision: int = 2) -> str:
        """Convert memory usage in bytes to a human-readable format."""
        if memory_bytes < 1024:  # noqa: PLR2004
            return f"{memory_bytes:.0f} B"
        if memory_bytes < 1024**2:
            return f"{memory_bytes / 1024:.0f} KB"
        if memory_bytes < 1024**3:
            return f"{memory_bytes / 1024**2:.{precision}f} MB"
        if memory_bytes < 1024**4:
            return f"{memory_bytes / 1024**3:.{precision}f} GB"
        if memory_bytes < 1024**5:
            return f"{memory_bytes / 1024**4:.{precision}f} TB"
        msg = "Memory size too large to convert"
        raise ValueError(msg)
