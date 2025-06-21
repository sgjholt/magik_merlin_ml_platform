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
    def __init__(self, config: DataSourceConfig):
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
        return True

    def get_data_preview(self, query: str, limit: int = 100) -> pd.DataFrame:
        return self.load_data(query, nrows=limit)

    def get_data_profile(self, df: pd.DataFrame) -> dict[str, Any]:
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
        }
