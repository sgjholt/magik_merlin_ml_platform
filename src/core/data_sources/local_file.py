from pathlib import Path
from typing import Any

import pandas as pd

from .base import DataSource, DataSourceConfig


class LocalFileDataSource(DataSource):
    SUPPORTED_FORMATS = {
        ".csv": pd.read_csv,
        ".parquet": pd.read_parquet,
        ".json": pd.read_json,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
    }

    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.base_path = Path(config.connection_params.get("base_path", "./data"))
        # Only create directory if it's a valid path and doesn't exist
        try:
            if not self.base_path.exists():
                self.base_path.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            # Handle cases where directory cannot be created (e.g., invalid paths)
            pass

    def connect(self) -> bool:
        return self.base_path.exists()

    def disconnect(self) -> None:
        pass

    def test_connection(self) -> bool:
        return self.base_path.exists() and self.base_path.is_dir()

    def load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        full_path = self.base_path / file_path

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {full_path}")

        file_extension = full_path.suffix.lower()
        if file_extension not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {file_extension}")

        reader_func = self.SUPPORTED_FORMATS[file_extension]

        cache_key = f"{file_path}_{hash(str(kwargs))}"
        if self.config.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            df = reader_func(full_path, **kwargs)
            if self.config.cache_enabled:
                self._cache[cache_key] = df
            return df
        except Exception as e:
            raise RuntimeError(f"Error loading file {full_path}: {e!s}")

    def get_schema(self, file_path: str) -> dict[str, Any]:
        df = self.load_data(file_path, nrows=1)
        return {
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "shape": df.shape,
        }

    def list_tables(self) -> list[str]:
        files = []
        for ext in self.SUPPORTED_FORMATS.keys():
            files.extend(self.base_path.glob(f"**/*{ext}"))

        return [str(f.relative_to(self.base_path)) for f in files]

    def save_data(self, df: pd.DataFrame, file_path: str, **kwargs) -> None:
        full_path = self.base_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        file_extension = full_path.suffix.lower()

        if file_extension == ".csv":
            df.to_csv(full_path, index=False, **kwargs)
        elif file_extension == ".parquet":
            df.to_parquet(full_path, index=False, **kwargs)
        elif file_extension == ".json":
            df.to_json(full_path, **kwargs)
        elif file_extension in [".xlsx", ".xls"]:
            df.to_excel(full_path, index=False, **kwargs)
        else:
            raise ValueError(f"Unsupported file format for saving: {file_extension}")
