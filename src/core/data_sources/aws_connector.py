import io
from typing import Any

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from ..logging import get_logger, log_performance
from .base import DataSource, DataSourceConfig


class AWSDataSource(DataSource):
    def __init__(self, config: DataSourceConfig) -> None:
        super().__init__(config)
        self.connection_params = config.connection_params
        self.s3_client = None
        self.bucket_name = self.connection_params.get("bucket_name")
        self.logger = get_logger(__name__, data_source="aws_s3", bucket=self.bucket_name)

    def connect(self) -> bool:
        self.logger.info("Attempting to connect to AWS S3", extra={
            "region": self.connection_params.get("region", "us-east-1"),
            "bucket": self.bucket_name
        })
        
        try:
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=self.connection_params.get("access_key_id"),
                aws_secret_access_key=self.connection_params.get("secret_access_key"),
                region_name=self.connection_params.get("region", "us-east-1"),
            )
            self.logger.info("Successfully connected to AWS S3")
            return True
        except Exception as e:
            self.logger.error("Failed to connect to AWS S3", exc_info=True, extra={
                "error_type": type(e).__name__
            })
            return False

    def disconnect(self) -> None:
        self.s3_client = None

    def test_connection(self) -> bool:
        if not self.s3_client:
            if not self.connect():
                return False

        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            return True
        except ClientError:
            return False

    @log_performance
    def load_data(self, s3_key: str, **kwargs) -> pd.DataFrame:
        if not self.s3_client:
            if not self.connect():
                raise ConnectionError("Failed to connect to AWS S3")

        cache_key = f"s3_{self.bucket_name}_{s3_key}"
        if self.config.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            content = response["Body"].read()

            # Determine file type and read appropriately
            if s3_key.endswith(".csv"):
                df = pd.read_csv(io.StringIO(content.decode("utf-8")), **kwargs)
            elif s3_key.endswith(".parquet"):
                df = pd.read_parquet(io.BytesIO(content), **kwargs)
            elif s3_key.endswith(".json"):
                df = pd.read_json(io.StringIO(content.decode("utf-8")), **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {s3_key}")

            if self.config.cache_enabled:
                self._cache[cache_key] = df
            return df

        except ClientError as e:
            raise RuntimeError(f"Error loading data from S3: {e!s}")

    def get_schema(self, s3_key: str) -> dict[str, Any]:
        df = self.load_data(s3_key, nrows=1)
        return {
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "file_size": self._get_object_size(s3_key),
        }

    def list_tables(self) -> list[str]:
        if not self.s3_client:
            if not self.connect():
                raise ConnectionError("Failed to connect to AWS S3")

        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
            objects = response.get("Contents", [])

            # Filter for supported file types
            supported_extensions = [".csv", ".parquet", ".json"]
            files = [
                obj["Key"]
                for obj in objects
                if any(obj["Key"].endswith(ext) for ext in supported_extensions)
            ]
            return files

        except ClientError as e:
            raise RuntimeError(f"Error listing S3 objects: {e!s}")

    def save_data(self, df: pd.DataFrame, s3_key: str, **kwargs) -> None:
        if not self.s3_client:
            if not self.connect():
                raise ConnectionError("Failed to connect to AWS S3")

        try:
            buffer = io.StringIO()

            if s3_key.endswith(".csv"):
                df.to_csv(buffer, index=False, **kwargs)
                content = buffer.getvalue()
                content_type = "text/csv"
            elif s3_key.endswith(".json"):
                df.to_json(buffer, **kwargs)
                content = buffer.getvalue()
                content_type = "application/json"
            elif s3_key.endswith(".parquet"):
                parquet_buffer = io.BytesIO()
                df.to_parquet(parquet_buffer, index=False, **kwargs)
                content = parquet_buffer.getvalue()
                content_type = "application/octet-stream"
            else:
                raise ValueError(f"Unsupported file format for saving: {s3_key}")

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=content,
                ContentType=content_type,
            )

        except ClientError as e:
            raise RuntimeError(f"Error saving data to S3: {e!s}")

    def _get_object_size(self, s3_key: str) -> int:
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return response["ContentLength"]
        except ClientError:
            return 0
