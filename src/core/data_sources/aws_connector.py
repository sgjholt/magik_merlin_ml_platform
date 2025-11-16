"""
AWS S3 Data Source Connector.

This module provides a production-ready connector for Amazon S3,
supporting secure connections, multi-format data loading, and
comprehensive bucket operations with proper error handling.

Example:
    >>> from src.core.data_sources.aws_connector import AWSS3DataSource
    >>> from src.core.data_sources.base import DataSourceConfig
    >>> config = DataSourceConfig(
    ...     name='s3_data_lake',
    ...     source_type='aws_s3',
    ...     connection_params={
    ...         'bucket_name': 'my-ml-bucket',
    ...         'access_key_id': 'AKIAIOSFODNN7EXAMPLE',
    ...         'secret_access_key': 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY',
    ...         'region': 'us-west-2'
    ...     }
    ... )
    >>> s3 = AWSS3DataSource(config)
    >>> s3.connect()
    >>> df = s3.load_data('data/customers.parquet')
"""

import io
from typing import Any

import pandas as pd

from .base import DataSource, DataSourceConfig

__all__ = ["AWSDataSource", "AWSS3DataSource"]


class AWSS3DataSource(DataSource):
    """
    Amazon S3 data source connector.

    Provides secure access to S3 buckets with support for:
    - Multiple authentication methods (IAM roles, access keys)
    - Multi-format file reading (CSV, Parquet, JSON, Excel)
    - Bulk operations and prefix-based filtering
    - Server-side encryption support
    - Comprehensive error handling

    Config Parameters:
        bucket_name (str): S3 bucket name
        access_key_id (str): AWS access key ID (optional if using IAM role)
        secret_access_key (str): AWS secret access key (optional if using IAM role)
        session_token (str): AWS session token for temporary credentials
        region (str): AWS region (default: 'us-east-1')
        endpoint_url (str): Custom endpoint URL (for S3-compatible services)
        use_ssl (bool): Whether to use SSL (default: True)
        verify (bool): Whether to verify SSL certificates (default: True)

    Example:
        >>> config = DataSourceConfig(
        ...     name='production_s3',
        ...     source_type='s3',
        ...     connection_params={
        ...         'bucket_name': 'ml-data-prod',
        ...         'region': 'us-east-1'
        ...     }
        ... )
        >>> s3 = AWSS3DataSource(config)
        >>> s3.connect()
        >>> df = s3.load_data('features/2024/user_features.parquet')
    """

    def __init__(self, config: DataSourceConfig) -> None:
        """Initialize AWS S3 data source."""
        super().__init__(config)
        self.bucket_name = config.connection_params.get("bucket_name", "")
        self.access_key_id = config.connection_params.get("access_key_id")
        self.secret_access_key = config.connection_params.get("secret_access_key")
        self.session_token = config.connection_params.get("session_token")
        self.region = config.connection_params.get("region", "us-east-1")
        self.endpoint_url = config.connection_params.get("endpoint_url")
        self.use_ssl = config.connection_params.get("use_ssl", True)
        self.verify = config.connection_params.get("verify", True)
        self.s3_client = None

    def connect(self) -> bool:
        """
        Establish connection to AWS S3.

        Returns:
            True if connection successful

        Raises:
            ImportError: If boto3 is not installed
            RuntimeError: If connection fails
        """
        try:
            import boto3  # type: ignore[import-untyped]  # noqa: PLC0415

            # Build connection parameters
            client_params: dict[str, Any] = {
                "service_name": "s3",
                "region_name": self.region,
                "use_ssl": self.use_ssl,
                "verify": self.verify,
            }

            if self.access_key_id and self.secret_access_key:
                client_params["aws_access_key_id"] = self.access_key_id
                client_params["aws_secret_access_key"] = self.secret_access_key

            if self.session_token:
                client_params["aws_session_token"] = self.session_token

            if self.endpoint_url:
                client_params["endpoint_url"] = self.endpoint_url

            self.s3_client = boto3.client(**client_params)
            self._connection = self.s3_client
            return True

        except ImportError as e:
            msg = "boto3 is not installed. Install it with: uv add boto3"
            raise ImportError(msg) from e
        except Exception as e:
            msg = f"Failed to connect to AWS S3: {e}"
            raise RuntimeError(msg) from e

    def disconnect(self) -> None:
        """Close S3 connection."""
        self.s3_client = None
        self._connection = None

    def test_connection(self) -> bool:
        """
        Test if connection is alive.

        Returns:
            True if can access bucket
        """
        if self.s3_client is None:
            return False

        try:
            from botocore.exceptions import (  # type: ignore[import-untyped]  # noqa: PLC0415
                ClientError,
            )

            self.s3_client.head_bucket(Bucket=self.bucket_name)
            return True
        except (ClientError, Exception):
            return False

    def load_data(self, s3_key: str, **kwargs: Any) -> pd.DataFrame:
        """
        Load data from S3 object.

        Args:
            s3_key: S3 object key (file path within bucket)
            **kwargs: Additional parameters for pandas read functions

        Returns:
            DataFrame containing the data

        Raises:
            RuntimeError: If not connected or load fails
        """
        if self.s3_client is None:
            msg = "Not connected to S3. Call connect() first."
            raise RuntimeError(msg)

        # Check cache
        cache_key = f"s3_{self.bucket_name}_{s3_key}"
        if self.config.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            from botocore.exceptions import (  # type: ignore[import-untyped]  # noqa: PLC0415
                ClientError,
            )

            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            content = response["Body"].read()

            # Determine file type and read appropriately
            if s3_key.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(content), **kwargs)
            elif s3_key.endswith(".parquet"):
                df = pd.read_parquet(io.BytesIO(content), **kwargs)
            elif s3_key.endswith(".json"):
                df = pd.read_json(io.BytesIO(content), **kwargs)
            elif s3_key.endswith((".xlsx", ".xls")):
                df = pd.read_excel(io.BytesIO(content), **kwargs)
            else:
                msg = f"Unsupported file format: {s3_key}"
                raise ValueError(msg)

            if self.config.cache_enabled:
                self._cache[cache_key] = df

            return df
        except ClientError as e:
            msg = f"Failed to load data from {s3_key}: {e}"
            raise RuntimeError(msg) from e
        except Exception as e:
            msg = f"Error loading data: {e}"
            raise RuntimeError(msg) from e

    def get_schema(self, s3_key: str) -> dict[str, Any]:
        """
        Get schema information for an S3 object.

        Args:
            s3_key: S3 object key

        Returns:
            Dictionary with column and metadata information
        """
        df = self.load_data(s3_key, nrows=1)

        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            size = response.get("ContentLength", 0)
            content_type = response.get("ContentType")
        except Exception:
            size = 0
            content_type = None

        return {
            "s3_key": s3_key,
            "columns": list(df.columns),
            "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
            "size_bytes": size,
            "content_type": content_type,
        }

    def list_tables(self) -> list[str]:
        """
        List all objects in the bucket.

        Returns:
            List of S3 object keys
        """
        if self.s3_client is None:
            msg = "Not connected to S3."
            raise RuntimeError(msg)

        try:
            from botocore.exceptions import (  # type: ignore[import-untyped]  # noqa: PLC0415
                ClientError,
            )

            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
            objects = response.get("Contents", [])

            # Filter for supported file types
            supported_extensions = (".csv", ".parquet", ".json", ".xlsx", ".xls")
            files = [
                obj["Key"]
                for obj in objects
                if any(obj["Key"].endswith(ext) for ext in supported_extensions)
            ]
            return files

        except ClientError as e:
            msg = f"Failed to list objects: {e}"
            raise RuntimeError(msg) from e

    def save_data(self, df: pd.DataFrame, s3_key: str, **kwargs: Any) -> None:
        """
        Save DataFrame to S3 object.

        Args:
            df: DataFrame to save
            s3_key: Target S3 object key
            **kwargs: Additional parameters for pandas write functions

        Raises:
            RuntimeError: If not connected or save fails
        """
        if self.s3_client is None:
            msg = "Not connected to S3. Call connect() first."
            raise RuntimeError(msg)

        try:
            from botocore.exceptions import (  # type: ignore[import-untyped]  # noqa: PLC0415
                ClientError,
            )

            buffer = io.BytesIO()

            if s3_key.endswith(".csv"):
                df.to_csv(buffer, index=False, **kwargs)
                content_type = "text/csv"
            elif s3_key.endswith(".parquet"):
                df.to_parquet(buffer, index=False, **kwargs)
                content_type = "application/octet-stream"
            elif s3_key.endswith(".json"):
                df.to_json(buffer, **kwargs)
                content_type = "application/json"
            else:
                msg = f"Unsupported file format for saving: {s3_key}"
                raise ValueError(msg)

            buffer.seek(0)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=buffer.getvalue(),
                ContentType=content_type,
            )

        except ClientError as e:
            msg = f"Failed to save data to {s3_key}: {e}"
            raise RuntimeError(msg) from e


# Backward compatibility alias
AWSDataSource = AWSS3DataSource
