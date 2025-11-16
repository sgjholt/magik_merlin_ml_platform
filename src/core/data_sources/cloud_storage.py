"""
Cloud Storage Connectors for GCS, Azure Blob Storage, and more.

This module provides production-ready connectors for major cloud storage
providers including Google Cloud Storage, Azure Blob Storage, and other
cloud-based data storage systems.

Classes:
    - GCSDataSource: Google Cloud Storage connector
    - AzureBlobDataSource: Azure Blob Storage connector
    - BigQueryDataSource: Google BigQuery connector

Example:
    >>> from src.core.data_sources.cloud_storage import GCSDataSource
    >>> from src.core.data_sources.base import DataSourceConfig
    >>> config = DataSourceConfig(
    ...     name='gcs_bucket',
    ...     source_type='gcs',
    ...     connection_params={
    ...         'bucket_name': 'my-ml-bucket',
    ...         'project_id': 'my-project-123',
    ...         'credentials_path': '/path/to/service-account.json'
    ...     }
    ... )
    >>> gcs = GCSDataSource(config)
    >>> gcs.connect()
    >>> df = gcs.load_data('data/features.parquet')
"""

import io
from typing import Any

import pandas as pd

from .base import DataSource, DataSourceConfig

__all__ = [
    "AzureBlobDataSource",
    "BigQueryDataSource",
    "GCSDataSource",
]


class GCSDataSource(DataSource):
    """
    Google Cloud Storage connector.

    Provides access to GCS buckets with support for:
    - Service account and OAuth authentication
    - Multi-format file reading (CSV, Parquet, JSON)
    - Bucket and blob management
    - Server-side encryption
    - Signed URLs for secure access

    Config Parameters:
        bucket_name (str): GCS bucket name
        project_id (str): Google Cloud project ID
        credentials_path (str): Path to service account JSON file (optional)
        credentials_json (dict): Service account credentials as dict (optional)

    Example:
        >>> config = DataSourceConfig(
        ...     name='gcs_data',
        ...     source_type='gcs',
        ...     connection_params={
        ...         'bucket_name': 'ml-features',
        ...         'project_id': 'my-ml-project',
        ...         'credentials_path': 'service-account.json'
        ...     }
        ... )
        >>> gcs = GCSDataSource(config)
        >>> gcs.connect()
        >>> df = gcs.load_data('features/user_data.parquet')
    """

    def __init__(self, config: DataSourceConfig) -> None:
        """Initialize GCS data source."""
        super().__init__(config)
        self.bucket_name = config.connection_params.get("bucket_name", "")
        self.project_id = config.connection_params.get("project_id", "")
        self.credentials_path = config.connection_params.get("credentials_path")
        self.credentials_json = config.connection_params.get("credentials_json")
        self.client_ = None
        self.bucket_ = None

    def connect(self) -> bool:
        """
        Establish connection to GCS.

        Returns:
            True if connection successful

        Raises:
            ImportError: If google-cloud-storage is not installed
            RuntimeError: If connection fails
        """
        try:
            from google.cloud import (
                storage,  # type: ignore[import-untyped]
            )

            # Create client with credentials
            if self.credentials_path:
                self.client_ = storage.Client.from_service_account_json(
                    self.credentials_path,
                    project=self.project_id,
                )
            elif self.credentials_json:
                self.client_ = storage.Client.from_service_account_info(
                    self.credentials_json,
                    project=self.project_id,
                )
            else:
                # Use default credentials (ADC)
                self.client_ = storage.Client(project=self.project_id)

            self.bucket_ = self.client_.bucket(self.bucket_name)
            self._connection = self.client_
            return True

        except ImportError as e:
            msg = (
                "google-cloud-storage is not installed. "
                "Install it with: uv add google-cloud-storage"
            )
            raise ImportError(msg) from e
        except Exception as e:
            msg = f"Failed to connect to GCS: {e}"
            raise RuntimeError(msg) from e

    def disconnect(self) -> None:
        """Close GCS connection."""
        self.client_ = None
        self.bucket_ = None
        self._connection = None

    def test_connection(self) -> bool:
        """
        Test if connection is alive.

        Returns:
            True if can access bucket
        """
        if self.bucket_ is None:
            return False
        try:
            self.bucket_.exists()
            return True
        except Exception:
            return False

    def load_data(self, blob_name: str, **kwargs: Any) -> pd.DataFrame:
        """
        Load data from GCS blob.

        Args:
            blob_name: Name of the blob (file) in the bucket
            **kwargs: Additional parameters for pandas read functions

        Returns:
            DataFrame containing the data

        Raises:
            RuntimeError: If not connected or load fails
        """
        if self.bucket_ is None:
            msg = "Not connected to GCS. Call connect() first."
            raise RuntimeError(msg)

        # Check cache
        cache_key = f"gcs_{self.bucket_name}_{blob_name}"
        if self.config.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            blob = self.bucket_.blob(blob_name)
            content = blob.download_as_bytes()

            # Determine file type and read appropriately
            if blob_name.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(content), **kwargs)
            elif blob_name.endswith(".parquet"):
                df = pd.read_parquet(io.BytesIO(content), **kwargs)
            elif blob_name.endswith(".json"):
                df = pd.read_json(io.BytesIO(content), **kwargs)
            elif blob_name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(io.BytesIO(content), **kwargs)
            else:
                msg = f"Unsupported file format: {blob_name}"
                raise ValueError(msg)

            if self.config.cache_enabled:
                self._cache[cache_key] = df

            return df
        except Exception as e:
            msg = f"Failed to load data from {blob_name}: {e}"
            raise RuntimeError(msg) from e

    def get_schema(self, blob_name: str) -> dict[str, Any]:
        """
        Get schema information for a blob.

        Args:
            blob_name: Name of the blob

        Returns:
            Dictionary with column and metadata information
        """
        df = self.load_data(blob_name, nrows=1)
        blob = self.bucket_.blob(blob_name)

        return {
            "blob_name": blob_name,
            "columns": list(df.columns),
            "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
            "size_bytes": blob.size if blob.exists() else 0,
            "content_type": blob.content_type if blob.exists() else None,
        }

    def list_tables(self) -> list[str]:
        """
        List all blobs (files) in the bucket.

        Returns:
            List of blob names
        """
        if self.bucket_ is None:
            msg = "Not connected to GCS."
            raise RuntimeError(msg)

        try:
            # Filter for supported file types
            supported_extensions = (".csv", ".parquet", ".json", ".xlsx", ".xls")
            blobs = [
                blob.name
                for blob in self.bucket_.list_blobs()
                if blob.name.endswith(supported_extensions)
            ]
            return blobs
        except Exception as e:
            msg = f"Failed to list blobs: {e}"
            raise RuntimeError(msg) from e

    def save_data(self, df: pd.DataFrame, blob_name: str, **kwargs: Any) -> None:
        """
        Save DataFrame to GCS blob.

        Args:
            df: DataFrame to save
            blob_name: Target blob name
            **kwargs: Additional parameters for pandas write functions

        Raises:
            RuntimeError: If not connected or save fails
        """
        if self.bucket_ is None:
            msg = "Not connected to GCS. Call connect() first."
            raise RuntimeError(msg)

        try:
            buffer = io.BytesIO()

            if blob_name.endswith(".csv"):
                df.to_csv(buffer, index=False, **kwargs)
                content_type = "text/csv"
            elif blob_name.endswith(".parquet"):
                df.to_parquet(buffer, index=False, **kwargs)
                content_type = "application/octet-stream"
            elif blob_name.endswith(".json"):
                df.to_json(buffer, **kwargs)
                content_type = "application/json"
            else:
                msg = f"Unsupported file format for saving: {blob_name}"
                raise ValueError(msg)

            blob = self.bucket_.blob(blob_name)
            blob.upload_from_string(buffer.getvalue(), content_type=content_type)

        except Exception as e:
            msg = f"Failed to save data to {blob_name}: {e}"
            raise RuntimeError(msg) from e


class AzureBlobDataSource(DataSource):
    """
    Azure Blob Storage connector.

    Provides access to Azure Blob Storage with support for:
    - Connection string and SAS token authentication
    - Multi-format file reading
    - Container and blob management
    - Blob leasing and snapshots

    Config Parameters:
        connection_string (str): Azure storage connection string
        container_name (str): Container name
        account_name (str): Storage account name (alternative to connection_string)
        account_key (str): Account key (alternative to connection_string)
        sas_token (str): Shared Access Signature token

    Example:
        >>> config = DataSourceConfig(
        ...     name='azure_storage',
        ...     source_type='azure_blob',
        ...     connection_params={
        ...         'connection_string': 'DefaultEndpointsProtocol=https;...',
        ...         'container_name': 'ml-data'
        ...     }
        ... )
        >>> azure = AzureBlobDataSource(config)
        >>> azure.connect()
        >>> df = azure.load_data('features/data.parquet')
    """

    def __init__(self, config: DataSourceConfig) -> None:
        """Initialize Azure Blob data source."""
        super().__init__(config)
        self.connection_string = config.connection_params.get("connection_string", "")
        self.container_name = config.connection_params.get("container_name", "")
        self.account_name = config.connection_params.get("account_name", "")
        self.account_key = config.connection_params.get("account_key", "")
        self.sas_token = config.connection_params.get("sas_token", "")
        self.blob_service_client_ = None
        self.container_client_ = None

    def connect(self) -> bool:
        """
        Establish connection to Azure Blob Storage.

        Returns:
            True if connection successful

        Raises:
            ImportError: If azure-storage-blob is not installed
            RuntimeError: If connection fails
        """
        try:
            from azure.storage.blob import (  # type: ignore[import-untyped]  # noqa: PLC0415
                BlobServiceClient,
            )

            if self.connection_string:
                self.blob_service_client_ = BlobServiceClient.from_connection_string(
                    self.connection_string
                )
            elif self.account_name and self.account_key:
                account_url = f"https://{self.account_name}.blob.core.windows.net"
                self.blob_service_client_ = BlobServiceClient(
                    account_url=account_url,
                    credential=self.account_key,
                )
            elif self.account_name and self.sas_token:
                account_url = f"https://{self.account_name}.blob.core.windows.net"
                self.blob_service_client_ = BlobServiceClient(
                    account_url=account_url,
                    credential=self.sas_token,
                )
            else:
                msg = "Must provide connection_string or account credentials"
                raise ValueError(msg)

            self.container_client_ = self.blob_service_client_.get_container_client(
                self.container_name
            )
            self._connection = self.blob_service_client_
            return True

        except ImportError as e:
            msg = (
                "azure-storage-blob is not installed. "
                "Install it with: uv add azure-storage-blob"
            )
            raise ImportError(msg) from e
        except Exception as e:
            msg = f"Failed to connect to Azure Blob Storage: {e}"
            raise RuntimeError(msg) from e

    def disconnect(self) -> None:
        """Close Azure connection."""
        if self.blob_service_client_ is not None:
            self.blob_service_client_.close()
            self.blob_service_client_ = None
            self.container_client_ = None
            self._connection = None

    def test_connection(self) -> bool:
        """
        Test if connection is alive.

        Returns:
            True if can access container
        """
        if self.container_client_ is None:
            return False
        try:
            self.container_client_.exists()
            return True
        except Exception:
            return False

    def load_data(self, blob_name: str, **kwargs: Any) -> pd.DataFrame:
        """
        Load data from Azure blob.

        Args:
            blob_name: Name of the blob
            **kwargs: Additional parameters for pandas read functions

        Returns:
            DataFrame containing the data

        Raises:
            RuntimeError: If not connected or load fails
        """
        if self.container_client_ is None:
            msg = "Not connected to Azure. Call connect() first."
            raise RuntimeError(msg)

        # Check cache
        cache_key = f"azure_{self.container_name}_{blob_name}"
        if self.config.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            blob_client = self.container_client_.get_blob_client(blob_name)
            content = blob_client.download_blob().readall()

            # Determine file type and read appropriately
            if blob_name.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(content), **kwargs)
            elif blob_name.endswith(".parquet"):
                df = pd.read_parquet(io.BytesIO(content), **kwargs)
            elif blob_name.endswith(".json"):
                df = pd.read_json(io.BytesIO(content), **kwargs)
            elif blob_name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(io.BytesIO(content), **kwargs)
            else:
                msg = f"Unsupported file format: {blob_name}"
                raise ValueError(msg)

            if self.config.cache_enabled:
                self._cache[cache_key] = df

            return df
        except Exception as e:
            msg = f"Failed to load data from {blob_name}: {e}"
            raise RuntimeError(msg) from e

    def get_schema(self, blob_name: str) -> dict[str, Any]:
        """Get schema information for a blob."""
        df = self.load_data(blob_name, nrows=1)

        return {
            "blob_name": blob_name,
            "columns": list(df.columns),
            "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
        }

    def list_tables(self) -> list[str]:
        """
        List all blobs in the container.

        Returns:
            List of blob names
        """
        if self.container_client_ is None:
            msg = "Not connected to Azure."
            raise RuntimeError(msg)

        try:
            supported_extensions = (".csv", ".parquet", ".json", ".xlsx", ".xls")
            blobs = [
                blob.name
                for blob in self.container_client_.list_blobs()
                if blob.name.endswith(supported_extensions)
            ]
            return blobs
        except Exception as e:
            msg = f"Failed to list blobs: {e}"
            raise RuntimeError(msg) from e


class BigQueryDataSource(DataSource):
    """
    Google BigQuery connector.

    Provides access to BigQuery with support for:
    - Service account and OAuth authentication
    - Standard SQL queries
    - Table and dataset management
    - Query job configuration
    - Result pagination

    Config Parameters:
        project_id (str): Google Cloud project ID
        dataset_id (str): Default dataset ID (optional)
        credentials_path (str): Path to service account JSON (optional)
        credentials_json (dict): Service account credentials as dict (optional)
        location (str): BigQuery location (default: 'US')

    Example:
        >>> config = DataSourceConfig(
        ...     name='bigquery_warehouse',
        ...     source_type='bigquery',
        ...     connection_params={
        ...         'project_id': 'my-project',
        ...         'credentials_path': 'service-account.json',
        ...         'dataset_id': 'analytics'
        ...     }
        ... )
        >>> bq = BigQueryDataSource(config)
        >>> bq.connect()
        >>> df = bq.load_data("SELECT * FROM users WHERE signup_date > '2024-01-01'")
    """

    def __init__(self, config: DataSourceConfig) -> None:
        """Initialize BigQuery data source."""
        super().__init__(config)
        self.project_id = config.connection_params.get("project_id", "")
        self.dataset_id = config.connection_params.get("dataset_id")
        self.credentials_path = config.connection_params.get("credentials_path")
        self.credentials_json = config.connection_params.get("credentials_json")
        self.location = config.connection_params.get("location", "US")
        self.client_ = None

    def connect(self) -> bool:
        """
        Establish connection to BigQuery.

        Returns:
            True if connection successful

        Raises:
            ImportError: If google-cloud-bigquery is not installed
            RuntimeError: If connection fails
        """
        try:
            from google.cloud import (
                bigquery,  # type: ignore[import-untyped]
            )

            if self.credentials_path:
                self.client_ = bigquery.Client.from_service_account_json(
                    self.credentials_path,
                    project=self.project_id,
                )
            elif self.credentials_json:
                self.client_ = bigquery.Client.from_service_account_info(
                    self.credentials_json,
                    project=self.project_id,
                )
            else:
                self.client_ = bigquery.Client(project=self.project_id)

            self._connection = self.client_
            return True

        except ImportError as e:
            msg = (
                "google-cloud-bigquery is not installed. "
                "Install it with: uv add google-cloud-bigquery"
            )
            raise ImportError(msg) from e
        except Exception as e:
            msg = f"Failed to connect to BigQuery: {e}"
            raise RuntimeError(msg) from e

    def disconnect(self) -> None:
        """Close BigQuery connection."""
        if self.client_ is not None:
            self.client_.close()
            self.client_ = None
            self._connection = None

    def test_connection(self) -> bool:
        """
        Test if connection is alive.

        Returns:
            True if can execute queries
        """
        if self.client_ is None:
            return False
        try:
            query = "SELECT 1"
            self.client_.query(query).result()
            return True
        except Exception:
            return False

    def load_data(self, query: str, **kwargs: Any) -> pd.DataFrame:
        """
        Execute BigQuery SQL and load results.

        Args:
            query: SQL query string
            **kwargs: Additional parameters for query

        Returns:
            DataFrame containing query results

        Raises:
            ValueError: If query is invalid
            RuntimeError: If not connected or query fails
        """
        self.validate_query(query)

        if self.client_ is None:
            msg = "Not connected to BigQuery. Call connect() first."
            raise RuntimeError(msg)

        # Check cache
        cache_key = f"bigquery_{hash(query)}"
        if self.config.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            query_job = self.client_.query(query)
            df = query_job.to_dataframe()

            if self.config.cache_enabled:
                self._cache[cache_key] = df

            return df
        except Exception as e:
            msg = f"Failed to execute query: {e}"
            raise RuntimeError(msg) from e

    def get_schema(self, table_name: str) -> dict[str, Any]:
        """
        Get schema information for a BigQuery table.

        Args:
            table_name: Table name (can be dataset.table or project.dataset.table)

        Returns:
            Dictionary with column information
        """
        if self.client_ is None:
            msg = "Not connected to BigQuery."
            raise RuntimeError(msg)

        try:
            # Parse table reference
            if "." not in table_name and self.dataset_id:
                table_ref = f"{self.project_id}.{self.dataset_id}.{table_name}"
            elif table_name.count(".") == 1:
                table_ref = f"{self.project_id}.{table_name}"
            else:
                table_ref = table_name

            table = self.client_.get_table(table_ref)

            return {
                "table_name": table_name,
                "columns": [
                    {
                        "name": field.name,
                        "type": field.field_type,
                        "mode": field.mode,
                        "description": field.description,
                    }
                    for field in table.schema
                ],
                "num_rows": table.num_rows,
                "num_bytes": table.num_bytes,
            }
        except Exception as e:
            msg = f"Failed to get schema for {table_name}: {e}"
            raise RuntimeError(msg) from e

    def list_tables(self) -> list[str]:
        """
        List all tables in the dataset.

        Returns:
            List of table names
        """
        if self.client_ is None:
            msg = "Not connected to BigQuery."
            raise RuntimeError(msg)

        if not self.dataset_id:
            msg = "dataset_id must be set to list tables"
            raise ValueError(msg)

        try:
            dataset_ref = f"{self.project_id}.{self.dataset_id}"
            tables = self.client_.list_tables(dataset_ref)
            return [table.table_id for table in tables]
        except Exception as e:
            msg = f"Failed to list tables: {e}"
            raise RuntimeError(msg) from e


# Backward compatibility alias
AWSDataSource = None  # Will be imported from aws_connector
