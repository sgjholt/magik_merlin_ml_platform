# Configuration

Learn how to configure the ML Platform for your environment.

## Configuration File

The platform uses a centralized configuration file: `src/config/settings.py`

### Environment Variables

Create a `.env` file in the project root:

```bash
# Application Settings
APP_NAME=ML Platform
DEBUG=true
HOST=localhost
PORT=5006

# MLflow Settings
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_ARTIFACT_ROOT=./mlartifacts

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json  # or 'console'

# Data Sources
DATA_ROOT=./data
```

## MLflow Configuration

### Basic Setup

```python
# In .env file
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=default
```

### Remote MLflow Server

```bash
# Point to remote server
MLFLOW_TRACKING_URI=https://mlflow.example.com
```

### Backend Store

```bash
# SQLite (default)
MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db

# PostgreSQL
MLFLOW_BACKEND_STORE_URI=postgresql://user:pass@localhost/mlflow
```

## Data Source Configuration

### Local Files

```python
[data_sources.local]
enabled = true
root_path = "./data"
allowed_formats = ["csv", "parquet", "xlsx"]
```

### Snowflake

```bash
# In .env
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema
```

### AWS S3

```bash
# In .env
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1
S3_BUCKET=your-bucket
```

## Advanced Configuration

See [settings.py](../../src/config/settings.py) for all available options.
