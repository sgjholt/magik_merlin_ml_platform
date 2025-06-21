from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # MLflow Configuration
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "ml-platform-experiments"

    # Data Source Configurations
    snowflake_user: str | None = None
    snowflake_password: str | None = None
    snowflake_account: str | None = None
    snowflake_warehouse: str | None = None
    snowflake_database: str | None = None
    snowflake_schema: str | None = None

    # AWS Configuration
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_default_region: str = "us-east-1"
    aws_s3_bucket: str | None = None

    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # Application Configuration
    app_host: str = "0.0.0.0"
    app_port: int = 5006
    debug: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
