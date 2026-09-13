from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    artifact_dir: Path = Path(__file__).resolve().parent.parent / "artifacts"
    feature_store_failure_rate: float = 0.2
    jwt_secret_key: str
    model_encryption_key: str
