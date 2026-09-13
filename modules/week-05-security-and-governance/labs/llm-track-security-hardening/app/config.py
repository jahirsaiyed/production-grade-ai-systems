from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    artifact_dir: Path = Path(__file__).resolve().parent.parent / "artifacts"
    openai_api_key: str | None = None
    jwt_secret_key: str = Field(..., min_length=1)
