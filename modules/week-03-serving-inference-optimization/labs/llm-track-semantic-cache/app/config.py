from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    artifact_dir: Path = Path(__file__).resolve().parent.parent / "artifacts"
    openai_api_key: str | None = None
