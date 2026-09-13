import pytest
from pydantic import ValidationError


def test_empty_jwt_secret_key_is_rejected(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "")
    monkeypatch.setenv("MODEL_ENCRYPTION_KEY", "x" * 32)
    from app.config import Settings

    with pytest.raises(ValidationError, match="at least 1 character"):
        Settings(_env_file=None)
