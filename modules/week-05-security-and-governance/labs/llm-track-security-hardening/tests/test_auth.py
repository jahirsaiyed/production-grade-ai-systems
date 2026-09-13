from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.api.auth import create_token, require_auth

_test_app = FastAPI()


@_test_app.get("/protected")
def _protected(subject: str = Depends(require_auth)) -> dict:
    return {"subject": subject}


_client = TestClient(_test_app)


def test_require_auth_rejects_missing_token():
    response = _client.get("/protected")
    assert response.status_code == 401


def test_require_auth_rejects_invalid_token():
    response = _client.get(
        "/protected", headers={"Authorization": "Bearer not-a-real-token"}
    )
    assert response.status_code == 401


def test_require_auth_accepts_valid_token():
    token = create_token(subject="test-user")
    response = _client.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["subject"] == "test-user"


def test_require_auth_rejects_expired_token():
    token = create_token(subject="test-user", expires_in_seconds=-1)
    response = _client.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
