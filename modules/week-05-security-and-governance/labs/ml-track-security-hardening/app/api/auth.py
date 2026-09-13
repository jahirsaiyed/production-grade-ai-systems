import time
from typing import Annotated

import jwt
from fastapi import Header, HTTPException

from app.config import Settings

settings = Settings()

JWT_ALGORITHM = "HS256"


def create_token(subject: str, expires_in_seconds: int = 3600) -> str:
    now = int(time.time())
    payload = {"sub": subject, "iat": now, "exp": now + expires_in_seconds}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=JWT_ALGORITHM)


def require_auth(authorization: Annotated[str | None, Header()] = None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")

    token = authorization.removeprefix("Bearer ")
    try:
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[JWT_ALGORITHM]
        )
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail="invalid token") from exc

    return payload["sub"]
