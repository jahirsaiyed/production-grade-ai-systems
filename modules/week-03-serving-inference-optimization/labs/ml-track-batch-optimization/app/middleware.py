from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.logging_utils import new_request_id, request_id_var


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        incoming = request.headers.get("x-request-id", new_request_id())
        token = request_id_var.set(incoming)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["x-request-id"] = incoming
        return response
