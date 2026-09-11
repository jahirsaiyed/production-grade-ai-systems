import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.adapters.llm_client import LlmClient
from app.api.routes import router
from app.config import Settings
from app.logging_utils import configure_logging
from app.middleware import RequestIdMiddleware

configure_logging()
logger = logging.getLogger(__name__)
settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.llm_client = LlmClient(api_key=settings.openai_api_key)
    logger.info(f"llm client ready, mock_mode={app.state.llm_client.is_mock}")
    yield


app = FastAPI(title="Q&A Service", lifespan=lifespan)
app.add_middleware(RequestIdMiddleware)
app.include_router(router)
