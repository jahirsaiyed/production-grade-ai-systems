import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.adapters.model_store import load_model
from app.api.routes import router
from app.config import Settings
from app.logging_utils import configure_logging
from app.middleware import RequestIdMiddleware

configure_logging()
logger = logging.getLogger(__name__)
settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    loaded = load_model(settings.artifact_dir, settings.model_encryption_key.encode())
    app.state.model = loaded.model
    logger.info(f"model loaded, artifact_version={loaded.manifest['artifact_version']}")
    yield


app = FastAPI(title="Fraud Detection Service (Security-Hardened)", lifespan=lifespan)
app.add_middleware(RequestIdMiddleware)
app.include_router(router)
