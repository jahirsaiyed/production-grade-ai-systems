import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from rank_bm25 import BM25Okapi

from app.adapters.embeddings import EmbeddingClient
from app.adapters.index_store import load_index
from app.adapters.llm_client import LlmClient
from app.api.routes import router
from app.config import Settings
from app.domain.semantic_cache import DEFAULT_SIMILARITY_THRESHOLD, SemanticCache
from app.logging_utils import configure_logging
from app.middleware import RequestIdMiddleware

configure_logging()
logger = logging.getLogger(__name__)
settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    loaded = load_index(settings.artifact_dir)
    app.state.chunks = loaded.chunks
    app.state.dense_vectors = loaded.dense_vectors
    app.state.bm25_index = BM25Okapi(loaded.bm25_tokenized_corpus)
    app.state.embedding_client = EmbeddingClient(api_key=settings.openai_api_key)
    app.state.llm_client = LlmClient(api_key=settings.openai_api_key)
    app.state.semantic_cache = SemanticCache(threshold=DEFAULT_SIMILARITY_THRESHOLD)
    logger.info(
        f"index loaded, artifact_version={loaded.manifest['artifact_version']}, "
        f"chunks={len(loaded.chunks)}"
    )
    yield


app = FastAPI(title="RAG Q&A Service", lifespan=lifespan)
app.add_middleware(RequestIdMiddleware)
app.include_router(router)
