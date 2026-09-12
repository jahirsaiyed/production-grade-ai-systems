import logging

from fastapi import APIRouter, Request

from app.adapters.embeddings import EmbeddingCallError
from app.adapters.llm_client import LlmCallError
from app.api.schemas import AskRequest, AskResponse, Citation
from app.domain.rag import build_citations, build_rag_prompt
from app.domain.retrieval import hybrid_search
from app.domain.tokenizing import tokenize

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request) -> AskResponse:
    state = request.app.state
    try:
        query_vector = state.embedding_client.embed(payload.question)
    except EmbeddingCallError as exc:
        logger.warning(
            f"embedding call failed after retries, serving fallback answer: {exc}"
        )
        return AskResponse(
            question=payload.question,
            answer="The assistant is temporarily unavailable. Please try again.",
            citations=[],
            source="mock" if state.embedding_client.is_mock else "llm",
            cache_hit=False,
        )

    cached = state.semantic_cache.lookup(query_vector)
    if cached is not None:
        return AskResponse(
            question=payload.question,
            answer=cached.answer,
            citations=[Citation(**c) for c in cached.citations],
            source="mock" if state.llm_client.is_mock else "llm",
            cache_hit=True,
        )

    bm25_scores = list(
        state.bm25_index.get_scores(tokenize(payload.question))
    )

    retrieved = hybrid_search(
        query_vector, state.chunks, state.dense_vectors, bm25_scores, k=3
    )

    prompt = build_rag_prompt(payload.question, retrieved)
    try:
        answer = state.llm_client.complete(prompt)
    except LlmCallError as exc:
        logger.warning(
            f"llm call failed after retries, serving fallback answer: {exc}"
        )
        answer = "The assistant is temporarily unavailable. Please try again."

    citation_dicts = build_citations(retrieved)
    state.semantic_cache.store(query_vector, answer, citation_dicts)

    return AskResponse(
        question=payload.question,
        answer=answer,
        citations=[Citation(**c) for c in citation_dicts],
        source="mock" if state.llm_client.is_mock else "llm",
        cache_hit=False,
    )


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
def readyz(request: Request) -> dict[str, str]:
    if getattr(request.app.state, "chunks", None) is None:
        return {"status": "not-ready"}
    return {"status": "ready"}
