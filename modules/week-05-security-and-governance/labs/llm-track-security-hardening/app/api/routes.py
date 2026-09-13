import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from app.adapters.audit_log import log_decision
from app.adapters.embeddings import EmbeddingCallError
from app.adapters.llm_client import LlmCallError
from app.api.auth import require_auth
from app.api.schemas import AskRequest, AskResponse, Citation
from app.domain.groundedness import score_groundedness
from app.domain.guardrails import check_citations_are_known, check_prompt_injection
from app.domain.pii_redaction import redact
from app.domain.rag import build_citations, build_rag_prompt
from app.domain.retrieval import hybrid_search
from app.domain.tokenizing import tokenize

router = APIRouter()
logger = logging.getLogger(__name__)
AUDIT_LOG_PATH = Path(__file__).resolve().parent.parent.parent / "audit.log"
KNOWN_SOURCES = {"product-faq.md", "hr-policy.md", "engineering-runbook.md"}


@router.post("/ask", response_model=AskResponse)
def ask(
    payload: AskRequest,
    request: Request,
    subject: str = Depends(require_auth),
) -> AskResponse:
    if check_prompt_injection(payload.question):
        raise HTTPException(status_code=400, detail="request blocked by input guardrail")

    redacted_question = redact(payload.question)
    state = request.app.state

    try:
        query_vector = state.embedding_client.embed(payload.question)
    except EmbeddingCallError as exc:
        logger.warning(
            f"embedding call failed after retries, serving fallback answer: {exc}"
        )
        log_decision(
            AUDIT_LOG_PATH,
            subject=subject,
            input_summary={"question": redacted_question},
            decision={"outcome": "embedding_failure_fallback"},
        )
        return AskResponse(
            question=payload.question,
            answer="The assistant is temporarily unavailable. Please try again.",
            citations=[],
            source="mock" if state.embedding_client.is_mock else "llm",
            groundedness_score=0.0,
        )

    bm25_scores = list(state.bm25_index.get_scores(tokenize(payload.question)))
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
    unknown_sources = check_citations_are_known(citation_dicts, KNOWN_SOURCES)
    if unknown_sources:
        logger.error(
            f"output guardrail caught fabricated citation sources: {unknown_sources}"
        )
        raise HTTPException(status_code=500, detail="response failed output guardrail check")

    citations = [Citation(**c) for c in citation_dicts]
    groundedness_score = score_groundedness(answer, [r.chunk.text for r in retrieved])

    log_decision(
        AUDIT_LOG_PATH,
        subject=subject,
        input_summary={"question": redacted_question},
        decision={
            "sources": [c.source for c in citations],
            "groundedness_score": groundedness_score,
        },
    )

    return AskResponse(
        question=payload.question,
        answer=answer,
        citations=citations,
        source="mock" if state.llm_client.is_mock else "llm",
        groundedness_score=groundedness_score,
    )


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
def readyz(request: Request) -> dict[str, str]:
    if getattr(request.app.state, "chunks", None) is None:
        return {"status": "not-ready"}
    return {"status": "ready"}
