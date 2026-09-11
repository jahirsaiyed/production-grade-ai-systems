from fastapi import APIRouter, Request

from app.adapters.llm_client import LlmCallError
from app.api.schemas import AskRequest, AskResponse
from app.domain.qa import build_prompt

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request) -> AskResponse:
    client = request.app.state.llm_client
    prompt = build_prompt(payload.question)
    try:
        answer = client.complete(prompt)
    except LlmCallError:
        answer = "The assistant is temporarily unavailable. Please try again."
    return AskResponse(
        question=payload.question,
        answer=answer,
        source="mock" if client.is_mock else "llm",
    )


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
def readyz(request: Request) -> dict[str, str]:
    if getattr(request.app.state, "llm_client", None) is None:
        return {"status": "not-ready"}
    return {"status": "ready"}
