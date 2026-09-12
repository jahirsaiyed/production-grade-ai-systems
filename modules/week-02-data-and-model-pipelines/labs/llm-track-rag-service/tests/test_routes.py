from unittest.mock import patch

from fastapi.testclient import TestClient

from app.adapters.embeddings import EmbeddingCallError
from app.adapters.llm_client import LlmCallError
from app.main import app


def test_healthz_returns_ok():
    with TestClient(app) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_returns_ready_after_startup():
    with TestClient(app) as client:
        response = client.get("/readyz")
    assert response.json() == {"status": "ready"}


def test_ask_returns_answer_with_citations():
    with TestClient(app) as client:
        response = client.post(
            "/ask", json={"question": "How many vacation days?"}
        )
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "mock"
    assert len(body["citations"]) >= 1
    assert body["citations"][0]["source"] == "hr-policy.md"


def test_ask_rejects_empty_question():
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_falls_back_gracefully_when_llm_call_fails():
    with TestClient(app) as client:
        with patch.object(
            client.app.state.llm_client, "complete", side_effect=LlmCallError("boom")
        ):
            response = client.post("/ask", json={"question": "What is RAG?"})
    assert response.status_code == 200
    assert (
        response.json()["answer"]
        == "The assistant is temporarily unavailable. Please try again."
    )


def test_ask_falls_back_gracefully_when_embedding_call_fails():
    with TestClient(app) as client:
        with patch.object(
            client.app.state.embedding_client,
            "embed",
            side_effect=EmbeddingCallError("boom"),
        ):
            response = client.post("/ask", json={"question": "What is RAG?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The assistant is temporarily unavailable. Please try again."
    assert body["citations"] == []
