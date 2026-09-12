from fastapi.testclient import TestClient

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
