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


def test_ask_returns_mock_answer_by_default():
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "What is RAG?"})
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "mock"
    assert "mock answer" in body["answer"].lower()


def test_ask_rejects_empty_question():
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422
