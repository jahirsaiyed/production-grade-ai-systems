from fastapi.testclient import TestClient

from app.api.auth import create_token
from app.main import app


def test_score_rejects_request_without_auth():
    with TestClient(app) as client:
        response = client.post(
            "/score",
            json={
                "transaction_id": "txn-1",
                "amount": 100.0,
                "merchant_category": "retail",
            },
        )
    assert response.status_code == 401


def test_score_accepts_request_with_valid_auth():
    token = create_token(subject="test-caller")
    with TestClient(app) as client:
        response = client.post(
            "/score",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "transaction_id": "txn-1",
                "amount": 100.0,
                "merchant_category": "retail",
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["transaction_id"] == "txn-1"
    assert 0.0 <= body["fraud_probability"] <= 1.0


def test_score_writes_an_audit_log_entry_without_leaking_the_raw_token(tmp_path, monkeypatch):
    import app.api.routes as routes_module

    log_path = tmp_path / "audit.log"
    monkeypatch.setattr(routes_module, "AUDIT_LOG_PATH", log_path)

    token = create_token(subject="test-caller")
    with TestClient(app) as client:
        client.post(
            "/score",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "transaction_id": "txn-2",
                "amount": 50.0,
                "merchant_category": "grocery",
            },
        )

    log_contents = log_path.read_text(encoding="utf-8")
    assert "test-caller" in log_contents
    assert token not in log_contents


def test_healthz_and_readyz_do_not_require_auth():
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 200
