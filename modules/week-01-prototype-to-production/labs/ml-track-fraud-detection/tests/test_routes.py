from unittest.mock import patch

from fastapi.testclient import TestClient

from app.adapters.feature_store import FeatureStoreUnavailable
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


def test_score_returns_fraud_probability():
    with TestClient(app) as client:
        response = client.post(
            "/score",
            json={
                "transaction_id": "txn-1",
                "amount": 42.5,
                "merchant_category": "electronics",
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["transaction_id"] == "txn-1"
    assert 0.0 <= body["fraud_probability"] <= 1.0


def test_score_rejects_negative_amount():
    with TestClient(app) as client:
        response = client.post(
            "/score",
            json={
                "transaction_id": "txn-1",
                "amount": -5,
                "merchant_category": "electronics",
            },
        )
    assert response.status_code == 422


def test_score_returns_503_when_feature_store_unavailable():
    with TestClient(app) as client:
        with patch(
            "app.api.routes.fetch_features",
            side_effect=FeatureStoreUnavailable("simulated outage"),
        ):
            response = client.post(
                "/score",
                json={
                    "transaction_id": "txn-1",
                    "amount": 42.5,
                    "merchant_category": "electronics",
                },
            )
    assert response.status_code == 503
    assert "detail" in response.json()
