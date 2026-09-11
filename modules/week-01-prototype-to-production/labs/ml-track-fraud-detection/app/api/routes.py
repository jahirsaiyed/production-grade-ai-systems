from fastapi import APIRouter, Request

from app.adapters.feature_store import fetch_features
from app.api.schemas import ScoreRequest, ScoreResponse
from app.config import Settings
from app.domain.scoring import score_transaction

router = APIRouter()
settings = Settings()


@router.post("/score", response_model=ScoreResponse)
def score(payload: ScoreRequest, request: Request) -> ScoreResponse:
    features = fetch_features(
        payload.transaction_id, failure_rate=settings.feature_store_failure_rate
    )
    result = score_transaction(features, request.app.state.model)
    return ScoreResponse(
        transaction_id=payload.transaction_id,
        fraud_probability=result.fraud_probability,
        is_fraud=result.is_fraud,
    )


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
def readyz(request: Request) -> dict[str, str]:
    if getattr(request.app.state, "model", None) is None:
        return {"status": "not-ready"}
    return {"status": "ready"}
