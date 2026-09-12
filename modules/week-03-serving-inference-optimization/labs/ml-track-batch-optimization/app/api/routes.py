import logging

from fastapi import APIRouter, HTTPException, Request

from app.adapters.feature_store import FeatureStoreUnavailable, fetch_features
from app.api.schemas import ScoreRequest, ScoreResponse
from app.config import Settings
from app.domain.scoring import score_transaction

router = APIRouter()
settings = Settings()
logger = logging.getLogger(__name__)


@router.post("/score", response_model=ScoreResponse)
def score(payload: ScoreRequest, request: Request) -> ScoreResponse:
    try:
        features = fetch_features(
            payload.transaction_id, failure_rate=settings.feature_store_failure_rate
        )
    except FeatureStoreUnavailable as exc:
        logger.error(f"feature store unavailable for transaction_id={payload.transaction_id}: {exc}")
        raise HTTPException(status_code=503, detail="feature store unavailable, please retry") from exc
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
