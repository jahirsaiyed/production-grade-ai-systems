import logging

from fastapi import APIRouter, HTTPException, Request

from app.adapters.feature_store import FeatureStoreUnavailable, fetch_features
from app.api.schemas import (
    BatchScoreRequest,
    BatchScoreResponse,
    ScoreRequest,
    ScoreResponse,
)
from app.config import Settings
from app.domain.batch_scoring import score_batch
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


@router.post("/score/batch", response_model=BatchScoreResponse)
def score_batch_endpoint(
    payload: BatchScoreRequest, request: Request
) -> BatchScoreResponse:
    try:
        feature_vectors = [
            fetch_features(
                txn.transaction_id,
                failure_rate=settings.feature_store_failure_rate,
            )
            for txn in payload.transactions
        ]
    except FeatureStoreUnavailable as exc:
        logger.error(f"feature store unavailable during batch scoring: {exc}")
        raise HTTPException(
            status_code=503, detail="feature store unavailable, please retry"
        ) from exc

    results = score_batch(feature_vectors, request.app.state.model)
    return BatchScoreResponse(
        results=[
            ScoreResponse(
                transaction_id=txn.transaction_id,
                fraud_probability=r.fraud_probability,
                is_fraud=r.is_fraud,
            )
            for txn, r in zip(payload.transactions, results)
        ]
    )


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
def readyz(request: Request) -> dict[str, str]:
    if getattr(request.app.state, "model", None) is None:
        return {"status": "not-ready"}
    return {"status": "ready"}
