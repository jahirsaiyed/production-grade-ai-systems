from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    transaction_id: str = Field(..., min_length=1, max_length=64)
    amount: float = Field(..., ge=0)
    merchant_category: str = Field(..., min_length=1, max_length=32)


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool


class BatchScoreRequest(BaseModel):
    transactions: list[ScoreRequest] = Field(..., min_length=1, max_length=1000)


class BatchScoreResponse(BaseModel):
    results: list[ScoreResponse]
