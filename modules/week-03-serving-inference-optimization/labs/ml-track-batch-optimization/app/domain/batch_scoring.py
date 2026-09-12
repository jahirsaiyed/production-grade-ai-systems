import numpy as np

from app.domain.scoring import FRAUD_THRESHOLD, ScoreResult


def score_batch(feature_vectors: list[list[float]], model) -> list[ScoreResult]:
    if not feature_vectors:
        return []
    probabilities = np.array(model.predict_proba(feature_vectors))[:, 1]
    return [
        ScoreResult(
            fraud_probability=float(p), is_fraud=float(p) >= FRAUD_THRESHOLD
        )
        for p in probabilities
    ]
