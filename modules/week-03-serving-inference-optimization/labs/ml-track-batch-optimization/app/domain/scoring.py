from dataclasses import dataclass

FRAUD_THRESHOLD = 0.5


@dataclass(frozen=True)
class ScoreResult:
    fraud_probability: float
    is_fraud: bool


def score_transaction(feature_vector: list[float], model) -> ScoreResult:
    probability = float(model.predict_proba([feature_vector])[0][1])
    return ScoreResult(
        fraud_probability=probability,
        is_fraud=probability >= FRAUD_THRESHOLD,
    )
