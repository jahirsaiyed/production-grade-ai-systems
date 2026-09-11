from app.domain.scoring import score_transaction


class _StubModel:
    def __init__(self, probability: float):
        self._probability = probability

    def predict_proba(self, X):
        return [[1 - self._probability, self._probability]]


def test_score_transaction_flags_high_probability_as_fraud():
    model = _StubModel(probability=0.9)
    result = score_transaction([1.0, 2.0, 3.0], model)
    assert result.is_fraud is True
    assert result.fraud_probability == 0.9


def test_score_transaction_does_not_flag_low_probability():
    model = _StubModel(probability=0.1)
    result = score_transaction([1.0, 2.0, 3.0], model)
    assert result.is_fraud is False
    assert result.fraud_probability == 0.1
