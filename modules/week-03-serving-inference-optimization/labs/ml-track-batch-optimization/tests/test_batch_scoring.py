from app.domain.batch_scoring import score_batch


class _StubModel:
    def __init__(self, probabilities: list[float]):
        self._probabilities = probabilities

    def predict_proba(self, X):
        return [[1 - p, p] for p in self._probabilities]


def test_score_batch_scores_all_rows_in_one_call():
    model = _StubModel(probabilities=[0.9, 0.1, 0.5])
    results = score_batch([[1.0], [2.0], [3.0]], model)
    assert len(results) == 3
    assert results[0].is_fraud is True
    assert results[0].fraud_probability == 0.9
    assert results[1].is_fraud is False
    assert results[2].is_fraud is True  # 0.5 >= FRAUD_THRESHOLD (0.5)


def test_score_batch_returns_empty_list_for_empty_input():
    model = _StubModel(probabilities=[])
    results = score_batch([], model)
    assert results == []
