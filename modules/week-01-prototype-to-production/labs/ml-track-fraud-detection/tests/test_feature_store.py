import pytest

from app.adapters.feature_store import FeatureStoreUnavailable, fetch_features


def test_fetch_features_retries_then_succeeds(monkeypatch):
    call_count = {"n": 0}

    def fake_random():
        call_count["n"] += 1
        return 0.0 if call_count["n"] < 3 else 1.0

    monkeypatch.setattr("app.adapters.feature_store.random.random", fake_random)
    features = fetch_features("txn-1", failure_rate=0.5)
    assert features == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert call_count["n"] == 3


def test_fetch_features_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr("app.adapters.feature_store.random.random", lambda: 0.0)
    with pytest.raises(FeatureStoreUnavailable):
        fetch_features("txn-1", failure_rate=1.0)
