import random

from tenacity import retry, stop_after_attempt, wait_fixed


class FeatureStoreUnavailable(RuntimeError):
    """Raised when the (simulated) feature store cannot be reached."""


def _call_feature_store(transaction_id: str, failure_rate: float) -> list[float]:
    if random.random() < failure_rate:
        raise FeatureStoreUnavailable(f"feature store timed out for {transaction_id}")
    # 6 values to match the 6-feature model trained by train_model.py.
    return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


@retry(stop=stop_after_attempt(3), wait=wait_fixed(0.1), reraise=True)
def fetch_features(transaction_id: str, failure_rate: float = 0.0) -> list[float]:
    return _call_feature_store(transaction_id, failure_rate)
