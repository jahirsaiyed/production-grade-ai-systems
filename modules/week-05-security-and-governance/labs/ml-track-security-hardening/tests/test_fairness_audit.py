import numpy as np
from cryptography.fernet import Fernet

from fairness_audit import compute_fairness_metrics, run


def test_compute_fairness_metrics_with_a_known_disparity():
    y_pred = np.array([1, 1, 0, 0])
    protected_group = np.array([0, 0, 1, 1])

    metrics = compute_fairness_metrics(y_pred, protected_group)

    assert metrics["selection_rate_group_0"] == 1.0
    assert metrics["selection_rate_group_1"] == 0.0
    assert metrics["demographic_parity_difference"] == -1.0
    assert metrics["disparate_impact_ratio"] == 0.0


def test_compute_fairness_metrics_with_no_disparity():
    y_pred = np.array([1, 0, 1, 0])
    protected_group = np.array([0, 0, 1, 1])

    metrics = compute_fairness_metrics(y_pred, protected_group)

    assert metrics["selection_rate_group_0"] == 0.5
    assert metrics["selection_rate_group_1"] == 0.5
    assert metrics["demographic_parity_difference"] == 0.0
    assert metrics["disparate_impact_ratio"] == 1.0


def test_run_produces_metrics_against_a_freshly_trained_disposable_artifact(tmp_path):
    # run() takes the encryption key as an explicit argument rather than reading
    # the environment itself, so this test builds its own disposable encrypted
    # artifact under tmp_path — it does not touch or need the real committed
    # artifact (which needs the actual secret key from Task 5's Step 9, not
    # available to this test, and shouldn't be needed just to check run()'s
    # output shape).
    import hashlib
    import io
    import json

    import joblib
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    from app.adapters.encryption import encrypt_bytes

    key = Fernet.generate_key()
    model = LogisticRegression(max_iter=1000)
    X, y = make_classification(
        n_samples=200, n_features=6, n_informative=4, weights=[0.9, 0.1], random_state=1
    )
    model.fit(X, y)

    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    encrypted = encrypt_bytes(buffer.getvalue(), key)

    (tmp_path / "model.joblib.enc").write_bytes(encrypted)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_version": "0.1.0-test",
                "sha256": hashlib.sha256(encrypted).hexdigest(),
            }
        )
    )

    metrics = run(key, artifact_dir=tmp_path)

    assert "demographic_parity_difference" in metrics
    assert "disparate_impact_ratio" in metrics
