import json
import os

import joblib
import pytest


class _StubModel:
    def predict_proba(self, X):
        return [[0.9, 0.1]]


@pytest.fixture(scope="session", autouse=True)
def _artifact_dir(tmp_path_factory):
    artifact_dir = tmp_path_factory.mktemp("artifacts")
    model_path = artifact_dir / "model.joblib"
    joblib.dump(_StubModel(), model_path)

    from app.adapters.model_store import _sha256_of

    manifest = {
        "artifact_version": "0.1.0-test",
        "sha256": _sha256_of(model_path),
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest))

    os.environ["ARTIFACT_DIR"] = str(artifact_dir)
    # Deterministic tests: never let the simulated flaky feature store fail.
    os.environ["FEATURE_STORE_FAILURE_RATE"] = "0"
    return artifact_dir
