import json
import os
import tempfile
from pathlib import Path

import joblib

from app.adapters.model_store import _sha256_of


class _StubModel:
    def predict_proba(self, X):
        return [[0.9, 0.1] for _ in X]


_artifact_dir = Path(tempfile.mkdtemp(prefix="ml-lab-test-artifacts-"))
_model_path = _artifact_dir / "model.joblib"
joblib.dump(_StubModel(), _model_path)

_manifest = {
    "artifact_version": "0.1.0-test",
    "sha256": _sha256_of(_model_path),
}
(_artifact_dir / "manifest.json").write_text(json.dumps(_manifest))

os.environ["ARTIFACT_DIR"] = str(_artifact_dir)
# Deterministic tests: never let the simulated flaky feature store fail.
os.environ["FEATURE_STORE_FAILURE_RATE"] = "0"
