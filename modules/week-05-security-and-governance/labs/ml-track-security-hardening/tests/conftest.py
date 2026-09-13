import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import joblib
from cryptography.fernet import Fernet

os.environ["JWT_SECRET_KEY"] = "test-secret-key-not-for-production"
os.environ["MODEL_ENCRYPTION_KEY"] = "kR9mZ3xQhT7vN2pL8wF5yB1cA6dE4gJ0sU3iO9nM7k8="
os.environ["FEATURE_STORE_FAILURE_RATE"] = "0"


class _StubModel:
    def predict_proba(self, X):
        return [[0.9, 0.1]]


_artifact_dir = Path(tempfile.mkdtemp(prefix="ml-security-lab-test-artifacts-"))

_buffer = io.BytesIO()
joblib.dump(_StubModel(), _buffer)
_key = os.environ["MODEL_ENCRYPTION_KEY"].encode()
_encrypted = Fernet(_key).encrypt(_buffer.getvalue())

_model_path = _artifact_dir / "model.joblib.enc"
_model_path.write_bytes(_encrypted)

_manifest = {
    "artifact_version": "0.1.0-test",
    "sha256": hashlib.sha256(_encrypted).hexdigest(),
}
(_artifact_dir / "manifest.json").write_text(json.dumps(_manifest))

os.environ["ARTIFACT_DIR"] = str(_artifact_dir)
