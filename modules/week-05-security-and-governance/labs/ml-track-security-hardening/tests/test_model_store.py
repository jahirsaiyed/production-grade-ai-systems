import hashlib
import io
import json

import joblib
from cryptography.fernet import Fernet

from app.adapters.model_store import ArtifactIntegrityError, load_model


class _StubModel:
    def predict_proba(self, X):
        return [[0.9, 0.1]]


def _write_encrypted_artifact(artifact_dir, key: bytes, model=None):
    model = model or _StubModel()
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    plaintext = buffer.getvalue()
    encrypted = Fernet(key).encrypt(plaintext)

    model_path = artifact_dir / "model.joblib.enc"
    model_path.write_bytes(encrypted)
    manifest = {
        "artifact_version": "0.1.0-test",
        "sha256": hashlib.sha256(encrypted).hexdigest(),
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest))
    return encrypted


def test_load_model_round_trips_a_valid_encrypted_artifact(tmp_path):
    key = Fernet.generate_key()
    _write_encrypted_artifact(tmp_path, key)

    loaded = load_model(tmp_path, key)

    assert loaded.model.predict_proba([[0.0] * 6]) == [[0.9, 0.1]]
    assert loaded.manifest["artifact_version"] == "0.1.0-test"


def test_load_model_raises_on_tampered_ciphertext(tmp_path):
    key = Fernet.generate_key()
    _write_encrypted_artifact(tmp_path, key)

    model_path = tmp_path / "model.joblib.enc"
    corrupted = bytearray(model_path.read_bytes())
    corrupted[0] ^= 0xFF
    model_path.write_bytes(bytes(corrupted))

    try:
        load_model(tmp_path, key)
        assert False, "expected ArtifactIntegrityError"
    except ArtifactIntegrityError:
        pass


def test_load_model_raises_on_wrong_key(tmp_path):
    key_a = Fernet.generate_key()
    key_b = Fernet.generate_key()
    _write_encrypted_artifact(tmp_path, key_a)

    try:
        load_model(tmp_path, key_b)
        assert False, "expected ArtifactIntegrityError"
    except ArtifactIntegrityError:
        pass
