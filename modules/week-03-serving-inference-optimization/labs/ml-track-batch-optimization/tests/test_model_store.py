import json

import joblib
import pytest

from app.adapters.model_store import ArtifactIntegrityError, _sha256_of, load_model


class _DummyModel:
    def predict_proba(self, X):
        return [[0.5, 0.5]]


def _write_artifact(tmp_path):
    model_path = tmp_path / "model.joblib"
    joblib.dump(_DummyModel(), model_path)
    manifest = {
        "artifact_version": "0.1.0",
        "sha256": _sha256_of(model_path),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_load_model_succeeds_when_hash_matches(tmp_path):
    artifact_dir = _write_artifact(tmp_path)
    loaded = load_model(artifact_dir)
    assert loaded.manifest["artifact_version"] == "0.1.0"


def test_load_model_raises_when_hash_mismatches(tmp_path):
    artifact_dir = _write_artifact(tmp_path)
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ArtifactIntegrityError):
        load_model(artifact_dir)
