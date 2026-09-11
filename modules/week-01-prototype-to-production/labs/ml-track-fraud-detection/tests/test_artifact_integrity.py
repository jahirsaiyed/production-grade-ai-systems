from pathlib import Path

from app.adapters.model_store import load_model


def test_committed_artifact_passes_integrity_check():
    artifact_dir = Path(__file__).resolve().parent.parent / "artifacts"
    loaded = load_model(artifact_dir)
    assert loaded.manifest["sha256"]
    assert loaded.manifest["artifact_version"]
