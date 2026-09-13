import json

from cryptography.fernet import Fernet

from app.adapters.model_store import load_model
from train_model import main


def test_main_writes_a_working_encrypted_artifact(tmp_path, monkeypatch):
    key = Fernet.generate_key()
    monkeypatch.setenv("MODEL_ENCRYPTION_KEY", key.decode())

    main(artifact_dir=tmp_path)

    assert (tmp_path / "model.joblib.enc").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["artifact_version"] == "0.1.0"
    assert manifest["encrypted"] is True

    loaded = load_model(tmp_path, key)
    probability = loaded.model.predict_proba([[0.1, 0.2, -0.3, 0.4, 0.5, -0.1]])[0][1]
    assert 0.0 <= probability <= 1.0
