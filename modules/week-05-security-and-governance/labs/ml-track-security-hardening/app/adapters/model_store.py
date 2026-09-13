import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
from cryptography.fernet import InvalidToken

from app.adapters.encryption import decrypt_bytes


class ArtifactIntegrityError(RuntimeError):
    """Raised when the model artifact does not match its manifest or fails decryption."""


@dataclass(frozen=True)
class LoadedModel:
    model: object
    manifest: dict


def load_model(artifact_dir: Path, encryption_key: bytes) -> LoadedModel:
    manifest_path = artifact_dir / "manifest.json"
    model_path = artifact_dir / "model.joblib.enc"
    manifest = json.loads(manifest_path.read_text())

    encrypted_bytes = model_path.read_bytes()
    actual_sha256 = hashlib.sha256(encrypted_bytes).hexdigest()
    if actual_sha256 != manifest["sha256"]:
        raise ArtifactIntegrityError(
            f"model.joblib.enc sha256 {actual_sha256} does not match "
            f"manifest sha256 {manifest['sha256']}"
        )

    try:
        decrypted_bytes = decrypt_bytes(encrypted_bytes, encryption_key)
    except InvalidToken as exc:
        raise ArtifactIntegrityError(
            "model.joblib.enc failed decryption: invalid key or corrupted ciphertext"
        ) from exc

    model = joblib.load(io.BytesIO(decrypted_bytes))
    return LoadedModel(model=model, manifest=manifest)
