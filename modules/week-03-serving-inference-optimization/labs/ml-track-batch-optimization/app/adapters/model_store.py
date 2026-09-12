import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib


class ArtifactIntegrityError(RuntimeError):
    """Raised when the model artifact does not match its manifest."""


@dataclass(frozen=True)
class LoadedModel:
    model: object
    manifest: dict


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def load_model(artifact_dir: Path) -> LoadedModel:
    manifest_path = artifact_dir / "manifest.json"
    model_path = artifact_dir / "model.joblib"
    manifest = json.loads(manifest_path.read_text())

    actual_sha256 = _sha256_of(model_path)
    if actual_sha256 != manifest["sha256"]:
        raise ArtifactIntegrityError(
            f"model.joblib sha256 {actual_sha256} does not match "
            f"manifest sha256 {manifest['sha256']}"
        )

    model = joblib.load(model_path)
    return LoadedModel(model=model, manifest=manifest)
