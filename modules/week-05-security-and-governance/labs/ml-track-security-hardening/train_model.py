"""
Generates the versioned fraud-detection model artifact, encrypted at rest.

Run this manually (`make train` or `python train_model.py`) whenever the
model needs to change. Requires MODEL_ENCRYPTION_KEY to be set — generate one
with:
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

The output (artifacts/model.joblib.enc and artifacts/manifest.json) is
committed to git, mirroring how a real team would publish a new encrypted
model version.
"""
import hashlib
import io
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from app.adapters.encryption import encrypt_bytes

ARTIFACT_DIR = Path(__file__).parent / "artifacts"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def main(artifact_dir: Path = ARTIFACT_DIR) -> None:
    encryption_key = os.environ["MODEL_ENCRYPTION_KEY"].encode()

    X, y = make_classification(
        n_samples=2000,
        n_features=6,
        n_informative=4,
        weights=[0.9, 0.1],
        random_state=42,
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    plaintext = buffer.getvalue()
    encrypted = encrypt_bytes(plaintext, encryption_key)

    artifact_dir.mkdir(exist_ok=True, parents=True)
    model_path = artifact_dir / "model.joblib.enc"
    model_path.write_bytes(encrypted)

    digest = hashlib.sha256(encrypted).hexdigest()
    manifest = {
        "artifact_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "sha256": digest,
        "encrypted": True,
        "python_version": platform.python_version(),
        "key_dependencies": {
            "scikit-learn": sklearn.__version__,
            "cryptography": "43.0.3",
        },
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {model_path} and manifest.json (sha256={digest[:12]}...)")


if __name__ == "__main__":
    main()
