"""
Generates the versioned fraud-detection model artifact.

Run this manually (`make train` or `python train_model.py`) whenever the
model needs to change. The output (artifacts/model.joblib and
artifacts/manifest.json) is committed to git, mirroring how a real team would
publish a new model version.
"""
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

ARTIFACT_DIR = Path(__file__).parent / "artifacts"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    X, y = make_classification(
        n_samples=2000,
        n_features=6,
        n_informative=4,
        weights=[0.9, 0.1],
        random_state=42,
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    ARTIFACT_DIR.mkdir(exist_ok=True)
    model_path = ARTIFACT_DIR / "model.joblib"
    joblib.dump(model, model_path)

    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    manifest = {
        "artifact_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "sha256": digest,
        "python_version": platform.python_version(),
        "key_dependencies": {"scikit-learn": sklearn.__version__},
    }
    (ARTIFACT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {model_path} and manifest.json (sha256={digest[:12]}...)")


if __name__ == "__main__":
    main()
