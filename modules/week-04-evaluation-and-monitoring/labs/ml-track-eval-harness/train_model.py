"""
Generates the versioned fraud-detection model artifact.

Run this manually (`make train` or `python train_model.py`) whenever the
model needs to change. The output (artifacts/model.joblib,
artifacts/manifest.json, and artifacts/held_out.json) is committed to git,
mirroring how a real team would publish a new model version.

The full synthetic dataset is generated once and split into a training
portion (used to fit the model) and a held-out portion (persisted so
eval_harness.py can score the model against data it never trained on, drawn
from the SAME distribution rather than a different synthetic problem).
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
from sklearn.model_selection import train_test_split

ARTIFACT_DIR = Path(__file__).parent / "artifacts"

# Seed for splitting the single rs=42 dataset into train/held-out portions.
# Independent of the data-generation seed (42) above it.
HELD_OUT_SPLIT_RANDOM_STATE = 7


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def main(artifact_dir: Path = ARTIFACT_DIR) -> None:
    X, y = make_classification(
        n_samples=2000,
        n_features=6,
        n_informative=4,
        weights=[0.9, 0.1],
        random_state=42,
    )
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=HELD_OUT_SPLIT_RANDOM_STATE,
        stratify=y,
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    artifact_dir.mkdir(exist_ok=True, parents=True)
    model_path = artifact_dir / "model.joblib"
    joblib.dump(model, model_path)

    held_out_path = artifact_dir / "held_out.json"
    held_out_path.write_text(
        json.dumps({"X": X_holdout.tolist(), "y": y_holdout.tolist()}, indent=2)
    )

    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    manifest = {
        "artifact_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "sha256": digest,
        "python_version": platform.python_version(),
        "key_dependencies": {"scikit-learn": sklearn.__version__},
    }
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {model_path} and manifest.json (sha256={digest[:12]}...)")
    print(f"Wrote {held_out_path} ({len(y_holdout)} held-out rows)")


if __name__ == "__main__":
    main()
