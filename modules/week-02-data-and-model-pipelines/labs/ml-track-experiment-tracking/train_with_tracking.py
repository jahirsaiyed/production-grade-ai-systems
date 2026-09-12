"""
Trains the fraud-detection classifier with MLflow experiment tracking and
writes a reproducibility manifest (commit, data digest, env lock, artifact
link) implementing the course's reproducibility checklist.

Run with `make train` or `python train_with_tracking.py`.
"""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from data_validation import validate_training_data

LAB_DIR = Path(__file__).parent
DEFAULT_MLRUNS_DIR = LAB_DIR / "mlruns"
DEFAULT_MANIFEST_PATH = LAB_DIR / "reproducibility_manifest.json"
REQUIREMENTS_PATH = LAB_DIR / "requirements.txt"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=LAB_DIR
        ).strip()
    except Exception:
        return "unknown"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main(
    tracking_uri: str | None = None, manifest_path: Path | None = None
) -> dict:
    tracking_uri = tracking_uri or f"file:{DEFAULT_MLRUNS_DIR}"
    manifest_path = manifest_path or DEFAULT_MANIFEST_PATH

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("fraud-detection-reproducibility")

    X, y = make_classification(
        n_samples=2000,
        n_features=6,
        n_informative=4,
        weights=[0.9, 0.1],
        random_state=42,
    )
    validate_training_data(X, y)

    data_digest = _sha256_bytes(X.tobytes() + y.tobytes())
    params = {
        "max_iter": 1000,
        "n_samples": 2000,
        "n_features": 6,
        "random_state": 42,
    }

    with mlflow.start_run() as run:
        mlflow.log_params(params)

        model = LogisticRegression(max_iter=params["max_iter"])
        model.fit(X, y)

        predictions = model.predict(X)
        metrics = {
            "accuracy": accuracy_score(y, predictions),
            "f1": f1_score(y, predictions),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        run_id = run.info.run_id
        artifact_uri = run.info.artifact_uri

    manifest = {
        "git_commit": _git_commit(),
        "data_digest": data_digest,
        "env_lock": _sha256_bytes(REQUIREMENTS_PATH.read_bytes()),
        "mlflow_run_id": run_id,
        "artifact_uri": artifact_uri,
        "metrics": metrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(
        f"Run {run_id} tracked at {tracking_uri}. "
        f"Reproducibility manifest written to {manifest_path}"
    )
    return manifest


if __name__ == "__main__":
    main()
