"""
Computes fairness metrics for the fraud-detection model against a synthetic
evaluation set that includes a synthetic protected-group attribute.

The protected_group column is generated independently of X/y — a random
binary coin flip uncorrelated with the label by construction — so this
script demonstrates HOW to measure disparity, not a rigged "look, we found
bias" example. A real fairness audit would use real demographic data; this
teaches the mechanics with a synthetic stand-in.

Run with `make fairness-audit` or `python fairness_audit.py` (requires
MODEL_ENCRYPTION_KEY to be set to the same key used to train the artifact).
"""
import json
import os
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification

from app.adapters.model_store import load_model
from app.domain.scoring import score_transaction

LAB_DIR = Path(__file__).parent
ARTIFACT_DIR = LAB_DIR / "artifacts"

FAIRNESS_EVAL_RANDOM_STATE = 99


def compute_fairness_metrics(y_pred: np.ndarray, protected_group: np.ndarray) -> dict:
    group_0_mask = protected_group == 0
    group_1_mask = protected_group == 1

    selection_rate_group_0 = float(y_pred[group_0_mask].mean())
    selection_rate_group_1 = float(y_pred[group_1_mask].mean())

    demographic_parity_difference = selection_rate_group_1 - selection_rate_group_0
    disparate_impact_ratio = (
        selection_rate_group_1 / selection_rate_group_0
        if selection_rate_group_0 > 0
        else float("inf")
    )

    return {
        "selection_rate_group_0": selection_rate_group_0,
        "selection_rate_group_1": selection_rate_group_1,
        "demographic_parity_difference": demographic_parity_difference,
        "disparate_impact_ratio": disparate_impact_ratio,
    }


def run(encryption_key: bytes, artifact_dir: Path = ARTIFACT_DIR) -> dict:
    loaded = load_model(artifact_dir, encryption_key)
    model = loaded.model

    X, y = make_classification(
        n_samples=2000,
        n_features=6,
        n_informative=4,
        weights=[0.9, 0.1],
        random_state=FAIRNESS_EVAL_RANDOM_STATE,
    )
    rng = np.random.default_rng(FAIRNESS_EVAL_RANDOM_STATE)
    protected_group = rng.integers(0, 2, size=len(y))

    y_pred = np.array(
        [1 if score_transaction(list(row), model).is_fraud else 0 for row in X]
    )

    return compute_fairness_metrics(y_pred, protected_group)


def main() -> None:
    encryption_key = os.environ["MODEL_ENCRYPTION_KEY"].encode()
    metrics = run(encryption_key)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
