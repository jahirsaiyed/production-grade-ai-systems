"""
Runs a genuine offline evaluation of the committed fraud-detection model
against a held-out labeled dataset it never saw during training.

Run with `make eval` or `python eval_harness.py`.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification

from app.adapters.model_store import load_model
from app.domain.scoring import score_transaction
from calibration import compute_brier_score, reliability_diagram_data
from metrics import compute_classification_metrics
from threshold_selection import best_threshold, sweep_thresholds

LAB_DIR = Path(__file__).parent
ARTIFACT_DIR = LAB_DIR / "artifacts"
EVAL_REPORT_PATH = LAB_DIR / "eval_report.json"

# Deliberately different from train_model.py's random_state=42 — this is what
# makes this a genuine held-out evaluation set, not a re-test on training data.
HELD_OUT_RANDOM_STATE = 123


def run() -> dict:
    loaded = load_model(ARTIFACT_DIR)
    model = loaded.model

    X, y = make_classification(
        n_samples=2000,
        n_features=6,
        n_informative=4,
        weights=[0.9, 0.1],
        random_state=HELD_OUT_RANDOM_STATE,
    )

    # Score through the exact same domain function the production service
    # uses (app/domain/scoring.py's score_transaction), not a shortcut — this
    # is the training/serving-parity lesson applied to evaluation itself.
    y_proba = []
    y_pred = []
    for row in X:
        result = score_transaction(list(row), model)
        y_proba.append(result.fraud_probability)
        y_pred.append(1 if result.is_fraud else 0)
    y_proba = np.array(y_proba)
    y_pred = np.array(y_pred)

    classification_metrics = compute_classification_metrics(y, y_pred, y_proba)
    brier_score = compute_brier_score(y, y_proba)
    reliability_bins = reliability_diagram_data(y, y_proba)
    threshold_sweep = sweep_thresholds(y, y_proba, cost_fp=1.0, cost_fn=25.0)
    chosen_threshold = best_threshold(y, y_proba, cost_fp=1.0, cost_fn=25.0)

    return {
        "metrics": classification_metrics,
        "brier_score": brier_score,
        "reliability_bins": reliability_bins,
        "threshold_sweep": threshold_sweep,
        "best_threshold": chosen_threshold,
        "held_out_random_state": HELD_OUT_RANDOM_STATE,
        "n_samples": len(y),
    }


def main() -> None:
    report = run()
    EVAL_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {EVAL_REPORT_PATH}")
    print(json.dumps(report["metrics"], indent=2))
    print(f"Brier score: {report['brier_score']:.4f}")
    print(f"Best threshold (cost-based): {report['best_threshold']}")


if __name__ == "__main__":
    main()
