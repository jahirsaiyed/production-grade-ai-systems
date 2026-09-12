import numpy as np


def _cost_at_threshold(
    y_true, y_proba, threshold: float, cost_fp: float, cost_fn: float
) -> float:
    y_pred = (y_proba >= threshold).astype(int)
    false_positives = int(((y_pred == 1) & (y_true == 0)).sum())
    false_negatives = int(((y_pred == 0) & (y_true == 1)).sum())
    return false_positives * cost_fp + false_negatives * cost_fn


def sweep_thresholds(
    y_true,
    y_proba,
    cost_fp: float,
    cost_fn: float,
    thresholds: list[float] | None = None,
) -> list[dict]:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.05, 1.0, 0.05)]

    results = []
    for threshold in thresholds:
        cost = _cost_at_threshold(y_true, y_proba, threshold, cost_fp, cost_fn)
        results.append({"threshold": float(threshold), "cost": float(cost)})
    return results


def best_threshold(y_true, y_proba, cost_fp: float, cost_fn: float) -> dict:
    results = sweep_thresholds(y_true, y_proba, cost_fp, cost_fn)
    return min(results, key=lambda r: r["cost"])
