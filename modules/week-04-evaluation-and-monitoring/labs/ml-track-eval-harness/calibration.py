import numpy as np
from sklearn.metrics import brier_score_loss


def compute_brier_score(y_true, y_proba) -> float:
    return float(brier_score_loss(y_true, y_proba))


def reliability_diagram_data(y_true, y_proba, n_bins: int = 10) -> list[dict]:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_proba >= lo) & (y_proba <= hi)
        else:
            mask = (y_proba >= lo) & (y_proba < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        bins.append(
            {
                "bin_low": float(lo),
                "bin_high": float(hi),
                "count": count,
                "mean_predicted": float(y_proba[mask].mean()),
                "actual_positive_fraction": float(y_true[mask].mean()),
            }
        )
    return bins
