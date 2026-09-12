import numpy as np

from metrics import compute_classification_metrics


def test_compute_classification_metrics_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.9, 0.8])

    result = compute_classification_metrics(y_true, y_pred, y_proba)

    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["roc_auc"] == 1.0
    assert result["pr_auc"] == 1.0


def test_compute_classification_metrics_imperfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    y_proba = np.array([0.2, 0.6, 0.7, 0.4])

    result = compute_classification_metrics(y_true, y_pred, y_proba)

    assert 0.0 < result["precision"] < 1.0
    assert 0.0 < result["recall"] < 1.0
    assert 0.0 < result["f1"] < 1.0
