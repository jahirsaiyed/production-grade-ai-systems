import numpy as np

from calibration import compute_brier_score, reliability_diagram_data


def test_compute_brier_score_is_zero_for_perfect_predictions():
    y_true = np.array([0, 1])
    y_proba = np.array([0.0, 1.0])
    assert compute_brier_score(y_true, y_proba) == 0.0


def test_compute_brier_score_is_positive_for_imperfect_predictions():
    y_true = np.array([0, 1])
    y_proba = np.array([0.5, 0.5])
    assert compute_brier_score(y_true, y_proba) > 0.0


def test_reliability_diagram_data_groups_into_bins_with_expected_fields():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.05, 0.15, 0.85, 0.95])

    bins = reliability_diagram_data(y_true, y_proba, n_bins=10)

    assert len(bins) > 0
    for b in bins:
        assert "bin_low" in b
        assert "bin_high" in b
        assert "count" in b
        assert "mean_predicted" in b
        assert "actual_positive_fraction" in b
    assert sum(b["count"] for b in bins) == 4
