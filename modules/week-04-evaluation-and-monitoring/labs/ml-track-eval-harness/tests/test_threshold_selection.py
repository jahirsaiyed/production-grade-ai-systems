import numpy as np

from threshold_selection import best_threshold, sweep_thresholds


def test_sweep_thresholds_returns_one_entry_per_threshold():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.6, 0.9])

    results = sweep_thresholds(
        y_true, y_proba, cost_fp=1.0, cost_fn=5.0, thresholds=[0.3, 0.5, 0.7]
    )

    assert len(results) == 3
    assert {r["threshold"] for r in results} == {0.3, 0.5, 0.7}


def test_best_threshold_picks_the_lowest_cost_option():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.2, 0.3, 0.55, 0.6])

    result = best_threshold(y_true, y_proba, cost_fp=1.0, cost_fn=100.0)

    assert result["threshold"] <= 0.55
    assert result["cost"] == 0.0
