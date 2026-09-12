from eval_harness import HELD_OUT_SPLIT_RANDOM_STATE, run


def test_run_produces_report_with_all_expected_keys():
    report = run()

    assert "metrics" in report
    assert "precision" in report["metrics"]
    assert "recall" in report["metrics"]
    assert "f1" in report["metrics"]
    assert "roc_auc" in report["metrics"]
    assert "pr_auc" in report["metrics"]
    assert "brier_score" in report
    assert "reliability_bins" in report
    assert "threshold_sweep" in report
    assert "best_threshold" in report
    assert report["held_out_random_state"] == HELD_OUT_SPLIT_RANDOM_STATE
    # Must differ from train_model.py's data-generation random_state (42) —
    # expresses that the held-out split is independent of that seed, i.e. a
    # genuine train/held-out split rather than a coincidental reuse of it.
    assert HELD_OUT_SPLIT_RANDOM_STATE != 42
