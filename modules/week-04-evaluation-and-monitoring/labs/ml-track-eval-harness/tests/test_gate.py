import json
from pathlib import Path

from eval_harness import run
from gate import check_gate


def test_gate_passes_against_this_labs_own_baseline():
    baseline = json.loads(
        (Path(__file__).parent.parent / "baseline_metrics.json").read_text()
    )
    report = run()

    result = check_gate(report, baseline)

    assert result.passed, result.failures


def test_gate_fails_when_a_metric_is_below_baseline():
    report = {
        "metrics": {
            "f1": 0.5,
            "precision": 0.9,
            "recall": 0.9,
            "roc_auc": 0.9,
            "pr_auc": 0.9,
        }
    }
    baseline = {"f1": 0.8}

    result = check_gate(report, baseline)

    assert result.passed is False
    assert any("f1" in f for f in result.failures)
