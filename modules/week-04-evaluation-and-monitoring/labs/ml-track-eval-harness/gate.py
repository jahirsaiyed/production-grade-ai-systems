"""
Compares an eval report against a committed baseline and fails (non-zero
exit) if any metric has regressed below its minimum. Enforced directly in
tests/test_gate.py, so the normal pytest run CI already executes is the
"gate that blocks bad releases" — no separate CI workflow step needed.

Run with `make gate` (after `make eval`) or `python gate.py`.
"""
import json
from dataclasses import dataclass
from pathlib import Path

LAB_DIR = Path(__file__).parent


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: list[str]


def check_gate(report: dict, baseline: dict) -> GateResult:
    failures = []
    for metric_name, minimum in baseline.items():
        actual = report["metrics"].get(metric_name)
        if actual is None:
            failures.append(f"metric '{metric_name}' missing from report")
            continue
        if actual < minimum:
            failures.append(
                f"{metric_name}={actual:.4f} is below baseline minimum {minimum:.4f}"
            )
    return GateResult(passed=not failures, failures=failures)


def main() -> None:
    report = json.loads((LAB_DIR / "eval_report.json").read_text(encoding="utf-8"))
    baseline = json.loads(
        (LAB_DIR / "baseline_metrics.json").read_text(encoding="utf-8")
    )

    result = check_gate(report, baseline)
    if result.passed:
        print("GATE PASSED")
    else:
        print("GATE FAILED:")
        for failure in result.failures:
            print(f"  - {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
