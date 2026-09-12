"""
Compares an eval report against a committed baseline and fails (non-zero
exit) if citation_match_rate has regressed below its minimum. Enforced
directly in tests/test_gate.py, so the normal pytest run CI already runs is
the "gate that blocks bad releases" — no separate CI workflow step needed.

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
    actual = report["metrics"].get("citation_match_rate")
    minimum = baseline.get("citation_match_rate")
    if actual is None:
        failures.append("metric 'citation_match_rate' missing from report")
    elif minimum is not None and actual < minimum:
        failures.append(
            f"citation_match_rate={actual:.2f} is below baseline minimum {minimum:.2f}"
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
