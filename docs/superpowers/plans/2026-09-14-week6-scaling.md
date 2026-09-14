# Week 6 Scaling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Week 6's concept README/exercises, mark the root README's course map as fully
complete (all 6 weeks), and build one unified capstone lab —
`capstone-scaling-platform` — simulating a canary rollout with SLO-based auto-rollback, a cost
tradeoff report, and a tightly-scoped human-in-the-loop-gated planner/worker agent demo.

**Architecture:** A single standalone, flat-layout Python lab (no FastAPI service, no Docker,
matching Week 4's harness-style shape) at `modules/week-06-scaling/labs/capstone-scaling-platform/`.
Everything is an in-process, deterministic, seeded simulation — no real Kubernetes, no real
network calls, no real cost data — disclosed as such throughout. This is this course's only
agentic-tool-use code, introduced once and tightly bounded to a single kind of proposal (a
rollout plan).

**Tech Stack:** Python 3.11+, numpy (for seeded outcome sampling), pytest + pytest-cov, ruff.

## Global Constraints

- Python 3.11+/plain venv/pip, `pytest`+`pytest-cov` with `--cov-fail-under=80` matched EXACTLY
  between this lab's `Makefile` `test` target and `.github/workflows/tests.yml`'s actual command
  (the Week 4/5 final-review lesson: a Makefile that doesn't match CI's real command is a false
  green).
- `ruff check .` must be run and verified directly by every implementer and reviewer before
  declaring a task clean — Week 5's final code review found `ruff check` failing despite every
  prior per-task review reporting clean, because no task had actually run the literal lint command.
- Exact-version pins in `requirements.txt`.
- Commit format `<type>: <description>` with the exact trailer
  `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` copied verbatim by every implementer
  regardless of which model executes the task.
- Every mechanism that claims to "catch" or "gate" something (SLO breach detection, the HITL
  approval gate) needs a test proving it actually fails/blocks under the condition it claims to
  guard against, not just a happy-path test — the single most repeated lesson across every prior
  week's final review.
- The simulation must be explicitly and honestly labeled as a simulation everywhere a learner
  would reasonably think otherwise (README, module docstrings) — no real Kubernetes, no real
  servers, no real network calls, no real cost data. All randomness is seeded for reproducibility.
- No PII/PHI in any committed file, log sample, or test fixture.
- Working directory for all commands below is the repo root:
  `D:\Learning\Build Production Grade AI Systems _ ByteByteGo Live\production-grade-ai-systems`.
  The repo already exists on GitHub (public, `origin/master`) — no `gh repo create`, just
  `git push` at the end.

---

### Task 1: Week 6 concept README, exercises, and course-map status update (course completion)

**Files:**
- Modify: `modules/week-06-scaling/README.md` (currently a "coming soon" stub)
- Create: `modules/week-06-scaling/exercises.md`
- Modify: `README.md` (repo root) — course-map row for Week 6
- Modify: `docs/course-outline.md` — Week 6 status line, plus one sentence noting course completion

**Interfaces:**
- Produces: a link to `labs/capstone-scaling-platform/README.md` (built in later tasks — a known
  forward reference, the same pattern used in every previous week's first task).

- [ ] **Step 1: Write `modules/week-06-scaling/README.md`**

Replace the stub with a concept README covering each topic below as its own `##` section, in this
order. For each: a 2-4 sentence plain-language explanation, why it matters in production, and (for
topics with a hands-on component) a pointer into `labs/capstone-scaling-platform/`. For
conceptual-only topics, say explicitly that no lab demonstrates them and briefly why (this course
has never had real cloud infrastructure to run against, the same constraint that made Week 4/5's
TLS/vector-store/regulatory sections conceptual-only).

1. **Kubernetes (EKS, GKE, AKS) and ECS; managed ML platforms (Vertex AI, SageMaker, Azure ML)** —
   conceptual only; name each platform and its role, no lab.
2. **Workflow orchestration (Airflow, Prefect, Dagster, Kubeflow, Ray)** — conceptual only; briefly
   distinguish general-purpose workflow orchestrators from ML-specific ones.
3. **GPU scheduling, node pools, autoscaling, capacity planning** — conceptual only.
4. **Canary and blue-green rollouts; auto-rollback gates** — hands-on; point at
   `labs/capstone-scaling-platform/canary.py` and `slo.py`; explain the distinction between canary
   (gradual traffic ramp) and blue-green (instant full cutover) even though the lab only
   demonstrates canary.
5. **SLOs (latency, error rate, groundedness)** — hands-on; point at `slo.py`; explicitly connect
   "groundedness" back to Week 5's `groundedness.py` as the same kind of signal used differently
   here — a release gate that can block a rollout, instead of a per-response score surfaced to a
   caller.
6. **Incident response and on-call basics** — conceptual only.
7. **Cost tradeoffs** — hands-on; point at `cost_report.py`; explain the core idea (a canary
   rollout that catches a regression early costs less than either extreme of "all traffic on the
   new version" or "all traffic on the old version," if the new version happens to be more
   expensive).
8. **Planner/worker patterns** — hands-on; point at `agent/planner.py` + `agent/worker.py`.
9. **HITL approval gates** — hands-on; point at `agent/hitl_gate.py`. State plainly that this is
   the course's ONLY agentic-tool-use code, introduced once, deliberately, in this final week —
   explain that every prior week (most recently Week 5, under OWASP's "excessive agency" item)
   explicitly kept agentic tool-use out of scope, and this lab is a one-time, tightly-bounded
   exception (one planner proposing exactly one kind of action) to make the syllabus's own
   "human-supervised agent" demo real rather than conceptual — not a general agent framework.
10. **Agent tracing** — hands-on; point at `agent/tracing.py`.
11. **When not to use agents; case studies (recommendation and search, fintech risk scoring,
    consumer assistants)** — conceptual only; 2-3 sentences on "when not to use agents" (the
    HITL-gate pattern this lab demonstrates is itself an answer: use a human gate rather than
    autonomous action whenever the cost of a wrong action is high and reversibility is low); then
    2-3 sentences per case study, each connecting back to this course's own ML/LLM tracks as a
    smaller-scale worked example of the same underlying problem.

Open the file with a `# Week 6: Scaling` heading and a short intro naming the unified capstone lab
(explain briefly why Week 6 breaks from the ML-track/LLM-track split every prior week used — the
syllabus's own "live demo" already describes one integrated scenario, not two separate ones) and
linking to `labs/capstone-scaling-platform/README.md`. Close with a "## Hands-on lab" section
linking it, and an "## Exercises" section linking `exercises.md`.

- [ ] **Step 2: Write `modules/week-06-scaling/exercises.md`**

```markdown
# Week 6 Exercises

Hints and acceptance criteria only — no answer key. If you get stuck, the reference implementation
is the lab's own code.

## Exercise 1: Make the regression sneak past a smaller canary step

`run_demo.py`'s regressed scenario is caught at the very first 10% traffic step. Lower
`run_demo.py`'s `N_REQUESTS_PER_STEP` constant to a small number like 20, then re-run `make demo`.
**Acceptance criteria:** you can explain, in your own words, why a smaller sample size per step
makes it statistically easier for a real regression to slip through a step without tripping the
SLO thresholds (hint: p95 latency and error rate are both less stable estimates with fewer
samples).

## Exercise 2: Change the cost assumption and watch the savings number move

`run_demo.py` assumes the regressed candidate costs `0.03` per request against `stable`'s `0.01`.
Change the candidate's cost to `0.10` (e.g. a much larger, more expensive model) and re-run
`make demo`. **Acceptance criteria:** state how `savings_vs_full_candidate` changed for the
auto-rolled-back scenario, and explain why an early rollback matters more, in dollar terms, the
more expensive the candidate is.

## Exercise 3: Deny the plan and watch the worker do nothing

In `run_demo.py`'s `run_scenario()` function, change one scenario's `require_approval(plan,
approve=True, ...)` call to `approve=False`, then re-run `make demo`. **Acceptance criteria:** the
printed output shows "Worker took no action" for that scenario, and you can point to the exact
line in `agent/worker.py` that prevents the rollout from ever starting.

## Exercise 4: Read the full trace

After running `make demo`, open the generated `demo_trace.log` file (one JSON line per agent
step). **Acceptance criteria:** you can identify, from the trace alone (without re-reading the
console output), which of the two scenarios rolled back and which succeeded.

## Exercise 5: Add a fourth SLO

`slo.py`'s `SloThresholds` currently checks p95 latency, error rate, and mean quality score. Add a
new threshold (e.g. `max_mean_latency_ms`, checking the average rather than the p95) to
`SloThresholds` and `check_slo_breach()`, add a test proving it can genuinely trigger a violation,
then re-run `make test`. **Acceptance criteria:** your new test fails before your
`check_slo_breach` change and passes after — proving your addition genuinely changes behavior, not
just adds an unused field.
```

- [ ] **Step 3: Update the root `README.md`'s course map**

Read the current file, find the course-map table's Week 6 row (currently
`| 6 | [Scaling](modules/week-06-scaling/README.md) | Coming soon |`), and change `Coming soon` to
`Fully built` — matching how the Week 1-5 rows already read. Do not touch any other row.

- [ ] **Step 4: Update `docs/course-outline.md`'s Week 6 status line and add a completion note**

Read the current file. Find the line under the Week 6 section reading
`**Status in this repo: skeleton only — see modules/week-06-scaling/.**`, and change
`skeleton only` to `fully built` — matching how the Week 1-5 sections' equivalent lines already
read. Do not touch any other section's status line.

Additionally, find the document's introductory paragraph near the top of the file (before the
Week 1 section) and add one sentence noting that all 6 weeks now ship with hands-on labs in this
repo. Match the existing paragraph's tone and phrasing style — read it first, then write a sentence
that fits naturally, rather than a generic templated announcement.

- [ ] **Step 5: Commit**

```bash
git add modules/week-06-scaling/README.md modules/week-06-scaling/exercises.md README.md docs/course-outline.md
git commit -m "$(cat <<'EOF'
docs: add Week 6 concept README/exercises and mark the course fully built

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Capstone lab skeleton — directory structure, dependencies, tooling

**Files:**
- Create: `modules/week-06-scaling/labs/capstone-scaling-platform/requirements.txt`
- Create: `.../capstone-scaling-platform/.gitignore`
- Create: `.../capstone-scaling-platform/.coveragerc`

**Interfaces:**
- Produces: a working venv with `numpy`/`pytest`/`pytest-cov`/`ruff` installed. Consumed by
  Tasks 3-11.

- [ ] **Step 1: Create the directory structure**

```bash
LAB="modules/week-06-scaling/labs/capstone-scaling-platform"
mkdir -p "$LAB/agent" "$LAB/tests"
touch "$LAB/agent/__init__.py" "$LAB/tests/__init__.py"
```

- [ ] **Step 2: Write `requirements.txt`**

```
numpy==2.1.3
pytest==8.3.4
pytest-cov==6.0.0
ruff==0.8.2
```

- [ ] **Step 3: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.coverage
demo_trace.log
```

- [ ] **Step 4: Write `.coveragerc`**

```ini
[run]
omit =
    .venv/*
    tests/*
```

- [ ] **Step 5: Create a fresh venv and install dependencies**

```bash
cd "modules/week-06-scaling/labs/capstone-scaling-platform"
python -m venv .venv
```
Windows: `.venv\Scripts\activate` — macOS/Linux: `source .venv/bin/activate`, then:
```bash
pip install -r requirements.txt
```

- [ ] **Step 6: Commit**

```bash
git add modules/week-06-scaling/labs/capstone-scaling-platform
git commit -m "$(cat <<'EOF'
feat: add capstone lab skeleton and tooling

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Deployment simulator (TDD)

**Files:**
- Create: `.../capstone-scaling-platform/deployment_simulator.py`
- Test: `.../capstone-scaling-platform/tests/test_deployment_simulator.py`

**Interfaces:**
- Produces: `RequestOutcome(latency_ms: float, succeeded: bool, quality_score: float)` (frozen
  dataclass), `ServiceVersion(name: str, mean_latency_ms: float, latency_stddev_ms: float,
  error_rate: float, mean_quality_score: float, quality_stddev: float)` (frozen dataclass) with a
  `.sample(n: int, rng: np.random.Generator) -> list[RequestOutcome]` method. Consumed by
  Tasks 4, 5, 9, 11.

All commands run from
`modules/week-06-scaling/labs/capstone-scaling-platform/`.

- [ ] **Step 1: Write the failing test**

`tests/test_deployment_simulator.py`:
```python
import numpy as np

from deployment_simulator import RequestOutcome, ServiceVersion


def test_sample_is_deterministic_given_the_same_seed():
    version = ServiceVersion(
        name="stable", mean_latency_ms=100.0, latency_stddev_ms=10.0,
        error_rate=0.05, mean_quality_score=0.9, quality_stddev=0.05,
    )
    outcomes_a = version.sample(50, np.random.default_rng(42))
    outcomes_b = version.sample(50, np.random.default_rng(42))

    assert outcomes_a == outcomes_b


def test_sample_returns_the_requested_count_of_outcomes():
    version = ServiceVersion(
        name="stable", mean_latency_ms=100.0, latency_stddev_ms=10.0,
        error_rate=0.05, mean_quality_score=0.9, quality_stddev=0.05,
    )
    outcomes = version.sample(200, np.random.default_rng(1))

    assert len(outcomes) == 200
    assert all(isinstance(o, RequestOutcome) for o in outcomes)


def test_sample_error_rate_is_approximately_correct_over_a_large_sample():
    version = ServiceVersion(
        name="flaky", mean_latency_ms=100.0, latency_stddev_ms=10.0,
        error_rate=0.3, mean_quality_score=0.9, quality_stddev=0.05,
    )
    outcomes = version.sample(5000, np.random.default_rng(7))
    observed_error_rate = sum(1 for o in outcomes if not o.succeeded) / len(outcomes)

    assert 0.25 < observed_error_rate < 0.35


def test_sample_quality_scores_are_clipped_to_valid_range():
    version = ServiceVersion(
        name="extreme", mean_latency_ms=100.0, latency_stddev_ms=10.0,
        error_rate=0.0, mean_quality_score=0.98, quality_stddev=0.5,
    )
    outcomes = version.sample(2000, np.random.default_rng(3))

    assert all(0.0 <= o.quality_score <= 1.0 for o in outcomes)


def test_sample_latencies_are_always_positive():
    version = ServiceVersion(
        name="fast", mean_latency_ms=5.0, latency_stddev_ms=10.0,
        error_rate=0.0, mean_quality_score=0.9, quality_stddev=0.05,
    )
    outcomes = version.sample(2000, np.random.default_rng(9))

    assert all(o.latency_ms >= 1.0 for o in outcomes)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_deployment_simulator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'deployment_simulator'`.

- [ ] **Step 3: Write minimal implementation**

`deployment_simulator.py`:
```python
"""
Simulates two service "versions" — stable (the incumbent) and candidate (what's
being rolled out) — as deterministic, seeded generators of per-request outcomes.
No real network calls, no real Kubernetes, no real service — a Python-only
stand-in so canary/SLO/rollback mechanics can be taught and tested without
cloud infrastructure this course doesn't have.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RequestOutcome:
    latency_ms: float
    succeeded: bool
    quality_score: float


@dataclass(frozen=True)
class ServiceVersion:
    name: str
    mean_latency_ms: float
    latency_stddev_ms: float
    error_rate: float
    mean_quality_score: float
    quality_stddev: float

    def sample(self, n: int, rng: np.random.Generator) -> list["RequestOutcome"]:
        if n == 0:
            return []

        latencies = rng.normal(self.mean_latency_ms, self.latency_stddev_ms, size=n)
        latencies = np.clip(latencies, a_min=1.0, a_max=None)
        failures = rng.random(n) < self.error_rate
        quality_scores = rng.normal(self.mean_quality_score, self.quality_stddev, size=n)
        quality_scores = np.clip(quality_scores, 0.0, 1.0)

        return [
            RequestOutcome(
                latency_ms=float(latencies[i]),
                succeeded=not bool(failures[i]),
                quality_score=float(quality_scores[i]),
            )
            for i in range(n)
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_deployment_simulator.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add modules/week-06-scaling/labs/capstone-scaling-platform/deployment_simulator.py modules/week-06-scaling/labs/capstone-scaling-platform/tests/test_deployment_simulator.py
git commit -m "$(cat <<'EOF'
feat: add deployment simulator for stable/candidate service versions

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: SLO thresholds and breach detection (TDD)

**Files:**
- Create: `.../capstone-scaling-platform/slo.py`
- Test: `.../capstone-scaling-platform/tests/test_slo.py`

**Interfaces:**
- Consumes: `RequestOutcome` (Task 3).
- Produces: `SloThresholds(max_p95_latency_ms: float, max_error_rate: float, min_quality_score:
  float)` (frozen dataclass), `check_slo_breach(outcomes: list[RequestOutcome], thresholds:
  SloThresholds) -> list[str]` (empty list = healthy). Consumed by Tasks 5, 9, 11.

All commands run from
`modules/week-06-scaling/labs/capstone-scaling-platform/`.

- [ ] **Step 1: Write the failing test**

`tests/test_slo.py`:
```python
from deployment_simulator import RequestOutcome
from slo import SloThresholds, check_slo_breach


def test_check_slo_breach_returns_empty_list_for_healthy_outcomes():
    outcomes = [
        RequestOutcome(latency_ms=50.0, succeeded=True, quality_score=0.95)
        for _ in range(100)
    ]
    thresholds = SloThresholds(max_p95_latency_ms=200.0, max_error_rate=0.1, min_quality_score=0.8)

    assert check_slo_breach(outcomes, thresholds) == []


def test_check_slo_breach_flags_high_latency():
    outcomes = [
        RequestOutcome(latency_ms=500.0, succeeded=True, quality_score=0.95)
        for _ in range(100)
    ]
    thresholds = SloThresholds(max_p95_latency_ms=200.0, max_error_rate=0.1, min_quality_score=0.8)

    violations = check_slo_breach(outcomes, thresholds)

    assert len(violations) == 1
    assert "latency" in violations[0]


def test_check_slo_breach_flags_high_error_rate():
    outcomes = [
        RequestOutcome(latency_ms=50.0, succeeded=(i % 2 == 0), quality_score=0.95)
        for i in range(100)
    ]
    thresholds = SloThresholds(max_p95_latency_ms=200.0, max_error_rate=0.1, min_quality_score=0.8)

    violations = check_slo_breach(outcomes, thresholds)

    assert any("error rate" in v for v in violations)


def test_check_slo_breach_flags_low_quality_score():
    outcomes = [
        RequestOutcome(latency_ms=50.0, succeeded=True, quality_score=0.2)
        for _ in range(100)
    ]
    thresholds = SloThresholds(max_p95_latency_ms=200.0, max_error_rate=0.1, min_quality_score=0.8)

    violations = check_slo_breach(outcomes, thresholds)

    assert any("quality" in v for v in violations)


def test_check_slo_breach_can_flag_multiple_violations_at_once():
    outcomes = [
        RequestOutcome(latency_ms=500.0, succeeded=False, quality_score=0.1)
        for _ in range(100)
    ]
    thresholds = SloThresholds(max_p95_latency_ms=200.0, max_error_rate=0.1, min_quality_score=0.8)

    violations = check_slo_breach(outcomes, thresholds)

    assert len(violations) == 3


def test_check_slo_breach_returns_empty_list_for_no_outcomes():
    thresholds = SloThresholds(max_p95_latency_ms=200.0, max_error_rate=0.1, min_quality_score=0.8)

    assert check_slo_breach([], thresholds) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_slo.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'slo'`.

- [ ] **Step 3: Write minimal implementation**

`slo.py`:
```python
"""
SLO threshold definitions and breach detection against a sample of
RequestOutcomes.
"""
from dataclasses import dataclass

import numpy as np

from deployment_simulator import RequestOutcome


@dataclass(frozen=True)
class SloThresholds:
    max_p95_latency_ms: float
    max_error_rate: float
    min_quality_score: float


def check_slo_breach(
    outcomes: list[RequestOutcome], thresholds: SloThresholds
) -> list[str]:
    if not outcomes:
        return []

    latencies = [o.latency_ms for o in outcomes]
    p95_latency = float(np.percentile(latencies, 95))
    error_rate = sum(1 for o in outcomes if not o.succeeded) / len(outcomes)
    mean_quality = sum(o.quality_score for o in outcomes) / len(outcomes)

    violations = []
    if p95_latency > thresholds.max_p95_latency_ms:
        violations.append(
            f"p95 latency {p95_latency:.1f}ms exceeds threshold "
            f"{thresholds.max_p95_latency_ms:.1f}ms"
        )
    if error_rate > thresholds.max_error_rate:
        violations.append(
            f"error rate {error_rate:.2%} exceeds threshold {thresholds.max_error_rate:.2%}"
        )
    if mean_quality < thresholds.min_quality_score:
        violations.append(
            f"mean quality score {mean_quality:.3f} is below threshold "
            f"{thresholds.min_quality_score:.3f}"
        )
    return violations
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_slo.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add modules/week-06-scaling/labs/capstone-scaling-platform/slo.py modules/week-06-scaling/labs/capstone-scaling-platform/tests/test_slo.py
git commit -m "$(cat <<'EOF'
feat: add SLO thresholds and breach detection

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Canary rollout with automatic rollback (TDD)

**Files:**
- Create: `.../capstone-scaling-platform/canary.py`
- Test: `.../capstone-scaling-platform/tests/test_canary.py`

**Interfaces:**
- Consumes: `ServiceVersion` (Task 3); `SloThresholds`, `check_slo_breach` (Task 4).
- Produces: `DEFAULT_RAMP_STEPS: list[int]`, `RolloutStep(traffic_pct: int, violations:
  list[str], rolled_back: bool)` (frozen dataclass), `RolloutResult(steps: list[RolloutStep],
  rolled_back: bool, final_traffic_pct: int)` (frozen dataclass),
  `run_canary_rollout(stable: ServiceVersion, candidate: ServiceVersion, thresholds:
  SloThresholds, n_requests_per_step: int = 200, ramp_steps: list[int] | None = None, seed: int =
  0) -> RolloutResult`. Consumed by Tasks 6, 9, 11.

All commands run from
`modules/week-06-scaling/labs/capstone-scaling-platform/`.

- [ ] **Step 1: Write the failing test**

`tests/test_canary.py`:
```python
from canary import DEFAULT_RAMP_STEPS, run_canary_rollout
from deployment_simulator import ServiceVersion
from slo import SloThresholds


def _thresholds():
    return SloThresholds(max_p95_latency_ms=200.0, max_error_rate=0.1, min_quality_score=0.7)


def test_healthy_candidate_completes_the_full_rollout():
    stable = ServiceVersion("stable", 80.0, 10.0, 0.02, 0.9, 0.05)
    candidate = ServiceVersion("candidate", 85.0, 10.0, 0.02, 0.88, 0.05)

    result = run_canary_rollout(
        stable, candidate, _thresholds(), n_requests_per_step=300, seed=1
    )

    assert result.rolled_back is False
    assert result.final_traffic_pct == 100
    assert result.steps[-1].traffic_pct == 100
    assert all(not s.rolled_back for s in result.steps)


def test_regressed_candidate_triggers_a_genuine_rollback():
    stable = ServiceVersion("stable", 80.0, 10.0, 0.02, 0.9, 0.05)
    # Deliberately regressed: much higher latency and error rate than the SLO allows.
    candidate = ServiceVersion("candidate", 400.0, 30.0, 0.4, 0.9, 0.05)

    result = run_canary_rollout(
        stable, candidate, _thresholds(), n_requests_per_step=300, seed=1
    )

    assert result.rolled_back is True
    assert result.final_traffic_pct == 0
    assert result.steps[-1].rolled_back is True
    assert len(result.steps) < len(DEFAULT_RAMP_STEPS)
    assert all(not s.rolled_back for s in result.steps[:-1])


def test_rollout_respects_custom_ramp_steps():
    stable = ServiceVersion("stable", 80.0, 10.0, 0.02, 0.9, 0.05)
    candidate = ServiceVersion("candidate", 82.0, 10.0, 0.02, 0.89, 0.05)

    result = run_canary_rollout(
        stable, candidate, _thresholds(), n_requests_per_step=300,
        ramp_steps=[50, 100], seed=2,
    )

    assert [s.traffic_pct for s in result.steps] == [50, 100]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_canary.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'canary'`.

- [ ] **Step 3: Write minimal implementation**

`canary.py`:
```python
"""
Simulates a canary rollout: traffic to `candidate` ramps up in steps, checking
SLOs after each step. SLOs are checked against the CANDIDATE's own traffic
only, not the blended total — a regression under a small traffic share would
otherwise get diluted by the much larger stable-traffic sample and might
never trip the threshold. The moment a step breaches SLOs, the rollout stops
and rolls back: all further traffic goes to `stable`, and `candidate`'s
traffic share returns to 0%.
"""
from dataclasses import dataclass

import numpy as np

from deployment_simulator import ServiceVersion
from slo import SloThresholds, check_slo_breach

DEFAULT_RAMP_STEPS = [10, 25, 50, 100]


@dataclass(frozen=True)
class RolloutStep:
    traffic_pct: int
    violations: list[str]
    rolled_back: bool


@dataclass(frozen=True)
class RolloutResult:
    steps: list[RolloutStep]
    rolled_back: bool
    final_traffic_pct: int


def run_canary_rollout(
    stable: ServiceVersion,
    candidate: ServiceVersion,
    thresholds: SloThresholds,
    n_requests_per_step: int = 200,
    ramp_steps: list[int] | None = None,
    seed: int = 0,
) -> RolloutResult:
    ramp_steps = ramp_steps if ramp_steps is not None else DEFAULT_RAMP_STEPS
    rng = np.random.default_rng(seed)

    steps: list[RolloutStep] = []
    for traffic_pct in ramp_steps:
        n_candidate = round(n_requests_per_step * traffic_pct / 100)
        n_stable = n_requests_per_step - n_candidate

        candidate_outcomes = candidate.sample(n_candidate, rng)
        stable.sample(n_stable, rng)  # exercised for realism; not used in the SLO check

        violations = check_slo_breach(candidate_outcomes, thresholds)
        rolled_back = bool(violations)

        steps.append(
            RolloutStep(
                traffic_pct=traffic_pct, violations=violations, rolled_back=rolled_back
            )
        )

        if rolled_back:
            return RolloutResult(steps=steps, rolled_back=True, final_traffic_pct=0)

    return RolloutResult(steps=steps, rolled_back=False, final_traffic_pct=ramp_steps[-1])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_canary.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add modules/week-06-scaling/labs/capstone-scaling-platform/canary.py modules/week-06-scaling/labs/capstone-scaling-platform/tests/test_canary.py
git commit -m "$(cat <<'EOF'
feat: add canary rollout with SLO-based automatic rollback

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Cost tradeoff report (TDD)

**Files:**
- Create: `.../capstone-scaling-platform/cost_report.py`
- Test: `.../capstone-scaling-platform/tests/test_cost_report.py`

**Interfaces:**
- Consumes: `RolloutResult`, `RolloutStep` (Task 5).
- Produces: `CostReport(full_stable_cost: float, full_candidate_cost: float,
  actual_canary_cost: float, savings_vs_full_candidate: float)` (frozen dataclass),
  `compute_cost_report(rollout_result: RolloutResult, n_requests_per_step: int,
  stable_cost_per_request: float, candidate_cost_per_request: float) -> CostReport`. Consumed by
  Task 11.

All commands run from
`modules/week-06-scaling/labs/capstone-scaling-platform/`.

- [ ] **Step 1: Write the failing test**

`tests/test_cost_report.py`:
```python
from canary import RolloutResult, RolloutStep
from cost_report import compute_cost_report


def test_compute_cost_report_when_rollback_happens_at_the_first_step():
    rollout_result = RolloutResult(
        steps=[RolloutStep(traffic_pct=10, violations=["x"], rolled_back=True)],
        rolled_back=True,
        final_traffic_pct=0,
    )

    report = compute_cost_report(
        rollout_result, n_requests_per_step=100,
        stable_cost_per_request=0.01, candidate_cost_per_request=0.05,
    )

    expected_actual = 90 * 0.01 + 10 * 0.05
    assert report.actual_canary_cost == expected_actual
    assert report.full_candidate_cost == 100 * 0.05
    assert report.full_stable_cost == 100 * 0.01
    assert report.savings_vs_full_candidate == report.full_candidate_cost - expected_actual


def test_compute_cost_report_for_a_full_successful_rollout():
    rollout_result = RolloutResult(
        steps=[
            RolloutStep(traffic_pct=10, violations=[], rolled_back=False),
            RolloutStep(traffic_pct=100, violations=[], rolled_back=False),
        ],
        rolled_back=False,
        final_traffic_pct=100,
    )

    report = compute_cost_report(
        rollout_result, n_requests_per_step=100,
        stable_cost_per_request=0.01, candidate_cost_per_request=0.05,
    )

    assert report.full_candidate_cost == 200 * 0.05
    expected_actual = (90 * 0.01 + 10 * 0.05) + (0 * 0.01 + 100 * 0.05)
    assert report.actual_canary_cost == expected_actual
    assert report.savings_vs_full_candidate == report.full_candidate_cost - expected_actual


def test_compute_cost_report_savings_is_positive_when_candidate_is_more_expensive_and_rollback_happens():
    rollout_result = RolloutResult(
        steps=[RolloutStep(traffic_pct=10, violations=["x"], rolled_back=True)],
        rolled_back=True,
        final_traffic_pct=0,
    )

    report = compute_cost_report(
        rollout_result, n_requests_per_step=1000,
        stable_cost_per_request=0.01, candidate_cost_per_request=0.10,
    )

    assert report.savings_vs_full_candidate > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cost_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cost_report'`.

- [ ] **Step 3: Write minimal implementation**

`cost_report.py`:
```python
"""
Compares three cost scenarios for a canary rollout: fully on stable, fully on
candidate, and the actual cost incurred by the canary's real traffic split
(cheaper than either extreme when a rollback happens early and the candidate
is the more expensive version).
"""
from dataclasses import dataclass

from canary import RolloutResult


@dataclass(frozen=True)
class CostReport:
    full_stable_cost: float
    full_candidate_cost: float
    actual_canary_cost: float
    savings_vs_full_candidate: float


def compute_cost_report(
    rollout_result: RolloutResult,
    n_requests_per_step: int,
    stable_cost_per_request: float,
    candidate_cost_per_request: float,
) -> CostReport:
    total_requests = n_requests_per_step * len(rollout_result.steps)

    full_stable_cost = total_requests * stable_cost_per_request
    full_candidate_cost = total_requests * candidate_cost_per_request

    actual_cost = 0.0
    for step in rollout_result.steps:
        n_candidate = round(n_requests_per_step * step.traffic_pct / 100)
        n_stable = n_requests_per_step - n_candidate
        actual_cost += n_candidate * candidate_cost_per_request
        actual_cost += n_stable * stable_cost_per_request

    return CostReport(
        full_stable_cost=full_stable_cost,
        full_candidate_cost=full_candidate_cost,
        actual_canary_cost=actual_cost,
        savings_vs_full_candidate=full_candidate_cost - actual_cost,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cost_report.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add modules/week-06-scaling/labs/capstone-scaling-platform/cost_report.py modules/week-06-scaling/labs/capstone-scaling-platform/tests/test_cost_report.py
git commit -m "$(cat <<'EOF'
feat: add cost tradeoff report

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Agent planner (TDD)

**Files:**
- Create: `.../capstone-scaling-platform/agent/planner.py`
- Test: `.../capstone-scaling-platform/tests/test_planner.py`

**Interfaces:**
- Produces: `RolloutPlan(candidate_name: str, target_traffic_pct: int, reason: str)` (frozen
  dataclass), `propose_rollout_plan(candidate_name: str, target_traffic_pct: int) -> RolloutPlan`
  (raises `ValueError` if `target_traffic_pct` is not in `(0, 100]`). Consumed by Tasks 8, 9, 11.

All commands run from
`modules/week-06-scaling/labs/capstone-scaling-platform/`.

- [ ] **Step 1: Write the failing test**

`tests/test_planner.py`:
```python
import pytest

from agent.planner import RolloutPlan, propose_rollout_plan


def test_propose_rollout_plan_returns_expected_fields():
    plan = propose_rollout_plan("candidate-v2", 50)

    assert isinstance(plan, RolloutPlan)
    assert plan.candidate_name == "candidate-v2"
    assert plan.target_traffic_pct == 50
    assert "candidate-v2" in plan.reason


def test_propose_rollout_plan_rejects_zero_traffic_percentage():
    with pytest.raises(ValueError):
        propose_rollout_plan("candidate-v2", 0)


def test_propose_rollout_plan_rejects_traffic_percentage_over_100():
    with pytest.raises(ValueError):
        propose_rollout_plan("candidate-v2", 101)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_planner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.planner'`.

- [ ] **Step 3: Write minimal implementation**

`agent/planner.py`:
```python
"""
A scripted (not LLM-backed) planner that proposes a rollout action. Kept
deliberately simple and inspectable — this course's only agentic-tool-use
code, scoped to exactly one kind of proposal.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RolloutPlan:
    candidate_name: str
    target_traffic_pct: int
    reason: str


def propose_rollout_plan(candidate_name: str, target_traffic_pct: int) -> RolloutPlan:
    if not 0 < target_traffic_pct <= 100:
        raise ValueError("target_traffic_pct must be between 1 and 100")

    return RolloutPlan(
        candidate_name=candidate_name,
        target_traffic_pct=target_traffic_pct,
        reason=(
            f"Roll out '{candidate_name}' up to {target_traffic_pct}% of traffic, "
            "monitoring SLOs at each ramp step and rolling back automatically on breach."
        ),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_planner.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add modules/week-06-scaling/labs/capstone-scaling-platform/agent/planner.py modules/week-06-scaling/labs/capstone-scaling-platform/tests/test_planner.py
git commit -m "$(cat <<'EOF'
feat: add agent planner that proposes rollout plans

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Human-in-the-loop approval gate (TDD)

**Files:**
- Create: `.../capstone-scaling-platform/agent/hitl_gate.py`
- Test: `.../capstone-scaling-platform/tests/test_hitl_gate.py`

**Interfaces:**
- Consumes: `RolloutPlan` (Task 7).
- Produces: `ApprovalDecision(approved: bool, reason: str)` (frozen dataclass),
  `require_approval(plan: RolloutPlan, approve: bool, reason: str = "") -> ApprovalDecision`.
  Consumed by Tasks 9, 11.

All commands run from
`modules/week-06-scaling/labs/capstone-scaling-platform/`.

- [ ] **Step 1: Write the failing test**

`tests/test_hitl_gate.py`:
```python
from agent.hitl_gate import ApprovalDecision, require_approval
from agent.planner import propose_rollout_plan


def test_require_approval_returns_approved_decision_when_approved():
    plan = propose_rollout_plan("candidate-v2", 50)

    decision = require_approval(plan, approve=True)

    assert isinstance(decision, ApprovalDecision)
    assert decision.approved is True


def test_require_approval_returns_denied_decision_when_not_approved():
    plan = propose_rollout_plan("candidate-v2", 50)

    decision = require_approval(plan, approve=False, reason="not enough baking time")

    assert decision.approved is False
    assert "baking" in decision.reason


def test_require_approval_fills_in_a_default_reason_when_none_given():
    plan = propose_rollout_plan("candidate-v2", 50)

    approved = require_approval(plan, approve=True)
    denied = require_approval(plan, approve=False)

    assert approved.reason != ""
    assert denied.reason != ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hitl_gate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.hitl_gate'`.

- [ ] **Step 3: Write minimal implementation**

`agent/hitl_gate.py`:
```python
"""
A human-in-the-loop approval gate. In a real system this would prompt a human
operator; here it takes an explicit `approve` argument so a caller (the demo
script, or a test) plays the human's role deterministically.
"""
from dataclasses import dataclass

from agent.planner import RolloutPlan


@dataclass(frozen=True)
class ApprovalDecision:
    approved: bool
    reason: str


def require_approval(
    plan: RolloutPlan, approve: bool, reason: str = ""
) -> ApprovalDecision:
    if approve:
        return ApprovalDecision(approved=True, reason=reason or "approved by operator")
    return ApprovalDecision(approved=False, reason=reason or "denied by operator")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hitl_gate.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add modules/week-06-scaling/labs/capstone-scaling-platform/agent/hitl_gate.py modules/week-06-scaling/labs/capstone-scaling-platform/tests/test_hitl_gate.py
git commit -m "$(cat <<'EOF'
feat: add human-in-the-loop approval gate

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Agent worker (TDD)

**Files:**
- Create: `.../capstone-scaling-platform/agent/worker.py`
- Test: `.../capstone-scaling-platform/tests/test_worker.py`

**Interfaces:**
- Consumes: `RolloutPlan` (Task 7); `ApprovalDecision` (Task 8); `ServiceVersion` (Task 3);
  `SloThresholds` (Task 4); `RolloutResult`, `run_canary_rollout` (Task 5).
- Produces: `execute_plan(plan: RolloutPlan, decision: ApprovalDecision, stable: ServiceVersion,
  candidate: ServiceVersion, thresholds: SloThresholds, n_requests_per_step: int = 200,
  seed: int = 0) -> RolloutResult | None` (returns `None` without calling `run_canary_rollout` at
  all if `decision.approved` is `False`). Consumed by Task 11.

All commands run from
`modules/week-06-scaling/labs/capstone-scaling-platform/`.

- [ ] **Step 1: Write the failing test**

`tests/test_worker.py`:
```python
from unittest.mock import patch

from agent.hitl_gate import require_approval
from agent.planner import propose_rollout_plan
from agent.worker import execute_plan
from deployment_simulator import ServiceVersion
from slo import SloThresholds


def _thresholds():
    return SloThresholds(max_p95_latency_ms=200.0, max_error_rate=0.1, min_quality_score=0.7)


def test_execute_plan_returns_none_and_never_calls_canary_when_not_approved():
    plan = propose_rollout_plan("candidate-v2", 50)
    decision = require_approval(plan, approve=False)
    stable = ServiceVersion("stable", 80.0, 10.0, 0.02, 0.9, 0.05)
    candidate = ServiceVersion("candidate", 85.0, 10.0, 0.02, 0.88, 0.05)

    with patch("agent.worker.run_canary_rollout") as mock_rollout:
        result = execute_plan(plan, decision, stable, candidate, _thresholds())

    assert result is None
    mock_rollout.assert_not_called()


def test_execute_plan_runs_the_rollout_when_approved():
    plan = propose_rollout_plan("candidate-v2", 100)
    decision = require_approval(plan, approve=True)
    stable = ServiceVersion("stable", 80.0, 10.0, 0.02, 0.9, 0.05)
    candidate = ServiceVersion("candidate", 85.0, 10.0, 0.02, 0.88, 0.05)

    result = execute_plan(plan, decision, stable, candidate, _thresholds(), seed=1)

    assert result is not None
    assert result.final_traffic_pct == 100


def test_execute_plan_stops_the_ramp_at_the_plans_target_traffic_pct():
    plan = propose_rollout_plan("candidate-v2", 25)
    decision = require_approval(plan, approve=True)
    stable = ServiceVersion("stable", 80.0, 10.0, 0.02, 0.9, 0.05)
    candidate = ServiceVersion("candidate", 82.0, 10.0, 0.02, 0.89, 0.05)

    result = execute_plan(plan, decision, stable, candidate, _thresholds(), seed=1)

    assert result is not None
    assert result.steps[-1].traffic_pct == 25
    assert max(s.traffic_pct for s in result.steps) == 25
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_worker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.worker'`.

- [ ] **Step 3: Write minimal implementation**

`agent/worker.py`:
```python
"""
Executes an approved rollout plan by running the canary rollout mechanics.
Structurally cannot act on an unapproved plan — the check happens before any
call into canary.run_canary_rollout, not as a policy a caller could forget.
"""
from agent.hitl_gate import ApprovalDecision
from agent.planner import RolloutPlan
from canary import RolloutResult, run_canary_rollout
from deployment_simulator import ServiceVersion
from slo import SloThresholds


def execute_plan(
    plan: RolloutPlan,
    decision: ApprovalDecision,
    stable: ServiceVersion,
    candidate: ServiceVersion,
    thresholds: SloThresholds,
    n_requests_per_step: int = 200,
    seed: int = 0,
) -> RolloutResult | None:
    if not decision.approved:
        return None

    ramp_steps = [s for s in [10, 25, 50, 100] if s <= plan.target_traffic_pct]
    if not ramp_steps or ramp_steps[-1] != plan.target_traffic_pct:
        ramp_steps.append(plan.target_traffic_pct)

    return run_canary_rollout(
        stable,
        candidate,
        thresholds,
        n_requests_per_step=n_requests_per_step,
        ramp_steps=ramp_steps,
        seed=seed,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_worker.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add modules/week-06-scaling/labs/capstone-scaling-platform/agent/worker.py modules/week-06-scaling/labs/capstone-scaling-platform/tests/test_worker.py
git commit -m "$(cat <<'EOF'
feat: add agent worker that only executes approved plans

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Agent trace logging (TDD)

**Files:**
- Create: `.../capstone-scaling-platform/agent/tracing.py`
- Test: `.../capstone-scaling-platform/tests/test_tracing.py`

**Interfaces:**
- Produces: `log_step(trace_path: Path, step: str, detail: dict) -> None` (appends one JSON line).
  Consumed by Task 11.

All commands run from
`modules/week-06-scaling/labs/capstone-scaling-platform/`.

- [ ] **Step 1: Write the failing test**

`tests/test_tracing.py`:
```python
import json

from agent.tracing import log_step


def test_log_step_appends_one_json_line_with_expected_fields(tmp_path):
    trace_path = tmp_path / "trace.log"

    log_step(trace_path, step="planner_proposed", detail={"candidate_name": "v2"})

    lines = trace_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["step"] == "planner_proposed"
    assert entry["detail"]["candidate_name"] == "v2"
    assert "timestamp" in entry


def test_log_step_appends_to_an_existing_file(tmp_path):
    trace_path = tmp_path / "trace.log"

    log_step(trace_path, step="a", detail={})
    log_step(trace_path, step="b", detail={})

    lines = trace_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["step"] == "a"
    assert json.loads(lines[1])["step"] == "b"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tracing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.tracing'`.

- [ ] **Step 3: Write minimal implementation**

`agent/tracing.py`:
```python
"""
Structured, append-only trace logging for each agent step — mirrors Week 5's
audit_log.py shape, giving the demo a full, inspectable record of what the
agent proposed, decided, and executed.
"""
import json
import time
from pathlib import Path


def log_step(trace_path: Path, step: str, detail: dict) -> None:
    entry = {"timestamp": time.time(), "step": step, "detail": detail}
    with trace_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tracing.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add modules/week-06-scaling/labs/capstone-scaling-platform/agent/tracing.py modules/week-06-scaling/labs/capstone-scaling-platform/tests/test_tracing.py
git commit -m "$(cat <<'EOF'
feat: add structured agent trace logging

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Live demo wiring, Makefile, README

**Files:**
- Create: `.../capstone-scaling-platform/run_demo.py`
- Create: `.../capstone-scaling-platform/Makefile`
- Create: `.../capstone-scaling-platform/README.md`

**Interfaces:**
- Consumes: `propose_rollout_plan` (Task 7); `require_approval` (Task 8); `execute_plan` (Task 9);
  `log_step` (Task 10); `compute_cost_report` (Task 6); `ServiceVersion` (Task 3);
  `SloThresholds` (Task 4).

- [ ] **Step 1: Write `run_demo.py`**

```python
"""
The Week 6 "live demo": a planner proposes a rollout plan, a human-in-the-loop
gate must approve it, and a worker executes it via the canary/SLO/rollback
mechanics — then a cost report is printed. Runs twice: once with a healthy
candidate (rollout succeeds) and once with a deliberately regressed candidate
(rollout auto-rolls-back) — so a learner sees both outcomes without editing
anything.

This is an in-process simulation — no real Kubernetes, no real network calls,
no real cost data. Run with `make demo` or `python run_demo.py`.
"""
from pathlib import Path

from agent.hitl_gate import require_approval
from agent.planner import propose_rollout_plan
from agent.tracing import log_step
from agent.worker import execute_plan
from cost_report import compute_cost_report
from deployment_simulator import ServiceVersion
from slo import SloThresholds

TRACE_PATH = Path(__file__).parent / "demo_trace.log"
THRESHOLDS = SloThresholds(
    max_p95_latency_ms=200.0, max_error_rate=0.1, min_quality_score=0.7
)
STABLE = ServiceVersion("stable-v1", 80.0, 10.0, 0.02, 0.90, 0.05)
N_REQUESTS_PER_STEP = 300
STABLE_COST_PER_REQUEST = 0.01
CANDIDATE_COST_PER_REQUEST = 0.03


def run_scenario(name: str, candidate: ServiceVersion, seed: int) -> None:
    print(f"\n=== Scenario: {name} ===")

    plan = propose_rollout_plan(candidate.name, target_traffic_pct=100)
    log_step(TRACE_PATH, "planner_proposed", {"plan": plan.__dict__})
    print(f"Planner proposes: {plan.reason}")

    decision = require_approval(plan, approve=True, reason="passed pre-rollout checklist")
    log_step(TRACE_PATH, "human_decided", {"decision": decision.__dict__})
    print(f"Human decision: {'APPROVED' if decision.approved else 'DENIED'} ({decision.reason})")

    result = execute_plan(
        plan, decision, STABLE, candidate, THRESHOLDS,
        n_requests_per_step=N_REQUESTS_PER_STEP, seed=seed,
    )
    log_step(
        TRACE_PATH, "worker_executed",
        {"rolled_back": result.rolled_back if result else None},
    )

    if result is None:
        print("Worker took no action (plan was not approved).")
        return

    for step in result.steps:
        status = "ROLLED BACK" if step.rolled_back else "OK"
        print(f"  step {step.traffic_pct}% traffic -> {status} {step.violations}")

    if result.rolled_back:
        print(f"Rollout ROLLED BACK. Final traffic on candidate: {result.final_traffic_pct}%.")
    else:
        print(f"Rollout SUCCEEDED. Final traffic on candidate: {result.final_traffic_pct}%.")

    report = compute_cost_report(
        result, N_REQUESTS_PER_STEP, STABLE_COST_PER_REQUEST, CANDIDATE_COST_PER_REQUEST
    )
    log_step(TRACE_PATH, "cost_report", {"report": report.__dict__})
    print(
        f"Cost report: full-stable=${report.full_stable_cost:.2f}, "
        f"full-candidate=${report.full_candidate_cost:.2f}, "
        f"actual-canary=${report.actual_canary_cost:.2f}, "
        f"savings-vs-full-candidate=${report.savings_vs_full_candidate:.2f}"
    )


def main() -> None:
    healthy_candidate = ServiceVersion("candidate-v2-healthy", 85.0, 10.0, 0.02, 0.88, 0.05)
    regressed_candidate = ServiceVersion(
        "candidate-v2-regressed", 400.0, 30.0, 0.4, 0.5, 0.1
    )

    run_scenario("Healthy rollout", healthy_candidate, seed=1)
    run_scenario("Regressed rollout (auto-rollback)", regressed_candidate, seed=2)

    print(f"\nFull trace written to {TRACE_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the demo to verify it works end to end**

```bash
python run_demo.py
```
Expected: prints both scenarios; the healthy one reaches "Rollout SUCCEEDED... 100%", the regressed
one reaches "Rollout ROLLED BACK... 0%"; a `demo_trace.log` file is created in the lab directory
(gitignored — this is a regenerable output, not a committed source of truth).

- [ ] **Step 3: Run the full test suite with coverage**

Run: `pytest --cov=. --cov-report=term-missing --cov-fail-under=80 --cov-config=.coveragerc`
Expected: all tests pass; coverage clears 80% (if it doesn't, check the term-missing report for
which lines are uncovered before proceeding — `run_demo.py` itself has no dedicated test, so if
coverage is short, that module's `if __name__ == "__main__":` guard should already exclude the
`main()` call from being required, but the `run_scenario` function body executing via `main()` is
not otherwise tested; this is acceptable and consistent with this repo's established pattern of
leaving CLI/demo-orchestration entry points untested by design, the same as every prior week's
`main()` functions in `eval_harness.py`/`fairness_audit.py`/etc. — do not add contrived tests just
to cover print statements).

- [ ] **Step 4: Write `Makefile`**

```makefile
.PHONY: setup test demo

setup:
	pip install -r requirements.txt

test:
	pytest --cov=. --cov-report=term-missing --cov-fail-under=80 --cov-config=.coveragerc

demo:
	python run_demo.py
```

- [ ] **Step 5: Write `README.md`**

```markdown
# Capstone Lab: Scaling — Canary Rollout, SLOs, Cost, and a Human-Supervised Agent

Companion to [Week 6's concept README](../../README.md). A single, standalone, in-process
simulation of the syllabus's own "live demo": deploy a canary release with an SLO-based
auto-rollback gate, report the cost tradeoff, and route the rollout decision through a
human-in-the-loop-gated planner/worker agent.

**This is a simulation.** No real Kubernetes, no real servers, no real network calls, no real
cost data — deterministic, seeded, and disclosed as such throughout. It exists to teach the
mechanics (how a canary rollout ramps traffic and rolls back on SLO breach; how a cost report
compares scenarios; how a HITL gate structurally blocks unapproved action), not to be a
production deployment tool.

## What's here

```
deployment_simulator.py   # simulates two service "versions" as seeded outcome generators
slo.py                     # SLO thresholds + breach detection
canary.py                   # ramped traffic rollout with automatic rollback on SLO breach
cost_report.py                # compares full-stable / full-candidate / actual-canary cost
agent/planner.py                # proposes a rollout plan (scripted, not LLM-backed)
agent/hitl_gate.py                # a human-in-the-loop approval gate
agent/worker.py                     # executes an approved plan; never acts on a denied one
agent/tracing.py                      # structured trace logging of each agent step
run_demo.py                            # the live demo: healthy rollout + regressed rollout
```

## Run it

If `make` isn't available (e.g. plain Windows without Git Bash), run the commands inside
`Makefile` directly — each target is a single command.

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows:     .venv\Scripts\activate
make setup
make test    # runs the full TDD'd test suite
make demo    # runs both scenarios end to end, prints a trace + cost report
```

## What to notice

- The canary rollout checks SLOs against the CANDIDATE's own traffic only, not the blended
  total — a regression under a small traffic share would otherwise get diluted by the much
  larger stable-traffic sample and might never trip the threshold.
- `agent/worker.py`'s `execute_plan()` structurally cannot act on an unapproved plan — the
  approval check happens before any call into the rollout mechanics, not as a policy a caller
  could forget to enforce. Verified directly in this lab's own tests via a mock that asserts
  the rollout function is never called when the plan is denied.
- This is this course's only agentic-tool-use code, introduced once, deliberately, and tightly
  scoped to a single kind of proposal (a rollout plan) — see Week 6's concept README for why
  every prior week explicitly kept this out of scope.
- `run_demo.py` runs the exact same code path twice with two different candidate configurations
  — one healthy, one deliberately regressed — so both the "rollout succeeds" and "rollout
  auto-rolls-back" outcomes are directly observable without editing anything.
```

- [ ] **Step 6: Commit**

```bash
git add modules/week-06-scaling/labs/capstone-scaling-platform/run_demo.py modules/week-06-scaling/labs/capstone-scaling-platform/Makefile modules/week-06-scaling/labs/capstone-scaling-platform/README.md
git commit -m "$(cat <<'EOF'
feat: wire the live demo (planner, HITL gate, worker, cost report)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: CI — add the capstone lab to the test matrix

**Files:**
- Modify: `.github/workflows/tests.yml`

**Interfaces:**
- Consumes: the capstone lab's `requirements.txt` and `tests/` directory (Tasks 2-11).

- [ ] **Step 1: Read the current `.github/workflows/tests.yml`**

Confirm its current `matrix.include` list of `{dir, cov}` pairs (10 entries after Week 5).

- [ ] **Step 2: Add one new entry**

```yaml
          - dir: modules/week-06-scaling/labs/capstone-scaling-platform
            cov: .
```

Uses `cov: .` since this lab is a flat file layout with no `app/` package (matching Week 4's
ML/LLM eval-harness labs' pattern, not Weeks 1-3/5's FastAPI-service `cov: app` pattern). Leave
every other part of the file unchanged.

- [ ] **Step 3: Validate the YAML**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/tests.yml'))"` (from the repo
root) and confirm it parses without error.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/tests.yml
git commit -m "$(cat <<'EOF'
ci: add the Week 6 capstone lab to the pytest matrix

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: Code review and security review pass

**Files:** none created — this task reviews everything from Tasks 2-12 and applies fixes in place.

- [ ] **Step 1: Run the code-reviewer agent**

Dispatch the `code-reviewer` agent against `modules/week-06-scaling/labs/capstone-scaling-platform/`.
Ask it to check code quality, error handling, and maintainability per this repo's standards, and
specifically to verify: the lab's `Makefile` `test` target matches
`.github/workflows/tests.yml`'s actual command EXACTLY (`pytest --cov=. --cov-report=term-missing
--cov-fail-under=80 --cov-config=.coveragerc`); `ruff check .` actually passes when run directly
(not just assumed from prior task reviews); `agent/worker.py`'s approval check is genuinely
enforced (read the code, don't just trust the mocked test); `canary.py`'s SLO check genuinely
evaluates only the candidate's outcomes, not a blended stable+candidate sample; run the full test
suite and read the actual coverage report, not just a pass/fail summary.

- [ ] **Step 2: Run the security-reviewer agent**

Dispatch the `security-reviewer` agent against the same directory. This lab has no HTTP-exposed
surface, no auth, no secrets, and no real network calls — focus on: confirm there is genuinely no
subprocess/network/filesystem-outside-the-lab-directory call anywhere in the codebase (grep for
`subprocess`, `requests`, `socket`, `urllib`, `os.system` — none should appear, since this is
supposed to be a pure in-process simulation with no real infrastructure interaction); confirm
`demo_trace.log` never contains anything PII/secret-shaped (it only logs plan/decision/rollout
dataclass contents, which are all synthetic).

- [ ] **Step 3: Fix CRITICAL and HIGH findings**

Apply fixes for any CRITICAL or HIGH severity finding directly in the affected files. Re-run the
affected test suite after each fix to confirm nothing broke.

- [ ] **Step 4: Commit fixes (if any)**

```bash
git add -A
git commit -m "$(cat <<'EOF'
fix: address code-reviewer and security-reviewer findings

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

If no findings required fixes, skip this commit and note that in your final report.

---

### Task 14: Push to GitHub

**Files:** none.

- [ ] **Step 1: Push**

From the repo root:
```bash
git push
```
Expected: pushes all Week 6 commits to `origin/master`.

- [ ] **Step 2: Verify**

Run: `gh repo view --json url,visibility,defaultBranchRef`
Expected: JSON showing the repo URL, `"visibility": "PUBLIC"`, default branch `master`.

- [ ] **Step 3: Report back**

Report the repo URL, the actual demo output from both scenarios (healthy rollout + regressed
rollout with rollback), and a one-line summary of Task 13's review findings.
