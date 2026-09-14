# Week 6: Scaling — Design Spec

## Overview

Week 6 is the final week of the course and covers platform architecture, release engineering, and
cost tradeoffs for production AI systems. Unlike Weeks 1-5, its syllabus describes a single
combined "live demo" (canary release + cost report + a human-supervised agent) rather than
separate ML/LLM tracks, so Week 6 is a **unified capstone lab**, not a track split.

Most of the syllabus (Kubernetes/EKS/GKE/AKS, managed ML platforms, GPU scheduling/autoscaling,
workflow orchestration engines, incident response, case studies) requires real cloud
infrastructure this course has never had and won't start requiring now — these stay
**conceptual-only** in the README, the same treatment Weeks 4-5 gave TLS, vector-store access
control, and regulatory frameworks. What IS genuinely demonstrable without cloud infrastructure —
canary rollout mechanics, SLO-based auto-rollback, cost tradeoff comparison, and a minimal
human-supervised agent — becomes one standalone, in-process, deterministic Python simulation: no
real Kubernetes, no real servers, no real cloud spend, the same "simplified, from-scratch, teaching
first" philosophy already used for this repo's simulated feature store (Week 1), simulated cost
constants (Week 3), and synthetic eval data (Week 4).

## A course-first: a small, real agentic demo

Every prior week's concept README has explicitly listed agentic tool-use as out of scope (most
recently Week 5, under OWASP's "excessive agency" item). Week 6 is a deliberate, one-time exception
to make the syllabus's own "human-supervised agent" demo real rather than conceptual — but it stays
tightly bounded: a planner proposes ONE kind of action (a rollout plan), a human-approval gate must
approve it, and a worker executes it only if approved. This is not a general agent framework, tool
library, or multi-step autonomous loop — it exists to teach the HITL-gate pattern concretely, not to
introduce agentic capability into the course's ongoing labs.

## Capstone lab: `capstone-scaling-platform`

New standalone lab under `modules/week-06-scaling/labs/capstone-scaling-platform/`. No FastAPI
service, no Docker — a CLI-driven demo (`run_demo.py`) plus a fully TDD'd library of modules
behind it, matching Week 4's harness-style shape (a `Makefile` with `setup`/`test`/`demo` targets,
`pytest`+`ruff` gated the same way every other lab is).

### `deployment_simulator.py`
Simulates two service "versions" — `stable` (the incumbent) and `candidate` (what's being rolled
out) — each a small, seeded, deterministic generator of per-request outcomes:
`RequestOutcome(latency_ms: float, succeeded: bool, quality_score: float)`. Each version has its
own configurable distribution parameters (mean/stddev latency, error rate, mean quality score) so a
demo scenario can construct a `candidate` that's either healthy (safe to roll out) or regressed
(should trigger rollback) — the same "prove the mechanism catches a real regression" standard this
repo has applied since Week 3's benchmark and Week 4's eval harnesses.

### `slo.py`
Defines an `SloThresholds` dataclass (`max_p95_latency_ms`, `max_error_rate`,
`min_quality_score`) and `check_slo_breach(outcomes: list[RequestOutcome], thresholds:
SloThresholds) -> list[str]` — returns which specific thresholds were violated (empty list = 
healthy), computing p95 latency, error rate, and mean quality score from the outcome sample.

### `canary.py`
`run_canary_rollout(stable, candidate, thresholds, n_requests, initial_traffic_pct,
ramp_steps) -> RolloutResult` — sends traffic to `candidate` in increasing steps (e.g. 10% → 25% →
50% → 100%), checking SLOs after each step; the moment a step breaches SLOs, the rollout stops and
reports a rollback (candidate traffic returns to 0%, all further traffic goes to `stable`).
`RolloutResult` records the step-by-step traffic history, whether a rollback occurred, and at which
step/traffic-percentage it happened.

### `cost_report.py`
`compute_cost_report(rollout_result, stable_cost_per_request, candidate_cost_per_request) ->
CostReport` — compares three scenarios: cost if 100% of traffic had stayed on `stable`, cost if
100% had gone to `candidate`, and the actual cost incurred by the canary rollout's real traffic
split (which is cheaper than either extreme if a rollback happened early, since less of the
possibly-more-expensive candidate traffic was served).

### `agent/` — the HITL-gated planner/worker demo
- **`planner.py`** — `propose_rollout_plan(candidate_name: str, target_traffic_pct: int) ->
  RolloutPlan` (a small, inspectable dataclass: what's being proposed, not a free-text LLM
  output — this is a scripted planner, not an LLM-backed one, keeping the "small, tightly bounded"
  agent promise literal).
- **`hitl_gate.py`** — `require_approval(plan: RolloutPlan, approve: bool) -> ApprovalDecision` —
  in a real system this would prompt a human; here it takes an explicit `approve` argument so the
  gate's behavior is deterministic and testable (a caller — the demo script or a test — plays the
  human's role). Records the decision with a reason string.
- **`worker.py`** — `execute_plan(plan: RolloutPlan, decision: ApprovalDecision, stable, candidate,
  thresholds) -> RolloutResult | None` — executes the plan (calls `canary.run_canary_rollout`) only
  if `decision.approved` is `True`; returns `None` and takes no action otherwise. This is the actual
  enforcement point: the worker structurally cannot act on an unapproved plan, mirroring how this
  course's Week 4/5 "guardrails hold by construction" pattern (verified via code inspection, not a
  policy someone could forget to check) applies here too.
- **`tracing.py`** — `log_step(trace_path: Path, step: str, detail: dict) -> None` — appends one
  structured JSON line per agent step (planner proposed, human decided, worker executed or
  skipped), the same audit-log shape Week 5 established, giving the demo a full, inspectable trace
  of what the agent did and why.

### `run_demo.py`
Ties everything together into the syllabus's own "live demo" framing: propose a plan → gate it
through HITL approval → execute via the worker (with the canary/SLO/rollback mechanics running
underneath) → print a cost report. Runs twice in the demo script: once with a healthy candidate
(rollout succeeds, reaches 100%) and once with a deliberately regressed candidate (rollout
auto-rolls-back partway through) — so a learner sees both outcomes without editing anything.

## Concept README (11 sections, mirroring Weeks 4-5's structure)

1. Kubernetes/managed ML platforms — conceptual only; name EKS/GKE/AKS/Vertex AI/SageMaker/Azure ML
   and what each is for; no lab (no cloud account this course can rely on).
2. Workflow orchestration (Airflow/Prefect/Dagster/Kubeflow/Ray) — conceptual only.
3. GPU scheduling, node pools, autoscaling, capacity planning — conceptual only.
4. Canary and blue-green rollouts; auto-rollback gates — hands-on, point at `canary.py`/`slo.py`.
5. SLOs (latency, error rate, groundedness) — hands-on, point at `slo.py`; connect "groundedness"
   explicitly back to Week 5's `groundedness.py` as the same kind of signal used differently here
   (a release gate instead of a per-response score).
6. Incident response and on-call basics — conceptual only.
7. Cost tradeoffs — hands-on, point at `cost_report.py`.
8. Planner/worker patterns — hands-on, point at `agent/planner.py` + `agent/worker.py`.
9. HITL approval gates — hands-on, point at `agent/hitl_gate.py`; state plainly this is the course's
   only agentic-tool-use code and explain why (see "A course-first" section above).
10. Agent tracing — hands-on, point at `agent/tracing.py`.
11. Case studies (recommendation/search, fintech risk scoring, consumer assistants) — conceptual
    only, 2-3 sentences each connecting back to this course's own ML/LLM tracks as worked examples
    of the same idea at smaller scale.

## Course completion

Since this is the last week, Task 1 also updates the root `README.md` to reflect the full 6-week
course as complete (Week 6's course-map row → "Fully built", same mechanism as every prior week),
and the intro section of `docs/course-outline.md` gets one sentence noting all 6 weeks now ship
with hands-on labs — no new "course complete" ceremony beyond that, consistent with this repo's
plain, unadorned documentation style.

## Build order

1. Concept README + exercises + root README/course-outline status update (bundled, per every
   prior week's Task 1 convention).
2. Lab skeleton: directory structure, `requirements.txt` (numpy for the outcome sampling; no new
   exotic dependency needed), `.gitignore`, `Makefile`, `pytest`+`ruff` wiring.
3. `deployment_simulator.py` (TDD).
4. `slo.py` (TDD).
5. `canary.py` (TDD) — including a genuine-regression test (a deliberately regressed candidate
   must trigger a real rollback, not just "the code runs").
6. `cost_report.py` (TDD).
7. `agent/planner.py` (TDD).
8. `agent/hitl_gate.py` (TDD).
9. `agent/worker.py` (TDD) — including a test proving an unapproved plan is genuinely never
   executed (mirroring Week 4/5's "prove the check isn't vacuous" standard).
10. `agent/tracing.py` (TDD).
11. `run_demo.py` wiring + `.coveragerc`/README finalize (both the healthy-rollout and
    regressed-rollout scenarios must be demonstrated).
12. CI matrix update (add the one new lab entry — 11 total).
13. Code review + security review pass (security review here is lighter than Weeks 1/5 since there's
    no API surface, no auth, no secrets — focus on: does the HITL gate genuinely block unapproved
    execution, is the simulation honestly disclosed as a simulation, no accidental real
    subprocess/network calls).
14. Push to GitHub.

## Global constraints carried over from prior weeks

- Python 3.11+, plain venv/pip, `pytest`+`pytest-cov` with `--cov-fail-under=80` matched exactly
  between the `Makefile`'s `test` target and `.github/workflows/tests.yml` (the Week 4/5 final-review
  lesson: a Makefile that doesn't match CI's real command is a false green).
  `ruff check .` must also be verified directly (not just assumed) before any task is considered
  reviewed clean — Week 5's final code review found `ruff check` failing despite every prior
  per-task review reporting clean, because no task had actually run the literal lint command.
- Exact-version pins in `requirements.txt`.
- Commit format `<type>: <description>` with the exact trailer
  `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.
- Every mechanism this week that claims to "catch" or "gate" something (SLO breach detection, the
  HITL approval gate) needs a test proving it actually fails/blocks under the condition it claims
  to guard against — not just a happy-path test. This is the single most repeated lesson across
  every prior week's final review (Week 4: gate metrics pinned at an unfailable floor; Week 5: an
  auth fix shipped with no regression test).
- The simulation must be explicitly and honestly labeled as a simulation everywhere a learner would
  reasonably think otherwise (README, module docstrings) — no real Kubernetes, no real network
  calls, no real cost data, deterministic and seeded for reproducibility.
- No PII/PHI in any committed file, log sample, or test fixture.
