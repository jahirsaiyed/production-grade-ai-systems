# Design: Week 4 — Evaluation and Monitoring

**Date**: 2026-09-13
**Status**: Approved

## Background

Weeks 1-3 (already shipped, each passing a final whole-branch review with fixes applied) built the
repo scaffold, concept README/exercises pattern, and six hands-on labs. This spec covers Week 4,
"Evaluation and Monitoring," per `docs/course-outline.md`'s Week 4 section, whose two sub-topics
are "Evaluating Production AI" and "Observability and Drift," with a live demo of "build eval
dashboards and a gate that blocks bad releases."

All Weeks 1-3 conventions carry forward unchanged: Python 3.11+/plain venv/pip, pytest +
pytest-cov (80%+ target), ruff, `<type>: <description>` commits with the same attribution line,
TDD build order, mock-mode-by-default for anything that could cost money, and the same
code-reviewer/security-reviewer + final whole-branch review gates before anything is pushed.

## Scope for this pass

Much of Week 4's syllabus content (OpenTelemetry/Prometheus/Grafana tracing, canary/shadow
progressive delivery, data/concept/embedding/prompt drift detection, continuous-training vs. CI/CD
orchestration) assumes real observability/deployment infrastructure that doesn't fit this course's
zero-cost, zero-infra lab philosophy. Per explicit user decision, these topics are covered
**conceptually only** in the Week 4 README.

Instead, the hands-on labs implement the syllabus's actual live demo goal — "build eval dashboards
and a gate that blocks bad releases" — as two **real, functioning evaluation harnesses with a real
CI gate**, not a simulation:

- Full concept README + exercises for Week 4.
- Root `README.md` course-map row and `docs/course-outline.md`'s Week 4 status line updated to
  "fully built" as part of the SAME task that writes the new content (not a separate step —
  this exact omission was caught by final review in both Week 1 and Week 2, and a whole-branch
  review caught cross-week doc drift again in Week 3, so this is now a standing, first-class step
  in every week's Task 1, not something to rediscover).
- Two hands-on labs, one per track, both TDD-built, both **standalone** (no FastAPI/API layer —
  offline evaluation doesn't need one), each reusing an earlier week's `app/domain/`+`app/adapters/`
  packages UNCHANGED (not modified) so no import-path edits are needed:
  - `ml-track-eval-harness` — reuses Week 1's `app/domain/scoring.py` +
    `app/adapters/model_store.py`.
  - `llm-track-eval-harness` — reuses Week 2's `app/domain/` (chunking, retrieval, rag, tokenizing)
    + `app/adapters/` (embeddings, llm_client, index_store).
- Weeks 5-6 remain untouched skeletons (out of scope for this pass).

## Repo structure

```
modules/week-04-evaluation-and-monitoring/
├── README.md
├── exercises.md
└── labs/
    ├── ml-track-eval-harness/
    └── llm-track-eval-harness/
```

## Week 4 concept README covers

**Evaluating Production AI** (the ML-track lab lives here):
- Offline vs. online evaluation — explain the distinction; both labs' harnesses are offline
  evaluation (scored against a held-out labeled set, not live production traffic).
- Classification metrics (precision, recall, F1, ROC-AUC vs. PR-AUC) — point at `metrics.py`.
- Threshold selection with cost-based sweeps — point at `threshold_selection.py`.
- Calibration (reliability diagrams, Brier score, Platt scaling, isotonic regression) — point at
  `calibration.py`; note Platt scaling/isotonic regression (calibration-*correction** techniques,
  as opposed to calibration *measurement*) are conceptual-only this week — the lab measures
  calibration, it doesn't correct it.
- Eval harnesses (RAGAS, DeepEval), LLM-as-judge calibration — point at the LLM-track lab's
  `eval_harness.py`/`judge.py`; explicitly name RAGAS/DeepEval as real off-the-shelf tools this
  lab's hand-rolled harness is a simplified stand-in for.
- Multi-turn continuity checks — conceptual only (this course's labs are single-turn Q&A, no
  conversation state to check continuity across).
- Red-team prompts in regression packs — point at `red_team.py`.

**Observability and Drift** (conceptual only this week — the CI gate is the one concrete,
hands-on artifact from this section):
- Tracing and metrics (OpenTelemetry, Prometheus, Grafana) — conceptual only.
- Drift classes: data, concept, embedding, prompt — conceptual only.
- Outcome-based alerting — conceptual only.
- Continuous training vs. CI/CD — conceptual only.
- Progressive delivery: shadow → canary → full, and rollback gates — conceptual only; connect to
  the ACTUAL gate this week's labs implement (`gate.py`, enforced via `tests/test_gate.py` in the
  normal CI-run pytest suite) as a simplified, single-stage version of a release gate.
- Prompt registries — conceptual only.

## ML track — `ml-track-eval-harness`

```
labs/ml-track-eval-harness/
├── README.md
├── Makefile                    # setup, train, eval, test
├── requirements.txt
├── .gitignore                  # excludes eval_report.json (regenerable, like benchmark_report.txt)
├── train_model.py              # copied from Week 1 unchanged, produces this lab's own artifact
├── metrics.py                   # NEW
├── calibration.py                # NEW
├── threshold_selection.py         # NEW
├── eval_harness.py                 # NEW
├── gate.py                          # NEW
├── baseline_metrics.json             # committed baseline
├── app/
│   ├── domain/scoring.py       # copied from Week 1 unchanged
│   └── adapters/model_store.py # copied from Week 1 unchanged
├── artifacts/                   # this lab's own trained model + manifest
└── tests/
```

- `metrics.py`: `compute_classification_metrics(y_true, y_pred, y_proba) -> dict` returns
  `{precision, recall, f1, roc_auc, pr_auc}` via `sklearn.metrics`.
- `calibration.py`: `reliability_diagram_data(y_true, y_proba, n_bins=10) -> list[dict]` (per-bin
  mean predicted probability vs. actual positive fraction) and
  `compute_brier_score(y_true, y_proba) -> float`.
- `threshold_selection.py`: `sweep_thresholds(y_true, y_proba, cost_fp, cost_fn) -> list[dict]`
  (cost at each candidate threshold) and `best_threshold(y_true, y_proba, cost_fp, cost_fn) ->
  dict` (the argmin).
- `eval_harness.py`: generates a genuinely held-out labeled dataset via
  `sklearn.datasets.make_classification` using a DIFFERENT `random_state` than `train_model.py`
  used (this is what makes it a real offline eval — the model never saw this exact data), loads
  the committed, sha256-verified model via `app.adapters.model_store.load_model`, scores it, and
  runs all three modules above, returning a single `dict` report (also printable/writable to
  `eval_report.json`, gitignored — it's deterministic and regenerable, like Week 3's benchmark
  reports, not a source-of-truth artifact).
- `gate.py`: `check_gate(report: dict, baseline: dict) -> GateResult` (a small frozen dataclass:
  `passed: bool`, `failures: list[str]`) comparing each of `report`'s metrics against
  `baseline_metrics.json`'s minimums. Exposed as a plain, importable function — NOT a
  separate CI workflow step. `tests/test_gate.py` calls `eval_harness.run()` then `check_gate(...)`
  and asserts `result.passed is True` — this means the EXISTING `pytest --cov-fail-under=80` CI
  step already blocks the build the moment a metric regresses below baseline, with zero bespoke
  CI workflow changes needed for either lab.

## LLM track — `llm-track-eval-harness`

```
labs/llm-track-eval-harness/
├── README.md
├── Makefile                    # setup, ingest, eval, test
├── requirements.txt             # no fastapi/uvicorn/pydantic-settings — no API layer needed
├── .env.example
├── .gitignore
├── docs/                         # same 3 corpus docs, copied from Week 2 unchanged
├── ingest.py                      # copied from Week 2 unchanged (imports app.domain.* — package
│                                  # structure preserved so this needs zero edits)
├── eval_set.py                     # NEW
├── judge.py                          # NEW
├── red_team.py                         # NEW
├── eval_harness.py                       # NEW
├── gate.py                                # NEW
├── baseline_metrics.json                   # committed baseline (citation-match rate only)
├── app/
│   ├── domain/                  # chunking.py, retrieval.py, rag.py, tokenizing.py — copied
│   │                            # from Week 2 unchanged
│   └── adapters/                # embeddings.py, llm_client.py, index_store.py — copied
│                                 # from Week 2 unchanged
├── artifacts/                    # this lab's own re-ingested index
└── tests/
```

- `eval_set.py`: ~8 labeled `{question, expected_source}` pairs spanning the 3 corpus docs (roughly
  2-3 per doc).
- `eval_harness.py`: for each `eval_set.py` entry, runs the retrieval pipeline directly (embed →
  BM25-score → `hybrid_search` → top citation), NOT through an HTTP layer (there is none in this
  lab) — a small `answer_question(question, ...) -> dict` orchestration function mirrors what
  Week 2's `/ask` route does, calling the same domain/adapter functions directly.
- The PRIMARY graded metric is **citation-match rate**: does the top-ranked retrieved chunk's
  `source` equal the eval example's `expected_source`? This is deterministic and equally
  meaningful in mock or real mode, since retrieval doesn't depend on the LLM completion at all —
  deliberately chosen over judging answer TEXT quality, which is meaningless in mock mode (the mock
  LLM always returns the same canned string).
- `judge.py`: a SEPARATE, best-effort LLM-as-judge answer-quality score — asks the LLM (via the
  same `LlmClient`) a second, distinct judge prompt: "does this answer correctly address the
  question, given it should discuss {expected_source}? Answer YES or NO." In mock mode, this
  returns the same canned answer every time and is explicitly reported as "not meaningful in mock
  mode" rather than pretending to measure something real — it becomes a genuine signal only with a
  real `OPENAI_API_KEY`. NOT gated on (only `citation_match_rate` feeds `gate.py`).
- `red_team.py`: a handful of adversarial/off-topic prompts (a prompt-injection attempt, an
  off-topic question, a nonsense/injection-flavored string). The check: after running each through
  the retrieval pipeline, assert every returned citation's `source` is one of the 3 known corpus
  filenames — never a fabricated source. This passes BY CONSTRUCTION given the citations-are-
  built-from-retrieval-metadata-not-LLM-output architecture already established in Week 2 — an
  honest, demonstrable "architectural defense" lesson, not a content filter or prompt-engineering
  trick.
- `gate.py`: same pattern as the ML lab — `check_gate(report, baseline) -> GateResult`, asserted in
  `tests/test_gate.py`, gating on `citation_match_rate` only (not the best-effort judge score).

## Tooling

Same conventions as Weeks 1-3. CI's matrix grows to 8 entries (both new labs added). Both use
`cov: app`, consistent with the FastAPI-lab entries, since both labs reuse an `app/domain/`+
`app/adapters/` package layout even though neither has an `app/api/` or `app/main.py` — see the
risk table below for why that package shape was kept.

## Build order

1. Write the Week 4 concept README + exercises, AND update the root `README.md`/
   `docs/course-outline.md` status lines for Week 4 — bundled into one task, per the standing
   convention established above.
2. `ml-track-eval-harness`: copy Week 1's `app/domain/scoring.py` + `app/adapters/model_store.py` +
   `train_model.py` (retrain to produce this lab's own artifact), then TDD-add `metrics.py` →
   `calibration.py` → `threshold_selection.py` → `eval_harness.py` → `baseline_metrics.json` →
   `gate.py` (+ `tests/test_gate.py` proving the gate passes against this lab's own real model).
3. `llm-track-eval-harness`: copy Week 2's `app/domain/`+`app/adapters/`+`docs/`+`ingest.py`
   (re-ingest to produce this lab's own artifact), then TDD-add `eval_set.py` → `eval_harness.py`
   (with its `answer_question` orchestration) → `judge.py` → `red_team.py` →
   `baseline_metrics.json` → `gate.py` (+ `tests/test_gate.py`).
4. Add both labs to the CI matrix.
5. Run `code-reviewer` and `security-reviewer` against both new labs; fix CRITICAL/HIGH findings.
6. Commit, push.
7. Final whole-branch review, same as every prior week — given three consecutive weeks' final
   reviews have each found genuine, previously-undetected bugs (Week 1: doc/behavior mismatch;
   Week 2: a retrieval bug that survived a first fix round; Week 3: an invalid benchmark
   measurement plus stale copied docs), budget for at least one round of post-review fixes here
   too, and treat a clean first pass as suspicious rather than reassuring.

## Explicitly out of scope for this pass

- Any real observability stack (OpenTelemetry, Prometheus, Grafana).
- Any real drift-detection implementation (data/concept/embedding/prompt).
- Any real progressive-delivery/canary/shadow deployment mechanism.
- Platt scaling / isotonic regression (calibration *correction*, not just measurement).
- Multi-turn conversation evaluation (this course's labs are single-turn).
- A literal "dashboard" UI — `eval_harness.py`'s report is a structured dict/JSON, not a rendered
  chart or web page, per the user's explicit choice not to add a dashboard viewer this pass.
- Weeks 5-6 (still skeleton-only).

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| The gate is gamed by a baseline set so loose it never fails | Medium | Set `baseline_metrics.json` thresholds from the ACTUAL measured performance of each lab's real (not stubbed) model/index, with a small realistic margin — not an arbitrary round number |
| LLM-as-judge score is presented as if meaningful in mock mode | Medium | Explicit, repeated documentation (README + code comments) that the judge score is only real evidence with a configured API key; gate never depends on it |
| Reusing `app/domain/`+`app/adapters/` without the `app/api/`/`app/main.py` layer leaves an oddly-shaped `app/` package (no `__init__.py` chain issues, but conceptually "why is there an app/ folder with no app") | Low | Note explicitly in each lab's README why the `app/` package exists here (preserving Week 1/2's exact import paths so the reused files need zero edits) despite there being no service to run |
| Held-out eval set for the ML lab isn't truly independent if `make_classification`'s different `random_state` still produces data too similar to training | Low | Use a clearly different `random_state` and document the reasoning; this is a teaching simplification (synthetic data), not a claim of rigorous train/test separation methodology |
