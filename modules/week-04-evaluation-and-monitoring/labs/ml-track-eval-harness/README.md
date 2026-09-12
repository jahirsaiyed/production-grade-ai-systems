# ML Track Lab: Evaluation Harness

Companion to [Week 4's concept README](../../README.md). A standalone offline-evaluation harness
for the Week 1 fraud-detection model — no API, no Docker, just real classification metrics,
calibration measurement, cost-based threshold selection, and a gate that would block a real CI run
on regression.

## What's here

```
app/domain/scoring.py       # reused from Week 1 unchanged — the SAME scoring function production uses
app/adapters/model_store.py # reused from Week 1 unchanged — sha256-verified model loading
train_model.py               # reused from Week 1 unchanged — trains this lab's own model
metrics.py                    # precision/recall/F1/ROC-AUC/PR-AUC
calibration.py                  # reliability diagram data + Brier score
threshold_selection.py           # cost-based threshold sweep
eval_harness.py                   # orchestrates all of the above against a held-out labeled set
gate.py                            # compares an eval run against baseline_metrics.json
baseline_metrics.json                # committed minimum-acceptable metrics
```

## Run it

If `make` isn't available (e.g. plain Windows without Git Bash), run the commands inside
`Makefile` directly — each target is a single command.

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows:     .venv\Scripts\activate
make setup
make train   # trains this lab's own model artifact
make eval    # runs the offline evaluation, prints metrics, writes eval_report.json
make test    # runs pytest, which includes the gate check as a normal assertion
```

## What to notice

- `eval_harness.py` scores the held-out set through `score_transaction` — the EXACT same domain
  function Week 1's production service calls — not a shortcut. Evaluating through the real
  production code path, not a reimplementation of it, is itself a training/serving-parity lesson.
- The held-out evaluation set uses `random_state=123`, deliberately different from
  `train_model.py`'s `random_state=42` — this model never saw this exact data during training.
- `gate.py`'s `check_gate()` is asserted directly in `tests/test_gate.py` — there's no separate CI
  workflow step for "the gate." The normal `pytest` run this repo's CI already executes for every
  lab IS the release gate. If a future change regresses a metric below `baseline_metrics.json`,
  CI goes red on the next push, the same way any other failing test would.
- Calibration here is MEASURED (reliability diagram, Brier score), not corrected — Platt
  scaling/isotonic regression (calibration correction techniques) are conceptual-only this week,
  see the Week 4 concept README.
- The measured metrics on this held-out set are notably weak — at the fixed `FRAUD_THRESHOLD = 0.5`
  in `app/domain/scoring.py`, the committed logistic-regression model produces precision, recall,
  and F1 of `0.0` (only 8 of 2000 transactions score above 0.5, and none of those 8 are true
  positives), with `roc_auc ≈ 0.556` and `pr_auc ≈ 0.112` — barely better than random. This is the
  real, reproducible output of `python eval_harness.py` against this lab's own committed model
  artifact, not a placeholder. `baseline_metrics.json` is set ~0.05 below these actual measured
  values (clamped to `[0, 1]`, so the already-zero metrics stay at `0.0`) — a genuine regression
  gate derived from a real run, not an arbitrary guess. It illustrates a real lesson: a model can
  clear a naive accuracy bar while its precision/recall at the deployed threshold are unusable —
  exactly the kind of gap an offline evaluation harness like this one exists to catch before
  shipping.
