# ML Track Lab: Evaluation Harness

Companion to [Week 4's concept README](../../README.md). A standalone offline-evaluation harness
for the Week 1 fraud-detection model — no API, no Docker, just real classification metrics,
calibration measurement, cost-based threshold selection, and a gate that would block a real CI run
on regression.

## What's here

```
app/domain/scoring.py       # reused from Week 1 unchanged — the SAME scoring function production uses
app/adapters/model_store.py # reused from Week 1 unchanged — sha256-verified model loading
train_model.py               # adapted from Week 1 — same model, plus the train/held-out split this lab evaluates against
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
- The held-out evaluation set is a genuine 80/20 split of the SAME `random_state=42` synthetic
  dataset `train_model.py` generates — not a different classification problem. `train_model.py`
  splits it with `train_test_split(..., test_size=0.2, random_state=7, stratify=y)`, fits the model
  only on the 80% training portion, and persists the 20% held-out portion to
  `artifacts/held_out.json`. `eval_harness.py` loads that exact file rather than regenerating data,
  so this model genuinely never saw these specific rows during training, while still being drawn
  from the distribution it was trained on.
- `gate.py`'s `check_gate()` is asserted directly in `tests/test_gate.py` — there's no separate CI
  workflow step for "the gate." The normal `pytest` run this repo's CI already executes for every
  lab IS the release gate. If a future change regresses a metric below `baseline_metrics.json`,
  CI goes red on the next push, the same way any other failing test would.
- Calibration here is MEASURED (reliability diagram, Brier score), not corrected — Platt
  scaling/isotonic regression (calibration correction techniques) are conceptual-only this week,
  see the Week 4 concept README.
- The measured discrimination on this held-out set is real and moderate, not spectacular — this is
  still a simple synthetic teaching example, not a production fraud model. `python eval_harness.py`
  against this lab's own committed model artifact reproducibly measures `roc_auc ≈ 0.760` and
  `pr_auc ≈ 0.349`: the model separates fraud from non-fraud meaningfully better than chance
  (`roc_auc = 0.5`) and meaningfully better than the ~10% positive-rate floor for `pr_auc`, but it's
  far from a strong classifier. ROC-AUC and PR-AUC are threshold-independent (they're computed
  across all thresholds), so they reflect the model's underlying discrimination, not the choice of
  `FRAUD_THRESHOLD`.
  At the fixed `FRAUD_THRESHOLD = 0.5` in `app/domain/scoring.py`, however, precision/recall/F1 ARE
  threshold-dependent and come out low (`precision = 0.5`, `recall = 0.1`, `f1 ≈ 0.167` on this
  held-out split) — few transactions cross 0.5, so recall in particular is weak. This is the
  specific, disclosed limitation of evaluating at one fixed operating point: `threshold_selection.py`'s
  cost-based sweep picks a much lower threshold (`~0.05`) as cheaper overall given `cost_fn=25.0`.
  `baseline_metrics.json` is set ~0.05 below each actual measured value — a genuine regression gate
  derived from a real run, not an arbitrary guess. It illustrates a real
  lesson: a model's threshold-independent discrimination (ROC-AUC/PR-AUC) can look reasonable while
  its precision/recall at a naively-chosen fixed threshold are weak — exactly the kind of gap an
  offline evaluation harness like this one, paired with cost-based threshold selection, exists to
  catch before shipping.
