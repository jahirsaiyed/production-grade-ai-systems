# ML Track Lab: Experiment Tracking and Reproducibility

Companion to [Week 2's concept README](../../README.md). Unlike Week 1's ML lab, this one has no
API and no Docker — it's scoped to the model-development/reproducibility topic itself: a data
validation gate, an MLflow-tracked training run, and a reproducibility manifest.

## What's here

```
data_validation.py       # validate_training_data() — a schema/range gate before training
train_with_tracking.py   # trains + logs to MLflow + writes reproducibility_manifest.json
tests/                   # pytest: both pieces above
```

## Run it

If `make` isn't available (e.g. plain Windows without Git Bash), run the commands inside
`Makefile` directly — each target is a single command.

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows:     .venv\Scripts\activate
make setup
make test    # runs pytest with coverage
make train   # trains, logs to MLflow, writes reproducibility_manifest.json
make mlflow-ui   # opens a local MLflow UI at the printed URL — inspect your run there
```

## What to notice

- `data_validation.py`'s `validate_training_data` runs *before* `mlflow.start_run()` in
  `train_with_tracking.py` — a bad dataset never gets an experiment logged for it, the same
  "gate before the expensive step" pattern Week 1 used for artifact integrity.
- `reproducibility_manifest.json` implements the course's reproducibility checklist literally:
  `git_commit` (what code produced this), `data_digest` (a sha256 of the exact training data —
  changes if the dataset generation changes even slightly), `env_lock` (a sha256 of
  `requirements.txt`), and `mlflow_run_id`/`artifact_uri` (the link back to the tracked run and
  its saved model artifact).
- MLflow's local file backend (`mlruns/`) requires no server — it's just a structured directory
  of run metadata, gitignored like a build artifact since it's fully regenerable by `make train`.
