# Week 2 Exercises

Hints and acceptance criteria only — no answer key. If you get stuck, the reference implementation
is the lab's own code.

## Exercise 1: Break the data validation gate on purpose

Edit `ml-track-experiment-tracking/train_with_tracking.py` to inject a single `float("nan")` into
the generated training data before calling `validate_training_data`, then run `make train`.
**Acceptance criteria:** the script raises `DataValidationError` and never reaches the MLflow
run — no experiment gets logged for bad data.

## Exercise 2: Inspect a run in the MLflow UI

Run `make mlflow-ui` in the ML lab after `make train`, and open the local URL it prints.
**Acceptance criteria:** you can find your most recent run, see its logged `accuracy`/`f1` metrics,
and open the `reproducibility_manifest.json` field values to explain what each one guards against.

## Exercise 3: Break the RAG lab's artifact hash on purpose

Edit one character inside `llm-track-rag-service/artifacts/index.json` (then revert it) and start
the service. **Acceptance criteria:** the service fails to start entirely with a clear
`ArtifactIntegrityError` in the logs — same fail-fast behavior as Week 1's model artifact check,
not a running server reporting `/readyz` as not-ready.

## Exercise 4: Change the hybrid retrieval weighting

`app/domain/retrieval.py`'s `hybrid_search` combines dense and BM25 scores 50/50. Change the
weighting to favor BM25 (e.g. 0.2 dense / 0.8 BM25), re-run the tests, and ask a question whose
answer depends on an exact keyword match versus one that depends on paraphrase understanding.
**Acceptance criteria:** you can describe, in your own words, one query where the reweighting
changed which chunk got retrieved first.

## Exercise 5: Add a fourth company doc and verify citations update

Add a new markdown file to `llm-track-rag-service/docs/`, re-run `make ingest`, restart the
service, and ask a question only your new document answers. **Acceptance criteria:** the
response's `citations` field includes your new file's name — proving citations come from the
actual retrieved chunks, not a hardcoded list.
