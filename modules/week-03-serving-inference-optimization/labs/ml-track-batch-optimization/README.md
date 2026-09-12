# ML Track Lab: Batch Optimization

Turns a notebook script into a layered, containerized, versioned FastAPI
service. Companion to [Week 3's concept README](../../README.md).

## Before: the prototype

`prototype.py` is the "notebook cell" version — no structure, no tests, no
versioning:

```bash
python prototype.py
```

## After: the production service

```
app/
├── main.py         # wiring: logging, middleware, routes, model loading
├── api/             # request/response schemas + routes (POST /score, POST /score/batch)
├── domain/          # scoring.py + batch_scoring.py — pure, framework-free business logic
└── adapters/        # model_store.py (versioned artifact loading),
                      # feature_store.py (simulated flaky dependency + retries)
```

## Run it

If `make` isn't available (e.g. plain Windows without Git Bash), run the
commands inside `Makefile` directly — each target is a single command.

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows:     .venv\Scripts\activate
make setup
make train      # generates artifacts/model.joblib + manifest.json
make test       # runs pytest with coverage
make run        # starts the service on http://localhost:8000
```

Try it:
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "txn-1", "amount": 42.50, "merchant_category": "electronics"}'
curl -X POST http://localhost:8000/score/batch \
  -H "Content-Type: application/json" \
  -d '{"transactions": [
        {"transaction_id": "txn-1", "amount": 42.50, "merchant_category": "electronics"},
        {"transaction_id": "txn-2", "amount": 9.99, "merchant_category": "groceries"}
      ]}'
curl http://localhost:8000/healthz
curl http://localhost:8000/readyz
```

## Run it in Docker

```bash
make docker-build
make docker-run
```

## What to notice

- `manifest.json` is checked against the model file's sha256 at startup — see
  `app/adapters/model_store.py`. Corrupt the model file and the service will
  fail to start entirely (uvicorn reports "Application startup failed" with a
  clear `ArtifactIntegrityError` in the logs) rather than silently serving wrong
  predictions.
- `app/adapters/feature_store.py` randomly fails (`FEATURE_STORE_FAILURE_RATE`
  in `.env`) to give retries (`tenacity`) something real to do.
- Every log line is JSON and carries the same `request_id` as the response's
  `x-request-id` header.

## Prove it: batching benchmark

Run `make benchmark` to compare sequential vs. batched scoring on 100 transactions
over 30 trials, printing p50/p95 latency for each:

```bash
make benchmark
```

This measures the throughput win from batching: a single vectorized call to the
model's `predict_proba` over all 100 feature vectors (converted internally to a
numpy array) vs. 100 separate single-row calls. The speedup comes from amortizing
the per-call overhead (FastAPI request handling and model invocation serialization)
across the entire batch, plus the vectorized `predict_proba` call itself.

Note: per-transaction feature fetching is deliberately **not** batched in this
lab — `app/api/routes.py`'s `/score/batch` handler still calls `fetch_features`
once per transaction in a loop, so that cost is not amortized. This is a
realistic limitation worth naming: a production batching layer would typically
need a batch-capable feature store API too, which is out of scope here.

This is a simplified, deterministic stand-in for real dynamic batching, which
accumulates concurrent requests over a short wait-time window and processes them
together. See the Week 3 concept README for the full dynamic batching mechanism
and its tradeoffs.
