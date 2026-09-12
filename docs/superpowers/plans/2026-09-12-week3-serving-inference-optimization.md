# Week 3 Serving, Inference, and Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Week 3's concept README/exercises, update the root README/course-outline status
lines, and build two hands-on labs — `ml-track-batch-optimization` (a copy of Week 1's ML lab plus
a batched scoring endpoint) and `llm-track-semantic-cache` (a copy of Week 2's RAG lab plus
semantic caching) — each with a deterministic benchmark proving a real speed/cost win, then add CI
coverage and push.

**Architecture:** Both new labs are wholesale copies of an existing lab (same layered
`api/domain/adapters` pattern, same conventions), with one new domain module and one new/modified
API route each, plus a `benchmark.py` script.

**Tech Stack:** Same as Weeks 1-2 (Python 3.11+, FastAPI/pydantic/tenacity for the LLM lab,
scikit-learn/joblib for the ML lab, numpy explicitly for both new domain modules), pytest +
pytest-cov, ruff.

## Global Constraints

- All Weeks 1-2 conventions carry forward unchanged: Python 3.11+/plain venv/pip, `pytest`+
  `pytest-cov` with an 80%+ target on `domain/`+`api/`, `ruff check` must pass, exact-version pins
  (same-minor-version substitution allowed if unavailable), commit format `<type>: <description>`
  with the exact trailer `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` copied verbatim
  by every implementer regardless of which model executes the task.
- Both `benchmark.py` scripts are manual, run-once-and-commit-the-report scripts (like Week 1's
  `train_model.py` / Week 2's `ingest.py`) — they are NOT pytest tests and must not be asserted on
  in the test suite. They use `fastapi.testclient.TestClient` in-process so no real server needs to
  be running.
- Benchmarks report percentiles (p50/p95) over many trials (30 for the ML lab, 20 for the LLM lab),
  not a single measurement, since absolute timing varies by machine — only the relative
  speedup direction is the teaching point.
- The semantic cache's benchmark must guarantee at least one real cache hit deterministically (by
  asking the exact same question twice within one `TestClient` session), not rely on a paraphrase
  clearing the similarity threshold by chance.
- Each new lab is a full, independent copy of its source lab (same duplication-over-sharing
  convention Week 2 already established for `logging_utils.py`/`middleware.py`/`llm_client.py`) —
  do not import from the Week 1/2 lab directories.
- After copying a lab wholesale, the copied artifact (`artifacts/model.joblib`+`manifest.json` for
  the ML lab, `artifacts/index.json`+`manifest.json` for the LLM lab) must be REGENERATED at the
  new location (by re-running `train_model.py`/`ingest.py`), not left as a stale copy — otherwise
  its `manifest.json`'s `git_commit` field would reference the wrong lab's history.
- Task 1 explicitly includes updating the root `README.md` course-map row and
  `docs/course-outline.md`'s Week 3 status line to "fully built" — this exact update was missed
  and caught by final review in BOTH Week 1 and Week 2, so it is a first-class step here, not an
  afterthought.
- Working directory for all commands below is the repo root:
  `D:\Learning\Build Production Grade AI Systems _ ByteByteGo Live\production-grade-ai-systems`.
  The repo already exists on GitHub (public, `origin/master`) — no `gh repo create`, just
  `git push` at the end.

---

### Task 1: Week 3 concept README, exercises, and course-map status update

**Files:**
- Modify: `modules/week-03-serving-inference-optimization/README.md` (currently a "coming soon" stub)
- Create: `modules/week-03-serving-inference-optimization/exercises.md`
- Modify: `README.md` (repo root) — course-map row for Week 3
- Modify: `docs/course-outline.md` — Week 3 status line

**Interfaces:**
- Produces: links to `labs/ml-track-batch-optimization/README.md` and
  `labs/llm-track-semantic-cache/README.md` (built in later tasks — a known forward reference,
  same pattern as Weeks 1-2's equivalent first task).

- [ ] **Step 1: Write `modules/week-03-serving-inference-optimization/README.md`**

Replace the stub with a concept README covering each topic below as its own `##` section, in this
order. For each: a 2-4 sentence plain-language explanation, why it matters in production, and (for
the two "Optimization Levers" topics that have a hands-on lab) a pointer into
`labs/ml-track-batch-optimization/` or `labs/llm-track-semantic-cache/`. For the conceptual-only
topics, say explicitly that no lab demonstrates them this week and why (GPU/real-infrastructure
requirement).

1. **Dedicated inference services** — explain the concept of separating a model-serving process
   from the rest of an application; conceptual only.
2. **Serving frameworks (Triton, TorchServe, Ray Serve, vLLM, SGLang)** — one or two sentences per
   framework naming what it's for and how it differs from the others (e.g. vLLM/SGLang for
   high-throughput LLM serving with continuous batching and PagedAttention-style memory
   management; Triton/TorchServe as general multi-framework model servers; Ray Serve for
   Python-native scaling). Conceptual only — no lab installs any of these.
3. **Prefill vs. decode; memory-bound vs. compute-bound inference** — explain the two phases of
   autoregressive LLM inference and why they have different performance characteristics.
   Conceptual only.
4. **Topologies: single model, router/gateway, cascades with fallbacks** — explain each topology;
   note that Week 1's "router/gateway" building-blocks section already touched on this, and this
   week's cascade-with-fallback pattern is conceptually similar to the graceful-degradation pattern
   already implemented in both labs' `except LlmCallError`/`FeatureStoreUnavailable` handling.
5. **Self-host vs. managed APIs (rate limits, data residency, unit economics)** — explain the
   tradeoffs; connect to the mock/real API-key switch both LLM labs already use as a concrete
   illustration of "self-host a mock, or pay a managed API."
6. **Profiling and baselines like p50/p95 latency** — explain what percentiles capture that an
   average hides (tail latency); point at both new labs' `benchmark.py` scripts as the concrete,
   runnable example.
7. **Semantic caching, dynamic vs. continuous batching, autoscaling signals** — explain semantic
   caching (cache by meaning, not exact string) and point at
   `llm-track-semantic-cache/app/domain/semantic_cache.py`; explain dynamic/continuous batching
   (accumulating concurrent requests to amortize per-call overhead) and point at
   `ml-track-batch-optimization/app/domain/batch_scoring.py`, explicitly noting the lab's
   `/score/batch` endpoint is a simplified, deterministic stand-in for real dynamic batching (which
   uses a request-accumulator with a wait-time window) — the lab demonstrates the throughput
   principle without the timing complexity. Autoscaling signals: conceptual only.
8. **Quantization (post-training, quantization-aware, INT8/INT4)** — explain the concept and the
   distinction between the two approaches; conceptual only, no lab quantizes a real model.
9. **Mixed precision (FP16/BF16)** — brief explanation; conceptual only.
10. **Pruning (structured, unstructured); distillation** — brief explanation of both; conceptual
    only.
11. **Runtimes and kernels (ONNX Runtime, TensorRT, OpenVINO)** — one or two sentences per runtime;
    conceptual only.
12. **KV Cache, FlashAttention, and speculative decoding** — explain each at a high level (why KV
    caching avoids recomputation, what FlashAttention optimizes, how speculative decoding trades
    a small model's guesses for large-model verification); conceptual only, note these are
    transformer-internals topics not applicable to this course's models so far.
13. **Packaging (checkpoint/ONNX → service image, gRPC contracts, etc.)** — connect back
    explicitly to Week 1's "packaging path: notebook → artifact → API → container image" lesson,
    noting this week's variant swaps in a converted/optimized model instead of the raw trained one;
    conceptual only.

Open the file with a `# Week 3: Serving, Inference, and Optimization` heading and a short intro
naming both tracks and linking to `labs/ml-track-batch-optimization/README.md` and
`labs/llm-track-semantic-cache/README.md`. Close with a "## Hands-on labs" section linking both,
and an "## Exercises" section linking `exercises.md`.

- [ ] **Step 2: Write `modules/week-03-serving-inference-optimization/exercises.md`**

```markdown
# Week 3 Exercises

Hints and acceptance criteria only — no answer key. If you get stuck, the reference implementation
is the lab's own code.

## Exercise 1: Prove batching's win at a different batch size

`ml-track-batch-optimization/benchmark.py` uses 100 transactions per trial. Change
`N_TRANSACTIONS` to 10 and re-run `make benchmark`. **Acceptance criteria:** you can explain, in
your own words, whether the speedup ratio grew, shrank, or stayed about the same, and why (hint:
per-call overhead is roughly fixed regardless of batch size, so its relative weight changes with
batch size).

## Exercise 2: Break the semantic cache's threshold on purpose

Lower `SemanticCache`'s `threshold` (in `llm-track-semantic-cache/app/main.py`, where it's
constructed) to `0.5`, restart the service, and ask two completely unrelated questions back to
back. **Acceptance criteria:** the second, unrelated question now incorrectly returns
`cache_hit: true` with the first question's answer — explain why a too-low threshold is worse than
no cache at all for a real product.

## Exercise 3: Measure the semantic cache's memory growth

The `SemanticCache` never evicts entries. Send 500 unique questions to a running
`llm-track-semantic-cache` instance (a small loop script counts as "sending"), then explain, in
your own words, what a production cache would need that this lab's doesn't have (hint: an eviction
policy — LRU, TTL, or a max-size bound).

## Exercise 4: Add a request-count guard to the batch endpoint

`POST /score/batch` currently accepts up to 1000 transactions per request (see
`BatchScoreRequest`'s `max_length` constraint). Lower it to 5, restart the service, and send a
batch of 10 transactions. **Acceptance criteria:** the request is rejected with a 422, and you can
point to the exact pydantic constraint that causes this.

## Exercise 5: Compare cold-start cost to a real managed-API price

`llm-track-semantic-cache/benchmark.py` uses an illustrative
`ASSUMED_COST_PER_LLM_CALL_USD` constant. Look up a real published price-per-1K-tokens for any
managed LLM API, estimate a realistic token count for this lab's prompts (see
`app/domain/rag.py`'s `MAX_CONTEXT_CHARS`), and recompute a more realistic per-call cost.
**Acceptance criteria:** you can state your estimated real per-call cost and how it compares to
the lab's placeholder constant.
```

- [ ] **Step 3: Update the root `README.md`'s course map**

Read the current file, find the course-map table's Week 3 row (currently
`| 3 | [Serving, Inference, and Optimization](...) | Coming soon |`), and change `Coming soon` to
`Fully built` — matching how the Week 1 and Week 2 rows already read. Do not touch any other row
or section.

- [ ] **Step 4: Update `docs/course-outline.md`'s Week 3 status line**

Read the current file, find the line under the Week 3 section reading
`**Status in this repo: skeleton only — see modules/week-03-serving-inference-optimization/.**`,
and change `skeleton only` to `fully built` — matching how the Week 1/Week 2 sections' equivalent
lines already read. Do not touch any other section.

- [ ] **Step 5: Commit**

```bash
git add modules/week-03-serving-inference-optimization/README.md modules/week-03-serving-inference-optimization/exercises.md README.md docs/course-outline.md
git commit -m "$(cat <<'EOF'
docs: add Week 3 concept README/exercises and mark it fully built

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: ML lab — copy Week 1's lab wholesale and regenerate its artifact

**Files:**
- Create (via copy): everything under
  `modules/week-03-serving-inference-optimization/labs/ml-track-batch-optimization/`, copied from
  `modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/`

**Interfaces:**
- Produces: an exact functional copy of Week 1's fraud-detection lab at the new path, with its own
  freshly-generated `artifacts/model.joblib`/`manifest.json`. Consumed by Tasks 3-5.

- [ ] **Step 1: Copy the lab directory**

From the repo root:
```bash
mkdir -p modules/week-03-serving-inference-optimization/labs
cp -r modules/week-01-prototype-to-production/labs/ml-track-fraud-detection modules/week-03-serving-inference-optimization/labs/ml-track-batch-optimization
```

- [ ] **Step 2: Clean out anything that shouldn't have been copied**

```bash
cd modules/week-03-serving-inference-optimization/labs/ml-track-batch-optimization
rm -rf .venv .pytest_cache .ruff_cache .coverage
find . -type d -name "__pycache__" -exec rm -rf {} +
```

- [ ] **Step 3: Create a fresh venv and install dependencies**

```bash
python -m venv .venv
```
Windows: `.venv\Scripts\activate` — macOS/Linux: `source .venv/bin/activate`, then:
```bash
pip install -r requirements.txt
```

- [ ] **Step 4: Regenerate the artifact so its manifest reflects THIS lab's own history**

Using this lab's own `.venv` Python explicitly (Windows: `.venv\Scripts\python.exe train_model.py`;
macOS/Linux: `.venv/bin/python train_model.py`):
```bash
python train_model.py
```
Expected: prints `Wrote .../model.joblib and manifest.json (sha256=...)`, and
`artifacts/model.joblib`/`manifest.json` are freshly written (the git_commit field will reference
the current HEAD at the time this is run — this is expected and matches Week 1/2's established
pattern where the manifest's commit is one commit behind the one that actually ships it).

- [ ] **Step 5: Run the copied test suite to confirm the copy works unmodified**

```bash
pytest -v --cov=app --cov-report=term-missing
```
Expected: all tests pass (same count as Week 1's ML lab — 12 tests), coverage 80%+ on
`domain/`+`api/`, exactly as in Week 1, since no code has changed yet.

- [ ] **Step 6: Commit**

```bash
git add modules/week-03-serving-inference-optimization/labs/ml-track-batch-optimization
git commit -m "$(cat <<'EOF'
feat: copy Week 1 fraud-detection lab as the base for batch optimization

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: ML lab — batch scoring domain logic (TDD)

**Files:**
- Modify: `modules/week-03-serving-inference-optimization/labs/ml-track-batch-optimization/requirements.txt`
  (add explicit `numpy` pin)
- Create: `modules/week-03-serving-inference-optimization/labs/ml-track-batch-optimization/app/domain/batch_scoring.py`
- Test: `modules/week-03-serving-inference-optimization/labs/ml-track-batch-optimization/tests/test_batch_scoring.py`

**Interfaces:**
- Consumes: `ScoreResult`, `FRAUD_THRESHOLD` (from `app/domain/scoring.py`, already present from
  the Task 2 copy).
- Produces: `score_batch(feature_vectors: list[list[float]], model) -> list[ScoreResult]`.
  Consumed by Task 4's `api/routes.py`.

All commands run from
`modules/week-03-serving-inference-optimization/labs/ml-track-batch-optimization/`.

- [ ] **Step 1: Add an explicit `numpy` pin to `requirements.txt`**

Read the current file, then add `numpy==2.1.3` as a new line (scikit-learn already pulls in numpy
transitively, but this lab's new code imports it directly, so it should be pinned explicitly like
every other direct dependency in this repo). Re-run `pip install -r requirements.txt` to confirm it
resolves (it should already be satisfied transitively).

- [ ] **Step 2: Write the failing test**

`tests/test_batch_scoring.py`:
```python
from app.domain.batch_scoring import score_batch


class _StubModel:
    def __init__(self, probabilities: list[float]):
        self._probabilities = probabilities

    def predict_proba(self, X):
        return [[1 - p, p] for p in self._probabilities]


def test_score_batch_scores_all_rows_in_one_call():
    model = _StubModel(probabilities=[0.9, 0.1, 0.5])
    results = score_batch([[1.0], [2.0], [3.0]], model)
    assert len(results) == 3
    assert results[0].is_fraud is True
    assert results[0].fraud_probability == 0.9
    assert results[1].is_fraud is False
    assert results[2].is_fraud is True  # 0.5 >= FRAUD_THRESHOLD (0.5)


def test_score_batch_returns_empty_list_for_empty_input():
    model = _StubModel(probabilities=[])
    results = score_batch([], model)
    assert results == []
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_batch_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.domain.batch_scoring'`.

- [ ] **Step 4: Write minimal implementation**

`app/domain/batch_scoring.py`:
```python
import numpy as np

from app.domain.scoring import FRAUD_THRESHOLD, ScoreResult


def score_batch(feature_vectors: list[list[float]], model) -> list[ScoreResult]:
    if not feature_vectors:
        return []
    probabilities = np.array(model.predict_proba(feature_vectors))[:, 1]
    return [
        ScoreResult(
            fraud_probability=float(p), is_fraud=float(p) >= FRAUD_THRESHOLD
        )
        for p in probabilities
    ]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_batch_scoring.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add modules/week-03-serving-inference-optimization/labs/ml-track-batch-optimization/requirements.txt modules/week-03-serving-inference-optimization/labs/ml-track-batch-optimization/app/domain/batch_scoring.py modules/week-03-serving-inference-optimization/labs/ml-track-batch-optimization/tests/test_batch_scoring.py
git commit -m "$(cat <<'EOF'
feat: add vectorized batch scoring domain logic

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: ML lab — `/score/batch` API route (TDD)

**Files:**
- Modify: `modules/week-03-serving-inference-optimization/labs/ml-track-batch-optimization/app/api/schemas.py`
- Modify: `modules/week-03-serving-inference-optimization/labs/ml-track-batch-optimization/app/api/routes.py`
- Modify: `modules/week-03-serving-inference-optimization/labs/ml-track-batch-optimization/tests/test_routes.py`

**Interfaces:**
- Consumes: `score_batch` (Task 3); `fetch_features`, `FeatureStoreUnavailable` (already present
  from Task 2's copy); `settings` (already present at module level in the copied `routes.py`).
- Produces: `POST /score/batch` returning `BatchScoreResponse`.

- [ ] **Step 1: Read the current `app/api/schemas.py` and `app/api/routes.py`**

These were copied unchanged from Week 1 in Task 2 — read them now to see the exact current
`ScoreRequest`/`ScoreResponse` shapes and the exact current `/score` route's structure (including
how it already imports `FeatureStoreUnavailable`, `HTTPException`, and a module-level `logger` and
`settings`), so your additions match the file's existing style precisely.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_routes.py` (do not remove or modify the existing tests in this file):
```python
from unittest.mock import patch


def test_score_batch_returns_results_for_all_transactions():
    with TestClient(app) as client:
        response = client.post(
            "/score/batch",
            json={
                "transactions": [
                    {
                        "transaction_id": "txn-1",
                        "amount": 10.0,
                        "merchant_category": "electronics",
                    },
                    {
                        "transaction_id": "txn-2",
                        "amount": 20.0,
                        "merchant_category": "groceries",
                    },
                ]
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 2
    assert body["results"][0]["transaction_id"] == "txn-1"
    assert body["results"][1]["transaction_id"] == "txn-2"


def test_score_batch_returns_503_when_feature_store_unavailable():
    with TestClient(app) as client:
        with patch(
            "app.api.routes.fetch_features",
            side_effect=FeatureStoreUnavailable("simulated outage"),
        ):
            response = client.post(
                "/score/batch",
                json={
                    "transactions": [
                        {
                            "transaction_id": "txn-1",
                            "amount": 10.0,
                            "merchant_category": "electronics",
                        }
                    ]
                },
            )
    assert response.status_code == 503


def test_score_batch_rejects_more_than_1000_transactions():
    with TestClient(app) as client:
        response = client.post(
            "/score/batch",
            json={
                "transactions": [
                    {
                        "transaction_id": f"txn-{i}",
                        "amount": 1.0,
                        "merchant_category": "electronics",
                    }
                    for i in range(1001)
                ]
            },
        )
    assert response.status_code == 422
```

(`FeatureStoreUnavailable` should already be imported at the top of this test file if the copied
Week 1 test file imports it for the existing `test_score_returns_503_when_feature_store_unavailable`
test — check and reuse that import; only add the `from unittest.mock import patch` import if not
already present.)

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_routes.py -v`
Expected: the 3 new tests FAIL (404, since `/score/batch` doesn't exist yet); all pre-existing
tests still PASS.

- [ ] **Step 4: Write minimal implementation**

Add to `app/api/schemas.py` (append, do not modify `ScoreRequest`/`ScoreResponse`):
```python
class BatchScoreRequest(BaseModel):
    transactions: list[ScoreRequest] = Field(..., min_length=1, max_length=1000)


class BatchScoreResponse(BaseModel):
    results: list[ScoreResponse]
```

Add to `app/api/routes.py` (append the import and the new route; do not modify the existing
`/score`, `/healthz`, `/readyz` routes):
```python
from app.api.schemas import BatchScoreRequest, BatchScoreResponse  # extend the existing schemas import line instead of duplicating it
from app.domain.batch_scoring import score_batch


@router.post("/score/batch", response_model=BatchScoreResponse)
def score_batch_endpoint(
    payload: BatchScoreRequest, request: Request
) -> BatchScoreResponse:
    try:
        feature_vectors = [
            fetch_features(
                txn.transaction_id,
                failure_rate=settings.feature_store_failure_rate,
            )
            for txn in payload.transactions
        ]
    except FeatureStoreUnavailable as exc:
        logger.error(f"feature store unavailable during batch scoring: {exc}")
        raise HTTPException(
            status_code=503, detail="feature store unavailable, please retry"
        ) from exc

    results = score_batch(feature_vectors, request.app.state.model)
    return BatchScoreResponse(
        results=[
            ScoreResponse(
                transaction_id=txn.transaction_id,
                fraud_probability=r.fraud_probability,
                is_fraud=r.is_fraud,
            )
            for txn, r in zip(payload.transactions, results)
        ]
    )
```

Adapt the exact import line merging to the file's real current import statements (e.g. if
`app/api/routes.py` already does `from app.api.schemas import ScoreRequest, ScoreResponse`, extend
that single line to include `BatchScoreRequest, BatchScoreResponse` rather than adding a second,
duplicate import line).

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_routes.py -v`
Expected: all tests pass (pre-existing + 3 new).

- [ ] **Step 6: Run the full suite with coverage**

Run: `pytest --cov=app --cov-report=term-missing`
Expected: all tests pass; coverage on `app/domain/` and `app/api/` still 80%+.

- [ ] **Step 7: Commit**

```bash
git add modules/week-03-serving-inference-optimization/labs/ml-track-batch-optimization/app/api modules/week-03-serving-inference-optimization/labs/ml-track-batch-optimization/tests/test_routes.py
git commit -m "$(cat <<'EOF'
feat: add POST /score/batch endpoint

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: ML lab — benchmark script, Makefile, README

**Files:**
- Create: `modules/week-03-serving-inference-optimization/labs/ml-track-batch-optimization/benchmark.py`
- Modify: `modules/week-03-serving-inference-optimization/labs/ml-track-batch-optimization/Makefile`
- Modify: `modules/week-03-serving-inference-optimization/labs/ml-track-batch-optimization/README.md`

**Interfaces:**
- Consumes: the running `app` from `app.main` (Task 2's copy), `/score` and `/score/batch` (Task 4).

- [ ] **Step 1: Write `benchmark.py`**

```python
"""
Benchmarks batched vs. sequential scoring to prove batching's throughput win.

Run with `make benchmark` or `python benchmark.py`. Uses an in-process TestClient
so no server needs to be running; absolute numbers vary by machine, but the
relative speedup direction is the point.
"""
import time

from fastapi.testclient import TestClient

from app.main import app

N_TRANSACTIONS = 100
N_TRIALS = 30


def _transaction(i: int) -> dict:
    return {
        "transaction_id": f"txn-{i}",
        "amount": 10.0 + i,
        "merchant_category": "electronics",
    }


def _time_sequential(client: TestClient) -> float:
    start = time.perf_counter()
    for i in range(N_TRANSACTIONS):
        client.post("/score", json=_transaction(i))
    return time.perf_counter() - start


def _time_batched(client: TestClient) -> float:
    payload = {"transactions": [_transaction(i) for i in range(N_TRANSACTIONS)]}
    start = time.perf_counter()
    client.post("/score/batch", json=payload)
    return time.perf_counter() - start


def _percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    index = min(int(len(values) * pct), len(values) - 1)
    return values[index]


def main() -> None:
    with TestClient(app) as client:
        sequential_times = [_time_sequential(client) for _ in range(N_TRIALS)]
        batched_times = [_time_batched(client) for _ in range(N_TRIALS)]

    seq_p50, seq_p95 = _percentile(sequential_times, 0.5), _percentile(
        sequential_times, 0.95
    )
    batch_p50, batch_p95 = _percentile(batched_times, 0.5), _percentile(
        batched_times, 0.95
    )

    report = (
        f"Benchmark: {N_TRANSACTIONS} transactions x {N_TRIALS} trials\n"
        f"{'':20}{'p50 (s)':>12}{'p95 (s)':>12}\n"
        f"{'sequential':20}{seq_p50:>12.4f}{seq_p95:>12.4f}\n"
        f"{'batched':20}{batch_p50:>12.4f}{batch_p95:>12.4f}\n"
        f"speedup (p50): {seq_p50 / batch_p50:.2f}x\n"
    )
    print(report)
    with open("benchmark_report.txt", "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it to confirm it works**

Run: `python benchmark.py`
Expected: prints the report table and a speedup ratio greater than 1.0x, and writes
`benchmark_report.txt`. The exact ratio will vary by machine — that's expected.

- [ ] **Step 3: Add a `benchmark` target and a `.gitignore` entry**

Read the current `Makefile`, then add a new target (keeping all existing targets unchanged):
```makefile
benchmark:
	python benchmark.py
```

Read the current `.gitignore` for this lab (copied from Week 1), then add a line so the generated
report isn't committed each time it's re-run:
```
benchmark_report.txt
```

- [ ] **Step 4: Update `README.md`**

Read the current file (copied from Week 1's ML lab), then add a new `## Prove it: batching
benchmark` section near the end (after the existing "What to notice" section) explaining:
- `make benchmark` runs 30 trials of 100 sequential `/score` calls vs. 100 transactions in one
  `/score/batch` call, and reports p50/p95 total latency for each.
- The speedup comes from calling the model's `predict_proba` once across all 100 rows instead of
  100 separate Python-level calls — the per-call overhead (feature fetch, FastAPI request
  handling, model invocation) is what gets amortized.
- This is a simplified, deterministic stand-in for real dynamic batching (which accumulates
  concurrent requests over a short wait-time window) — see the Week 3 concept README for the real
  mechanism.

- [ ] **Step 5: Commit**

```bash
git add modules/week-03-serving-inference-optimization/labs/ml-track-batch-optimization/benchmark.py modules/week-03-serving-inference-optimization/labs/ml-track-batch-optimization/Makefile modules/week-03-serving-inference-optimization/labs/ml-track-batch-optimization/.gitignore modules/week-03-serving-inference-optimization/labs/ml-track-batch-optimization/README.md
git commit -m "$(cat <<'EOF'
docs: add batching benchmark script, Makefile target, and README section

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: LLM lab — copy Week 2's RAG lab wholesale and regenerate its artifact

**Files:**
- Create (via copy): everything under
  `modules/week-03-serving-inference-optimization/labs/llm-track-semantic-cache/`, copied from
  `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/`

**Interfaces:**
- Produces: an exact functional copy of Week 2's RAG lab at the new path, with its own freshly
  re-ingested `artifacts/index.json`/`manifest.json`. Consumed by Tasks 7-9.

- [ ] **Step 1: Copy the lab directory**

From the repo root:
```bash
cp -r modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service modules/week-03-serving-inference-optimization/labs/llm-track-semantic-cache
```

- [ ] **Step 2: Clean out anything that shouldn't have been copied**

```bash
cd modules/week-03-serving-inference-optimization/labs/llm-track-semantic-cache
rm -rf .venv .pytest_cache .ruff_cache .coverage
find . -type d -name "__pycache__" -exec rm -rf {} +
```

- [ ] **Step 3: Create a fresh venv and install dependencies**

```bash
python -m venv .venv
```
Windows: `.venv\Scripts\activate` — macOS/Linux: `source .venv/bin/activate`, then:
```bash
pip install -r requirements.txt
```

- [ ] **Step 4: Re-ingest so the manifest reflects THIS lab's own history**

Using this lab's own `.venv` Python explicitly (Windows: `.venv\Scripts\python.exe ingest.py`;
macOS/Linux: `.venv/bin/python ingest.py`):
```bash
python ingest.py
```
Expected: prints `Wrote .../index.json and manifest.json (6 chunks, sha256=...)` (6 chunks, since
Week 2's final review fixed the chunk size to produce 2 chunks per doc across 3 docs).

- [ ] **Step 5: Run the copied test suite to confirm the copy works unmodified**

```bash
pytest -v --cov=app --cov-report=term-missing
```
Expected: all tests pass (same count as Week 2's RAG lab — 22 tests), coverage 80%+ on
`domain/`+`api/`.

- [ ] **Step 6: Commit**

```bash
git add modules/week-03-serving-inference-optimization/labs/llm-track-semantic-cache
git commit -m "$(cat <<'EOF'
feat: copy Week 2 RAG lab as the base for semantic caching

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: LLM lab — semantic cache domain logic (TDD)

**Files:**
- Create: `modules/week-03-serving-inference-optimization/labs/llm-track-semantic-cache/app/domain/semantic_cache.py`
- Test: `modules/week-03-serving-inference-optimization/labs/llm-track-semantic-cache/tests/test_semantic_cache.py`

**Interfaces:**
- Produces: `CachedAnswer(answer: str, citations: list[dict])` (frozen dataclass),
  `SemanticCache(threshold: float = 0.95)` with `.lookup(query_vector: np.ndarray) -> CachedAnswer
  | None` and `.store(query_vector: np.ndarray, answer: str, citations: list[dict]) -> None`.
  Consumed by Task 8.

All commands run from
`modules/week-03-serving-inference-optimization/labs/llm-track-semantic-cache/`.

- [ ] **Step 1: Write the failing test**

`tests/test_semantic_cache.py`:
```python
import numpy as np

from app.domain.semantic_cache import SemanticCache


def test_lookup_returns_none_when_cache_is_empty():
    cache = SemanticCache()
    assert cache.lookup(np.array([1.0, 0.0])) is None


def test_lookup_returns_cached_answer_for_identical_vector():
    cache = SemanticCache(threshold=0.95)
    cache.store(
        np.array([1.0, 0.0]),
        "answer one",
        [{"source": "doc.md", "chunk_id": 0, "snippet": "..."}],
    )

    result = cache.lookup(np.array([1.0, 0.0]))

    assert result is not None
    assert result.answer == "answer one"
    assert result.citations == [{"source": "doc.md", "chunk_id": 0, "snippet": "..."}]


def test_lookup_returns_none_below_similarity_threshold():
    cache = SemanticCache(threshold=0.95)
    cache.store(np.array([1.0, 0.0]), "answer one", [])

    result = cache.lookup(np.array([0.0, 1.0]))

    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_semantic_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.domain.semantic_cache'`.

- [ ] **Step 3: Write minimal implementation**

`app/domain/semantic_cache.py`:
```python
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CachedAnswer:
    answer: str
    citations: list[dict]


@dataclass
class _CacheEntry:
    query_vector: np.ndarray
    answer: str
    citations: list[dict]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class SemanticCache:
    def __init__(self, threshold: float = 0.95):
        self._threshold = threshold
        self._entries: list[_CacheEntry] = []

    def lookup(self, query_vector: np.ndarray) -> CachedAnswer | None:
        best_score = -1.0
        best_entry: _CacheEntry | None = None
        for entry in self._entries:
            score = _cosine_similarity(query_vector, entry.query_vector)
            if score > best_score:
                best_score = score
                best_entry = entry
        if best_entry is not None and best_score >= self._threshold:
            return CachedAnswer(
                answer=best_entry.answer, citations=best_entry.citations
            )
        return None

    def store(
        self, query_vector: np.ndarray, answer: str, citations: list[dict]
    ) -> None:
        self._entries.append(
            _CacheEntry(
                query_vector=query_vector, answer=answer, citations=citations
            )
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_semantic_cache.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add modules/week-03-serving-inference-optimization/labs/llm-track-semantic-cache/app/domain/semantic_cache.py modules/week-03-serving-inference-optimization/labs/llm-track-semantic-cache/tests/test_semantic_cache.py
git commit -m "$(cat <<'EOF'
feat: add semantic cache domain logic

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: LLM lab — wire semantic cache into `/ask` (TDD)

**Files:**
- Modify: `modules/week-03-serving-inference-optimization/labs/llm-track-semantic-cache/app/api/schemas.py`
- Modify: `modules/week-03-serving-inference-optimization/labs/llm-track-semantic-cache/app/api/routes.py`
- Modify: `modules/week-03-serving-inference-optimization/labs/llm-track-semantic-cache/app/main.py`
- Modify: `modules/week-03-serving-inference-optimization/labs/llm-track-semantic-cache/tests/test_routes.py`

**Interfaces:**
- Consumes: `SemanticCache`, `CachedAnswer` (Task 7).
- Produces: `AskResponse` gains `cache_hit: bool`; `/ask` checks the cache before retrieval+LLM.

- [ ] **Step 1: Read the current `app/api/schemas.py`, `app/api/routes.py`, and `app/main.py`**

These were copied unchanged from Week 2 in Task 6 — read them now to see the exact current
`AskResponse` shape, the exact current `/ask` route (including its existing `except
EmbeddingCallError` and `except LlmCallError` handling from Week 2's final review fixes — do not
remove or weaken either), and the exact current `lifespan` function in `main.py`, so your edits
match the file's existing style precisely.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_routes.py` (do not remove or modify existing tests):
```python
def test_ask_returns_cache_hit_false_on_first_call():
    with TestClient(app) as client:
        response = client.post(
            "/ask", json={"question": "How many vacation days do I get?"}
        )
    assert response.json()["cache_hit"] is False


def test_ask_returns_cache_hit_true_on_repeated_identical_question():
    with TestClient(app) as client:
        client.post("/ask", json={"question": "How many vacation days do I get?"})
        response = client.post(
            "/ask", json={"question": "How many vacation days do I get?"}
        )
    assert response.json()["cache_hit"] is True
```

(Both requests must be sent within the SAME `with TestClient(app) as client:` block so the same
`app.state.semantic_cache` instance persists across both calls — this mirrors how a real running
server behaves, since the cache is in-memory per-process.)

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_routes.py -v`
Expected: the 2 new tests FAIL (`KeyError`/`AssertionError` on the missing `cache_hit` field); all
pre-existing tests still PASS.

- [ ] **Step 4: Write minimal implementation**

Add to `app/api/schemas.py` (add the field to the existing `AskResponse` class, do not create a
new class):
```python
class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    source: str
    cache_hit: bool
```

Update `app/api/routes.py`'s `ask` function. The existing function (from Week 2's final review
fixes) already has this overall shape — read it first, then modify it to add cache-check-first and
cache-store-on-miss, while preserving both existing exception handlers exactly:
```python
from app.domain.semantic_cache import SemanticCache  # only if not already needed directly; the cache instance itself lives on app.state, constructed in main.py


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request) -> AskResponse:
    state = request.app.state
    try:
        query_vector = state.embedding_client.embed(payload.question)
    except EmbeddingCallError as exc:
        logger.warning(
            f"embedding call failed after retries, serving fallback answer: {exc}"
        )
        return AskResponse(
            question=payload.question,
            answer="The assistant is temporarily unavailable. Please try again.",
            citations=[],
            source="mock" if state.embedding_client.is_mock else "llm",
            cache_hit=False,
        )

    cached = state.semantic_cache.lookup(query_vector)
    if cached is not None:
        return AskResponse(
            question=payload.question,
            answer=cached.answer,
            citations=[Citation(**c) for c in cached.citations],
            source="mock" if state.llm_client.is_mock else "llm",
            cache_hit=True,
        )

    bm25_scores = list(
        state.bm25_index.get_scores(tokenize(payload.question))
    )
    retrieved = hybrid_search(
        query_vector, state.chunks, state.dense_vectors, bm25_scores, k=3
    )

    prompt = build_rag_prompt(payload.question, retrieved)
    try:
        answer = state.llm_client.complete(prompt)
    except LlmCallError as exc:
        logger.warning(
            f"llm call failed after retries, serving fallback answer: {exc}"
        )
        answer = "The assistant is temporarily unavailable. Please try again."

    citation_dicts = build_citations(retrieved)
    state.semantic_cache.store(query_vector, answer, citation_dicts)

    return AskResponse(
        question=payload.question,
        answer=answer,
        citations=[Citation(**c) for c in citation_dicts],
        source="mock" if state.llm_client.is_mock else "llm",
        cache_hit=False,
    )
```

Adapt to the file's real current imports (e.g. `tokenize` should already be imported from
`app.domain.tokenizing` per Week 2's final review fix — reuse that import, don't duplicate it; the
`SemanticCache` import may not be needed directly in `routes.py` at all if only `main.py`
constructs it — check whether the route body references the class itself anywhere, and only import
what's actually used).

Update `app/main.py`'s `lifespan` function to construct the cache on `app.state`:
```python
from app.domain.semantic_cache import SemanticCache

# inside the lifespan function, alongside the existing state assignments:
    app.state.semantic_cache = SemanticCache(threshold=0.95)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_routes.py -v`
Expected: all tests pass (pre-existing + 2 new).

- [ ] **Step 6: Run the full suite with coverage**

Run: `pytest --cov=app --cov-report=term-missing`
Expected: all tests pass; coverage on `app/domain/` and `app/api/` still 80%+.

- [ ] **Step 7: Commit**

```bash
git add modules/week-03-serving-inference-optimization/labs/llm-track-semantic-cache/app modules/week-03-serving-inference-optimization/labs/llm-track-semantic-cache/tests/test_routes.py
git commit -m "$(cat <<'EOF'
feat: wire semantic cache into /ask, skipping retrieval and the LLM call on a hit

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: LLM lab — benchmark script, Makefile, README

**Files:**
- Create: `modules/week-03-serving-inference-optimization/labs/llm-track-semantic-cache/benchmark.py`
- Modify: `modules/week-03-serving-inference-optimization/labs/llm-track-semantic-cache/Makefile`
- Modify: `modules/week-03-serving-inference-optimization/labs/llm-track-semantic-cache/README.md`

**Interfaces:**
- Consumes: the running `app` from `app.main` (Task 6's copy), `/ask` with `cache_hit` (Task 8).

- [ ] **Step 1: Write `benchmark.py`**

```python
"""
Benchmarks semantic-cache hit vs. miss latency and illustrates cost savings.

Run with `make benchmark` or `python benchmark.py`. Uses an in-process TestClient
so no server needs to be running.
"""
import time

from fastapi.testclient import TestClient

from app.main import app

N_TRIALS = 20
REPEATED_QUESTION = "How many vacation days do I get?"
# Illustrative gpt-4o-mini-class estimate, not a real pricing guarantee — see
# Week 3's Exercise 5 for computing a more realistic figure.
ASSUMED_COST_PER_LLM_CALL_USD = 0.0006


def _percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    index = min(int(len(values) * pct), len(values) - 1)
    return values[index]


def main() -> None:
    with TestClient(app) as client:
        miss_times = []
        hit_times = []
        for i in range(N_TRIALS):
            start = time.perf_counter()
            client.post("/ask", json={"question": f"unique question number {i}"})
            miss_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            client.post("/ask", json={"question": REPEATED_QUESTION})
            hit_times.append(time.perf_counter() - start)

    miss_p50 = _percentile(miss_times, 0.5)
    hit_p50 = _percentile(hit_times, 0.5)
    cost_saved = N_TRIALS * ASSUMED_COST_PER_LLM_CALL_USD

    report = (
        f"Benchmark: {N_TRIALS} cache-miss vs. cache-hit trials\n"
        f"{'':20}{'p50 (s)':>12}\n"
        f"{'cache miss':20}{miss_p50:>12.4f}\n"
        f"{'cache hit':20}{hit_p50:>12.4f}\n"
        f"speedup (p50): {miss_p50 / hit_p50:.2f}x\n"
        f"Illustrative cost saved over {N_TRIALS} avoided LLM calls "
        f"(at ${ASSUMED_COST_PER_LLM_CALL_USD}/call): ${cost_saved:.4f}\n"
    )
    print(report)
    with open("benchmark_report.txt", "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it to confirm it works**

Run: `python benchmark.py`
Expected: prints the report table and a speedup ratio greater than 1.0x (a cache hit skips
retrieval and the LLM call entirely, so it should be meaningfully faster even in mock mode), and
writes `benchmark_report.txt`.

- [ ] **Step 3: Add a `benchmark` target and a `.gitignore` entry**

Read the current `Makefile` (copied from Week 2), then add:
```makefile
benchmark:
	python benchmark.py
```

Read the current `.gitignore` for this lab, then add:
```
benchmark_report.txt
```

- [ ] **Step 4: Update `README.md`**

Read the current file (copied from Week 2's RAG lab), then add a new `## Prove it: semantic-cache
benchmark` section near the end explaining:
- `make benchmark` measures the latency of a cache miss (a brand-new question, requiring
  retrieval + an LLM call) vs. a cache hit (a repeated question, answered immediately from the
  cache) and reports the speedup.
- It also prints an illustrative cost estimate for the number of LLM calls avoided — genuinely
  meaningful once a real `OPENAI_API_KEY` is configured, since a cache hit skips the real API call
  entirely.
- The cache is in-memory and per-process, not persisted — restarting the service clears it; this
  is a deliberate simplification, not a production-ready cache store (see the Week 3 concept
  README's "semantic caching" section and Exercise 3).

- [ ] **Step 5: Commit**

```bash
git add modules/week-03-serving-inference-optimization/labs/llm-track-semantic-cache/benchmark.py modules/week-03-serving-inference-optimization/labs/llm-track-semantic-cache/Makefile modules/week-03-serving-inference-optimization/labs/llm-track-semantic-cache/.gitignore modules/week-03-serving-inference-optimization/labs/llm-track-semantic-cache/README.md
git commit -m "$(cat <<'EOF'
docs: add semantic-cache benchmark script, Makefile target, and README section

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: CI — add both new labs to the test matrix

**Files:**
- Modify: `.github/workflows/tests.yml`

**Interfaces:**
- Consumes: both new labs' `requirements.txt` and `tests/` directories (Tasks 2-9).

- [ ] **Step 1: Read the current `.github/workflows/tests.yml`**

Week 2's final review restructured this file's matrix to `include: [{dir, cov}, ...]` pairs (so the
ML-lab-style flat `--cov=.` and the FastAPI-lab-style `--cov=app` can share one step). Confirm this
structure is what's currently there before editing.

- [ ] **Step 2: Add two new matrix entries**

Add these two entries to the existing `matrix.include` list, alongside the existing four (do not
remove or reorder the existing four):
```yaml
          - dir: modules/week-03-serving-inference-optimization/labs/ml-track-batch-optimization
            cov: app
          - dir: modules/week-03-serving-inference-optimization/labs/llm-track-semantic-cache
            cov: app
```

Leave every other part of the file (checkout, setup-python, install, lint, the
`pytest --cov=${{ matrix.cov }} ...` step) exactly as-is.

- [ ] **Step 3: Validate the YAML**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/tests.yml'))"` (from the repo
root) and confirm it parses without error.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/tests.yml
git commit -m "$(cat <<'EOF'
ci: add both Week 3 labs to the pytest matrix

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Code review and security review pass

**Files:** none created — this task reviews everything from Tasks 2-10 and applies fixes in place.

- [ ] **Step 1: Run the code-reviewer agent**

Dispatch the `code-reviewer` agent against both new labs'
`modules/week-03-serving-inference-optimization/labs/*/` directories. Ask it to check code
quality, error handling, and maintainability per this repo's standards, and specifically to check
whether Weeks 1-2's post-review lessons (logging before every graceful-degradation fallback, no CI
step that trains/ingests something tests don't need, `docker-run` not requiring a `.env` file, a
regression test existing for every previously-found bug class) were correctly carried over into the
copied labs, not just the new `batch_scoring.py`/`semantic_cache.py` code.

- [ ] **Step 2: Run the security-reviewer agent**

Dispatch the `security-reviewer` agent against the same directories, specifically the new
`/score/batch` route and the semantic cache's in-memory storage of past questions/answers (check:
does the cache ever grow unbounded in a way that's a resource-exhaustion concern worth noting, even
if out of scope to fix this week? does `/score/batch`'s `max_length=1000` constraint meaningfully
bound the batch size, or could a very large per-transaction payload still cause an issue?).

- [ ] **Step 3: Fix CRITICAL and HIGH findings**

Apply fixes for any CRITICAL or HIGH severity finding directly in the affected files. Re-run the
affected lab's test suite after each fix to confirm nothing broke.

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

### Task 12: Push to GitHub

**Files:** none.

- [ ] **Step 1: Push**

From the repo root:
```bash
git push
```
Expected: pushes all Week 3 commits to `origin/master`.

- [ ] **Step 2: Verify**

Run: `gh repo view --json url,visibility,defaultBranchRef`
Expected: JSON showing the repo URL, `"visibility": "PUBLIC"`, default branch `master`.

- [ ] **Step 3: Report back**

Report the repo URL, both benchmark scripts' actual measured speedup ratios from this run, and a
one-line summary of Task 11's review findings.
