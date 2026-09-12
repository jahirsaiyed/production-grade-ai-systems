# Design: Week 3 — Serving, Inference, and Optimization

**Date**: 2026-09-12
**Status**: Approved

## Background

Weeks 1-2 (already shipped) built the repo scaffold, concept README/exercises pattern, and four
hands-on labs (two per week, one ML track + one LLM track) following a layered
`api/domain/adapters` architecture with structured logging, request IDs, retries, and
sha256-versioned artifacts. This spec covers Week 3, "Serving, Inference, and Optimization," per
`docs/course-outline.md`'s Week 3 section, whose two sub-topics are "Serving Architecture" and
"Optimization Levers," with a live demo of "make the assistant faster and cheaper, proven with
benchmarks."

All Weeks 1-2 conventions carry forward unchanged: Python 3.11+/plain venv/pip, pytest +
pytest-cov (80%+ target on domain/api), ruff, `<type>: <description>` commits with the same
attribution line, TDD build order, mock-mode-by-default for anything that could cost money or
need real infrastructure, and the same code-reviewer/security-reviewer + final whole-branch review
gates before anything is pushed.

## Scope for this pass

Much of Week 3's syllabus content (Triton, vLLM, SGLang, TorchServe, TensorRT, ONNX Runtime,
GPU-based quantization-aware training, KV cache internals, FlashAttention, speculative decoding,
GPU scheduling) assumes real GPU infrastructure that doesn't fit this course's established
zero-cost, zero-GPU, self-contained lab philosophy. Per explicit user decision, these topics are
covered **conceptually only** in the Week 3 README — no hands-on lab attempts to stand up a real
inference server or a real quantized model.

Instead, the hands-on labs demonstrate the syllabus's actual live demo goal — "make it faster and
cheaper, proven with benchmarks" — by adding real, measurable, CPU-only optimizations on top of
copies of the existing Week 1/2 services, each with a deterministic benchmark proving the
before/after improvement:

- Full concept README + exercises for Week 3.
- Two hands-on labs, one per track, both TDD-built, each a self-contained copy of an earlier
  week's lab (matching the established "each lab independently runnable" pattern — Week 2 already
  duplicated Week 1's `llm_client.py`/`logging_utils.py` rather than sharing them):
  - `ml-track-batch-optimization` — copies Week 1's ML lab, adds a batched scoring endpoint.
  - `llm-track-semantic-cache` — copies Week 2's RAG lab, adds semantic caching.
- Weeks 4-6 remain untouched skeletons (out of scope for this pass).

## Repo structure

```
modules/week-03-serving-inference-optimization/
├── README.md                          # concept explanations for every Week 3 topic
├── exercises.md                       # hints + acceptance criteria, no answer key
└── labs/
    ├── ml-track-batch-optimization/
    └── llm-track-semantic-cache/
```

## Week 3 concept README covers

**Serving Architecture** (conceptual only — no lab demonstrates these directly):
- Dedicated inference services
- Serving frameworks (Triton, TorchServe, Ray Serve, vLLM, SGLang) — explained and compared, not
  installed
- Prefill vs. decode; memory-bound vs. compute-bound inference
- Topologies: single model, router/gateway, cascades with fallbacks
- Self-host vs. managed APIs (rate limits, data residency, unit economics)

**Optimization Levers** (the two hands-on labs live here):
- Profiling and baselines like p50/p95 latency — both labs' `benchmark.py` scripts are the
  concrete example
- Semantic caching — `llm-track-semantic-cache`'s concrete implementation
- Dynamic vs. continuous batching, autoscaling signals — `ml-track-batch-optimization`'s batched
  endpoint is a simplified, deterministic stand-in for real dynamic batching (a request-accumulator
  with a wait-time window); the README explains the real mechanism and notes the lab demonstrates
  the throughput principle (amortizing per-call overhead) without the timing complexity
- Quantization (post-training, quantization-aware, INT8/INT4) — conceptual only
- Mixed precision (FP16/BF16) — conceptual only
- Pruning (structured, unstructured); distillation — conceptual only
- Runtimes and kernels (ONNX Runtime, TensorRT, OpenVINO) — conceptual only
- KV Cache, FlashAttention, speculative decoding — conceptual only
- Packaging (checkpoint/ONNX → service image, gRPC contracts, etc.) — conceptual only, connects
  back to Week 1's packaging-path lesson

## ML track — `ml-track-batch-optimization`

Deliberately a copy of Week 1's `ml-track-fraud-detection` (same layered architecture, structured
logging, request IDs, sha256-verified artifact — copied wholesale, not reinvented) with one
addition: batched scoring.

```
labs/ml-track-batch-optimization/
├── README.md
├── Makefile                    # setup, train, test, run, benchmark, docker-build, docker-run
├── requirements.txt
├── .env.example
├── Dockerfile
├── prototype.py                 # copied from Week 1 unchanged
├── train_model.py                # copied from Week 1 unchanged
├── benchmark.py                   # NEW: proves batching's throughput win
├── app/
│   ├── main.py, config.py, logging_utils.py, middleware.py   # copied from Week 1 unchanged
│   ├── api/
│   │   ├── schemas.py             # existing ScoreRequest/Response + new BatchScoreRequest/Response
│   │   └── routes.py               # existing /score + new POST /score/batch
│   ├── domain/
│   │   ├── scoring.py              # copied from Week 1 unchanged
│   │   └── batch_scoring.py         # NEW
│   └── adapters/                    # model_store.py, feature_store.py copied from Week 1 unchanged
├── artifacts/                        # freshly trained, this lab's own versioned artifact
└── tests/
```

- `app/domain/batch_scoring.py`: `score_batch(feature_vectors: list[list[float]], model) ->
  list[ScoreResult]` calls `model.predict_proba(feature_vectors)` ONCE across all rows (a single
  vectorized call), rather than looping `score_transaction` once per item — this is the actual
  mechanism that makes batching faster (amortizing Python-level and model-invocation overhead
  across many rows).
- `POST /score/batch` accepts `{"transactions": [...]}` (list of the same fields as `/score`),
  fetches features for each (still one `fetch_features` call per transaction — batching the
  feature-store call itself is out of scope, this lab isolates the model-inference batching
  benefit specifically), and returns a list of per-transaction results via `score_batch`.
- `benchmark.py`: runs N=30 trials of two approaches against a running instance of the service (or
  in-process via `TestClient`, to keep it dependency-free and deterministic) —
  (a) 100 sequential `POST /score` calls, one transaction each;
  (b) 1 `POST /score/batch` call with all 100 transactions.
  Reports p50/p95 total wall-clock time per trial for each approach, and the speedup ratio, to a
  plain-text report file. No timing windows, no sleeps, no flakiness — the comparison is between
  two deterministic code paths, repeated enough times to get a stable percentile.

## LLM track — `llm-track-semantic-cache`

Deliberately a copy of Week 2's `llm-track-rag-service` (same domain/adapters/api layers, same
corpus and index artifact — copied wholesale) with one addition: semantic caching in front of
retrieval and the LLM call.

```
labs/llm-track-semantic-cache/
├── README.md
├── Makefile                    # setup, ingest, test, run, benchmark, docker-build, docker-run
├── requirements.txt
├── .env.example
├── Dockerfile
├── docs/                         # same 3 corpus docs, copied from Week 2 unchanged
├── ingest.py                      # copied from Week 2 unchanged
├── benchmark.py                    # NEW: proves caching's latency/cost win
├── app/
│   ├── main.py, config.py, logging_utils.py, middleware.py   # copied from Week 2 unchanged
│   ├── api/
│   │   ├── schemas.py             # AskResponse gains a `cache_hit: bool` field
│   │   └── routes.py               # /ask now checks the cache before retrieval+LLM
│   ├── domain/
│   │   ├── chunking.py, retrieval.py, rag.py, tokenizing.py   # copied from Week 2 unchanged
│   │   └── semantic_cache.py        # NEW
│   └── adapters/                    # embeddings.py, llm_client.py, index_store.py copied unchanged
├── artifacts/                        # freshly ingested, this lab's own versioned index
└── tests/
```

- `app/domain/semantic_cache.py`: a `SemanticCache` class holding a list of
  `(query_vector, question, answer, citations)` entries. `lookup(query_vector, threshold=0.95) ->
  CachedAnswer | None` computes cosine similarity against every cached entry and returns the best
  match's answer/citations if it clears the threshold, else `None`. `store(query_vector, question,
  answer, citations)` appends a new entry. Pure, framework-free, unit-testable in isolation (no
  FastAPI, no I/O).
- `/ask`'s route: computes the query embedding (as it already does for retrieval), checks
  `app.state.semantic_cache.lookup(query_vector)` FIRST — on a hit, returns immediately with
  `cache_hit=true` and the cached answer/citations, skipping retrieval and the LLM call entirely;
  on a miss, runs the existing retrieval+LLM flow, then calls `.store(...)` on the result before
  returning with `cache_hit=false`.
- The cache is in-memory, per-process, reset on restart — explicitly not persisted, since this
  lab's point is to teach the caching mechanism, not build a production cache store (that would be
  a real vector DB with TTLs, which is out of scope here and already partially covered
  conceptually in Week 2's "vector DB operations" section).
- `benchmark.py`: sends the same question twice (or two paraphrases embedding-close enough to
  clear the threshold) via `TestClient`, measuring latency for the cold (cache-miss) call vs. the
  warm (cache-hit) call, and reports the wall-clock speedup. Also computes an illustrative cost
  saving: given an assumed per-real-LLM-call cost constant (e.g. $0.0006, documented as a rough
  illustrative gpt-4o-mini-class estimate, not a real pricing guarantee), reports the dollar amount
  saved for N repeated/similar queries avoided by the cache. This works identically whether the lab
  runs in mock mode or with a real API key — the cache-hit path skips the LLM call either way, so
  the latency and cost savings are genuinely demonstrated regardless of mode.

## Tooling

Same conventions as Weeks 1-2: Python 3.11+, plain venv/pip, pytest + pytest-cov (80%+ on
domain/api), ruff, exact-version pins with same-minor-version substitution allowed, `<type>:
<description>` commits with the same attribution trailer. CI's matrix grows to 6 entries (both new
labs added, using the same `{dir, cov}` pair structure the Week 2 final review introduced so the
ML lab's flat `--cov=.` and the FastAPI labs' `--cov=app` both work under one shared step).

## Build order

1. Write the Week 3 concept README + exercises, AND update the root `README.md` course-map row and
   `docs/course-outline.md`'s Week 3 status line from "coming soon"/"skeleton only" to "fully
   built" — bundled into the same task, not a separate afterthought, since forgetting exactly this
   update is what both Week 1's and Week 2's final reviews caught as a Critical/Important finding.
2. `ml-track-batch-optimization`: copy Week 1's ML lab wholesale, retrain to produce this lab's own
   artifact, then TDD-add `batch_scoring.py` → `/score/batch` route → `benchmark.py`.
3. `llm-track-semantic-cache`: copy Week 2's RAG lab wholesale, re-ingest to produce this lab's own
   artifact, then TDD-add `semantic_cache.py` → wire into `/ask` → `benchmark.py`.
4. Add both labs to the CI matrix.
5. Run `code-reviewer` and `security-reviewer` against both new labs; fix CRITICAL/HIGH findings.
6. Commit, push.
7. Final whole-branch review across everything changed since Week 2 shipped, same as before —
   given the last two final reviews each caught real, previously-undetected bugs (a stale
   "coming soon" doc, and a hybrid-retrieval tokenization bug that survived one fix round), budget
   for at least one round of post-review fixes here too.

## Explicitly out of scope for this pass

- Any real serving-framework installation (Triton, vLLM, SGLang, TorchServe, Ray Serve).
- Any real quantization, pruning, or distillation of a model.
- ONNX Runtime / TensorRT / OpenVINO runtime usage.
- KV cache, FlashAttention, or speculative decoding implementation (transformer-internals topics,
  not applicable to this course's sklearn/API-based models anyway).
- A production-grade (persistent, TTL'd, size-bounded) semantic cache — the lab's cache is
  intentionally a minimal in-memory teaching example.
- Real dynamic/continuous batching with a wait-time-window accumulator — the ML lab demonstrates
  the throughput principle via a simpler, deterministic "one batched call vs N sequential calls"
  comparison instead.
- Weeks 4-6 (still skeleton-only).

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Benchmark results are noisy/non-deterministic on different hardware, undermining "proven with benchmarks" | Medium | Report percentiles (p50/p95) over many trials rather than a single measurement; explicitly document that absolute numbers vary by machine but the *relative* speedup direction is what's being taught |
| Semantic cache's fixed similarity threshold (0.95) never triggers a hit with mock embeddings on the benchmark's chosen questions | Medium | Benchmark asks the literal same question twice for the "guaranteed hit" case (cosine similarity 1.0 against itself), rather than relying on a paraphrase to clear the threshold |
| Copying two entire labs wholesale roughly doubles this week's line count vs. Weeks 1-2 | Low (accepted) | Consistent with the established "each lab independently runnable, duplication over sharing" convention; per-file review effort is lower since most files are unchanged copies |
| README/course-outline status lines forgotten again (this exact bug was caught in both Weeks 1 and 2's final reviews) | Medium | Explicitly a plan task this time (see Build order step 1 doesn't cover it — add an explicit task in the implementation plan to update root README/course-outline status lines to "fully built" for Week 3) |
