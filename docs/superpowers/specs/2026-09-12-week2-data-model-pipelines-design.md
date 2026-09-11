# Design: Week 2 — Data and Model Pipelines

**Date**: 2026-09-12
**Status**: Approved

## Background

Week 1 (already shipped) fully built the repo scaffold, concept README/exercises pattern, and two
hands-on labs following a layered `api/domain/adapters` architecture with structured logging,
request IDs, retries, and sha256-versioned artifacts. This spec covers Week 2 of the same course:
"Data and Model Pipelines," per `docs/course-outline.md`'s Week 2 section, whose two sub-topics are
"Data Pipelines and Enterprise Retrieval" and "Model Development and Reproducibility," with a live
demo of "build a RAG pipeline that answers from company docs with citations."

All Week 1 conventions carry forward unchanged: Python 3.11+/plain venv/pip, pytest + pytest-cov
(80%+ target on domain/api), ruff, `<type>: <description>` commits with the same attribution line,
TDD build order, the disclaimer already in the root README/course-outline, and the same
code-reviewer/security-reviewer gate before anything is pushed.

## Scope for this pass

- Full concept README + exercises for Week 2 (replaces the "coming soon" stub from Week 1's pass).
- Two hands-on labs, one per track, both TDD-built:
  - `ml-track-experiment-tracking` — scoped to the model-development/reproducibility topic, NOT a
    rebuild of Week 1's serving work.
  - `llm-track-rag-service` — a new FastAPI service implementing the syllabus's Week 2 live demo.
- Weeks 3-6 remain untouched skeletons (out of scope for this pass, per the original course design).

## Repo structure

```
modules/week-02-data-and-model-pipelines/
├── README.md                          # concept explanations for every Week 2 topic
├── exercises.md                       # hints + acceptance criteria, no answer key
└── labs/
    ├── ml-track-experiment-tracking/
    └── llm-track-rag-service/
```

## Week 2 concept README covers

**Data Pipelines and Enterprise Retrieval:**
- Ingestion patterns (batch vs. streaming, ETL vs. ELT)
- Data validation gates and schema enforcement
- Data lineage and versioning
- Training/serving parity; leakage detection
- Deep-learning data paths (loaders, batching, augmentation, distributed training) — conceptual
  only, no lab exercises this week
- Document parsing (OCR, PDF, HTML) and chunking strategies
- Vector DB operations (indexing, upserts, metadata filters)
- Context packaging (ordering, token caps, citations)
- Hybrid retrieval: BM25 + dense embeddings + reranking; GraphRAG (GraphRAG conceptual only)

**Model Development and Reproducibility:**
- Experiment tracking (MLflow, Weights & Biases)
- Model registry stages
- Hyperparameter tuning
- Reproducibility checklist (commit, data digest, env lock, artifact link)
- Adaptation decision framework: prompting vs. RAG vs. PEFT (LoRA) vs. full fine-tuning

Each topic gets the same treatment as Week 1: plain-language explanation, why it matters in
production, and a concrete pointer into one of this week's two labs where the reader can see it in
running code.

## ML track — `ml-track-experiment-tracking`

Deliberately scoped to the topic this week actually teaches (reproducible model development), not
a re-implementation of Week 1's FastAPI serving stack. No API, no Docker — a small, focused,
TDD-tested pipeline project.

```
labs/ml-track-experiment-tracking/
├── README.md
├── Makefile                    # setup, test, train, mlflow-ui
├── requirements.txt             # pins scikit-learn, mlflow, pytest, pytest-cov, ruff
├── data_validation.py           # schema/range gate on the training data
├── train_with_tracking.py       # trains + logs to MLflow + writes reproducibility manifest
├── tests/
│   ├── test_data_validation.py
│   └── test_train_with_tracking.py
└── mlruns/                      # MLflow's local file-backend store (gitignored)
```

- `data_validation.py`: `validate_training_data(X, y)` raises `DataValidationError` on NaNs,
  infinite values, or out-of-range features — a real "gate" a bad dataset can't get past, run
  before training starts. Mirrors Week 1's "reliability primitives" theme applied to data instead
  of network calls.
- `train_with_tracking.py`: trains the same style of synthetic fraud classifier as Week 1
  (`sklearn.datasets.make_classification`) but wraps the run in an MLflow run
  (`mlflow.set_tracking_uri("file:./mlruns")`, no server required), logging hyperparameters,
  metrics (accuracy, F1), and the trained model as an MLflow artifact.
- After training, it writes `reproducibility_manifest.json` implementing the syllabus's checklist
  literally: `git_commit`, `data_digest` (sha256 of the generated training data, so a rebuild can
  be verified byte-for-byte), `env_lock` (sha256 of `requirements.txt`), and `mlflow_run_id` /
  `artifact_uri` (the link back to the tracked run).
- Tests: `data_validation` edge cases (each rejected condition gets its own test), and a test that
  runs `train_with_tracking` against a temporary MLflow tracking URI and asserts the resulting run
  has the expected params/metrics logged and the manifest file has all required keys.
- `mlruns/` is added to this lab's own `.gitignore` (MLflow's local store is regenerable, like a
  build artifact — the manifest and code that produce it are what's committed).

## LLM track — `llm-track-rag-service`

Implements the syllabus's Week 2 live demo directly. Same layered architecture, structured
logging, request-ID middleware, sha256-verified versioned artifact, and mock-mode-by-default
philosophy as Week 1's labs.

```
labs/llm-track-rag-service/
├── README.md
├── Makefile                    # setup, ingest, test, run, docker-build, docker-run
├── requirements.txt             # + rank-bm25, numpy (openai already needed for mock/real switch)
├── .env.example
├── Dockerfile
├── docs/                        # the synthetic "company docs" corpus (committed, small)
│   ├── product-faq.md
│   ├── hr-policy.md
│   └── engineering-runbook.md
├── ingest.py                    # chunks docs/, embeds, builds indices, writes artifacts/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── logging_utils.py         # identical pattern to Week 1 labs (intentional duplication)
│   ├── middleware.py
│   ├── api/
│   │   ├── schemas.py           # AskRequest, AskResponse (with citations)
│   │   └── routes.py            # POST /ask, GET /healthz, GET /readyz
│   ├── domain/
│   │   ├── chunking.py          # chunk_text(text, chunk_size, overlap) -> list[Chunk]
│   │   ├── retrieval.py         # hybrid_search(query, ...) -> list[RetrievedChunk]
│   │   └── rag.py               # build_rag_prompt(question, retrieved) -> str, with citations
│   └── adapters/
│       ├── embeddings.py        # mock (deterministic hash-based) / real (OpenAI) switch
│       ├── llm_client.py        # same mock/real completion pattern as Week 1
│       └── index_store.py       # loads + sha256-verifies the index artifact (mirrors model_store.py)
├── artifacts/
│   ├── index.json                # chunks + dense vectors + bm25 stats, versioned
│   └── manifest.json             # same schema as Week 1's manifest, applied to a retrieval index
└── tests/
```

### Corpus and ingestion

`docs/` holds 3-5 short, deliberately fictional markdown files (a product FAQ, an HR policy
excerpt, an engineering runbook) — small enough to read in full, self-contained (no external
download), and varied enough that hybrid retrieval and citations are demonstrable.

`ingest.py` (run manually via `make ingest`, mirroring Week 1's `make train`):
1. Reads every file in `docs/`, chunks each with `domain/chunking.py` (fixed-size chunks with
   overlap, each chunk tagged with its source filename and a chunk index).
2. Embeds every chunk via `adapters/embeddings.py` (mock or real, same switch logic as the LLM
   client).
3. Builds a BM25 index (`rank_bm25.BM25Okapi`) over the same chunks for keyword scoring.
4. Serializes chunks + dense vectors + BM25 stats into `artifacts/index.json`, and writes
   `artifacts/manifest.json` with the same schema as Week 1's model manifest (`artifact_version`,
   `created_at`, `git_commit`, `sha256`, `python_version`, `key_dependencies`) computed over
   `index.json`.

### Retrieval and answering

- `domain/chunking.py`: pure function, chunk size and overlap as parameters, no I/O.
- `domain/retrieval.py`: `hybrid_search(query, chunks, dense_vectors, bm25_index, embed_fn, k)`
  computes a dense cosine-similarity score and a BM25 score per chunk, normalizes and combines
  them with a fixed weight (e.g. 0.5/0.5), and returns the top-`k` chunks by combined score. This
  combined-score step stands in for a real cross-encoder reranker — documented explicitly as a
  simplification in the concept README, not presented as state-of-the-art.
- `domain/rag.py`: `build_rag_prompt(question, retrieved_chunks)` assembles a prompt that includes
  each retrieved chunk's text tagged with a citation marker (e.g. `[1] (source: hr-policy.md)`),
  caps total included context by a token/character budget (context packaging), and instructs the
  model to cite sources inline. `AskResponse.citations` is populated directly from the
  retrieved-chunk metadata (source file, chunk index, a short snippet) — independent of whatever
  the LLM actually outputs, so citations are always accurate even in mock mode.
- `adapters/embeddings.py`: `EmbeddingClient` with the same `is_mock` shape as Week 1's
  `LlmClient` — a deterministic mock embedding (e.g. a fixed-dimension vector derived from a hash
  of the input text, so identical text always embeds identically and different text produces
  different vectors) requires zero setup; if `OPENAI_API_KEY` is set, real embeddings
  (`text-embedding-3-small`) are used instead, with the same tenacity retry pattern as the LLM
  client.
- `adapters/index_store.py`: loads `artifacts/index.json`, verifies its sha256 against
  `manifest.json` before use (identical integrity pattern and identical fail-fast startup behavior
  to Week 1's `model_store.py` — verified as correct and intentional in Week 1's review, not
  changed here).
- `POST /ask` returns `{question, answer, citations: [{source, chunk_id, snippet}], source}` where
  `source` is `"mock"` or `"llm"` exactly as in Week 1.

### Tooling

- `requirements.txt` adds `rank-bm25` and `numpy` to Week 1's LLM-lab dependency set (`fastapi`,
  `uvicorn`, `pydantic`/`pydantic-settings`, `tenacity`, `openai`, `pytest`/`pytest-cov`, `ruff`),
  all pinned to exact versions per the existing convention.
- `Makefile` gets a `make ingest` target (runs `ingest.py`) alongside the existing
  `setup`/`test`/`run`/`docker-build`/`docker-run` targets.
- CI (`.github/workflows/tests.yml`) gets a third matrix entry, `llm-track-rag-service`, alongside
  the existing two Week 1 labs — installs deps, lints, runs the real committed-artifact integrity
  test (mirroring Week 1's fix #5), then runs pytest with the 80% coverage gate. No live-ingestion
  step in CI (same reasoning as Week 1: tests use their own fixture-built index, not the committed
  one; a dedicated artifact-integrity test covers the committed artifact directly instead).

## Tests (both labs)

Same TDD discipline as Week 1: tests written before implementation for every domain/adapter/api
piece, `pytest-cov` enforcing 80%+ on `domain/`+`api/` (RAG lab) or on the whole small codebase
(experiment-tracking lab, which has no `api/` layer). No live network calls in any test —
mock-mode is the test default for both the LLM client and the embeddings client, exactly as Week 1
established (and as Week 1's final review fixed for the case of an ambient real API key).

## Build order

1. Write the Week 2 concept README + exercises (replacing the "coming soon" stub).
2. `ml-track-experiment-tracking`, TDD-first: `data_validation.py` → `train_with_tracking.py` →
   reproducibility manifest → README.
3. `llm-track-rag-service`, TDD-first: `domain/chunking.py` → `domain/retrieval.py` →
   `domain/rag.py` → `adapters/embeddings.py` → `adapters/index_store.py` →
   `adapters/llm_client.py` (adapted from Week 1's) → corpus + `ingest.py` (run once to produce
   the committed artifact) → API/config/logging/middleware/main → Dockerfile/Makefile/README.
4. Add the CI matrix entry for the new lab.
5. Run `code-reviewer` and `security-reviewer` against both new labs; fix CRITICAL/HIGH findings.
6. Commit, push (repo already exists — no `gh repo create` this time, just `git push`).
7. Final whole-branch review across everything changed since Week 1 shipped, same as before.

## Explicitly out of scope for this pass

- GraphRAG, deep-learning data loaders/distributed training, OCR/PDF parsing — covered
  conceptually in the README only, no hands-on lab (the syllabus itself only demos RAG + citations
  and reproducibility, not every bullet).
- A real cross-encoder reranker — the hybrid combined-score step is an explicit, documented
  simplification.
- Weights & Biases — MLflow only, per the user's selection.
- Weeks 3-6 (still skeleton-only from Week 1's pass).

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Mock embeddings are "too good" (e.g. random vectors) and hybrid retrieval looks meaningless in mock mode | Medium | Use a deterministic hash-based embedding with enough structure that semantically similar mock inputs (e.g. shared words) land closer together than unrelated ones — call this out explicitly as a teaching simplification, not real semantic search |
| `rank-bm25` or another new dependency fails to resolve on some platform | Low | Same "closest patch, same minor version" substitution policy as Week 1 |
| Citations become inaccurate if the LLM ignores retrieved context | Low | Citations are built from retrieval metadata directly, not parsed from the LLM's output — accuracy doesn't depend on the model behaving |
| `mlruns/` (MLflow's local store) accidentally gets committed | Medium | Explicit per-lab `.gitignore` entry, checked in review |
