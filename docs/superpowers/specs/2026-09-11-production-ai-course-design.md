# Design: Production-Grade AI Systems — Self-Study Course & Repo

**Date**: 2026-09-11
**Status**: Approved

## Background

The source material is the ByteByteGo Live marketing/syllabus page for their 6-week
"Build Production Grade AI Systems" cohort (instructor: Tanya Roosta, Director of AI
@ AMD). It lists topics per week but contains no lecture content, code, or
explanations — it's an outline, not a course. The extracted outline is preserved at
`docs/course-outline.md`.

The goal of this project is to use that outline as a curriculum skeleton and author
an original, self-contained learning course: detailed concept explanations plus
runnable hands-on labs, published as a public GitHub repo.

## Scope for this pass

- Full repo skeleton for all 6 weeks (structure + topic-list README stubs for
  weeks 2–6).
- Week 1 ("From Prototype to Production") fully built out: concept explanations,
  exercises, and two complete hands-on labs (one per track).
- Weeks 2–6 detailed content is explicitly **out of scope** for this pass and will
  be tackled as follow-up per-module passes.
- No deployed docs site, no managed cloud infra — everything runs locally via
  `venv` + Docker.

## Repo structure

```
production-grade-ai-systems/
├── README.md                          # course overview, how to use the repo, prerequisites
├── LICENSE (MIT)
├── .gitignore
├── .github/workflows/tests.yml        # CI: run pytest for both Week 1 labs
├── docs/
│   ├── course-outline.md              # full 6-week syllabus, extracted from the source PDF
│   └── superpowers/specs/             # design docs (this file)
└── modules/
    ├── week-01-prototype-to-production/     ← FULLY BUILT
    │   ├── README.md                  # concept explanations for every Week 1 topic
    │   ├── exercises.md               # practice exercises tied to the concepts
    │   └── labs/
    │       ├── ml-track-fraud-detection/
    │       └── llm-track-qa-service/
    ├── week-02-data-and-model-pipelines/     ← SKELETON
    ├── week-03-serving-inference-optimization/  ← SKELETON
    ├── week-04-evaluation-and-monitoring/    ← SKELETON
    ├── week-05-security-and-governance/      ← SKELETON
    └── week-06-scaling/                      ← SKELETON
```

Concept explanations for a week live in one `README.md` shared across both tracks
(the architecture ideas are identical); only the hands-on `labs/` split by track,
since that's where the code differs. Skeleton weeks get a README with the topic
list from the syllabus and a pointer to `docs/course-outline.md`, so the full
6-week shape is visible in the repo from day one.

## Week 1 concept README covers

- Prototype vs. production systems; training/serving skew
- AI lifecycle: data → packaging → serving → monitoring; handoff artifacts
- Batch vs. online inference; monolith vs. microservices vs. event-driven
- Production building blocks (API gateways, feature stores, vector DBs, model
  registries)
- Reliability primitives: retries, timeouts, queues, graceful degradation
- REST vs. gRPC; streaming responses
- Layered service architecture (API / domain / model adapters)
- Configuration management, secrets separation, request IDs, retry budgets
- Async programming for I/O-bound LLM/retrieval calls
- Structured logging; health and readiness endpoints
- Model serialization: versioned artifacts with sha256 metadata
- Packaging path: notebook → artifact → API → container image
- Docker fundamentals; Kubernetes overview (conceptual only, no cluster this week)

Both tracks — traditional ML (fraud detection, recommenders) and LLM applications
(RAG, agents) — are introduced here since the syllabus applies every production
pattern to both throughout the course.

## Week 1 hands-on labs

Both labs follow the same narrative: **notebook script → layered service →
versioned artifact → container image**, each with retries, health/readiness
checks, structured logging, and config/secrets separation. Both are built
test-first (TDD): tests for domain logic and API contracts are written before
the implementation.

Shared lab layout:

```
labs/<track>/
├── README.md               # walkthrough: run the prototype, then the service, then Docker
├── prototype.py            # the messy "before": a notebook-style script, no structure
├── app/
│   ├── main.py              # FastAPI app wiring
│   ├── api/                  # routes + request/response schemas (pydantic)
│   ├── domain/                # business logic, framework-agnostic
│   └── adapters/              # model loading / LLM client, external calls
├── artifacts/
│   └── manifest.json          # version, sha256, created_at for the packaged model/prompt
├── tests/                     # pytest: domain logic + API contract tests
├── Dockerfile                 # commented line-by-line (audience is Docker-beginner)
├── requirements.txt
└── .env.example
```

### ML track — fraud detection

- Synthetic dataset via `sklearn.datasets.make_classification` — no external
  download, fully self-contained and reproducible.
- `prototype.py`: a deliberately unstructured "before" script that trains a
  classifier inline.
- Production version: `domain/scoring.py` (pure, unit-testable scoring logic),
  `adapters/model_store.py` (loads the versioned `joblib` artifact and verifies
  its sha256 against `manifest.json`), `api/routes.py` exposing `POST /score`.
- A simulated flaky "feature store" call in the adapter layer motivates real use
  of retries/backoff (`tenacity`) rather than a contrived example.

### LLM track — Q&A service

- Deliberately *not* RAG yet (that's Week 2 material) — a clean LLM-backed
  service, matching what Week 1 actually teaches.
- `adapters/llm_client.py` wraps an OpenAI-compatible call with retries on
  transient errors.
- Runs in **mock mode by default** (deterministic canned responses) so the lab
  works with zero API key; if `OPENAI_API_KEY` is set in `.env`, it calls the
  real API. This keeps the hands-on guide usable without requiring a paid key.
- `api/routes.py` exposes `POST /ask`.

### Both labs get

- `/healthz` (liveness) and `/readyz` (readiness — confirms the model/client is
  loaded).
- JSON structured logs carrying a per-request ID (middleware + contextvars).
- `pydantic-settings` for config; `.env.example` committed, `.env` gitignored.

## Tooling & conventions

- Python 3.11+, plain `venv` + `pip install -r requirements.txt` (matches the
  syllabus's stated prerequisites — no Poetry/uv, to keep the barrier low).
- `pytest` for tests, `ruff` for lint/format.
- Commit style: `<type>: <description>` (feat/fix/docs/chore).
- CI: `.github/workflows/tests.yml` runs `pytest` for both Week 1 labs on push
  and pull request (install each lab's `requirements.txt`, run its `tests/`).

## Build order

1. Scaffold the full directory tree (all 6 week folders, skeleton READMEs for
   weeks 2–6, `docs/course-outline.md`).
2. Write the Week 1 concept `README.md` and `exercises.md`.
3. Build both labs, TDD-first: failing tests for domain logic + API contracts
   (RED) → implement `domain/` → `adapters/` → `api/` → `main.py` until green →
   add Dockerfile, manifest/versioning, structured logging, retries.
4. Add the GitHub Actions workflow.
5. Run `code-reviewer` and `security-reviewer` agents against both labs (API
   endpoints + external calls trigger the mandatory security-review rule);
   address CRITICAL/HIGH findings.
6. Commit, then `gh repo create production-grade-ai-systems --public
   --source=. --push`.
7. Report the repo URL back.

## Explicitly out of scope for this pass

- Full detailed content for Weeks 2–6 (skeleton only).
- Deployed docs site (e.g. GitHub Pages / MkDocs).
- Real cloud deployment (Kubernetes cluster, managed ML platforms) — Week 1
  covers Kubernetes and gRPC/streaming only conceptually.
- Non-mock LLM evaluation — no real API key is assumed to exist.

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| LLM lab is untestable without an API key | High if not handled | Default to mock mode; real key is optional |
| Docker not installed/runnable in this environment | Medium | Provide Dockerfile + instructions; note local `docker build`/`run` may need to be verified by the user if the sandbox can't run Docker |
| Scope creep into Weeks 2–6 during this pass | Medium | Explicitly time-box to skeleton-only for those weeks per this spec |
