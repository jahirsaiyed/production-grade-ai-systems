# Design: Production-Grade AI Systems — Self-Study Course & Repo

**Date**: 2026-09-11 (revised)
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

## Attribution & disclaimer

Because the repo is public and reuses a third party's course title, instructor
name, and topic structure, the root `README.md` must open with a clear,
prominent disclaimer, substantially:

> This is an independent, unofficial study companion inspired by the publicly
> published syllabus of ByteByteGo Live's "Build Production Grade AI Systems"
> course. It is not affiliated with, endorsed by, or produced by ByteByteGo or
> the course instructor. No proprietary lecture content, slides, or recordings
> are reproduced here — only the publicly listed topic outline is used as a
> curriculum skeleton; all explanations, exercises, and code are written
> independently.

`docs/course-outline.md` (the extracted syllabus) carries the same disclaimer
plus a link to the original public course page.

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

### Root `README.md` contents

- The disclaimer above (first thing in the file).
- One-paragraph course overview and who it's for.
- Prerequisites (Python 3.11+, git, basic pip/venv — mirrors the source
  syllabus).
- Expected time commitment (4–7 hrs/week, matching the source syllabus's FAQ,
  since it's a realistic number worth keeping).
- Quickstart: clone → `cd modules/week-01.../labs/<track>` → `make setup`.
- A table of contents linking every week's README (skeleton weeks marked
  "coming soon").

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

## Exercises approach

`exercises.md` gives hints and acceptance criteria, not full solutions —
forcing active recall is the point of a self-study course. Where a worked
answer materially helps (e.g. a tricky retry/backoff calculation), it can link
to the relevant section of the lab's own code as the reference implementation,
rather than a separate answer key.

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
├── Makefile                 # setup, test, run, docker-build, docker-run targets
├── app/
│   ├── main.py              # FastAPI app wiring
│   ├── api/                  # routes + request/response schemas (pydantic)
│   ├── domain/                # business logic, framework-agnostic
│   └── adapters/              # model loading / LLM client, external calls
├── artifacts/
│   └── manifest.json          # see "Artifact manifest schema" below
├── tests/                     # pytest: domain logic + API contract tests
├── Dockerfile                 # commented line-by-line (audience is Docker-beginner)
├── requirements.txt          # pinned exact versions (`pip freeze` style)
└── .env.example
```

### Developer ergonomics

Each lab ships a small `Makefile` so the hands-on guide can say `make setup`,
`make test`, `make run`, `make docker-build`, `make docker-run` instead of
retyping long commands — lowers friction for the stated beginner-Docker
audience.

### Artifact manifest schema

`manifest.json` foreshadows the Week 2 "reproducibility checklist" (commit,
data digest, env lock, artifact link) so the idea isn't introduced cold later:

```json
{
  "artifact_version": "0.1.0",
  "created_at": "2026-09-11T00:00:00Z",
  "git_commit": "<sha of the commit that produced this artifact>",
  "sha256": "<hash of the artifact file>",
  "python_version": "3.11.x",
  "key_dependencies": {"scikit-learn": "1.5.x"}
}
```

The adapter layer verifies the artifact's sha256 against this manifest at
startup and fails fast (readiness probe reports not-ready) on a mismatch.

### Security & logging hygiene

- Structured logs never include the raw `OPENAI_API_KEY`/secrets, and request
  bodies are logged with sensitive/free-text fields excluded or truncated —
  this is worth establishing as a habit now, even though full PII
  redaction is a Week 5 topic.
- Both `/score` and `/ask` validate and cap input size (pydantic field
  constraints) before it reaches domain/adapter code — request bodies are
  untrusted input.
- `requirements.txt` pins exact versions for reproducibility and predictable
  `pip install`.

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
- `pytest` + `pytest-cov` for tests; `ruff` for lint/format.
- Test coverage target: 80%+ on each lab's `domain/` and `api/` code (the
  actual business logic), measured by `pytest-cov`. `adapters/` (thin
  I/O wrappers) are covered via the API contract tests, not chased for
  coverage percentage on their own.
- Commit style: `<type>: <description>` (feat/fix/docs/chore).
- `.gitignore`: `.venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`,
  `.ruff_cache/`, `.coverage`, `.env` (the small synthetic model artifact and
  its manifest ARE committed — intentionally, since they're tiny and are the
  point of the versioning lesson).

## CI

`.github/workflows/tests.yml` uses a matrix over the two lab directories so
one job definition covers both instead of duplicating YAML:

- Matrix: `{ lab: [ml-track-fraud-detection, llm-track-qa-service] }`
- Per matrix entry: checkout → set up Python 3.11 → `pip install -r
  modules/week-01-prototype-to-production/labs/${{ matrix.lab }}/requirements.txt`
  → `ruff check` → `pytest --cov` with a minimum coverage threshold (fail
  under 80% on `domain/` + `api/`).
- Triggers: push and pull_request.

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
| Repo could look like it's redistributing ByteByteGo's paid course | Low but high-impact if it happens | Prominent disclaimer in README + course-outline.md; no proprietary content reproduced, only the public topic list |
| Secrets or raw user input leak into structured logs | Medium if not deliberate | Explicit logging-hygiene rule: exclude/truncate sensitive fields, never log the API key |
