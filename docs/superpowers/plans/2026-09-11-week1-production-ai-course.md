# Week 1 Production AI Course Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold the full 6-week `production-grade-ai-systems` repo and fully build Week 1 ("From Prototype to Production"): concept README, exercises, and two TDD-built, Dockerized FastAPI labs (fraud-detection ML track, Q&A LLM track), then push the repo to GitHub.

**Architecture:** Each lab is an independently runnable FastAPI service with a layered `api/ domain/ adapters/` structure, a versioned+sha256-verified artifact, structured JSON logging with request IDs, and tenacity-based retries. Weeks 2-6 get skeleton READMEs only.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, pydantic v2 / pydantic-settings, scikit-learn + joblib (ML track), openai SDK (LLM track, mock-mode default), tenacity, pytest + pytest-cov, ruff, Docker, GitHub Actions.

## Global Constraints

- Python 3.11+, plain `venv` + `pip install -r requirements.txt` — no Poetry/uv.
- `requirements.txt` files pin exact versions.
- Test coverage target: 80%+ on each lab's `domain/` and `api/` code, via `pytest-cov`.
- Lint: `ruff check` must pass with zero errors.
- Commit message format: `<type>: <description>` (feat/fix/docs/chore/test/ci), each ending with the attribution line `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` per this session's convention.
- The root `README.md` and `docs/course-outline.md` must both open with the attribution/disclaimer text specified in Task 1.
- The LLM lab must run in mock mode with zero API key by default.
- Both labs must verify their model/prompt artifact's sha256 against `manifest.json` at startup and expose `/healthz` and `/readyz`.
- Structured logs are JSON, carry a per-request ID, and never include secrets or raw free-text request bodies.
- `.gitignore` must exclude `.venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.ruff_cache/`, `.coverage`, `.env` — but artifact files (`model.joblib`, `manifest.json`) ARE committed.
- Working directory for all commands below is the repo root: `D:\Learning\Build Production Grade AI Systems _ ByteByteGo Live\production-grade-ai-systems`. Git is already initialized there with two commits (the design spec and its revision).

---

### Task 1: Repo scaffold — root files, course outline, skeleton weeks

**Files:**
- Create: `README.md`
- Create: `LICENSE`
- Create: `.gitignore`
- Create: `docs/course-outline.md`
- Create: `modules/week-02-data-and-model-pipelines/README.md`
- Create: `modules/week-03-serving-inference-optimization/README.md`
- Create: `modules/week-04-evaluation-and-monitoring/README.md`
- Create: `modules/week-05-security-and-governance/README.md`
- Create: `modules/week-06-scaling/README.md`

**Interfaces:**
- Produces: the disclaimer text block (below), reused verbatim in Task 2's Week 1 README front matter is NOT required — only root README and course-outline.md need it per Global Constraints.

- [ ] **Step 1: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.coverage
.env
```

- [ ] **Step 2: Write `LICENSE`** (MIT, copyright the current year and the repo owner)

```
MIT License

Copyright (c) 2026 jahirsaiyed

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 3: Write `docs/course-outline.md`**

```markdown
# Course Outline (Source Syllabus)

> **Disclaimer:** This page is an independent, unofficial transcription of the
> publicly published syllabus for ByteByteGo Live's "Build Production Grade AI
> Systems" course
> (https://live.bytebytego.com/courses/production-ai), preserved here as the
> curriculum skeleton for this repo. This repo is not affiliated with,
> endorsed by, or produced by ByteByteGo or the course instructor. No
> proprietary lecture content, slides, or recordings are reproduced — only the
> public topic list.

## Week 1 — From Prototype to Production
**Production Architecture and the AI Lifecycle**
- Two tracks throughout the course: traditional ML systems (fraud detection,
  recommenders) and LLM applications (RAG, agents)
- Prototype vs production systems; training/serving skew
- AI lifecycle from data to packaging, serving and monitoring; handoff artifacts
- Batch vs online inference; monolith vs microservices vs event-driven
- Production building blocks (API gateways, feature stores, vector DBs, model registries)
- Reliability primitives: retries, timeouts, queues, graceful degradation
- REST vs gRPC; streaming responses

**Production Services, Packaging, and Containers**
- Layered service architecture (API / domain / model adapters)
- Configuration management, secrets separation, request IDs, retry budgets
- Async programming for I/O-bound LLM and retrieval calls
- Structured logging; health and readiness endpoints
- Model serialization: versioned artifacts with sha256 metadata
- Packaging path: notebook -> artifact -> API -> container image
- Docker fundamentals; Kubernetes overview

*Live demo: turn a Python script into a versioned, containerized AI service.*

**Status in this repo: fully built — see `modules/week-01-prototype-to-production/`.**

## Week 2 — Data and Model Pipelines
**Data Pipelines and Enterprise Retrieval**
- Ingestion patterns (batch vs streaming, ETL vs ELT)
- Data validation gates and schema enforcement
- Data lineage and versioning
- Training/serving parity; leakage detection
- Deep-learning data paths (loaders, batching, augmentation, distributed training)
- Document parsing (OCR, PDF, HTML) and chunking strategies
- Vector DB operations (indexing, upserts, metadata filters)
- Context packaging (ordering, token caps, citations)
- Hybrid retrieval: BM25 + dense embeddings + reranking; GraphRAG

**Model Development and Reproducibility**
- Experiment tracking (MLflow, Weights & Biases)
- Model registry stages
- Hyperparameter tuning
- Reproducibility checklist (commit, data digest, env lock, artifact link)
- Adaptation decision framework: prompting vs RAG vs PEFT (LoRA) vs full fine-tuning

*Live demo: build a RAG pipeline that answers from company docs with citations.*

**Status in this repo: skeleton only — see `modules/week-02-data-and-model-pipelines/`.**

## Week 3 — Serving, Inference, and Optimization
**Serving Architecture**
- Dedicated inference services
- Serving frameworks (Triton, TorchServe, Ray Serve, vLLM, SGLang)
- Prefill vs decode; memory-bound vs compute-bound inference
- Topologies: single model, router/gateway, cascades with fallbacks
- Self-host vs managed APIs (rate limits, data residency, unit economics)

**Optimization Levers**
- Profiling and baselines like p50/p95 latency
- Semantic caching, dynamic vs continuous batching, autoscaling signals
- Quantization (post-training, quantization-aware, INT8/INT4)
- Mixed precision (FP16/BF16)
- Pruning (structured, unstructured); distillation
- Runtimes and kernels (ONNX Runtime, TensorRT, OpenVINO)
- KV Cache, FlashAttention and speculative decoding
- Packaging (checkpoint/ONNX -> service image, gRPC contracts, etc.)

*Live demo: make the assistant faster and cheaper, proven with benchmarks.*

**Status in this repo: skeleton only — see `modules/week-03-serving-inference-optimization/`.**

## Week 4 — Evaluation and Monitoring
**Evaluating Production AI**
- Offline vs online evaluation
- Classification metrics (precision, recall, F1, ROC-AUC vs PR-AUC)
- Threshold selection with cost-based sweeps
- Calibration (reliability diagrams, Brier score, Platt scaling, isotonic regression)
- Eval harnesses (RAGAS, DeepEval), LLM-as-judge calibration
- Multi-turn continuity checks
- Red-team prompts in regression packs

**Observability and Drift**
- Tracing and metrics (OpenTelemetry, Prometheus, Grafana)
- Drift classes: data, concept, embedding, prompt
- Outcome-based alerting
- Continuous training vs CI/CD
- Progressive delivery: shadow -> canary -> full, and rollback gates
- Prompt registries

*Live demo: build eval dashboards and a gate that blocks bad releases.*

**Status in this repo: skeleton only — see `modules/week-04-evaluation-and-monitoring/`.**

## Week 5 — Security and Governance
**Threats, Authentication, and Privacy**
- Threat landscape; OWASP LLM Top 10
- API authentication (tokens, JWT)
- PII detection and redaction (prompts, logs, responses)
- Encryption in transit and at rest
- Vector store protection

**Guardrails and Prompt Registry**
- Layered defense: input/output filters, structured-output contracts, audit logging
- Groundedness checks
- Model cards, datasheets, fairness audits
- Explainability and citations as RAG explanations
- Regulatory context (EU AI Act, NIST AI RMF)
- Prompt/policy registry: versioning, review, offline checks, release, rollback

*Live demo: lock down the assistant with auth, role-based retrieval, and audited policy releases.*

**Status in this repo: skeleton only — see `modules/week-05-security-and-governance/`.**

## Week 6 — Scaling
**Platform Architecture and Release Engineering**
- Kubernetes (EKS, GKE, AKS) and ECS; managed ML platforms (Vertex AI, SageMaker, Azure ML)
- Workflow orchestration (Airflow, Prefect, Dagster, Kubeflow, Ray)
- GPU scheduling, node pools, autoscaling, capacity planning
- Canary and blue-green rollouts; auto-rollback gates
- SLOs (latency, error rate, groundedness)
- Incident response and on-call basics

**Cost Tradeoffs and Case Studies**
- Planner/worker patterns
- HITL approval gates; agent tracing; when not to use agents
- Case studies: recommendation and search, fintech risk scoring, consumer assistants

*Live demo: deploy the full platform with a canary release, a cost report, and a human-supervised agent.*

**Status in this repo: skeleton only — see `modules/week-06-scaling/`.**
```

- [ ] **Step 4: Write skeleton READMEs for weeks 2-6**

`modules/week-02-data-and-model-pipelines/README.md`:
```markdown
# Week 2 — Data and Model Pipelines

**Status: coming soon.**

This module will cover data pipelines, enterprise retrieval/RAG, and
reproducible model development. See the full topic list in
[`docs/course-outline.md`](../../docs/course-outline.md#week-2--data-and-model-pipelines).
```

`modules/week-03-serving-inference-optimization/README.md`:
```markdown
# Week 3 — Serving, Inference, and Optimization

**Status: coming soon.**

This module will cover serving architectures and inference optimization
levers. See the full topic list in
[`docs/course-outline.md`](../../docs/course-outline.md#week-3--serving-inference-and-optimization).
```

`modules/week-04-evaluation-and-monitoring/README.md`:
```markdown
# Week 4 — Evaluation and Monitoring

**Status: coming soon.**

This module will cover offline/online evaluation and production observability.
See the full topic list in
[`docs/course-outline.md`](../../docs/course-outline.md#week-4--evaluation-and-monitoring).
```

`modules/week-05-security-and-governance/README.md`:
```markdown
# Week 5 — Security and Governance

**Status: coming soon.**

This module will cover AI security threats, guardrails, and governance. See
the full topic list in
[`docs/course-outline.md`](../../docs/course-outline.md#week-5--security-and-governance).
```

`modules/week-06-scaling/README.md`:
```markdown
# Week 6 — Scaling

**Status: coming soon.**

This module will cover platform architecture, release engineering, and cost
tradeoffs at scale. See the full topic list in
[`docs/course-outline.md`](../../docs/course-outline.md#week-6--scaling).
```

- [ ] **Step 5: Write root `README.md`**

```markdown
# Production-Grade AI Systems — Self-Study Course

> **Disclaimer:** This is an independent, unofficial study companion inspired
> by the publicly published syllabus of ByteByteGo Live's "Build Production
> Grade AI Systems" course
> (https://live.bytebytego.com/courses/production-ai). It is not affiliated
> with, endorsed by, or produced by ByteByteGo or the course instructor. No
> proprietary lecture content, slides, or recordings are reproduced here —
> only the publicly listed topic outline is used as a curriculum skeleton;
> all explanations, exercises, and code in this repo are written
> independently.

A hands-on, self-study course on shipping AI systems to production: turning
prototypes into layered services, building enterprise RAG pipelines, cutting
inference cost and latency, evaluating and monitoring production AI, and
securing and scaling it.

## Who this is for

Engineers with basic Python (creating a virtual environment, installing
packages with pip) who want a structured, hands-on path from "it works in a
notebook" to "it's a production service." No extensive ML background required.

## Prerequisites

- Python 3.11+
- Git
- Docker (for the containerization labs)

## Time commitment

Roughly 4-7 hours per week, self-paced — work through a module whenever fits
your schedule.

## Course map

| Week | Topic | Status |
|---|---|---|
| 1 | [From Prototype to Production](modules/week-01-prototype-to-production/README.md) | Fully built |
| 2 | [Data and Model Pipelines](modules/week-02-data-and-model-pipelines/README.md) | Coming soon |
| 3 | [Serving, Inference, and Optimization](modules/week-03-serving-inference-optimization/README.md) | Coming soon |
| 4 | [Evaluation and Monitoring](modules/week-04-evaluation-and-monitoring/README.md) | Coming soon |
| 5 | [Security and Governance](modules/week-05-security-and-governance/README.md) | Coming soon |
| 6 | [Scaling](modules/week-06-scaling/README.md) | Coming soon |

The full source syllabus is preserved at [`docs/course-outline.md`](docs/course-outline.md).

## Quickstart (Week 1)

```bash
git clone <this-repo-url>
cd production-grade-ai-systems/modules/week-01-prototype-to-production/labs/ml-track-fraud-detection
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows:     .venv\Scripts\activate
make setup
make train   # generates the versioned model artifact
make test
make run
```

Repeat inside `labs/llm-track-qa-service` for the LLM track — it works with
zero API key (mock mode) out of the box.

## License

MIT — see [`LICENSE`](LICENSE).
```

- [ ] **Step 6: Commit**

```bash
git add README.md LICENSE .gitignore docs/course-outline.md modules/week-02-data-and-model-pipelines modules/week-03-serving-inference-optimization modules/week-04-evaluation-and-monitoring modules/week-05-security-and-governance modules/week-06-scaling
git commit -m "$(cat <<'EOF'
docs: scaffold repo with course outline and skeleton weeks 2-6

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Week 1 concept README and exercises

**Files:**
- Create: `modules/week-01-prototype-to-production/README.md`
- Create: `modules/week-01-prototype-to-production/exercises.md`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: links to `labs/ml-track-fraud-detection/` and `labs/llm-track-qa-service/` (built in later tasks — link them even though the directories don't exist until Task 3+; they will by the time this task's commit is pushed).

- [ ] **Step 1: Write `modules/week-01-prototype-to-production/README.md`**

Write a concept README covering each topic below as its own `##` section.
For each, include: a 2-4 sentence explanation in plain language, why it
matters in production (not just theory), and one concrete example grounded in
the two hands-on labs (`labs/ml-track-fraud-detection/` and
`labs/llm-track-qa-service/`) so the reader can immediately see the concept in
running code. Cover, in this order:

1. **Prototype vs. production systems; training/serving skew** — contrast a
   notebook script against a service; explain skew as "the features/inputs
   your model sees at inference time differing from what it saw in training,"
   with a concrete example (e.g., a feature computed differently in training
   code vs. serving code).
2. **The AI lifecycle: data -> packaging -> serving -> monitoring** — an
   ASCII diagram of the stages, and where "handoff artifacts" (a trained
   model + its manifest) sit between stages.
3. **Batch vs. online inference; monolith vs. microservices vs.
   event-driven** — a short comparison table (latency, throughput,
   complexity, when to choose each); note both Week 1 labs are online
   inference behind a synchronous API.
4. **Production building blocks** (API gateways, feature stores, vector DBs,
   model registries) — one sentence per building block, noting that
   `adapters/feature_store.py` in the fraud-detection lab is a deliberately
   simplified stand-in for a real feature store.
5. **Reliability primitives**: retries, timeouts, queues, graceful
   degradation — explain each, then point to the concrete retry
   implementation in `adapters/feature_store.py` (ML lab) and
   `adapters/llm_client.py` (LLM lab), and the graceful-degradation fallback
   answer in the LLM lab's `/ask` route.
6. **REST vs. gRPC; streaming responses** — explain the tradeoff (gRPC:
   binary, faster, strongly typed, harder to debug by hand; REST: simpler,
   human-readable, what both labs use); note streaming responses are relevant
   for LLM token-by-token output but out of scope for this week's lab.
7. **Layered service architecture (API / domain / model adapters)** — an
   ASCII diagram of the three layers and the dependency direction (API
   depends on domain, domain has no framework dependencies, adapters isolate
   I/O), pointing at the actual `app/api/`, `app/domain/`, `app/adapters/`
   folders in both labs.
8. **Configuration management, secrets separation, request IDs, retry
   budgets** — explain `.env` + `pydantic-settings`, why `.env` is gitignored
   but `.env.example` is committed, and how the request-ID middleware works.
9. **Async programming for I/O-bound calls** — explain why LLM/network calls
   benefit from `async def` routes even though this week's labs keep the
   adapters synchronous for simplicity, and note this as a natural extension
   exercise.
10. **Structured logging; health and readiness endpoints** — explain the
    difference between `/healthz` (is the process alive) and `/readyz` (is it
    able to serve traffic, e.g. model loaded), and show a sample JSON log line
    from either lab.
11. **Model serialization: versioned artifacts with sha256 metadata** —
    explain why a hash check at load time matters (catches corrupted or
    mismatched deploys), referencing `manifest.json` and
    `adapters/model_store.py`.
12. **Packaging path: notebook -> artifact -> API -> container image** — walk
    through the exact transformation: `prototype.py` -> `train_model.py`
    produces `artifacts/model.joblib` + `manifest.json` -> `app/` wraps it in
    a FastAPI service -> `Dockerfile` packages it as an image.
13. **Docker fundamentals; Kubernetes overview** — explain images vs.
    containers, what each `Dockerfile` instruction does (point to the
    labs' commented Dockerfiles), and give a 3-4 sentence conceptual overview
    of what Kubernetes adds on top of a single container (scheduling,
    scaling, self-healing) without requiring a cluster this week.

Open the file with an `# Week 1: From Prototype to Production` heading and a
short intro paragraph naming the two tracks (ML: fraud detection; LLM: Q&A
service) and linking to `labs/ml-track-fraud-detection/README.md` and
`labs/llm-track-qa-service/README.md`. Close with a "## Hands-on labs"
section linking both lab READMEs, and a "## Exercises" section linking
`exercises.md`.

- [ ] **Step 2: Write `modules/week-01-prototype-to-production/exercises.md`**

```markdown
# Week 1 Exercises

Hints and acceptance criteria only — no answer key. If you get stuck, the
reference implementation is the lab's own code.

## Exercise 1: Break the artifact hash on purpose

Edit one byte of `labs/ml-track-fraud-detection/artifacts/model.joblib` (then
revert it) and start the service. **Acceptance criteria:** the service fails
to start (or `/readyz` reports not-ready) with a clear `ArtifactIntegrityError`
message, not a silent wrong prediction.

## Exercise 2: Make the flaky feature store flakier

Raise `feature_store_failure_rate` in `.env` for the fraud-detection lab to
`0.9` and call `/score` several times. **Acceptance criteria:** you can
observe (via logs or by adding a print) that `fetch_features` is retried up
to 3 times before giving up, and that a request occasionally still fails after
exhausting retries — explain in your own words why that's the correct
tradeoff versus retrying forever.

## Exercise 3: Add a request-size guardrail

The `/ask` endpoint caps question length, and `/score`'s `transaction_id`
and `merchant_category` are length-capped too — but `amount` has no upper
bound (only `ge=0`). Add a reasonable upper bound (e.g. `le=1_000_000`) and
write a test proving an over-limit value is rejected with a 422.
**Acceptance criteria:** a new passing test in `tests/`, and `make test`
still shows 80%+ coverage on `domain/` and `api/`.

## Exercise 4: Turn on the real LLM

Get an OpenAI API key, set `OPENAI_API_KEY` in `.env` for the LLM lab, restart
the service, and call `/ask`. **Acceptance criteria:** the response's
`"source"` field changes from `"mock"` to `"llm"`, and you can explain why the
mock/real switch required no code change — only a config change.

## Exercise 5: Trace a request end-to-end

Send a request with a custom `X-Request-ID` header to either service and find
that same ID in the structured JSON log output. **Acceptance criteria:** you
can point to the exact line of middleware code that makes this work.
```

- [ ] **Step 3: Commit**

```bash
git add modules/week-01-prototype-to-production/README.md modules/week-01-prototype-to-production/exercises.md
git commit -m "$(cat <<'EOF'
docs: add Week 1 concept README and exercises

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: ML lab — domain scoring logic (TDD)

**Files:**
- Create: `modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/requirements.txt`
- Create: `modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/app/__init__.py` (empty)
- Create: `modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/app/domain/__init__.py` (empty)
- Create: `modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/app/domain/scoring.py`
- Test: `modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/tests/__init__.py` (empty)
- Test: `modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/tests/test_scoring.py`

**Interfaces:**
- Produces: `score_transaction(feature_vector: list[float], model) -> ScoreResult`, `ScoreResult(fraud_probability: float, is_fraud: bool)`, `FRAUD_THRESHOLD: float = 0.5`. Consumed by Task 7's `api/routes.py`.

All commands in this task run from
`modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/`.

- [ ] **Step 1: Write `requirements.txt`**

```
fastapi==0.115.5
uvicorn[standard]==0.32.1
pydantic==2.10.3
pydantic-settings==2.6.1
scikit-learn==1.5.2
joblib==1.4.2
tenacity==9.0.0
pytest==8.3.4
pytest-cov==6.0.0
httpx==0.28.1
ruff==0.8.2
```

- [ ] **Step 2: Create the venv and install dependencies**

```bash
python -m venv .venv
```
Windows: `.venv\Scripts\activate` — macOS/Linux: `source .venv/bin/activate`, then:
```bash
pip install -r requirements.txt
```
Expected: all packages install without error. If an exact pin is unavailable
for your platform, install the closest available patch version of that same
minor version.

- [ ] **Step 3: Write the failing test**

`tests/test_scoring.py`:
```python
from app.domain.scoring import score_transaction


class _StubModel:
    def __init__(self, probability: float):
        self._probability = probability

    def predict_proba(self, X):
        return [[1 - self._probability, self._probability]]


def test_score_transaction_flags_high_probability_as_fraud():
    model = _StubModel(probability=0.9)
    result = score_transaction([1.0, 2.0, 3.0], model)
    assert result.is_fraud is True
    assert result.fraud_probability == 0.9


def test_score_transaction_does_not_flag_low_probability():
    model = _StubModel(probability=0.1)
    result = score_transaction([1.0, 2.0, 3.0], model)
    assert result.is_fraud is False
    assert result.fraud_probability == 0.1
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.domain.scoring'` (or similar import error).

- [ ] **Step 5: Write minimal implementation**

`app/domain/scoring.py`:
```python
from dataclasses import dataclass

FRAUD_THRESHOLD = 0.5


@dataclass(frozen=True)
class ScoreResult:
    fraud_probability: float
    is_fraud: bool


def score_transaction(feature_vector: list[float], model) -> ScoreResult:
    probability = float(model.predict_proba([feature_vector])[0][1])
    return ScoreResult(
        fraud_probability=probability,
        is_fraud=probability >= FRAUD_THRESHOLD,
    )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_scoring.py -v`
Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
cd "modules/week-01-prototype-to-production/labs/ml-track-fraud-detection" 2>/dev/null || true
git add modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/requirements.txt modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/app modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/tests
git commit -m "$(cat <<'EOF'
feat: add fraud-detection domain scoring logic

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

(Run `git add`/`git commit` from the repo root, as shown, regardless of which directory your shell is in for the `pytest`/`pip` steps.)

---

### Task 4: ML lab — model_store adapter with sha256 verification (TDD)

**Files:**
- Create: `modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/app/adapters/__init__.py` (empty)
- Create: `modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/app/adapters/model_store.py`
- Test: `modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/tests/test_model_store.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `load_model(artifact_dir: Path) -> LoadedModel`, `LoadedModel(model: object, manifest: dict)`, `ArtifactIntegrityError`, `_sha256_of(path: Path) -> str`. Consumed by Task 7 (`main.py` lifespan).

All commands run from `modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/` with the venv active.

- [ ] **Step 1: Write the failing test**

`tests/test_model_store.py`:
```python
import json

import joblib
import pytest

from app.adapters.model_store import ArtifactIntegrityError, _sha256_of, load_model


class _DummyModel:
    def predict_proba(self, X):
        return [[0.5, 0.5]]


def _write_artifact(tmp_path):
    model_path = tmp_path / "model.joblib"
    joblib.dump(_DummyModel(), model_path)
    manifest = {
        "artifact_version": "0.1.0",
        "sha256": _sha256_of(model_path),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_load_model_succeeds_when_hash_matches(tmp_path):
    artifact_dir = _write_artifact(tmp_path)
    loaded = load_model(artifact_dir)
    assert loaded.manifest["artifact_version"] == "0.1.0"


def test_load_model_raises_when_hash_mismatches(tmp_path):
    artifact_dir = _write_artifact(tmp_path)
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ArtifactIntegrityError):
        load_model(artifact_dir)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.adapters.model_store'`.

- [ ] **Step 3: Write minimal implementation**

`app/adapters/model_store.py`:
```python
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib


class ArtifactIntegrityError(RuntimeError):
    """Raised when the model artifact does not match its manifest."""


@dataclass(frozen=True)
class LoadedModel:
    model: object
    manifest: dict


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def load_model(artifact_dir: Path) -> LoadedModel:
    manifest_path = artifact_dir / "manifest.json"
    model_path = artifact_dir / "model.joblib"
    manifest = json.loads(manifest_path.read_text())

    actual_sha256 = _sha256_of(model_path)
    if actual_sha256 != manifest["sha256"]:
        raise ArtifactIntegrityError(
            f"model.joblib sha256 {actual_sha256} does not match "
            f"manifest sha256 {manifest['sha256']}"
        )

    model = joblib.load(model_path)
    return LoadedModel(model=model, manifest=manifest)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model_store.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/app/adapters modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/tests/test_model_store.py
git commit -m "$(cat <<'EOF'
feat: add model_store adapter with sha256 artifact verification

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: ML lab — flaky feature_store adapter with retries (TDD)

**Files:**
- Create: `modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/app/adapters/feature_store.py`
- Test: `modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/tests/test_feature_store.py`

**Interfaces:**
- Produces: `fetch_features(transaction_id: str, failure_rate: float = 0.0) -> list[float]` (returns a 6-element list — must match the 6-feature model Task 6 trains), `FeatureStoreUnavailable`. Consumed by Task 7 (`api/routes.py`).

- [ ] **Step 1: Write the failing test**

`tests/test_feature_store.py`:
```python
import pytest

from app.adapters.feature_store import FeatureStoreUnavailable, fetch_features


def test_fetch_features_retries_then_succeeds(monkeypatch):
    call_count = {"n": 0}

    def fake_random():
        call_count["n"] += 1
        return 0.0 if call_count["n"] < 3 else 1.0

    monkeypatch.setattr("app.adapters.feature_store.random.random", fake_random)
    features = fetch_features("txn-1", failure_rate=0.5)
    assert features == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert call_count["n"] == 3


def test_fetch_features_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr("app.adapters.feature_store.random.random", lambda: 0.0)
    with pytest.raises(FeatureStoreUnavailable):
        fetch_features("txn-1", failure_rate=1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_feature_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.adapters.feature_store'`.

- [ ] **Step 3: Write minimal implementation**

`app/adapters/feature_store.py`:
```python
import random

from tenacity import retry, stop_after_attempt, wait_fixed


class FeatureStoreUnavailable(RuntimeError):
    """Raised when the (simulated) feature store cannot be reached."""


def _call_feature_store(transaction_id: str, failure_rate: float) -> list[float]:
    if random.random() < failure_rate:
        raise FeatureStoreUnavailable(f"feature store timed out for {transaction_id}")
    # 6 values to match the 6-feature model trained by train_model.py.
    return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


@retry(stop=stop_after_attempt(3), wait=wait_fixed(0.1), reraise=True)
def fetch_features(transaction_id: str, failure_rate: float = 0.0) -> list[float]:
    return _call_feature_store(transaction_id, failure_rate)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_feature_store.py -v`
Expected: 2 passed (the second test takes ~0.2s longer due to retry waits).

- [ ] **Step 5: Commit**

```bash
git add modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/app/adapters/feature_store.py modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/tests/test_feature_store.py
git commit -m "$(cat <<'EOF'
feat: add flaky feature_store adapter with tenacity retries

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: ML lab — prototype script and artifact training

**Files:**
- Create: `modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/prototype.py`
- Create: `modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/train_model.py`
- Create (generated, then committed): `modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/artifacts/model.joblib`
- Create (generated, then committed): `modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/artifacts/manifest.json`

**Interfaces:**
- Produces: `artifacts/model.joblib` + `artifacts/manifest.json` on disk, matching the schema `{artifact_version, created_at, git_commit, sha256, python_version, key_dependencies}`. Consumed by Task 7 (`main.py` lifespan calls `load_model` from Task 4 against this directory).

- [ ] **Step 1: Write `prototype.py`**

```python
"""
BEFORE: this is how the fraud model started life as a notebook cell.
No structure, no tests, no versioning, no API. Compare this to app/ to see
the same problem solved with a layered, production-ready service.
"""
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=2000, n_features=6, weights=[0.9, 0.1])
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

new_transaction = [0.1, 0.2, -0.3, 0.4, 0.5, -0.1]
probability = model.predict_proba([new_transaction])[0][1]
print(f"Fraud probability: {probability:.2f}")
if probability > 0.5:
    print("FLAGGED AS FRAUD")
```

- [ ] **Step 2: Write `train_model.py`**

```python
"""
Generates the versioned fraud-detection model artifact.

Run this manually (`make train` or `python train_model.py`) whenever the
model needs to change. The output (artifacts/model.joblib and
artifacts/manifest.json) is committed to git, mirroring how a real team would
publish a new model version.
"""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

ARTIFACT_DIR = Path(__file__).parent / "artifacts"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    X, y = make_classification(
        n_samples=2000,
        n_features=6,
        n_informative=4,
        weights=[0.9, 0.1],
        random_state=42,
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    ARTIFACT_DIR.mkdir(exist_ok=True)
    model_path = ARTIFACT_DIR / "model.joblib"
    joblib.dump(model, model_path)

    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    manifest = {
        "artifact_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "sha256": digest,
        "python_version": "3.11",
        "key_dependencies": {"scikit-learn": sklearn.__version__},
    }
    (ARTIFACT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {model_path} and manifest.json (sha256={digest[:12]}...)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it to generate the committed artifact**

Run: `python train_model.py`
Expected: prints `Wrote .../model.joblib and manifest.json (sha256=...)`, and
`artifacts/model.joblib` + `artifacts/manifest.json` now exist on disk.

- [ ] **Step 4: Commit (including the binary artifact)**

```bash
git add modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/prototype.py modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/train_model.py modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/artifacts
git commit -m "$(cat <<'EOF'
feat: add prototype script and versioned fraud-detection model artifact

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: ML lab — API, config, logging, middleware, main (TDD)

**Files:**
- Create: `modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/app/api/__init__.py` (empty)
- Create: `modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/app/api/schemas.py`
- Create: `modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/app/api/routes.py`
- Create: `modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/app/config.py`
- Create: `modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/app/logging_utils.py`
- Create: `modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/app/middleware.py`
- Create: `modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/app/main.py`
- Create: `modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/tests/conftest.py`
- Test: `modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/tests/test_routes.py`

**Interfaces:**
- Consumes: `score_transaction`, `ScoreResult` (Task 3); `load_model`, `LoadedModel`, `ArtifactIntegrityError` (Task 4); `fetch_features` (Task 5); `artifacts/` directory (Task 6).
- Produces: running FastAPI `app` object with routes `POST /score`, `GET /healthz`, `GET /readyz`.

- [ ] **Step 1: Write the failing tests**

`tests/conftest.py`:
```python
import json
import os

import joblib
import pytest


class _StubModel:
    def predict_proba(self, X):
        return [[0.9, 0.1]]


@pytest.fixture(scope="session", autouse=True)
def _artifact_dir(tmp_path_factory):
    artifact_dir = tmp_path_factory.mktemp("artifacts")
    model_path = artifact_dir / "model.joblib"
    joblib.dump(_StubModel(), model_path)

    from app.adapters.model_store import _sha256_of

    manifest = {
        "artifact_version": "0.1.0-test",
        "sha256": _sha256_of(model_path),
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest))

    os.environ["ARTIFACT_DIR"] = str(artifact_dir)
    # Deterministic tests: never let the simulated flaky feature store fail.
    os.environ["FEATURE_STORE_FAILURE_RATE"] = "0"
    return artifact_dir
```

`tests/test_routes.py`:
```python
from fastapi.testclient import TestClient

from app.main import app


def test_healthz_returns_ok():
    with TestClient(app) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_returns_ready_after_startup():
    with TestClient(app) as client:
        response = client.get("/readyz")
    assert response.json() == {"status": "ready"}


def test_score_returns_fraud_probability():
    with TestClient(app) as client:
        response = client.post(
            "/score",
            json={
                "transaction_id": "txn-1",
                "amount": 42.5,
                "merchant_category": "electronics",
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["transaction_id"] == "txn-1"
    assert 0.0 <= body["fraud_probability"] <= 1.0


def test_score_rejects_negative_amount():
    with TestClient(app) as client:
        response = client.post(
            "/score",
            json={
                "transaction_id": "txn-1",
                "amount": -5,
                "merchant_category": "electronics",
            },
        )
    assert response.status_code == 422
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_routes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.main'`.

- [ ] **Step 3: Write minimal implementation**

`app/config.py`:
```python
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    artifact_dir: Path = Path(__file__).resolve().parent.parent / "artifacts"
    feature_store_failure_rate: float = 0.2
```

`app/logging_utils.py`:
```python
import contextvars
import json
import logging
import uuid

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
        }
        return json.dumps(payload)


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RequestIdFilter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)


def new_request_id() -> str:
    return str(uuid.uuid4())
```

`app/middleware.py`:
```python
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.logging_utils import new_request_id, request_id_var


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        incoming = request.headers.get("x-request-id", new_request_id())
        token = request_id_var.set(incoming)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["x-request-id"] = incoming
        return response
```

`app/api/schemas.py`:
```python
from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    transaction_id: str = Field(..., min_length=1, max_length=64)
    amount: float = Field(..., ge=0)
    merchant_category: str = Field(..., min_length=1, max_length=32)


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
```

`app/api/routes.py`:
```python
from fastapi import APIRouter, Request

from app.adapters.feature_store import fetch_features
from app.api.schemas import ScoreRequest, ScoreResponse
from app.config import Settings
from app.domain.scoring import score_transaction

router = APIRouter()
settings = Settings()


@router.post("/score", response_model=ScoreResponse)
def score(payload: ScoreRequest, request: Request) -> ScoreResponse:
    features = fetch_features(
        payload.transaction_id, failure_rate=settings.feature_store_failure_rate
    )
    result = score_transaction(features, request.app.state.model)
    return ScoreResponse(
        transaction_id=payload.transaction_id,
        fraud_probability=result.fraud_probability,
        is_fraud=result.is_fraud,
    )


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
def readyz(request: Request) -> dict[str, str]:
    if getattr(request.app.state, "model", None) is None:
        return {"status": "not-ready"}
    return {"status": "ready"}
```

`app/main.py`:
```python
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.adapters.model_store import load_model
from app.api.routes import router
from app.config import Settings
from app.logging_utils import configure_logging
from app.middleware import RequestIdMiddleware

configure_logging()
logger = logging.getLogger(__name__)
settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    loaded = load_model(settings.artifact_dir)
    app.state.model = loaded.model
    logger.info(f"model loaded, artifact_version={loaded.manifest['artifact_version']}")
    yield


app = FastAPI(title="Fraud Detection Service", lifespan=lifespan)
app.add_middleware(RequestIdMiddleware)
app.include_router(router)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_routes.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run the full test suite with coverage**

Run: `pytest --cov=app --cov-report=term-missing`
Expected: all tests pass; coverage on `app/domain/` and `app/api/` is 80%+.
If below 80%, add the missing test cases before proceeding (e.g. a test for
`readyz` returning not-ready when `app.state.model` is unset).

- [ ] **Step 6: Commit**

```bash
git add modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/app modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/tests
git commit -m "$(cat <<'EOF'
feat: add fraud-detection FastAPI service with structured logging and request IDs

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: ML lab — Dockerfile, Makefile, env example, lab README

**Files:**
- Create: `modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/Dockerfile`
- Create: `modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/Makefile`
- Create: `modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/.env.example`
- Create: `modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/README.md`

**Interfaces:**
- Consumes: everything from Tasks 3-7 (this task packages and documents it).

- [ ] **Step 1: Write `.env.example`**

```
FEATURE_STORE_FAILURE_RATE=0.2
```

- [ ] **Step 2: Write `Dockerfile`**

```dockerfile
# Base image: slim keeps the image small while still having a full Python runtime.
FROM python:3.11-slim

# Set a working directory inside the container so paths are predictable.
WORKDIR /app

# Copy only the dependency list first so Docker can cache this layer and skip
# re-installing packages when only application code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the application code and the versioned model artifact.
COPY app/ ./app/
COPY artifacts/ ./artifacts/

# Document the port the service listens on (informational; doesn't publish it).
EXPOSE 8000

# Run the service with uvicorn. --host 0.0.0.0 is required so the server is
# reachable from outside the container, not just from localhost inside it.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Write `Makefile`**

```makefile
.PHONY: setup train test run docker-build docker-run

setup:
	pip install -r requirements.txt

train:
	python train_model.py

test:
	pytest --cov=app --cov-report=term-missing

run:
	uvicorn app.main:app --reload

docker-build:
	docker build -t fraud-detection-service .

docker-run:
	docker run --rm -p 8000:8000 fraud-detection-service
```

- [ ] **Step 4: Write `README.md`**

```markdown
# ML Track Lab: Fraud Detection

Turns a notebook script into a layered, containerized, versioned FastAPI
service. Companion to [Week 1's concept README](../../README.md).

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
├── api/             # request/response schemas + routes (POST /score)
├── domain/          # scoring.py — pure, framework-free business logic
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
  `app/adapters/model_store.py`. Corrupt the model file and watch `/readyz`
  report `not-ready`.
- `app/adapters/feature_store.py` randomly fails (`FEATURE_STORE_FAILURE_RATE`
  in `.env`) to give retries (`tenacity`) something real to do.
- Every log line is JSON and carries the same `request_id` as the response's
  `x-request-id` header.
```

- [ ] **Step 5: Verify Docker build (best-effort)**

Run: `docker build -t fraud-detection-service .`
Expected: image builds successfully. If Docker isn't available in this
environment, skip this step and note it in your final report so the user can
verify locally.

- [ ] **Step 6: Commit**

```bash
git add modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/Dockerfile modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/Makefile modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/.env.example modules/week-01-prototype-to-production/labs/ml-track-fraud-detection/README.md
git commit -m "$(cat <<'EOF'
docs: add Dockerfile, Makefile, and README for fraud-detection lab

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: LLM lab — domain Q&A logic (TDD)

**Files:**
- Create: `modules/week-01-prototype-to-production/labs/llm-track-qa-service/requirements.txt`
- Create: `modules/week-01-prototype-to-production/labs/llm-track-qa-service/app/__init__.py` (empty)
- Create: `modules/week-01-prototype-to-production/labs/llm-track-qa-service/app/domain/__init__.py` (empty)
- Create: `modules/week-01-prototype-to-production/labs/llm-track-qa-service/app/domain/qa.py`
- Test: `modules/week-01-prototype-to-production/labs/llm-track-qa-service/tests/__init__.py` (empty)
- Test: `modules/week-01-prototype-to-production/labs/llm-track-qa-service/tests/test_qa.py`

**Interfaces:**
- Produces: `build_prompt(question: str) -> str`, `QuestionTooLongError`, `MAX_QUESTION_LENGTH: int = 500`. Consumed by Task 11's `api/routes.py`.

All commands run from `modules/week-01-prototype-to-production/labs/llm-track-qa-service/`.

- [ ] **Step 1: Write `requirements.txt`**

```
fastapi==0.115.5
uvicorn[standard]==0.32.1
pydantic==2.10.3
pydantic-settings==2.6.1
tenacity==9.0.0
openai==1.57.0
pytest==8.3.4
pytest-cov==6.0.0
httpx==0.28.1
ruff==0.8.2
```

- [ ] **Step 2: Create the venv and install dependencies**

```bash
python -m venv .venv
```
Windows: `.venv\Scripts\activate` — macOS/Linux: `source .venv/bin/activate`, then:
```bash
pip install -r requirements.txt
```

- [ ] **Step 3: Write the failing test**

`tests/test_qa.py`:
```python
import pytest

from app.domain.qa import MAX_QUESTION_LENGTH, QuestionTooLongError, build_prompt


def test_build_prompt_includes_the_question():
    prompt = build_prompt("What is training/serving skew?")
    assert "What is training/serving skew?" in prompt


def test_build_prompt_rejects_overly_long_questions():
    with pytest.raises(QuestionTooLongError):
        build_prompt("x" * (MAX_QUESTION_LENGTH + 1))
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_qa.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.domain.qa'`.

- [ ] **Step 5: Write minimal implementation**

`app/domain/qa.py`:
```python
from dataclasses import dataclass

MAX_QUESTION_LENGTH = 500


class QuestionTooLongError(ValueError):
    """Raised when a question exceeds the allowed length."""


@dataclass(frozen=True)
class Answer:
    question: str
    answer: str
    source: str  # "mock" or "llm"


def build_prompt(question: str) -> str:
    if len(question) > MAX_QUESTION_LENGTH:
        raise QuestionTooLongError(
            f"question exceeds {MAX_QUESTION_LENGTH} characters"
        )
    return (
        "You are a concise assistant for a production AI systems course.\n"
        f"Question: {question.strip()}\nAnswer:"
    )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_qa.py -v`
Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add modules/week-01-prototype-to-production/labs/llm-track-qa-service/requirements.txt modules/week-01-prototype-to-production/labs/llm-track-qa-service/app modules/week-01-prototype-to-production/labs/llm-track-qa-service/tests
git commit -m "$(cat <<'EOF'
feat: add Q&A domain logic with prompt building and length validation

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: LLM lab — llm_client adapter with mock mode and retries (TDD)

**Files:**
- Create: `modules/week-01-prototype-to-production/labs/llm-track-qa-service/app/adapters/__init__.py` (empty)
- Create: `modules/week-01-prototype-to-production/labs/llm-track-qa-service/app/adapters/llm_client.py`
- Test: `modules/week-01-prototype-to-production/labs/llm-track-qa-service/tests/test_llm_client.py`

**Interfaces:**
- Produces: `LlmClient(api_key: str | None)` with `.is_mock: bool` and `.complete(prompt: str) -> str`; `LlmCallError`. Consumed by Task 11 (`main.py`, `api/routes.py`).

- [ ] **Step 1: Write the failing test**

`tests/test_llm_client.py`:
```python
from unittest.mock import patch

from app.adapters.llm_client import LlmClient


def test_complete_returns_mock_answer_when_no_api_key():
    client = LlmClient(api_key=None)
    assert client.is_mock is True
    assert "mock answer" in client.complete("anything").lower()


def test_complete_calls_real_api_when_key_present():
    client = LlmClient(api_key="fake-key")
    with patch.object(client, "_call_real_api", return_value="real answer") as mock_call:
        result = client.complete("hello")
    assert result == "real answer"
    mock_call.assert_called_once_with("hello")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.adapters.llm_client'`.

- [ ] **Step 3: Write minimal implementation**

`app/adapters/llm_client.py`:
```python
import logging

from tenacity import retry, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)

_MOCK_ANSWER = (
    "This is a mock answer. Set OPENAI_API_KEY in .env to call a real model."
)


class LlmCallError(RuntimeError):
    """Raised when the upstream LLM call fails after retries."""


class LlmClient:
    def __init__(self, api_key: str | None):
        self._api_key = api_key

    @property
    def is_mock(self) -> bool:
        return not self._api_key

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(0.1), reraise=True)
    def complete(self, prompt: str) -> str:
        if self.is_mock:
            return _MOCK_ANSWER
        return self._call_real_api(prompt)

    def _call_real_api(self, prompt: str) -> str:
        from openai import OpenAI

        try:
            client = OpenAI(api_key=self._api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("llm call failed, will retry if attempts remain")
            raise LlmCallError(str(exc)) from exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_client.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add modules/week-01-prototype-to-production/labs/llm-track-qa-service/app/adapters modules/week-01-prototype-to-production/labs/llm-track-qa-service/tests/test_llm_client.py
git commit -m "$(cat <<'EOF'
feat: add llm_client adapter with mock mode and retry logic

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: LLM lab — API, config, logging, middleware, main (TDD)

**Files:**
- Create: `modules/week-01-prototype-to-production/labs/llm-track-qa-service/app/api/__init__.py` (empty)
- Create: `modules/week-01-prototype-to-production/labs/llm-track-qa-service/app/api/schemas.py`
- Create: `modules/week-01-prototype-to-production/labs/llm-track-qa-service/app/api/routes.py`
- Create: `modules/week-01-prototype-to-production/labs/llm-track-qa-service/app/config.py`
- Create: `modules/week-01-prototype-to-production/labs/llm-track-qa-service/app/logging_utils.py`
- Create: `modules/week-01-prototype-to-production/labs/llm-track-qa-service/app/middleware.py`
- Create: `modules/week-01-prototype-to-production/labs/llm-track-qa-service/app/main.py`
- Test: `modules/week-01-prototype-to-production/labs/llm-track-qa-service/tests/test_routes.py`

**Interfaces:**
- Consumes: `build_prompt`, `QuestionTooLongError` (Task 9); `LlmClient`, `LlmCallError` (Task 10).
- Produces: running FastAPI `app` object with routes `POST /ask`, `GET /healthz`, `GET /readyz`.

- [ ] **Step 1: Write the failing tests**

`tests/test_routes.py`:
```python
from fastapi.testclient import TestClient

from app.main import app


def test_healthz_returns_ok():
    with TestClient(app) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_returns_ready_after_startup():
    with TestClient(app) as client:
        response = client.get("/readyz")
    assert response.json() == {"status": "ready"}


def test_ask_returns_mock_answer_by_default():
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "What is RAG?"})
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "mock"
    assert "mock answer" in body["answer"].lower()


def test_ask_rejects_empty_question():
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_routes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.main'`.

- [ ] **Step 3: Write minimal implementation**

`app/config.py`:
```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str | None = None
```

`app/logging_utils.py` — identical to the ML lab's version (Task 7):
```python
import contextvars
import json
import logging
import uuid

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
        }
        return json.dumps(payload)


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RequestIdFilter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)


def new_request_id() -> str:
    return str(uuid.uuid4())
```

`app/middleware.py` — identical to the ML lab's version (Task 7):
```python
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.logging_utils import new_request_id, request_id_var


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        incoming = request.headers.get("x-request-id", new_request_id())
        token = request_id_var.set(incoming)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["x-request-id"] = incoming
        return response
```

(These two files are intentionally duplicated rather than shared between labs
— each lab is meant to be independently copyable/runnable, per the course
design.)

`app/api/schemas.py`:
```python
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)


class AskResponse(BaseModel):
    question: str
    answer: str
    source: str
```

`app/api/routes.py`:
```python
from fastapi import APIRouter, Request

from app.adapters.llm_client import LlmCallError
from app.api.schemas import AskRequest, AskResponse
from app.domain.qa import build_prompt

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request) -> AskResponse:
    client = request.app.state.llm_client
    prompt = build_prompt(payload.question)
    try:
        answer = client.complete(prompt)
    except LlmCallError:
        answer = "The assistant is temporarily unavailable. Please try again."
    return AskResponse(
        question=payload.question,
        answer=answer,
        source="mock" if client.is_mock else "llm",
    )


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
def readyz(request: Request) -> dict[str, str]:
    if getattr(request.app.state, "llm_client", None) is None:
        return {"status": "not-ready"}
    return {"status": "ready"}
```

`app/main.py`:
```python
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.adapters.llm_client import LlmClient
from app.api.routes import router
from app.config import Settings
from app.logging_utils import configure_logging
from app.middleware import RequestIdMiddleware

configure_logging()
logger = logging.getLogger(__name__)
settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.llm_client = LlmClient(api_key=settings.openai_api_key)
    logger.info(f"llm client ready, mock_mode={app.state.llm_client.is_mock}")
    yield


app = FastAPI(title="Q&A Service", lifespan=lifespan)
app.add_middleware(RequestIdMiddleware)
app.include_router(router)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_routes.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run the full test suite with coverage**

Run: `pytest --cov=app --cov-report=term-missing`
Expected: all tests pass; coverage on `app/domain/` and `app/api/` is 80%+.
If below 80%, add the missing test case(s) (e.g. `readyz` returning
not-ready when `app.state.llm_client` is unset) before proceeding.

- [ ] **Step 6: Commit**

```bash
git add modules/week-01-prototype-to-production/labs/llm-track-qa-service/app modules/week-01-prototype-to-production/labs/llm-track-qa-service/tests
git commit -m "$(cat <<'EOF'
feat: add Q&A FastAPI service with structured logging and request IDs

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: LLM lab — Dockerfile, Makefile, env example, lab README

**Files:**
- Create: `modules/week-01-prototype-to-production/labs/llm-track-qa-service/prototype.py`
- Create: `modules/week-01-prototype-to-production/labs/llm-track-qa-service/Dockerfile`
- Create: `modules/week-01-prototype-to-production/labs/llm-track-qa-service/Makefile`
- Create: `modules/week-01-prototype-to-production/labs/llm-track-qa-service/.env.example`
- Create: `modules/week-01-prototype-to-production/labs/llm-track-qa-service/README.md`

- [ ] **Step 1: Write `prototype.py`**

```python
"""
BEFORE: this is how the Q&A assistant started life as a notebook cell —
a single hardcoded call, no structure, no error handling, no way to run it
as a service.
"""
import os

question = "What is training/serving skew?"
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    print("(no API key set) Mock answer: This is a mock answer.")
else:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
    )
    print(response.choices[0].message.content)
```

- [ ] **Step 2: Write `.env.example`**

```
OPENAI_API_KEY=
```

- [ ] **Step 3: Write `Dockerfile`**

```dockerfile
# Base image: slim keeps the image small while still having a full Python runtime.
FROM python:3.11-slim

# Set a working directory inside the container so paths are predictable.
WORKDIR /app

# Copy only the dependency list first so Docker can cache this layer and skip
# re-installing packages when only application code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the application code.
COPY app/ ./app/

# Document the port the service listens on (informational; doesn't publish it).
EXPOSE 8000

# Run the service with uvicorn. --host 0.0.0.0 is required so the server is
# reachable from outside the container, not just from localhost inside it.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 4: Write `Makefile`**

```makefile
.PHONY: setup test run docker-build docker-run

setup:
	pip install -r requirements.txt

test:
	pytest --cov=app --cov-report=term-missing

run:
	uvicorn app.main:app --reload

docker-build:
	docker build -t qa-service .

docker-run:
	docker run --rm -p 8000:8000 --env-file .env qa-service
```

- [ ] **Step 5: Write `README.md`**

```markdown
# LLM Track Lab: Q&A Service

Turns a hardcoded LLM call into a layered, containerized FastAPI service.
Companion to [Week 1's concept README](../../README.md). Deliberately *not*
RAG yet — that's Week 2.

## Before: the prototype

```bash
python prototype.py
```

## After: the production service

```
app/
├── main.py         # wiring: logging, middleware, routes, client setup
├── api/             # request/response schemas + routes (POST /ask)
├── domain/          # qa.py — pure prompt-building + validation logic
└── adapters/        # llm_client.py — mock-mode-by-default LLM wrapper with retries
```

## Run it (no API key needed)

If `make` isn't available (e.g. plain Windows without Git Bash), run the
commands inside `Makefile` directly — each target is a single command.

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows:     .venv\Scripts\activate
make setup
make test       # runs pytest with coverage
make run        # starts the service on http://localhost:8000
```

Try it:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is training/serving skew?"}'
```
The response's `"source"` field will be `"mock"`.

## Use a real model

Copy `.env.example` to `.env`, set `OPENAI_API_KEY=<your key>`, restart the
service. `"source"` will now be `"llm"` — no code change required, only
config.

## Run it in Docker

```bash
make docker-build
make docker-run
```

## What to notice

- The mock/real switch is entirely a config decision (`app/adapters/llm_client.py`),
  which is why the hands-on guide works with zero API key.
- `/ask` validates and caps question length before it reaches the domain
  layer — see `app/domain/qa.py`.
- Every log line is JSON and carries the same `request_id` as the response's
  `x-request-id` header.
```

- [ ] **Step 6: Verify Docker build (best-effort)**

Run: `docker build -t qa-service .`
Expected: image builds successfully. If Docker isn't available in this
environment, skip this step and note it in your final report so the user can
verify locally.

- [ ] **Step 7: Commit**

```bash
git add modules/week-01-prototype-to-production/labs/llm-track-qa-service/prototype.py modules/week-01-prototype-to-production/labs/llm-track-qa-service/Dockerfile modules/week-01-prototype-to-production/labs/llm-track-qa-service/Makefile modules/week-01-prototype-to-production/labs/llm-track-qa-service/.env.example modules/week-01-prototype-to-production/labs/llm-track-qa-service/README.md
git commit -m "$(cat <<'EOF'
docs: add prototype script, Dockerfile, Makefile, and README for Q&A lab

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: CI workflow

**Files:**
- Create: `.github/workflows/tests.yml`

**Interfaces:**
- Consumes: both labs' `requirements.txt` and `tests/` directories (Tasks 3-12).

- [ ] **Step 1: Write `.github/workflows/tests.yml`**

```yaml
name: tests

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        lab:
          - ml-track-fraud-detection
          - llm-track-qa-service
    defaults:
      run:
        working-directory: modules/week-01-prototype-to-production/labs/${{ matrix.lab }}
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Lint
        run: ruff check .

      - name: Generate model artifact
        if: matrix.lab == 'ml-track-fraud-detection'
        run: python train_model.py

      - name: Run tests with coverage
        run: pytest --cov=app --cov-report=term-missing --cov-fail-under=80
```

Note on the last step: `--cov-fail-under=80` applies to the whole `--cov=app`
measurement (which includes `domain/` and `api/`, the modules the 80% target
applies to); this keeps the workflow simple while still enforcing the
Global Constraints coverage target.

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/tests.yml
git commit -m "$(cat <<'EOF'
ci: add pytest workflow for both Week 1 labs

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: Code review and security review pass

**Files:** none created — this task reviews everything from Tasks 3-13 and applies fixes in place.

- [ ] **Step 1: Run the code-reviewer agent**

Dispatch the `code-reviewer` agent against both labs'
`modules/week-01-prototype-to-production/labs/*/app/` and `tests/`
directories. Ask it to check for code quality, error handling, and
maintainability issues per this repo's coding standards (functions <50
lines, no deep nesting, explicit error handling, no hardcoded secrets).

- [ ] **Step 2: Run the security-reviewer agent**

Dispatch the `security-reviewer` agent against the same directories,
specifically the API routes (`api/routes.py`) and the adapters that make
external calls (`adapters/llm_client.py`, `adapters/feature_store.py`,
`adapters/model_store.py`). Ask it to check for injection risks, secret
leakage into logs, unvalidated input, and unsafe deserialization
(`joblib.load` on the committed artifact is trusted/first-party, but note
this assumption explicitly in the review).

- [ ] **Step 3: Fix CRITICAL and HIGH findings**

Apply fixes for any CRITICAL or HIGH severity finding directly in the
affected files. Re-run the affected lab's `pytest --cov=app` after each fix
to confirm nothing broke.

- [ ] **Step 4: Commit fixes (if any)**

```bash
git add -A
git commit -m "$(cat <<'EOF'
fix: address code-reviewer and security-reviewer findings

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

If no findings required fixes, skip this commit and note that in your final
report.

---

### Task 15: Push to GitHub

**Files:** none.

- [ ] **Step 1: Verify `gh` auth**

Run: `gh auth status`
Expected: shows logged in as `jahirsaiyed` with `repo` scope (already
confirmed during planning).

- [ ] **Step 2: Create the repo and push**

From the repo root:
```bash
gh repo create production-grade-ai-systems --public --source=. --remote=origin --push
```
Expected: prints the new repo URL and pushes the `master` branch (or
`main`, whichever this repo's default branch is) with all commits from
Tasks 1-14.

- [ ] **Step 3: Verify**

Run: `gh repo view --web=false --json url,visibility,defaultBranchRef`
Expected: JSON showing the repo URL, `"visibility": "PUBLIC"`, and the
pushed branch.

- [ ] **Step 4: Report back**

Report the repo URL to the user, along with: which steps (if any) from
Task 8/Task 12's Docker-build verification were skipped because Docker
wasn't available in this environment, and a one-line summary of Task 14's
review findings.
