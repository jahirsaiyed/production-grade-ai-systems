# Week 2 Data and Model Pipelines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Week 2's concept README/exercises and two hands-on labs — `ml-track-experiment-tracking` (data validation + MLflow reproducibility) and `llm-track-rag-service` (a hybrid-retrieval RAG service with citations) — then add CI coverage and push.

**Architecture:** The ML lab is a standalone training pipeline (no API), TDD-tested. The LLM lab follows Week 1's exact layered `api/domain/adapters` pattern, structured JSON logging, request IDs, and a sha256-verified versioned artifact — now applied to a retrieval index instead of a model, and stored as JSON (no pickle) rather than joblib.

**Tech Stack:** Python 3.11+, `numpy`, `rank-bm25`, `mlflow` (local file backend), `scikit-learn`, FastAPI/pydantic/tenacity/openai (same as Week 1), pytest + pytest-cov, ruff.

## Global Constraints

- All Week 1 conventions carry forward unchanged: Python 3.11+/plain venv/pip, `pytest`+`pytest-cov`
  with an 80%+ target on `domain/`+`api/` (or the whole codebase for the ML lab, which has no
  `api/` layer), `ruff check` must pass, exact-version pins in `requirements.txt` (same-minor-version
  substitution allowed if a pin doesn't resolve locally), commit format `<type>: <description>` with
  the exact trailer `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` copied verbatim by
  every implementer regardless of which model executes the task.
- No live network calls in any test. The LLM lab's `tests/conftest.py` must set both
  `ARTIFACT_DIR` (to a fixture-built temp index) and `OPENAI_API_KEY=""` at MODULE level (not
  inside a fixture function), exactly mirroring Week 1's two hard-won fixes: (a) module-level env
  vars execute before pytest's collection-time import of `app.main`, unlike fixture bodies which
  run too late; (b) forcing `OPENAI_API_KEY=""` neutralizes any ambient real key in the developer's
  shell so tests never hit the real OpenAI API.
- The LLM lab's retrieval index is stored as **JSON, not pickle/joblib** — deliberately, so the
  artifact never requires trusting a deserializer. Still versioned and sha256-verified the same
  way Week 1's model artifact is, and the adapter that loads it fails fast (raises, uncaught, at
  FastAPI lifespan startup) on a hash mismatch — same behavior as Week 1's `model_store.py`, so no
  README should ever claim `/readyz` observably reports "not-ready" for an artifact integrity
  failure (Week 1's final review corrected exactly this mistake once already).
- Bake in Week 1's post-review lessons from the start rather than rediscovering them: the
  `except LlmCallError` fallback path must log a warning before returning the degraded answer; CI
  must not retrain/re-ingest anything the tests don't actually need — instead a dedicated test
  loads and hash-checks the real COMMITTED artifact directly (bypassing the `ARTIFACT_DIR` test
  override) so CI actually protects the thing that ships; `make docker-run` must use
  `-e OPENAI_API_KEY` (host env passthrough), never `--env-file .env` (no `.env` file exists in a
  fresh clone).
- Mock embeddings must have real semantic structure (shared words → non-zero vector overlap), not
  pure randomness — a hash-based bag-of-words embedding, not `np.random`.
- Citations in `AskResponse` are built directly from retrieval metadata, never parsed out of the
  LLM's own output — so citations stay accurate even in mock mode or when the LLM hallucinates.
- Working directory for all commands below is the repo root:
  `D:\Learning\Build Production Grade AI Systems _ ByteByteGo Live\production-grade-ai-systems`.
  The repo already exists on GitHub (`jahirsaiyed/production-grade-ai-systems`, public,
  `origin/master`) — no `gh repo create` this time, just `git push` at the end.

---

### Task 1: Week 2 concept README and exercises

**Files:**
- Modify: `modules/week-02-data-and-model-pipelines/README.md` (currently a "coming soon" stub)
- Create: `modules/week-02-data-and-model-pipelines/exercises.md`

**Interfaces:**
- Produces: links to `labs/ml-track-experiment-tracking/README.md` and
  `labs/llm-track-rag-service/README.md` (built in later tasks — a known forward reference, same
  pattern as Week 1's Task 2).

- [ ] **Step 1: Write `modules/week-02-data-and-model-pipelines/README.md`**

Replace the stub with a concept README covering each topic below as its own `##` section, in this
order. For each: a 2-4 sentence plain-language explanation, why it matters in production, and one
concrete pointer into `labs/ml-track-experiment-tracking/` or `labs/llm-track-rag-service/` so the
reader can see the concept in running code.

1. **Ingestion patterns (batch vs. streaming, ETL vs. ELT)** — explain the distinction and note
   `ingest.py` in the RAG lab is a batch, ETL-style job (transform-then-load: chunk and embed
   before writing the artifact) run on demand, not a streaming pipeline.
2. **Data validation gates and schema enforcement** — explain the concept, then point directly at
   `data_validation.py` in the ML lab as a concrete, runnable gate.
3. **Data lineage and versioning** — explain what lineage means (being able to answer "what data
   and code produced this artifact"), then point at the ML lab's reproducibility manifest
   (`git_commit`, `data_digest`) and the RAG lab's index `manifest.json` as two concrete lineage
   records.
4. **Training/serving parity; leakage detection** — explain both concepts briefly (this week's labs
   don't have a serving/training split to demonstrate leakage directly, so keep this section
   conceptual and note it's revisited when Week 1's serving patterns apply to a trained model).
5. **Deep-learning data paths (loaders, batching, augmentation, distributed training)** — brief,
   explicitly conceptual-only paragraph; no lab this week demonstrates it, say so directly.
6. **Document parsing (OCR, PDF, HTML) and chunking strategies** — explain chunking's purpose
   (fitting text into a model's context window while preserving semantic coherence), then point at
   `app/domain/chunking.py`'s fixed-size-with-overlap strategy in the RAG lab, noting this repo's
   corpus is already plain markdown so OCR/PDF/HTML parsing isn't exercised, only chunking is.
7. **Vector DB operations (indexing, upserts, metadata filters)** — explain the concepts a real
   vector DB (Pinecone, Weaviate, pgvector, etc.) provides, then contrast with this week's
   deliberately simplified in-process index (`ingest.py` + `adapters/index_store.py`) — no upserts,
   no metadata filters, just enough to teach retrieval mechanics without standing up infrastructure.
8. **Context packaging (ordering, token caps, citations)** — point directly at
   `app/domain/rag.py`'s `build_rag_prompt` (character-budget capping, numbered citation markers)
   and `build_citations` (metadata-based, not LLM-output-parsed).
9. **Hybrid retrieval: BM25 + dense embeddings + reranking; GraphRAG** — explain BM25 (keyword/
   exact-match strength) vs. dense embeddings (semantic/paraphrase strength) and why combining them
   often beats either alone; point at `app/domain/retrieval.py`'s `hybrid_search`. Explicitly note
   the combined-score step is a simplified stand-in for a real cross-encoder reranker, not a
   state-of-the-art reranking implementation. Explain GraphRAG in one paragraph, conceptual only.
10. **Experiment tracking (MLflow, Weights & Biases)** — point at `train_with_tracking.py`'s MLflow
    integration (local file backend, no server) as the concrete example; mention W&B as the
    SaaS alternative not used here.
11. **Model registry stages** — explain the concept (staging → production → archived) briefly;
    this week's labs don't implement a registry, say so and connect it to Week 1's model-versioning
    manifest as a lighter-weight relative of the same idea.
12. **Hyperparameter tuning** — brief conceptual paragraph; note `train_with_tracking.py` logs a
    fixed hyperparameter set to MLflow rather than searching over one, and suggest sweeping
    `max_iter` or `n_features` as a natural extension exercise.
13. **Reproducibility checklist (commit, data digest, env lock, artifact link)** — walk through
    `train_with_tracking.py`'s `reproducibility_manifest.json` field by field, matching each field
    to this exact checklist item.
14. **Adaptation decision framework: prompting vs. RAG vs. PEFT (LoRA) vs. full fine-tuning** —
    explain the framework and where this week's RAG lab sits on it (a concrete example of the "RAG"
    branch), contrasted with Week 1's LLM lab (plain "prompting", no retrieval).

Open the file with a `# Week 2: Data and Model Pipelines` heading and a short intro naming both
tracks and linking to `labs/ml-track-experiment-tracking/README.md` and
`labs/llm-track-rag-service/README.md`. Close with a "## Hands-on labs" section linking both, and
an "## Exercises" section linking `exercises.md`.

- [ ] **Step 2: Write `modules/week-02-data-and-model-pipelines/exercises.md`**

```markdown
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
```

- [ ] **Step 3: Commit**

```bash
git add modules/week-02-data-and-model-pipelines/README.md modules/week-02-data-and-model-pipelines/exercises.md
git commit -m "$(cat <<'EOF'
docs: add Week 2 concept README and exercises

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: ML lab — data validation gate (TDD)

**Files:**
- Create: `modules/week-02-data-and-model-pipelines/labs/ml-track-experiment-tracking/requirements.txt`
- Create: `modules/week-02-data-and-model-pipelines/labs/ml-track-experiment-tracking/data_validation.py`
- Create: `modules/week-02-data-and-model-pipelines/labs/ml-track-experiment-tracking/tests/__init__.py` (empty)
- Test: `modules/week-02-data-and-model-pipelines/labs/ml-track-experiment-tracking/tests/test_data_validation.py`

**Interfaces:**
- Produces: `validate_training_data(X: np.ndarray, y: np.ndarray) -> None` (raises
  `DataValidationError` on failure), `DataValidationError`. Consumed by Task 3's
  `train_with_tracking.py`.

All commands run from
`modules/week-02-data-and-model-pipelines/labs/ml-track-experiment-tracking/`.

- [ ] **Step 1: Write `requirements.txt`**

```
scikit-learn==1.5.2
numpy==2.1.3
mlflow==2.18.0
pytest==8.3.4
pytest-cov==6.0.0
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
If an exact pin is unavailable for your platform, install the closest available patch version of
that same minor version.

- [ ] **Step 3: Write the failing test**

`tests/test_data_validation.py`:
```python
import numpy as np
import pytest

from data_validation import DataValidationError, validate_training_data


def test_validate_training_data_passes_for_clean_data():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1])
    validate_training_data(X, y)


def test_validate_training_data_rejects_mismatched_row_counts():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0])
    with pytest.raises(DataValidationError):
        validate_training_data(X, y)


def test_validate_training_data_rejects_nan():
    X = np.array([[1.0, np.nan], [3.0, 4.0]])
    y = np.array([0, 1])
    with pytest.raises(DataValidationError):
        validate_training_data(X, y)


def test_validate_training_data_rejects_infinite_values():
    X = np.array([[1.0, np.inf], [3.0, 4.0]])
    y = np.array([0, 1])
    with pytest.raises(DataValidationError):
        validate_training_data(X, y)


def test_validate_training_data_rejects_unexpected_labels():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 2])
    with pytest.raises(DataValidationError):
        validate_training_data(X, y)
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_data_validation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_validation'`.

- [ ] **Step 5: Write minimal implementation**

`data_validation.py`:
```python
import numpy as np


class DataValidationError(ValueError):
    """Raised when training data fails a validation gate."""


def validate_training_data(X: np.ndarray, y: np.ndarray) -> None:
    if X.shape[0] != y.shape[0]:
        raise DataValidationError(
            f"X has {X.shape[0]} rows but y has {y.shape[0]} rows"
        )
    if np.isnan(X).any():
        raise DataValidationError("X contains NaN values")
    if np.isinf(X).any():
        raise DataValidationError("X contains infinite values")
    unique_labels = set(np.unique(y).tolist())
    if not unique_labels.issubset({0, 1}):
        raise DataValidationError(
            f"y contains labels outside {{0, 1}}: {unique_labels}"
        )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_data_validation.py -v`
Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add modules/week-02-data-and-model-pipelines/labs/ml-track-experiment-tracking/requirements.txt modules/week-02-data-and-model-pipelines/labs/ml-track-experiment-tracking/data_validation.py modules/week-02-data-and-model-pipelines/labs/ml-track-experiment-tracking/tests
git commit -m "$(cat <<'EOF'
feat: add data validation gate for fraud training data

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: ML lab — train with MLflow tracking + reproducibility manifest (TDD)

**Files:**
- Create: `modules/week-02-data-and-model-pipelines/labs/ml-track-experiment-tracking/train_with_tracking.py`
- Test: `modules/week-02-data-and-model-pipelines/labs/ml-track-experiment-tracking/tests/test_train_with_tracking.py`
- Create: `modules/week-02-data-and-model-pipelines/labs/ml-track-experiment-tracking/.gitignore`

**Interfaces:**
- Consumes: `validate_training_data`, `DataValidationError` (Task 2).
- Produces: `main(tracking_uri: str | None = None, manifest_path: Path | None = None) -> dict`
  returning the reproducibility manifest dict; also writes it to `manifest_path` (defaults to
  `reproducibility_manifest.json` in this lab's root) and to MLflow at `tracking_uri` (defaults to
  `file:./mlruns` in this lab's root).

- [ ] **Step 1: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.coverage
mlruns/
reproducibility_manifest.json
```

(`reproducibility_manifest.json` is generated output, like the MLflow store — not committed. The
*code* that produces it is what's committed.)

- [ ] **Step 2: Write the failing test**

`tests/test_train_with_tracking.py`:
```python
import json

from train_with_tracking import main


def test_main_logs_expected_metrics_and_writes_manifest(tmp_path):
    manifest_path = tmp_path / "reproducibility_manifest.json"
    tracking_uri = f"file:{tmp_path / 'mlruns'}"

    manifest = main(tracking_uri=tracking_uri, manifest_path=manifest_path)

    assert manifest_path.exists()
    written = json.loads(manifest_path.read_text())
    assert written["mlflow_run_id"] == manifest["mlflow_run_id"]
    assert "accuracy" in manifest["metrics"]
    assert "f1" in manifest["metrics"]
    assert 0.0 <= manifest["metrics"]["accuracy"] <= 1.0
    assert len(manifest["data_digest"]) == 64
    assert len(manifest["env_lock"]) == 64
    assert manifest["artifact_uri"].startswith(tracking_uri.replace("file:", "file://"))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_train_with_tracking.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'train_with_tracking'`.

- [ ] **Step 4: Write minimal implementation**

`train_with_tracking.py`:
```python
"""
Trains the fraud-detection classifier with MLflow experiment tracking and
writes a reproducibility manifest (commit, data digest, env lock, artifact
link) implementing the course's reproducibility checklist.

Run with `make train` or `python train_with_tracking.py`.
"""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from data_validation import validate_training_data

LAB_DIR = Path(__file__).parent
DEFAULT_MLRUNS_DIR = LAB_DIR / "mlruns"
DEFAULT_MANIFEST_PATH = LAB_DIR / "reproducibility_manifest.json"
REQUIREMENTS_PATH = LAB_DIR / "requirements.txt"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main(
    tracking_uri: str | None = None, manifest_path: Path | None = None
) -> dict:
    tracking_uri = tracking_uri or f"file:{DEFAULT_MLRUNS_DIR}"
    manifest_path = manifest_path or DEFAULT_MANIFEST_PATH

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("fraud-detection-reproducibility")

    X, y = make_classification(
        n_samples=2000,
        n_features=6,
        n_informative=4,
        weights=[0.9, 0.1],
        random_state=42,
    )
    validate_training_data(X, y)

    data_digest = _sha256_bytes(X.tobytes() + y.tobytes())
    params = {
        "max_iter": 1000,
        "n_samples": 2000,
        "n_features": 6,
        "random_state": 42,
    }

    with mlflow.start_run() as run:
        mlflow.log_params(params)

        model = LogisticRegression(max_iter=params["max_iter"])
        model.fit(X, y)

        predictions = model.predict(X)
        metrics = {
            "accuracy": accuracy_score(y, predictions),
            "f1": f1_score(y, predictions),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        run_id = run.info.run_id
        artifact_uri = run.info.artifact_uri

    manifest = {
        "git_commit": _git_commit(),
        "data_digest": data_digest,
        "env_lock": _sha256_bytes(REQUIREMENTS_PATH.read_bytes()),
        "mlflow_run_id": run_id,
        "artifact_uri": artifact_uri,
        "metrics": metrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(
        f"Run {run_id} tracked at {tracking_uri}. "
        f"Reproducibility manifest written to {manifest_path}"
    )
    return manifest


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_train_with_tracking.py -v`
Expected: 1 passed. (May take a few seconds — MLflow writes real run files to the temp directory.)

- [ ] **Step 6: Commit**

```bash
git add modules/week-02-data-and-model-pipelines/labs/ml-track-experiment-tracking/train_with_tracking.py modules/week-02-data-and-model-pipelines/labs/ml-track-experiment-tracking/tests/test_train_with_tracking.py modules/week-02-data-and-model-pipelines/labs/ml-track-experiment-tracking/.gitignore
git commit -m "$(cat <<'EOF'
feat: add MLflow-tracked training run with reproducibility manifest

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: ML lab — Makefile, README

**Files:**
- Create: `modules/week-02-data-and-model-pipelines/labs/ml-track-experiment-tracking/Makefile`
- Create: `modules/week-02-data-and-model-pipelines/labs/ml-track-experiment-tracking/README.md`

- [ ] **Step 1: Write `Makefile`**

```makefile
.PHONY: setup train test mlflow-ui

setup:
	pip install -r requirements.txt

train:
	python train_with_tracking.py

test:
	pytest --cov=. --cov-report=term-missing --cov-config=.coveragerc

mlflow-ui:
	mlflow ui --backend-store-uri file:./mlruns
```

- [ ] **Step 2: Write `.coveragerc`** (so coverage excludes the venv and doesn't try to measure
  MLflow's own internals when run with `--cov=.`)

```ini
[run]
omit =
    .venv/*
    tests/*
```

- [ ] **Step 3: Write `README.md`**

```markdown
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
```

- [ ] **Step 4: Commit**

```bash
git add modules/week-02-data-and-model-pipelines/labs/ml-track-experiment-tracking/Makefile modules/week-02-data-and-model-pipelines/labs/ml-track-experiment-tracking/.coveragerc modules/week-02-data-and-model-pipelines/labs/ml-track-experiment-tracking/README.md
git commit -m "$(cat <<'EOF'
docs: add Makefile and README for experiment-tracking lab

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: RAG lab — chunking (TDD)

**Files:**
- Create: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/requirements.txt`
- Create: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/app/__init__.py` (empty)
- Create: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/app/domain/__init__.py` (empty)
- Create: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/app/domain/chunking.py`
- Create: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/tests/__init__.py` (empty)
- Test: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/tests/test_chunking.py`

**Interfaces:**
- Produces: `Chunk(text: str, source: str, chunk_id: int)` (frozen dataclass),
  `chunk_text(text: str, source: str, chunk_size: int = 400, overlap: int = 50) -> list[Chunk]`.
  Consumed by Tasks 6, 7, 9, 11.

All commands run from
`modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/`.

- [ ] **Step 1: Write `requirements.txt`**

```
fastapi==0.115.5
uvicorn[standard]==0.32.1
pydantic==2.10.3
pydantic-settings==2.6.1
tenacity==9.0.0
openai==1.57.0
numpy==2.1.3
rank-bm25==0.2.2
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

`tests/test_chunking.py`:
```python
import pytest

from app.domain.chunking import Chunk, chunk_text


def test_chunk_text_splits_long_text_into_multiple_chunks():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, source="doc.md", chunk_size=400, overlap=50)
    assert len(chunks) > 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.source == "doc.md" for c in chunks)


def test_chunk_text_returns_single_chunk_for_short_text():
    text = "just a few words here"
    chunks = chunk_text(text, source="doc.md", chunk_size=400, overlap=50)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].chunk_id == 0


def test_chunk_text_rejects_overlap_greater_than_or_equal_to_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text", source="doc.md", chunk_size=10, overlap=10)
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_chunking.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.domain.chunking'`.

- [ ] **Step 5: Write minimal implementation**

`app/domain/chunking.py`:
```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    chunk_id: int


def chunk_text(
    text: str, source: str, chunk_size: int = 400, overlap: int = 50
) -> list[Chunk]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    words = text.split()
    chunks: list[Chunk] = []
    start = 0
    chunk_id = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(
            Chunk(text=" ".join(chunk_words), source=source, chunk_id=chunk_id)
        )
        chunk_id += 1
        if end >= len(words):
            break
        start = end - overlap
    return chunks
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_chunking.py -v`
Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/requirements.txt modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/app modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/tests
git commit -m "$(cat <<'EOF'
feat: add chunking domain logic for RAG lab

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: RAG lab — hybrid retrieval (TDD)

**Files:**
- Create: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/app/domain/retrieval.py`
- Test: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/tests/test_retrieval.py`

**Interfaces:**
- Consumes: `Chunk` (Task 5).
- Produces: `RetrievedChunk(chunk: Chunk, score: float)` (frozen dataclass),
  `hybrid_search(query_vector: np.ndarray, chunks: list[Chunk], dense_vectors: list[np.ndarray], bm25_scores: list[float], k: int = 3, dense_weight: float = 0.5) -> list[RetrievedChunk]`.
  Consumed by Tasks 7, 12.

- [ ] **Step 1: Write the failing test**

`tests/test_retrieval.py`:
```python
import numpy as np
import pytest

from app.domain.chunking import Chunk
from app.domain.retrieval import hybrid_search


def test_hybrid_search_returns_top_k_by_combined_score():
    chunks = [
        Chunk(text="a", source="doc.md", chunk_id=0),
        Chunk(text="b", source="doc.md", chunk_id=1),
        Chunk(text="c", source="doc.md", chunk_id=2),
    ]
    dense_vectors = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.9, 0.1]),
    ]
    query_vector = np.array([1.0, 0.0])
    bm25_scores = [5.0, 0.0, 1.0]

    results = hybrid_search(query_vector, chunks, dense_vectors, bm25_scores, k=2)

    assert len(results) == 2
    assert results[0].chunk.chunk_id == 0
    assert results[0].score >= results[1].score


def test_hybrid_search_rejects_mismatched_lengths():
    chunks = [Chunk(text="a", source="doc.md", chunk_id=0)]
    with pytest.raises(ValueError):
        hybrid_search(np.array([1.0]), chunks, [], [], k=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_retrieval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.domain.retrieval'`.

- [ ] **Step 3: Write minimal implementation**

`app/domain/retrieval.py`:
```python
from dataclasses import dataclass

import numpy as np

from app.domain.chunking import Chunk


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def hybrid_search(
    query_vector: np.ndarray,
    chunks: list[Chunk],
    dense_vectors: list[np.ndarray],
    bm25_scores: list[float],
    k: int = 3,
    dense_weight: float = 0.5,
) -> list[RetrievedChunk]:
    if not (len(chunks) == len(dense_vectors) == len(bm25_scores)):
        raise ValueError(
            "chunks, dense_vectors, and bm25_scores must be the same length"
        )

    dense_scores = [_cosine_similarity(query_vector, v) for v in dense_vectors]
    dense_norm = _normalize(dense_scores)
    bm25_norm = _normalize(list(bm25_scores))

    combined = [
        dense_weight * d + (1 - dense_weight) * b
        for d, b in zip(dense_norm, bm25_norm)
    ]

    ranked = sorted(zip(chunks, combined), key=lambda pair: pair[1], reverse=True)
    return [RetrievedChunk(chunk=c, score=s) for c, s in ranked[:k]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_retrieval.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/app/domain/retrieval.py modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/tests/test_retrieval.py
git commit -m "$(cat <<'EOF'
feat: add hybrid BM25/dense retrieval domain logic

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: RAG lab — prompt assembly and citations (TDD)

**Files:**
- Create: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/app/domain/rag.py`
- Test: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/tests/test_rag.py`

**Interfaces:**
- Consumes: `RetrievedChunk` (Task 6).
- Produces: `MAX_CONTEXT_CHARS: int = 2000`, `build_rag_prompt(question: str, retrieved: list[RetrievedChunk]) -> str`, `build_citations(retrieved: list[RetrievedChunk]) -> list[dict]`. Consumed by Task 12.

- [ ] **Step 1: Write the failing test**

`tests/test_rag.py`:
```python
from app.domain.chunking import Chunk
from app.domain.rag import build_citations, build_rag_prompt
from app.domain.retrieval import RetrievedChunk


def _sample_retrieved():
    return [
        RetrievedChunk(
            chunk=Chunk(
                text="Vacation policy is 20 days.",
                source="hr-policy.md",
                chunk_id=0,
            ),
            score=0.9,
        )
    ]


def test_build_rag_prompt_includes_question_and_context():
    prompt = build_rag_prompt("How many vacation days?", _sample_retrieved())
    assert "How many vacation days?" in prompt
    assert "Vacation policy is 20 days." in prompt
    assert "hr-policy.md" in prompt


def test_build_citations_returns_source_metadata():
    citations = build_citations(_sample_retrieved())
    assert citations == [
        {
            "source": "hr-policy.md",
            "chunk_id": 0,
            "snippet": "Vacation policy is 20 days.",
        }
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rag.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.domain.rag'`.

- [ ] **Step 3: Write minimal implementation**

`app/domain/rag.py`:
```python
from app.domain.retrieval import RetrievedChunk

MAX_CONTEXT_CHARS = 2000


def build_rag_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    context_parts = []
    total_chars = 0
    for i, item in enumerate(retrieved, start=1):
        snippet = item.chunk.text
        if total_chars + len(snippet) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(f"[{i}] (source: {item.chunk.source}) {snippet}")
        total_chars += len(snippet)

    context = "\n\n".join(context_parts)
    return (
        "You are a helpful assistant answering from the provided company "
        "documents.\nCite sources using [n] markers matching the numbered "
        f"context below.\n\nContext:\n{context}\n\n"
        f"Question: {question.strip()}\nAnswer:"
    )


def build_citations(retrieved: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "source": item.chunk.source,
            "chunk_id": item.chunk.chunk_id,
            "snippet": item.chunk.text[:200],
        }
        for item in retrieved
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rag.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/app/domain/rag.py modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/tests/test_rag.py
git commit -m "$(cat <<'EOF'
feat: add RAG prompt assembly and citation building

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: RAG lab — embeddings adapter, mock/real switch (TDD)

**Files:**
- Create: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/app/adapters/__init__.py` (empty)
- Create: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/app/adapters/embeddings.py`
- Test: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/tests/test_embeddings.py`

**Interfaces:**
- Produces: `MOCK_EMBEDDING_DIM: int = 64`, `EmbeddingCallError`,
  `EmbeddingClient(api_key: str | None)` with `.is_mock: bool` and `.embed(text: str) -> np.ndarray`.
  Consumed by Tasks 11, 12.

- [ ] **Step 1: Write the failing test**

`tests/test_embeddings.py`:
```python
from unittest.mock import patch

import numpy as np

from app.adapters.embeddings import EmbeddingClient


def test_embed_is_deterministic_in_mock_mode():
    client = EmbeddingClient(api_key=None)
    assert client.is_mock is True
    vec1 = client.embed("hello world")
    vec2 = client.embed("hello world")
    assert np.array_equal(vec1, vec2)


def test_embed_produces_different_vectors_for_different_text():
    client = EmbeddingClient(api_key=None)
    vec1 = client.embed("hello world")
    vec2 = client.embed("completely different sentence")
    assert not np.array_equal(vec1, vec2)


def test_embed_gives_shared_words_more_overlap_than_unrelated_text():
    client = EmbeddingClient(api_key=None)
    vec_a = client.embed("vacation policy days")
    vec_b = client.embed("vacation policy allowance")
    vec_c = client.embed("deploy rollback incident")

    def cosine(a, b):
        return float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

    assert cosine(vec_a, vec_b) > cosine(vec_a, vec_c)


def test_embed_calls_real_api_when_key_present():
    client = EmbeddingClient(api_key="fake-key")
    fake_vector = np.array([0.1, 0.2])
    with patch.object(
        client, "_call_real_api", return_value=fake_vector
    ) as mock_call:
        result = client.embed("hello")
    assert np.array_equal(result, fake_vector)
    mock_call.assert_called_once_with("hello")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_embeddings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.adapters.embeddings'`.

- [ ] **Step 3: Write minimal implementation**

`app/adapters/embeddings.py`:
```python
import hashlib

import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed

MOCK_EMBEDDING_DIM = 64


class EmbeddingCallError(RuntimeError):
    """Raised when the upstream embeddings call fails after retries."""


def _mock_embed(text: str) -> np.ndarray:
    vector = np.zeros(MOCK_EMBEDDING_DIM)
    for word in text.lower().split():
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % MOCK_EMBEDDING_DIM
        vector[index] += 1.0
    return vector


class EmbeddingClient:
    def __init__(self, api_key: str | None):
        self._api_key = api_key

    @property
    def is_mock(self) -> bool:
        return not self._api_key

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(0.1), reraise=True)
    def embed(self, text: str) -> np.ndarray:
        if self.is_mock:
            return _mock_embed(text)
        return self._call_real_api(text)

    def _call_real_api(self, text: str) -> np.ndarray:
        from openai import OpenAI

        try:
            client = OpenAI(api_key=self._api_key)
            response = client.embeddings.create(
                model="text-embedding-3-small", input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as exc:
            raise EmbeddingCallError(str(exc)) from exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_embeddings.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/app/adapters/__init__.py modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/app/adapters/embeddings.py modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/tests/test_embeddings.py
git commit -m "$(cat <<'EOF'
feat: add embeddings adapter with deterministic mock mode

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: RAG lab — index store with sha256 verification (TDD)

**Files:**
- Create: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/app/adapters/index_store.py`
- Test: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/tests/test_index_store.py`

**Interfaces:**
- Consumes: `Chunk` (Task 5).
- Produces: `ArtifactIntegrityError`, `LoadedIndex(chunks: list[Chunk], dense_vectors: list[np.ndarray], bm25_tokenized_corpus: list[list[str]], manifest: dict)` (frozen dataclass), `load_index(artifact_dir: Path) -> LoadedIndex`, `_sha256_of(path: Path) -> str`. Consumed by Tasks 11, 12.

- [ ] **Step 1: Write the failing test**

`tests/test_index_store.py`:
```python
import json

import numpy as np
import pytest

from app.adapters.index_store import ArtifactIntegrityError, _sha256_of, load_index


def _write_artifact(tmp_path):
    payload = {
        "chunks": [{"text": "hello world", "source": "doc.md", "chunk_id": 0}],
        "dense_vectors": [[0.1, 0.2, 0.3]],
        "bm25_tokenized_corpus": [["hello", "world"]],
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(payload))

    manifest = {
        "artifact_version": "0.1.0",
        "sha256": _sha256_of(index_path),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_load_index_succeeds_when_hash_matches(tmp_path):
    artifact_dir = _write_artifact(tmp_path)
    loaded = load_index(artifact_dir)
    assert loaded.chunks[0].source == "doc.md"
    assert np.array_equal(loaded.dense_vectors[0], np.array([0.1, 0.2, 0.3]))
    assert loaded.bm25_tokenized_corpus == [["hello", "world"]]


def test_load_index_raises_when_hash_mismatches(tmp_path):
    artifact_dir = _write_artifact(tmp_path)
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ArtifactIntegrityError):
        load_index(artifact_dir)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_index_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.adapters.index_store'`.

- [ ] **Step 3: Write minimal implementation**

`app/adapters/index_store.py`:
```python
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.domain.chunking import Chunk


class ArtifactIntegrityError(RuntimeError):
    """Raised when the index artifact does not match its manifest."""


@dataclass(frozen=True)
class LoadedIndex:
    chunks: list[Chunk]
    dense_vectors: list[np.ndarray]
    bm25_tokenized_corpus: list[list[str]]
    manifest: dict


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def load_index(artifact_dir: Path) -> LoadedIndex:
    manifest_path = artifact_dir / "manifest.json"
    index_path = artifact_dir / "index.json"
    manifest = json.loads(manifest_path.read_text())

    actual_sha256 = _sha256_of(index_path)
    if actual_sha256 != manifest["sha256"]:
        raise ArtifactIntegrityError(
            f"index.json sha256 {actual_sha256} does not match "
            f"manifest sha256 {manifest['sha256']}"
        )

    payload = json.loads(index_path.read_text())
    chunks = [
        Chunk(text=c["text"], source=c["source"], chunk_id=c["chunk_id"])
        for c in payload["chunks"]
    ]
    dense_vectors = [np.array(v) for v in payload["dense_vectors"]]
    bm25_tokenized_corpus = payload["bm25_tokenized_corpus"]

    return LoadedIndex(
        chunks=chunks,
        dense_vectors=dense_vectors,
        bm25_tokenized_corpus=bm25_tokenized_corpus,
        manifest=manifest,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_index_store.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/app/adapters/index_store.py modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/tests/test_index_store.py
git commit -m "$(cat <<'EOF'
feat: add index_store adapter with sha256 artifact verification

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: RAG lab — LLM client adapter (TDD)

**Files:**
- Create: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/app/adapters/llm_client.py`
- Test: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/tests/test_llm_client.py`

**Interfaces:**
- Produces: `LlmClient(api_key: str | None)` with `.is_mock: bool` and `.complete(prompt: str) -> str`; `LlmCallError`. Consumed by Task 12. This is the SAME code as Week 1's LLM lab
  (`modules/week-01-prototype-to-production/labs/llm-track-qa-service/app/adapters/llm_client.py`)
  — each lab is independently copyable, so it's duplicated rather than shared, exactly like Week
  1's `logging_utils.py`/`middleware.py` pattern.

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
    with patch.object(
        client, "_call_real_api", return_value="real answer"
    ) as mock_call:
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
                max_tokens=300,
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
git add modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/app/adapters/llm_client.py modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/tests/test_llm_client.py
git commit -m "$(cat <<'EOF'
feat: add llm_client adapter with mock mode and retries

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: RAG lab — corpus, ingestion script, and committed artifact

**Files:**
- Create: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/docs/product-faq.md`
- Create: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/docs/hr-policy.md`
- Create: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/docs/engineering-runbook.md`
- Create: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/ingest.py`
- Create (generated, then committed): `.../artifacts/index.json`, `.../artifacts/manifest.json`

**Interfaces:**
- Consumes: `chunk_text` (Task 5), `EmbeddingClient` (Task 8).
- Produces: `artifacts/index.json` + `artifacts/manifest.json` on disk, matching the schema
  `{chunks, dense_vectors, bm25_tokenized_corpus}` / `{artifact_version, created_at, git_commit, sha256, python_version, key_dependencies}`. Consumed by Task 12 (`main.py` lifespan calls `load_index` against this directory).

- [ ] **Step 1: Write `docs/product-faq.md`**

```markdown
# Product FAQ

## What does the product do?
Our platform helps small businesses track inventory in real time across
multiple warehouses. Orders sync automatically every 5 minutes.

## Is there a free trial?
Yes, every new account gets a 14-day free trial with full feature access.
No credit card is required to start the trial.

## How do I cancel my subscription?
Go to Account Settings > Billing > Cancel Subscription. Cancellations take
effect at the end of the current billing period.
```

- [ ] **Step 2: Write `docs/hr-policy.md`**

```markdown
# HR Policy Excerpt

## Vacation Policy
Full-time employees accrue 20 days of paid vacation per year, accrued
monthly. Unused vacation days roll over up to a maximum of 10 days into the
next calendar year.

## Remote Work
Employees may work remotely up to 3 days per week with manager approval.
Fully remote arrangements require VP-level sign-off.

## Parental Leave
Employees are eligible for 16 weeks of paid parental leave after 6 months
of continuous employment.
```

- [ ] **Step 3: Write `docs/engineering-runbook.md`**

```markdown
# Engineering Runbook Excerpt

## Deploying a Hotfix
1. Create a branch from `main` named `hotfix/<ticket-id>`.
2. Open a PR and get one approval from the on-call engineer.
3. Merge triggers an automatic canary deploy to 5% of traffic.
4. Monitor the error-rate dashboard for 15 minutes before promoting to 100%.

## Rolling Back a Deploy
Run `deploy rollback --service <name> --to <previous-version>` from the
deploy CLI. Rollbacks complete within 2 minutes and do not require approval.

## On-Call Rotation
On-call shifts are one week long, starting Monday at 9am. The on-call
engineer is the first responder for all P1 and P2 incidents.
```

- [ ] **Step 4: Write `ingest.py`**

```python
"""
Chunks the docs/ corpus, embeds every chunk (mock mode by default — real
embeddings are a per-request runtime choice, not an ingestion-time one, so
ingestion always uses the deterministic mock embedder), builds a BM25 index
over the same chunks, and writes a versioned, sha256-verified artifact.

Run with `make ingest` or `python ingest.py`.
"""
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from app.adapters.embeddings import EmbeddingClient
from app.domain.chunking import chunk_text

LAB_DIR = Path(__file__).parent
DOCS_DIR = LAB_DIR / "docs"
ARTIFACT_DIR = LAB_DIR / "artifacts"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def main() -> None:
    embedding_client = EmbeddingClient(api_key=None)

    all_chunks = []
    for doc_path in sorted(DOCS_DIR.glob("*.md")):
        text = doc_path.read_text(encoding="utf-8")
        all_chunks.extend(chunk_text(text, source=doc_path.name))

    dense_vectors = [
        embedding_client.embed(chunk.text).tolist() for chunk in all_chunks
    ]
    tokenized_corpus = [_tokenize(chunk.text) for chunk in all_chunks]

    payload = {
        "chunks": [
            {"text": c.text, "source": c.source, "chunk_id": c.chunk_id}
            for c in all_chunks
        ],
        "dense_vectors": dense_vectors,
        "bm25_tokenized_corpus": tokenized_corpus,
    }

    ARTIFACT_DIR.mkdir(exist_ok=True)
    index_path = ARTIFACT_DIR / "index.json"
    index_path.write_text(json.dumps(payload))

    digest = hashlib.sha256(index_path.read_bytes()).hexdigest()
    manifest = {
        "artifact_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "sha256": digest,
        "python_version": platform.python_version(),
        "key_dependencies": {"rank-bm25": "0.2.2"},
    }
    (ARTIFACT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"Wrote {index_path} and manifest.json "
        f"({len(all_chunks)} chunks, sha256={digest[:12]}...)"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run it to generate the committed artifact**

Run: `python ingest.py`
Expected: prints `Wrote .../index.json and manifest.json (N chunks, sha256=...)`, and
`artifacts/index.json` + `artifacts/manifest.json` now exist on disk.

- [ ] **Step 6: Commit**

```bash
git add modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/docs modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/ingest.py modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/artifacts
git commit -m "$(cat <<'EOF'
feat: add company-docs corpus, ingestion script, and versioned RAG index

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: RAG lab — API, config, logging, middleware, main (TDD)

**Files:**
- Create: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/app/api/__init__.py` (empty)
- Create: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/app/api/schemas.py`
- Create: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/app/api/routes.py`
- Create: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/app/config.py`
- Create: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/app/logging_utils.py`
- Create: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/app/middleware.py`
- Create: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/app/main.py`
- Create: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/tests/conftest.py`
- Test: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/tests/test_routes.py`
- Test: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/tests/test_artifact_integrity.py`

**Interfaces:**
- Consumes: `Chunk`, `chunk_text` (Task 5); `hybrid_search`, `RetrievedChunk` (Task 6);
  `build_rag_prompt`, `build_citations` (Task 7); `EmbeddingClient` (Task 8); `load_index`,
  `LoadedIndex`, `_sha256_of` (Task 9); `LlmClient`, `LlmCallError` (Task 10); `artifacts/` (Task 11).
- Produces: running FastAPI `app` with `POST /ask`, `GET /healthz`, `GET /readyz`.

- [ ] **Step 1: Write the failing tests**

`tests/conftest.py` (module-level, NOT a fixture function — this executes at import time, before
pytest's collection-time import of `app.main`, avoiding Week 1's fixture-timing bug from the
start):
```python
import json
import os
import tempfile
from pathlib import Path

from app.adapters.embeddings import EmbeddingClient
from app.adapters.index_store import _sha256_of
from app.domain.chunking import chunk_text

_artifact_dir = Path(tempfile.mkdtemp(prefix="rag-lab-test-artifacts-"))

_client = EmbeddingClient(api_key=None)
_chunks = chunk_text(
    "Vacation policy is 20 days per year for full-time employees.",
    source="hr-policy.md",
)
_dense_vectors = [_client.embed(c.text).tolist() for c in _chunks]
_tokenized_corpus = [c.text.lower().split() for c in _chunks]

_payload = {
    "chunks": [
        {"text": c.text, "source": c.source, "chunk_id": c.chunk_id}
        for c in _chunks
    ],
    "dense_vectors": _dense_vectors,
    "bm25_tokenized_corpus": _tokenized_corpus,
}

_index_path = _artifact_dir / "index.json"
_index_path.write_text(json.dumps(_payload))

_manifest = {
    "artifact_version": "0.1.0-test",
    "sha256": _sha256_of(_index_path),
}
(_artifact_dir / "manifest.json").write_text(json.dumps(_manifest))

os.environ["ARTIFACT_DIR"] = str(_artifact_dir)
os.environ["OPENAI_API_KEY"] = ""
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


def test_ask_returns_answer_with_citations():
    with TestClient(app) as client:
        response = client.post(
            "/ask", json={"question": "How many vacation days?"}
        )
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "mock"
    assert len(body["citations"]) >= 1
    assert body["citations"][0]["source"] == "hr-policy.md"


def test_ask_rejects_empty_question():
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422
```

`tests/test_artifact_integrity.py` (bypasses the `ARTIFACT_DIR` test override entirely — reads the
REAL committed artifact from Task 11 directly, giving CI real regression protection for the
artifact that actually ships, per the lesson from Week 1's final review):
```python
from pathlib import Path

from app.adapters.index_store import load_index


def test_committed_artifact_passes_integrity_check():
    artifact_dir = Path(__file__).resolve().parent.parent / "artifacts"
    loaded = load_index(artifact_dir)
    assert loaded.manifest["sha256"]
    assert loaded.manifest["artifact_version"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_routes.py tests/test_artifact_integrity.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.main'` (`app.main` is the only module
this task creates; `app.adapters.index_store` already exists from Task 9 and is not the cause of
the failure).

- [ ] **Step 3: Write minimal implementation**

`app/config.py`:
```python
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    artifact_dir: Path = Path(__file__).resolve().parent.parent / "artifacts"
    openai_api_key: str | None = None
```

`app/logging_utils.py` (identical pattern to Week 1's labs):
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

`app/middleware.py` (identical pattern to Week 1's labs):
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


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)


class Citation(BaseModel):
    source: str
    chunk_id: int
    snippet: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    source: str
```

`app/api/routes.py`:
```python
import logging

from fastapi import APIRouter, Request

from app.adapters.llm_client import LlmCallError
from app.api.schemas import AskRequest, AskResponse, Citation
from app.domain.rag import build_citations, build_rag_prompt
from app.domain.retrieval import hybrid_search

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request) -> AskResponse:
    state = request.app.state
    query_vector = state.embedding_client.embed(payload.question)
    bm25_scores = list(
        state.bm25_index.get_scores(payload.question.lower().split())
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

    citations = [Citation(**c) for c in build_citations(retrieved)]

    return AskResponse(
        question=payload.question,
        answer=answer,
        citations=citations,
        source="mock" if state.llm_client.is_mock else "llm",
    )


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
def readyz(request: Request) -> dict[str, str]:
    if getattr(request.app.state, "chunks", None) is None:
        return {"status": "not-ready"}
    return {"status": "ready"}
```

`app/main.py`:
```python
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from rank_bm25 import BM25Okapi

from app.adapters.embeddings import EmbeddingClient
from app.adapters.index_store import load_index
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
    loaded = load_index(settings.artifact_dir)
    app.state.chunks = loaded.chunks
    app.state.dense_vectors = loaded.dense_vectors
    app.state.bm25_index = BM25Okapi(loaded.bm25_tokenized_corpus)
    app.state.embedding_client = EmbeddingClient(api_key=settings.openai_api_key)
    app.state.llm_client = LlmClient(api_key=settings.openai_api_key)
    logger.info(
        f"index loaded, artifact_version={loaded.manifest['artifact_version']}, "
        f"chunks={len(loaded.chunks)}"
    )
    yield


app = FastAPI(title="RAG Q&A Service", lifespan=lifespan)
app.add_middleware(RequestIdMiddleware)
app.include_router(router)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_routes.py tests/test_artifact_integrity.py -v`
Expected: 5 passed (4 route tests + 1 artifact-integrity test).

- [ ] **Step 5: Run the full test suite with coverage**

Run: `pytest --cov=app --cov-report=term-missing`
Expected: all tests pass; coverage on `app/domain/` and `app/api/` is 80%+. If below 80%, add the
missing test case(s) (e.g. `readyz` returning not-ready when `app.state.chunks` is unset, or a
test for the `except LlmCallError` fallback path mirroring Week 1's
`test_ask_falls_back_gracefully_when_llm_call_fails`) before proceeding.

- [ ] **Step 6: Commit**

```bash
git add modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/app modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/tests
git commit -m "$(cat <<'EOF'
feat: add RAG FastAPI service with structured logging and request IDs

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: RAG lab — Dockerfile, Makefile, env example, README

**Files:**
- Create: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/Dockerfile`
- Create: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/Makefile`
- Create: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/.env.example`
- Create: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/.gitignore`
- Create: `modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/README.md`

- [ ] **Step 1: Write `.env.example`**

```
OPENAI_API_KEY=
```

- [ ] **Step 2: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.coverage
.env
```

(`artifacts/index.json` and `artifacts/manifest.json` ARE committed — same reasoning as Week 1's
model artifact: they're small and are the point of the versioning lesson.)

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

# Copy the application code and the versioned retrieval index. docs/ and
# ingest.py aren't needed at runtime — ingestion already happened at commit
# time, the same way Week 1's ML lab doesn't ship prototype.py/train_model.py.
COPY app/ ./app/
COPY artifacts/ ./artifacts/

# Document the port the service listens on (informational; doesn't publish it).
EXPOSE 8000

# Run the service with uvicorn. --host 0.0.0.0 is required so the server is
# reachable from outside the container, not just from localhost inside it.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 4: Write `Makefile`**

```makefile
.PHONY: setup ingest test run docker-build docker-run

setup:
	pip install -r requirements.txt

ingest:
	python ingest.py

test:
	pytest --cov=app --cov-report=term-missing

run:
	uvicorn app.main:app --reload

docker-build:
	docker build -t rag-service .

docker-run:
	docker run --rm -p 8000:8000 -e OPENAI_API_KEY rag-service
```

- [ ] **Step 5: Write `README.md`**

```markdown
# LLM Track Lab: RAG Q&A Service

Answers questions from a small corpus of company documents, with citations, using hybrid
BM25 + dense-embedding retrieval. Companion to [Week 2's concept README](../../README.md) and the
syllabus's Week 2 live demo.

## What's here

```
docs/               # the "company docs" corpus (product FAQ, HR policy, eng runbook)
ingest.py            # chunks + embeds docs/, builds the versioned retrieval index
app/
├── main.py           # wiring: logging, middleware, routes, index/client loading
├── api/               # routes (POST /ask) + schemas (with Citation)
├── domain/            # chunking.py, retrieval.py (hybrid search), rag.py (prompt + citations)
└── adapters/           # embeddings.py, llm_client.py (both mock-by-default), index_store.py
artifacts/            # the committed, sha256-verified retrieval index
```

## Run it (no API key needed)

If `make` isn't available (e.g. plain Windows without Git Bash), run the commands inside
`Makefile` directly — each target is a single command.

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
  -d '{"question": "How many vacation days do I get?"}'
curl http://localhost:8000/healthz
curl http://localhost:8000/readyz
```

The response's `citations` field names the exact source file and chunk that answered your
question — built from retrieval metadata, not parsed from the model's own output, so it stays
accurate even in mock mode.

## Re-ingest after changing the docs

```bash
make ingest
```

This regenerates `artifacts/index.json`/`manifest.json`. Commit the regenerated files the same
way Week 1's ML lab commits its retrained model artifact.

## Use real embeddings and a real LLM

Copy `.env.example` to `.env`, set `OPENAI_API_KEY=<your key>`, restart the service. Both the
embeddings client and the LLM client switch to the real OpenAI API — no code change, only config,
exactly like Week 1's LLM lab.

## Run it in Docker

```bash
make docker-build
make docker-run
```

## What to notice

- The retrieval index is stored as **JSON, not pickle** — a deliberate choice so this artifact
  never requires trusting a deserializer, unlike Week 1's `joblib`-based model artifact (which is
  fine there because it's first-party/trusted, but this lab shows the alternative when you don't
  need pickle at all).
- Corrupt `artifacts/index.json` and the service fails to start entirely with a clear
  `ArtifactIntegrityError` — the same fail-fast behavior as Week 1's model artifact check, verified
  and documented there.
- `app/domain/retrieval.py`'s combined BM25+dense score is a simplified stand-in for a real
  cross-encoder reranker — good enough to demonstrate hybrid retrieval, not state-of-the-art.
```

- [ ] **Step 6: Verify Docker build (best-effort)**

Run: `docker build -t rag-service .`
Expected: image builds successfully. If Docker isn't available in this environment, skip this step
and note it in your final report so the user can verify locally.

- [ ] **Step 7: Commit**

```bash
git add modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/Dockerfile modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/Makefile modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/.env.example modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/.gitignore modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service/README.md
git commit -m "$(cat <<'EOF'
docs: add Dockerfile, Makefile, and README for RAG lab

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: CI — add the RAG lab to the test matrix

**Files:**
- Modify: `.github/workflows/tests.yml`

**Interfaces:**
- Consumes: `llm-track-rag-service`'s `requirements.txt` and `tests/` (Tasks 5-13).

- [ ] **Step 1: Read the current `.github/workflows/tests.yml`**, then add
  `week-02-data-and-model-pipelines/labs/llm-track-rag-service` to the matrix's `lab` list. The
  final matrix should look like:

```yaml
    strategy:
      matrix:
        lab:
          - modules/week-01-prototype-to-production/labs/ml-track-fraud-detection
          - modules/week-01-prototype-to-production/labs/llm-track-qa-service
          - modules/week-02-data-and-model-pipelines/labs/llm-track-rag-service
```

Since the matrix values are now full relative paths from the repo root (rather than bare lab
names), also update the `defaults.run.working-directory` line from
`modules/week-01-prototype-to-production/labs/${{ matrix.lab }}` to simply `${{ matrix.lab }}`,
and update the `test_artifact_integrity`-style Week 1 ML-lab-specific logic if any remains (there
should be none left after Week 1's final review removed the retrain step — verify this and leave
the file otherwise unchanged). Read the file first and make the smallest edit that achieves this,
preserving every other step (checkout, setup-python, install, ruff, pytest with coverage gate)
exactly as-is.

- [ ] **Step 2: Validate the YAML**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/tests.yml'))"` (from the repo
root) and confirm it parses without error.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/tests.yml
git commit -m "$(cat <<'EOF'
ci: add RAG lab to the pytest matrix

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 15: Code review and security review pass

**Files:** none created — this task reviews everything from Tasks 2-14 and applies fixes in place.

- [ ] **Step 1: Run the code-reviewer agent**

Dispatch the `code-reviewer` agent against both new labs'
`modules/week-02-data-and-model-pipelines/labs/*/` directories (both `app/`+`tests/` for the RAG
lab, and the flat `*.py`+`tests/` layout for the experiment-tracking lab). Ask it to check code
quality, error handling, and maintainability per this repo's standards (functions <50 lines, no
deep nesting, explicit error handling, no hardcoded secrets), and specifically to check whether
the two Week 1 labs' post-review lessons (logging in the `LlmCallError` fallback path, no CI step
that trains/ingests something tests don't need, `docker-run` not requiring a `.env` file) were
correctly applied here from the start rather than needing a second round of fixes.

- [ ] **Step 2: Run the security-reviewer agent**

Dispatch the `security-reviewer` agent against the same directories, specifically the RAG lab's
`api/routes.py` and the adapters making external calls (`adapters/llm_client.py`,
`adapters/embeddings.py`). Ask it to check for injection risks, secret leakage into logs,
unvalidated input, and to explicitly confirm the index artifact's JSON-only (no pickle)
deserialization has no unsafe-deserialization finding at all (unlike Week 1's `joblib.load`, which
required an explicit trust-boundary note — this artifact needs no such caveat).

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

### Task 16: Push to GitHub

**Files:** none.

- [ ] **Step 1: Push**

The repo already exists (`jahirsaiyed/production-grade-ai-systems`, public). From the repo root:
```bash
git push
```
Expected: pushes all Week 2 commits to `origin/master`.

- [ ] **Step 2: Verify**

Run: `gh repo view --json url,visibility,defaultBranchRef`
Expected: JSON showing the repo URL, `"visibility": "PUBLIC"`, default branch `master`.

- [ ] **Step 3: Report back**

Report the repo URL, which Docker-build verification steps (Task 13) were skipped because Docker
wasn't available in this environment, and a one-line summary of Task 15's review findings.
