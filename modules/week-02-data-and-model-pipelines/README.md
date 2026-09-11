# Week 2: Data and Model Pipelines

Week 1 got a model and an LLM app into a service with retries, health checks,
and a versioned artifact. Week 2 goes one stage upstream: how does data get
into a shape a model can train on or an LLM can retrieve from, and how do you
prove *which* data and *which* code produced the artifact that's now running?
You'll work through the same two tracks again, once per concept set: **ML**
(a data-validation-gated, MLflow-tracked training pipeline in
[`labs/ml-track-experiment-tracking/README.md`](labs/ml-track-experiment-tracking/README.md))
and **LLM** (a hybrid-retrieval RAG service with citations in
[`labs/llm-track-rag-service/README.md`](labs/llm-track-rag-service/README.md)).

## Ingestion patterns (batch vs. streaming, ETL vs. ELT)

**Batch** ingestion processes a bounded chunk of data on a schedule or on
demand (an hourly job, a nightly job, a one-off run); **streaming** ingestion
processes an unbounded sequence of events continuously, as they arrive.
Separately, **ETL** (extract, transform, load) transforms data *before*
writing it to its destination, while **ELT** (extract, load, transform) lands
raw data first and transforms it later, usually inside the destination
system (a warehouse or lake) where transformation can be replayed or redone.

This matters in production because the choice isn't cosmetic: streaming
buys you freshness at the cost of needing a broker, backpressure handling,
and out-of-order-event logic; ELT buys you flexibility to reprocess history
at the cost of storing (and governing) raw data you haven't validated yet.
This week's RAG lab's `ingest.py` is a **batch, ETL-style job**: it reads the
whole `docs/` corpus, chunks and embeds it (the "transform" step), and only
then writes the finished artifact (the "load" step) — run on demand via
`make ingest`, not a continuously running pipeline.

## Data validation gates and schema enforcement

A data validation gate is a checkpoint that runs *before* an expensive step
(training, indexing) and refuses to proceed if the input data violates known
constraints — wrong shape, missing values, out-of-range labels, a schema
that silently drifted. Without one, bad data doesn't announce itself: it
trains a model that's subtly wrong, or gets discovered three steps later
when someone's confused why accuracy dropped.

The ML lab makes this concrete: `data_validation.py`'s
`validate_training_data(X, y)` checks that `X` and `y` have matching row
counts, that `X` contains no `NaN`/infinite values, and that `y`'s labels are
within the expected set — and it raises `DataValidationError` rather than
letting a bad dataset reach `mlflow.start_run()`. That ordering (validate,
*then* spend the expensive step) is exactly what Exercise 1 asks you to
break on purpose and observe.

## Data lineage and versioning

Lineage means being able to answer, after the fact, "what data and what code
produced this specific artifact?" — a question that's easy to ignore until
a stakeholder asks why last Tuesday's model behaves differently from
today's, and "I don't know, we didn't record it" is not an acceptable
answer in a regulated or high-stakes setting. Versioning is the mechanism
that makes lineage answerable: every meaningful artifact carries a stamp
tying it back to the exact inputs and code revision that built it.

Both of this week's labs produce a lineage record, in different shapes. The
ML lab's `reproducibility_manifest.json` carries `git_commit` (the exact
code revision) and `data_digest` (a sha256 hash of the exact training data),
so any two runs can be compared byte-for-byte. The RAG lab's
`artifacts/manifest.json` plays the same role for the retrieval index: it
records the `git_commit` and a sha256 of `index.json`, so you always know
which corpus and which ingestion code produced the index a running service
loaded.

## Training/serving parity; leakage detection

**Training/serving parity** means the exact same feature computation runs at
training time and at inference time — Week 1 called this out as
"training/serving skew" when the two differ. **Leakage** is a related but
distinct failure: information that wouldn't be available at real prediction
time (a future value, a label-derived feature, a duplicate row that ended up
in both the train and test split) sneaks into training and makes offline
metrics look better than the model will ever perform in production.

Neither of this week's labs has a serving/training split to demonstrate
leakage directly — the ML lab only trains, it doesn't serve, and the RAG
lab doesn't train a model at all. Keep this section conceptual for now: it
becomes concrete once a model trained with this week's reproducibility
practices gets wrapped in a serving layer using Week 1's patterns, and
you're checking that the two agree on how features are computed.

## Deep-learning data paths (loaders, batching, augmentation, distributed training)

Deep learning at scale needs its own data-plumbing layer: **data loaders**
that stream examples off disk without holding an entire dataset in memory,
**batching** to group examples for parallel GPU computation, **augmentation**
to synthetically expand a dataset (random crops, token masking) and reduce
overfitting, and **distributed training** to split a model or dataset across
multiple GPUs or machines when either is too large for one device.

This section is explicitly conceptual only — neither lab this week trains a
deep neural network (the ML lab trains a `LogisticRegression`; the RAG lab
doesn't train anything). None of these four mechanisms is demonstrated in
running code this week; they're named here so you recognize them by name
when you encounter a real deep-learning training pipeline.

## Document parsing (OCR, PDF, HTML) and chunking strategies

Before a document can be retrieved, its raw form — a scanned PDF needing OCR,
an HTML page full of markup noise, a Word document — has to become clean
text. **Chunking** then splits that text into pieces small enough to fit a
model's context window (and an embedding model's input limit) while trying
to preserve semantic coherence, so a chunk doesn't cut a sentence or an idea
in half at an arbitrary boundary.

This repo's RAG corpus (`docs/*.md`) is already plain markdown, so OCR/PDF/
HTML parsing isn't exercised this week — only chunking is. `chunking.py`'s
`chunk_text` implements a **fixed-size-with-overlap** strategy: it splits
text into word-count windows of `chunk_size` words, and each subsequent
window starts `overlap` words before the previous one ended, so a sentence
near a boundary usually appears whole in at least one chunk instead of being
truncated in every chunk that touches it.

## Vector DB operations (indexing, upserts, metadata filters)

A real vector database (Pinecone, Weaviate, pgvector, and similar) provides
three things this week's lab deliberately does *not* build: persistent
**indexing** structures (e.g. HNSW) for fast approximate nearest-neighbor
search over millions of vectors, **upserts** so a single changed document
can be added or replaced without rebuilding the whole index, and **metadata
filters** so a query can be restricted to a subset of vectors (by tenant,
document type, date range) before or during the similarity search.

The RAG lab's `ingest.py` plus `adapters/index_store.py` are a deliberately
simplified stand-in: an in-process, in-memory index built fresh on every
`make ingest` run, with no incremental upserts and no metadata filtering —
just enough machinery to teach retrieval mechanics (embedding, scoring,
ranking) without standing up real infrastructure. Exercise 5 (adding a
fourth document and re-running `make ingest`) is what a real vector DB's
upsert would replace with an incremental call instead of a full rebuild.

## Context packaging (ordering, token caps, citations)

Once you've retrieved candidate chunks, you still have to assemble them into
a prompt: in what order, capped at what size (so the prompt plus the
question stays inside the model's context window and cost budget), and with
what markers so the model's answer — and your service's response — can be
traced back to a specific source chunk.

The RAG lab's `app/domain/rag.py` does exactly this. `build_rag_prompt`
walks retrieved chunks in score order, stops adding chunks once a running
character budget (`MAX_CONTEXT_CHARS`) would be exceeded, and prefixes each
included chunk with a numbered marker (`[1]`, `[2]`, ...) and its source
file. `build_citations` builds the response's citation list directly from
retrieval metadata — not by parsing the LLM's own output — which is what
keeps citations accurate even when the LLM hallucinates or (in mock mode)
returns a canned string.

## Hybrid retrieval: BM25 + dense embeddings + reranking; GraphRAG

**BM25** is a keyword-based ranking function: it scores a document highly
when it contains the query's exact terms (or close variants), which makes it
strong at exact-match and rare-term queries but blind to paraphrase — it
won't connect "time off" to "vacation days" unless the words literally
overlap. **Dense embeddings** score by semantic similarity in vector space,
so they *do* catch paraphrases and synonyms, but can miss a query that hinges
on an exact, distinctive term buried in a long document. Combining both —
hybrid retrieval — typically outperforms either alone, because each covers
the other's blind spot.

`app/domain/retrieval.py`'s `hybrid_search` implements this: it computes a
dense cosine-similarity score and a BM25 score per chunk, min-max normalizes
each list independently, and combines them with a configurable weight
(50/50 by default). Be clear-eyed about what this is and isn't: the combined-
score step is a simplified stand-in for a real reranker — production hybrid
pipelines typically run a cross-encoder model over the top candidates for a
final, much more accurate reordering, which this lab does not implement.
Exercise 4 asks you to shift the weighting toward BM25 and observe how that
changes which chunk wins for a keyword-heavy versus a paraphrase-heavy
question.

**GraphRAG**, briefly and conceptually only (not built this week): instead of
retrieving isolated chunks, it first builds a knowledge graph of entities and
relationships extracted from the corpus, then retrieves and reasons over
connected subgraphs — useful when an answer depends on relationships that
span multiple documents (e.g. "which team owns the service that the
on-call runbook says to page for a payments incident?") rather than a
single, self-contained passage that a plain hybrid search can hand you
directly.

## Experiment tracking (MLflow, Weights & Biases)

Experiment tracking records the inputs, outputs, and metadata of a training
run — hyperparameters, metrics, the model artifact itself — in a queryable
store, so you can compare runs, find the best one, and reconstruct what
produced a given result without digging through scattered print statements
or spreadsheets.

The ML lab's `train_with_tracking.py` uses **MLflow** with its local file
backend (`file:./mlruns`, no server to stand up) — it logs the run's
hyperparameters (`mlflow.log_params`), its `accuracy`/`f1` metrics
(`mlflow.log_metrics`), and the trained model itself
(`mlflow.sklearn.log_model`). **Weights & Biases (W&B)** is the SaaS
alternative — hosted, with richer dashboards and team collaboration
features out of the box — not used in this lab, but worth knowing as the
name that comes up most often alongside MLflow.

## Model registry stages

A model registry is a versioned catalog of trained models with **stage**
metadata — typically *staging* (candidate, being validated), *production*
(currently serving live traffic), and *archived* (retired, kept for
rollback/audit) — plus who approved each promotion and when. It's what lets
a team answer "which model version is live right now, and can we roll back
to the one before it?" without spelunking through file timestamps.

Neither of this week's labs implements a registry — that's out of scope
here. The closest relative already in this repo is Week 1's
`artifacts/manifest.json` in the fraud-detection lab: a single-model,
single-file version of the same idea (a version string and metadata tied to
one artifact), just without the multi-model catalog or stage transitions a
real registry like MLflow's Model Registry manages at scale.

## Hyperparameter tuning

Hyperparameter tuning is the search for the configuration values a training
algorithm doesn't learn on its own (learning rate, regularization strength,
number of trees, and similar) — usually via grid search, random search, or a
more sample-efficient method like Bayesian optimization, evaluated against a
validation metric.

`train_with_tracking.py` doesn't search: it logs one fixed hyperparameter
set (`max_iter=1000`, `n_features=6`, and similar) to MLflow rather than
sweeping over a range. A natural extension exercise is to wrap the training
call in a loop over a few values of `max_iter` or `n_features`, log each as
its own MLflow run, and use the MLflow UI (Exercise 2) to compare them
side by side.

## Reproducibility checklist (commit, data digest, env lock, artifact link)

A reproducibility checklist is the minimum set of facts you need recorded
alongside any trained model to answer "can we reproduce this exact result,
or explain exactly why we can't?" Four items make up the version used in
this course: the **commit** that ran, a **data digest** of the exact
training data, an **env lock** of the exact dependency versions, and an
**artifact link** back to the trained model itself.

`train_with_tracking.py`'s `reproducibility_manifest.json` implements this
checklist field by field: `git_commit` is the commit hash (via
`git rev-parse HEAD`); `data_digest` is a sha256 hash of the generated
training data's bytes, so any change to the dataset — even one flipped
value — changes the digest; `env_lock` is a sha256 hash of
`requirements.txt`, standing in for a full dependency lock; and
`mlflow_run_id`/`artifact_uri` are the link back to the tracked MLflow run
and its saved model artifact. Exercise 2 has you open this file's fields
after a real run and explain what each one guards against.

## Adaptation decision framework: prompting vs. RAG vs. PEFT (LoRA) vs. full fine-tuning

When an LLM needs to behave differently than it does out of the box, there's
a spectrum of ways to get there, roughly ordered by increasing cost and
decreasing turnaround time: **prompting** (just ask better, zero training,
cheapest, fastest to iterate) → **RAG** (retrieve relevant context and feed
it to the prompt at request time — no training, but needs a retrieval
pipeline and up-to-date data) → **PEFT/LoRA** (train a small number of
additional parameters on top of a frozen base model — cheaper and faster
than full fine-tuning, good for adapting style or a narrow skill) → **full
fine-tuning** (update all of the base model's weights — most expensive,
most capable of deep behavioral change, and the slowest to iterate on).

This week's RAG lab is a concrete example of the "RAG" branch: it doesn't
touch model weights at all, it retrieves relevant chunks and packages them
into the prompt at request time. That's a direct contrast with Week 1's
LLM lab, which sits one branch to the left, at plain "prompting" — it
answers straight from the model with no retrieval and no training at all.
Neither of this week's labs demonstrates the PEFT/LoRA or full fine-tuning
branches; they're named here as the next steps up the cost/capability curve.

## Hands-on labs

- [`labs/ml-track-experiment-tracking/README.md`](labs/ml-track-experiment-tracking/README.md) —
  ML track: a data-validation-gated training pipeline with MLflow tracking
  and a reproducibility manifest.
- [`labs/llm-track-rag-service/README.md`](labs/llm-track-rag-service/README.md) —
  LLM track: a hybrid-retrieval (BM25 + dense) RAG FastAPI service that
  answers from company docs with citations.

## Exercises

See [`exercises.md`](exercises.md) for five hands-on exercises that build on
both labs above.
