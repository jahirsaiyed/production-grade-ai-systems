# Week 1: From Prototype to Production

This module is about the gap between "it works in a notebook" and "it's a
service other systems can depend on." You'll work through the same ideas
twice, once per track: **ML** (a fraud-detection scoring service in
[`labs/ml-track-fraud-detection/README.md`](labs/ml-track-fraud-detection/README.md))
and **LLM** (a Q&A service in
[`labs/llm-track-qa-service/README.md`](labs/llm-track-qa-service/README.md)).
Both labs start from a throwaway script and end as a layered, containerized
FastAPI service with versioned artifacts, retries, structured logs, and
health checks — the same production skeleton you'll reuse for the rest of
the course.

## Prototype vs. production systems, and training/serving skew

A prototype is a script: it runs top to bottom, in one process, on your
machine, with hardcoded inputs, and if it crashes you just rerun it. A
production system is a long-running service that strangers (or other
services) call over the network, with inputs it has never seen, that has to
keep working when a dependency is slow, when the input is malformed, and
when nobody is watching the terminal. Getting from one to the other is what
this whole week is about.

One of the most common ways prototypes quietly fail in production is
**training/serving skew**: the features or inputs your model sees at
inference time differ from what it saw during training, even though nobody
changed the model. This isn't an exotic edge case — it's usually a plumbing
bug. For example, imagine a "days since last transaction" feature computed
in a training notebook using calendar days from a batch SQL query, but
computed in the live service using elapsed seconds divided by 86400 from a
slightly different clock source or timezone. The two computations *look*
equivalent but produce subtly different numbers, so the model — trained on
one distribution — scores live traffic against a different one. Both labs
sidestep this specific failure mode by keeping the feature computation
outside the model's control (`adapters/feature_store.py` in the fraud lab
returns a fixed-shape feature vector used identically at both train and
serve time), but it's worth watching for any time your training pipeline
and serving pipeline are two different codebases.

## The AI lifecycle: data -> packaging -> serving -> monitoring

Every production AI system, whether it's a fraud model or an LLM app, moves
through the same four stages. What passes between stages matters as much as
what happens inside each one — the thing that crosses the boundary is a
**handoff artifact**: a concrete, versioned file (or pair of files) that the
next stage can consume without needing to know how it was produced.

```
   DATA                 PACKAGING                SERVING                MONITORING
 ┌────────┐          ┌───────────────┐        ┌───────────────┐      ┌────────────────┐
 │ raw    │  train    │ train_model.py│ wraps  │  FastAPI app  │ logs │ structured logs │
 │ events,│ ───────▶  │  produces:    │──────▶ │  loads the    │────▶ │ request IDs,    │
 │ labels │           │  model.joblib │        │  artifact,    │      │ latency, error  │
 └────────┘           │  manifest.json│        │  exposes      │      │ rates, drift     │
                       └───────┬───────┘        │  /score,/ask  │      │ (Week 4)         │
                               │                └───────────────┘      └────────┬────────┘
                               │  HANDOFF ARTIFACT                                │
                               │  (versioned model + manifest,                    │
                               │   sha256-checked at load time)                   │
                               └───────────────────────────────────────────────────┘
                                          feedback loop back into data/training
```

Why this matters in production: if "packaging" isn't a distinct, versioned
step, you can't answer "which model is actually running right now?" or
"can I safely roll back?" In the fraud-detection lab, `train_model.py` is
the packaging step — it turns a training run into `artifacts/model.joblib`
plus `artifacts/manifest.json`, and that pair (not the training script, not
the notebook) is exactly what `app/main.py` loads at startup. The LLM lab's
handoff artifact is lighter-weight — the prompt-building logic in
`app/domain/qa.py` — since Week 1's Q&A service doesn't fine-tune or version
a model file (fine-tuning and retrieval come in later weeks).

## Batch vs. online inference; monolith vs. microservices vs. event-driven

Two independent decisions shape how a model gets used: *when* you run
inference, and *how* the surrounding system is structured.

| | Latency | Throughput | Complexity | When to choose it |
|---|---|---|---|---|
| **Batch inference** | Minutes to hours (not user-facing) | High — process millions of rows in one job | Lower — no live API to keep up | Nightly risk re-scoring, precomputed recommendations, reporting |
| **Online inference** | Milliseconds to low seconds, on the request path | Bounded by concurrent request capacity | Higher — needs uptime, retries, monitoring | A user or system is waiting on the answer right now |

| | Latency overhead | Throughput/scaling | Operational complexity | When to choose it |
|---|---|---|---|---|
| **Monolith** | Lowest — everything in one process | Scale the whole app together | Simplest to run and deploy | Small team, one service, this week's labs |
| **Microservices** | Extra network hop(s) between services | Scale each service independently | Higher — service discovery, versioning, more moving parts | Independent teams/scaling needs per capability |
| **Event-driven** | Often asynchronous, not request/response | Very high, decoupled producers/consumers | Highest — need a broker, ordering, retries, dead-letter handling | Fan-out workloads, decoupling slow downstream steps |

Both Week 1 labs are deliberately the simplest cell in each table: **online
inference behind a synchronous, monolithic API** (`POST /score` and
`POST /ask` each do one model/LLM call and return one answer, no queues,
no separate services). That's the right starting point for learning the
mechanics — Week 3 and Week 6 revisit batching, caching, and
service-oriented topologies once the fundamentals are solid.

## Production building blocks

A handful of building blocks show up in almost every production AI
architecture. This week's labs only actually build one of them (a
simplified feature store) — the rest are named here so you recognize them
in the wild and in later weeks:

- **API gateway** — a shared front door for all API traffic that centralizes
  auth, rate limiting, and routing so individual services don't reimplement
  them; neither lab has one, they're accessed directly for simplicity.
- **Feature store** — a system that computes and serves consistent features
  for both training and inference, which is precisely the mechanism that
  prevents the training/serving skew described above. `adapters/feature_store.py`
  in the fraud-detection lab is a **deliberately simplified stand-in**: it
  returns a fixed, fake feature vector (with a configurable random failure
  rate to make retries observable) instead of querying a real online store
  like Feast or a Redis-backed cache.
- **Vector database** — stores embeddings for similarity search, the backbone
  of retrieval-augmented generation; not used this week (the Q&A lab
  answers directly from the LLM, no retrieval) — it shows up in Week 2's RAG
  pipeline.
- **Model registry** — a versioned catalog of trained models with stage
  metadata (staging/production, who approved it); `artifacts/manifest.json`
  in the fraud lab is a one-model, single-file analogy to what a real
  registry like MLflow's Model Registry tracks at scale.

## Reliability primitives: retries, timeouts, queues, graceful degradation

Production dependencies fail — networks drop packets, downstream services
get slow, third-party APIs rate-limit you. Four primitives are the standard
toolkit for handling that:

- **Retries** — automatically re-attempt a failed call, usually with a cap
  and a wait between attempts, because many failures are transient. Both
  labs implement this concretely with `tenacity`: `fetch_features()` in the
  fraud lab's `adapters/feature_store.py` attempts the call up to 3 times
  total before giving up
  (`@retry(stop=stop_after_attempt(3), wait=wait_fixed(0.1))`) — that's the
  first try plus 2 retries, and `LlmClient.complete()` in the Q&A lab's `adapters/llm_client.py`
  uses the identical pattern around the OpenAI call.
- **Timeouts** — bound how long you'll wait for a dependency before giving up,
  so one slow call can't stall your whole service; neither lab configures an
  explicit network timeout (they lean on the retry library's bounded attempt
  count instead), which is worth calling out as a simplification rather than
  a best practice to copy as-is.
- **Queues** — decouple a slow or bursty producer from a consumer by buffering
  work between them; out of scope for this week's synchronous request/response
  labs, but essential once you need to smooth out load spikes or run batch
  jobs (see the event-driven row in the table above).
- **Graceful degradation** — when a dependency is unavailable, return a
  degraded-but-useful response instead of an error, when a soft answer is an
  acceptable substitute for a real one. The Q&A lab's `POST /ask` route
  demonstrates this directly: if `client.complete()` raises `LlmCallError`
  after exhausting retries, the route catches it, logs a warning, and
  returns a fallback answer ("The assistant is temporarily unavailable.
  Please try again.") with a normal 200 response, rather than surfacing a
  500 — a "fail soft" strategy that fits a conversational UI, where a
  degraded answer beats an error page. The fraud lab's `/score` route takes
  the opposite approach on purpose: it *does* catch `FeatureStoreUnavailable`
  after retries are exhausted, but rather than degrading, it logs the error
  through the structured logger (so it's visible with a request ID) and
  returns an explicit `503 Service Unavailable`. That's "fail loud, but
  cleanly" — appropriate here because a fraud score computed without real
  features wouldn't be a harmless degraded answer, it would be an actively
  wrong one, and silently returning it (or guessing) is worse than telling
  the caller to retry.

## REST vs. gRPC; streaming responses

REST (what both labs use, over plain HTTP/JSON) sends human-readable text
that you can `curl` and read with your eyes, has broad tooling and library
support, and is easy to debug by hand — at the cost of some serialization
overhead and looser type guarantees than a schema-first protocol. gRPC uses
a binary wire format (protobuf) over HTTP/2, is typically faster and more
bandwidth-efficient, and enforces a strongly-typed contract between client
and server — at the cost of needing generated stubs and specialized tooling
to poke at it manually, which slows down debugging and quick iteration.

For services calling each other internally at high volume, gRPC's
efficiency and type safety often win. For a public-facing API, or one you
want engineers (and instructors) to be able to `curl` directly while
learning, REST wins — which is why both Week 1 labs expose plain REST/JSON
endpoints (`POST /score`, `POST /ask`).

**Streaming responses** — sending a response incrementally instead of all at
once — matter most for LLM output, where you want to render tokens to the
user as the model generates them instead of waiting for the full completion.
Neither lab streams: `POST /ask` waits for the complete OpenAI response (or
mock string) and returns it in one JSON payload. Token-by-token streaming is
a natural next step once you're comfortable with the synchronous version
here.

## Layered service architecture: API, domain, adapters

Both labs are organized into three layers with a strict, one-directional
dependency rule, so business logic never gets tangled up with HTTP or I/O
concerns:

```
                    app/api/
        (routes.py, schemas.py — FastAPI routes,
         request/response validation)
              │                      │
   depends on │                      │ depends on
              ▼                      ▼
   app/domain/                app/adapters/
   (scoring.py / qa.py —      (model_store.py, feature_store.py,
   pure business logic,       llm_client.py — isolate all I/O:
   no FastAPI/HTTP imports)   files, the network, retries)
```

`app/api/` is the only layer that knows about FastAPI, HTTP status codes, and
request/response shapes — it depends on both other layers to do the actual
work. `app/domain/` (`scoring.py` in the fraud lab, `qa.py` in the Q&A lab)
holds pure business logic with zero framework imports, which is exactly why
Task 3/9's domain tests can import and test it directly with plain stub
objects, no running server required. `app/adapters/` isolates everything
that touches the outside world — a file on disk (`model_store.py`), a
simulated flaky dependency (`feature_store.py`), or a real network call
(`llm_client.py`) — so that swapping a real feature store or a different LLM
provider in later weeks only touches one file, not the whole codebase.

## Configuration management, secrets separation, request IDs, and retry budgets

Both labs load configuration through a `pydantic-settings` `Settings` class
(`app/config.py`) that reads from environment variables and an optional
`.env` file, validating types (e.g., `feature_store_failure_rate: float`)
at startup instead of failing deep inside request handling. `.env` is
git-ignored on purpose — it's where you'd put a real `OPENAI_API_KEY` or any
other secret — while `.env.example` **is** committed, listing the variable
names (with placeholder or default values, never real secrets) so anyone
cloning the repo knows exactly what to configure without ever seeing an
actual credential.

Every request also gets a **request ID**, handled by `app/middleware.py`'s
`RequestIdMiddleware`: it reads an incoming `X-Request-ID` header if the
caller supplied one, otherwise generates a fresh UUID, stores it in a
`contextvars.ContextVar` for the duration of the request (so any log line
emitted while handling it can pick the ID up automatically), and echoes it
back on the response's `x-request-id` header. That's what makes Exercise 5
possible: trace one request across a client, the service logs, and the
response.

**Retry budgets** are the cap that keeps "retries" from becoming "retry
forever": `stop_after_attempt(3)` on both `fetch_features()` and
`LlmClient.complete()` is a retry budget — it bounds the worst-case latency
and cost of a single request to at most 3 attempts, trading a slightly
higher chance of a single request failing for a hard guarantee that no
request can hang or hammer a struggling dependency indefinitely (Exercise 2
asks you to reason through exactly this tradeoff).

## Async programming for I/O-bound calls

Network calls and LLM completions are I/O-bound: the CPU is idle while
waiting on a response, so an `async def` route (paired with an async client
library) lets a single process handle many concurrent requests without
spinning up a thread per request — a big win when a service spends most of
its time waiting on an upstream API. Both Week 1 labs keep their route
handlers and adapters **synchronous** (`def`, not `async def`) on purpose,
because FastAPI transparently runs sync route handlers in a worker thread
pool, which is simpler to reason about while you're still learning the
layered structure, and it's plenty fast for the mock/simulated calls this
week's labs make.

This is a real simplification, though: under real production load with a
real, sometimes-slow LLM API, converting `POST /ask` to `async def` and
swapping `LlmClient`'s OpenAI calls for the SDK's async client
(`AsyncOpenAI`) would let the service handle far more concurrent requests
per worker process. Making that change without breaking the existing tests
is a good self-directed extension exercise once you've finished the labs as
written.

## Structured logging, health, and readiness endpoints

Both labs configure logging (`app/logging_utils.py`) to emit one JSON object
per log line instead of free-text, which is what lets log aggregation tools
(and humans doing `grep`/`jq`) query logs by field instead of parsing
strings. Every line carries the current request ID via the same context
variable the middleware sets, so a single request's logs can be pulled out
by ID even under concurrent traffic. A sample line from the fraud lab at
startup looks like this:

```json
{"level": "INFO", "message": "model loaded, artifact_version=0.1.0", "logger": "app.main", "request_id": "-"}
```

(`request_id` is `"-"` here because model loading happens at startup,
outside any single request's context; during an actual `/score` call it
would carry that request's UUID instead.)

The two health endpoints answer different questions. **`/healthz`** answers
"is the process alive at all?" — it always returns `{"status": "ok"}` the
instant the server can respond, regardless of whether anything is fully
initialized, which is what an orchestrator uses to decide whether to kill
and restart a stuck process. **`/readyz`** answers "is this instance able to
serve real traffic right now?" — it returns `{"status": "ready"}` only once
the fraud lab's model (or the Q&A lab's LLM client) has finished loading, and
`{"status": "not-ready"}` otherwise, which is what an orchestrator or load
balancer uses to decide whether to route traffic to this instance yet.

In practice, in this week's two labs, `/readyz`'s not-ready branch is
defensive code that isn't currently reachable — the ML lab fails the whole
process at startup on artifact corruption rather than starting in a
not-ready state (see the lab's own README), and the LLM lab's client can't
fail to construct. A production system with a slower-loading dependency
(e.g. downloading a large model from remote storage) is where this branch
would actually fire — see if you can design a scenario for one of these
labs where it would.

## Model serialization: versioned artifacts with sha256 metadata

The fraud-detection lab never loads `artifacts/model.joblib` on trust alone.
`app/adapters/model_store.py` computes the sha256 hash of the model file on
disk and compares it against the `sha256` field recorded in
`artifacts/manifest.json` at the moment the artifact was produced; if they
don't match, `load_model()` raises `ArtifactIntegrityError` instead of
silently loading a possibly-corrupted or mismatched file.

Why this matters in production: a model artifact can get truncated by a
failed upload, overwritten by the wrong build, or shipped alongside a
manifest describing a different training run — and a corrupted or swapped
model doesn't usually crash on load, it just quietly produces wrong
predictions. A hash check turns that silent failure into a loud, immediate
one at startup (or at `/readyz`), which is exactly what Exercise 1 asks you
to go verify by hand.

## Packaging path: notebook -> artifact -> API -> container image

The fraud-detection lab makes this transformation concrete and walkable,
file by file:

1. **`prototype.py`** — the "notebook cell": trains a model inline and
   prints one prediction, with no structure, no tests, no versioning.
2. **`train_model.py`** — takes the same idea and makes it repeatable and
   auditable: it trains the model, then writes `artifacts/model.joblib`
   (the serialized model) and `artifacts/manifest.json` (version, timestamp,
   git commit, sha256, and key dependency versions) — the handoff artifact
   from the "packaging" stage of the lifecycle diagram above.
3. **`app/`** — wraps that artifact in a FastAPI service: `main.py` loads and
   hash-verifies it once at startup, and `api/routes.py` exposes it as
   `POST /score` over HTTP.
4. **`Dockerfile`** — packages the service (code + artifact + dependencies)
   into a single container image that runs identically on any machine with
   a container runtime, whether that's your laptop or a production cluster.

The Q&A lab follows the same shape minus the model-artifact step (no
training involved): `prototype.py` -> `app/` -> `Dockerfile`.

## Docker fundamentals and a Kubernetes overview

A Docker **image** is a read-only, versioned bundle of your application code,
its dependencies, and enough of an OS layer to run it anywhere; a
**container** is a running instance of that image — you can start many
containers from the same image, each an isolated process with its own
filesystem view. Both labs' `Dockerfile`s are commented line-by-line; the
key instructions are:

- `FROM python:3.11-slim` — start from a small official Python base image.
- `WORKDIR /app` — set a consistent working directory inside the image.
- `COPY requirements.txt .` followed by `RUN pip install -r requirements.txt`
  **before** copying application code — this orders the build so Docker's
  layer cache can skip reinstalling dependencies when only your code
  changes, not your dependency list.
- `COPY app/ ./app/` (and `COPY artifacts/ ./artifacts/` in the fraud lab) —
  copy in the actual application and its versioned artifact.
- `EXPOSE 8000` — documents the port the service listens on.
- `CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]` —
  the command that runs when a container starts; `--host 0.0.0.0` is
  required so the server accepts connections from outside the container,
  not just from `localhost` inside it.

**Kubernetes**, in brief, is what you reach for once running one container by
hand isn't enough: it's a system for running many containers across many
machines and automatically handling the things you'd otherwise script by
hand — *scheduling* (deciding which machine runs which container),
*scaling* (adding or removing running copies of a service based on load),
and *self-healing* (restarting or rescheduling a container that crashes or
fails its readiness check, which is exactly what `/healthz` and `/readyz`
exist to feed into). This week's labs run as a single container each, no
cluster required — Week 6 is where a real Kubernetes deployment enters the
picture.

## Hands-on labs

- [`labs/ml-track-fraud-detection/README.md`](labs/ml-track-fraud-detection/README.md) —
  ML track: a fraud-scoring FastAPI service.
- [`labs/llm-track-qa-service/README.md`](labs/llm-track-qa-service/README.md) —
  LLM track: a Q&A FastAPI service that runs in mock mode by default.

## Exercises

See [`exercises.md`](exercises.md) for five hands-on exercises that build on
both labs above.
