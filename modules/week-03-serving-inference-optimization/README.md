# Week 3: Serving, Inference, and Optimization

Weeks 1-2 got a model and an LLM app packaged into a service and gave that
service well-governed data and reproducible training. Week 3 asks a
different question: once the service is running, how do you make it fast,
cheap, and scalable enough to actually put in front of real traffic? You'll
work through two tracks: **ML** (a batch-scoring throughput lab in
[`labs/ml-track-batch-optimization/README.md`](labs/ml-track-batch-optimization/README.md))
and **LLM** (a semantic-caching Q&A lab in
[`labs/llm-track-semantic-cache/README.md`](labs/llm-track-semantic-cache/README.md)).
This week is unusually conceptual: most of "Serving Architecture" and several
"Optimization Levers" topics need real GPUs, real model-serving frameworks,
or real production traffic to demonstrate properly, which this course's
CPU-only, mock-friendly labs can't provide. Each section below says plainly
whether a lab backs it up or whether it's explained conceptually only.

## Dedicated inference services

A dedicated inference service is a process whose only job is loading a model
(or calling an LLM) and answering prediction requests — separate from the
web app, the batch job, or whatever else in your system needs predictions.
Splitting it out means the model can be scaled, deployed, and rolled back
independently of the application code that calls it, and a spike in
prediction traffic doesn't compete for the same process's memory and CPU as
unrelated request handling. This is conceptual only this week: both new labs
already run as their own single-purpose FastAPI services (as every lab in
this course has since Week 1), so there's nothing new to build to
demonstrate the idea — it's named here because "inference service" as a
distinct architectural role is the foundation the rest of this week builds
on.

## Serving frameworks (Triton, TorchServe, Ray Serve, vLLM, SGLang)

Purpose-built serving frameworks exist because "wrap a model in a FastAPI
route" (this course's approach) doesn't scale to production LLM or
high-throughput ML workloads without a lot of hand-rolled machinery. **NVIDIA
Triton Inference Server** and **TorchServe** are general-purpose,
multi-framework model servers — they load models trained in PyTorch,
TensorFlow, ONNX, and others behind a common serving API, with built-in
batching and multi-model hosting. **Ray Serve** is a Python-native scaling
layer built on Ray, aimed at composing multiple models and business logic
into one deployment graph without leaving Python. **vLLM** and **SGLang** are
specialized for high-throughput LLM serving specifically: both implement
continuous batching (folding new requests into an in-flight batch rather than
waiting for the current batch to finish) and PagedAttention-style memory
management (treating the KV cache like paged virtual memory so it's used
efficiently across many concurrent sequences). This is conceptual only — no
lab in this course installs or runs any of these frameworks; both labs stay
with plain FastAPI so the underlying throughput and latency concepts stay
visible without a heavyweight dependency.

## Prefill vs. decode; memory-bound vs. compute-bound inference

Autoregressive LLM inference happens in two phases with different
performance profiles. **Prefill** processes the entire input prompt in one
pass, computing attention over every input token at once — it's highly
parallel and tends to be **compute-bound** (limited by how fast the GPU can
do matrix multiplication). **Decode** then generates output tokens one at a
time, each new token depending on all previous ones, and tends to be
**memory-bound** (limited by how fast the growing KV cache can be read from
GPU memory for each single-token step, not by raw compute). This distinction
is why LLM-serving frameworks batch and schedule prefill and decode requests
differently — a service that only optimizes for one phase leaves the other's
bottleneck unaddressed. This is conceptual only: reasoning about GPU memory
bandwidth versus compute throughput requires a real GPU and a real
autoregressive decoding loop, neither of which either lab this week runs.

## Topologies: single model, router/gateway, cascades with fallbacks

Three shapes cover most serving topologies. A **single model** setup is what
every lab in this course has used so far: one service, one model (or LLM
call) answers every request. A **router/gateway** sits in front of multiple
backends and directs each request to the right one — Week 1's production
building blocks section already introduced the API gateway idea (a shared
front door centralizing auth, rate limiting, and routing); a serving router
is the same concept applied to picking which model or provider handles a
given request. A **cascade with fallbacks** tries a primary option first and
falls back to a secondary (often cheaper or more available) option if the
primary fails or is unavailable — conceptually the same pattern as the
graceful degradation this course already implements: Week 1's Q&A lab
catching `LlmCallError` and returning a fallback answer, and the fraud lab
catching `FeatureStoreUnavailable` and failing loud instead. A serving
cascade just applies that same "primary fails, do something defined instead
of crashing" idea one layer up, at the level of which model or endpoint
answers the request. This is conceptual only this week — neither new lab
adds a second model or provider to route or cascade between.

## Self-host vs. managed APIs (rate limits, data residency, unit economics)

Calling a managed LLM API (OpenAI, Anthropic, and similar) means no
infrastructure to run, but you inherit the provider's rate limits, have
limited control over where your data physically goes (a real concern for
data-residency regulations), and pay per token — a unit economic that scales
linearly with usage and can dominate a product's costs at volume. Self-
hosting a model gives you control over all three at the cost of owning GPU
infrastructure, scaling it, and keeping it patched and available yourself.
This course's labs have used a lightweight, concrete version of this
tradeoff since Week 1: both LLM labs run in **mock mode** by default (no API
key needed, deterministic canned responses) and switch to a **real managed
API** call the moment an API key is set in `.env` — a small-scale stand-in
for "self-host a mock, or pay a managed provider" that makes the cost/control
tradeoff tangible without actually standing up self-hosted infrastructure.
The semantic-cache lab (this week) uses the identical switch.

## Profiling and baselines like p50/p95 latency

An average latency hides exactly the numbers you need to make a serving
decision: if 95% of requests return in 50ms but the slowest 5% take 2
seconds, the average might look fine while a meaningful share of real users
have a bad experience. **Percentiles** fix this: **p50** (median) tells you
the typical request's latency, while **p95** and **p99** tell you about tail
latency — the slow outliers that matter disproportionately because a single
slow dependency call, one unlucky retry, or a garbage-collection pause can
push a request into that tail. Production systems set SLOs on percentiles,
not averages, for exactly this reason. Both of this week's labs make this
runnable and concrete: `ml-track-batch-optimization/benchmark.py` and
`llm-track-semantic-cache/benchmark.py` each measure and report latency
percentiles for their respective optimization, so profiling isn't just a
concept here — it's the thing that proves the lab's optimization actually
worked.

## Semantic caching, dynamic vs. continuous batching, autoscaling signals

**Semantic caching** caches responses by *meaning* rather than exact string
match — a cache keyed on embedding similarity returns a cached answer for "How
do I reset my password?" even when a later question arrives as "password
reset steps," which an exact-string cache would treat as a complete miss.
`llm-track-semantic-cache/app/domain/semantic_cache.py` implements exactly
this: it embeds each incoming question, compares it against previously cached
questions by cosine similarity, and returns the cached answer once similarity
clears a configured threshold. **Dynamic (or continuous) batching** is the
serving-side counterpart on the ML side: instead of scoring one request at a
time, a server accumulates several concurrent requests and scores them
together in one model call, amortizing fixed per-call overhead (model
dispatch, framework overhead) across more work.
`ml-track-batch-optimization/app/domain/batch_scoring.py` and its
`POST /score/batch` endpoint demonstrate the throughput principle — many
transactions scored together beat the same number scored one at a time — but
it is a **deliberately simplified, deterministic stand-in**: real dynamic
batching uses a request-accumulator that waits up to some short time window
for more concurrent requests to arrive before running the batch, trading a
small added latency for a larger, more efficient batch. This lab's endpoint
skips that timing complexity entirely — the caller decides the batch size
up front — so it teaches the throughput math without the scheduling problem.
**Autoscaling signals** (queue depth, GPU utilization, request rate, and
similar, used to decide when to add or remove serving replicas) are
conceptual only — no lab this week runs multiple replicas or an autoscaler
to generate real signals from.

## Quantization (post-training, quantization-aware, INT8/INT4)

Quantization shrinks a model's numeric precision — typically from 32-bit
floats down to 8-bit or 4-bit integers — which cuts memory footprint and
often speeds up inference, at some cost to accuracy. **Post-training
quantization** converts an already-trained model's weights after the fact,
which is fast and simple but can lose more accuracy since the model never
adapted to reduced precision. **Quantization-aware training** simulates
reduced precision *during* training, so the model learns weights that
tolerate the eventual precision drop better, at the cost of needing to retrain
(or fine-tune) rather than just convert. This is conceptual only: no lab this
week quantizes a real model — doing so meaningfully needs a large enough
model and real accuracy benchmarks to show the tradeoff, which is out of
scope for this course's small, CPU-friendly models.

## Mixed precision (FP16/BF16)

Mixed precision training and inference use a 16-bit floating-point format
(FP16 or BF16) for most computation instead of full 32-bit floats, cutting
memory usage roughly in half and speeding up matrix operations on hardware
with dedicated 16-bit support, while keeping select operations (like certain
accumulations) in 32-bit precision where reduced range or precision would
cause numerical instability. It's one of the cheapest wins available for
GPU-trained deep learning models, which is why it's default-on in most modern
training frameworks. This is conceptual only — it's a training/inference-time
GPU technique, and neither this week's labs nor any lab so far in this course
trains or runs a model on a GPU.

## Pruning (structured, unstructured); distillation

**Pruning** removes parts of a trained model that contribute little to its
output. **Unstructured pruning** zeroes out individual weights wherever they
fall below some importance threshold, which shrinks the model on disk but
usually needs specialized sparse-computation support to actually speed up
inference. **Structured pruning** removes whole structural units — entire
neurons, attention heads, or layers — which is less surgical but yields a
smaller, genuinely faster model on ordinary hardware because there's simply
less to compute. **Distillation** takes a different approach entirely:
instead of shrinking one model, you train a smaller "student" model to mimic
a larger "teacher" model's outputs, often producing a compact model that
retains much of the teacher's behavior at a fraction of its size and
inference cost. All three are conceptual only this week — evaluating whether
a pruned or distilled model still performs acceptably needs a real model and
a real accuracy benchmark to prune or distill against, which is beyond this
week's scope.

## Runtimes and kernels (ONNX Runtime, TensorRT, OpenVINO)

Once a model is trained, a dedicated **runtime** can often serve it faster
than the framework it was trained in, by compiling the model's computation
graph into optimized, hardware-specific kernels. **ONNX Runtime** executes
models converted to the framework-agnostic ONNX format across a range of
hardware backends (CPU, GPU, and specialized accelerators) from one common
entry point. **TensorRT** is NVIDIA's GPU-specific inference optimizer and
runtime, applying kernel fusion and precision calibration (including the
quantization and mixed-precision ideas above) tuned specifically for NVIDIA
hardware. **OpenVINO** plays a similar role for Intel hardware (CPUs,
integrated GPUs, and Intel's own accelerators). This is conceptual only — no
lab this week converts a model into any of these runtimes; that conversion
step is exactly what Section 13 below names as the packaging step this
week's labs don't build.

## KV Cache, FlashAttention, and speculative decoding

Three techniques address specific inefficiencies inside transformer-based
LLM inference. The **KV cache** stores the key and value tensors computed for
already-generated tokens so that generating the next token doesn't require
recomputing attention over the entire sequence from scratch — without it,
decode-phase inference would get quadratically slower as a response grows.
**FlashAttention** is an optimized attention-computation algorithm that
restructures the computation to minimize slow GPU memory reads and writes,
producing mathematically identical attention output significantly faster and
with less memory. **Speculative decoding** speeds up generation by having a
small, fast "draft" model guess several tokens ahead, which the large target
model then verifies (and accepts or corrects) in a single parallel pass —
trading extra compute on the cheap draft model for fewer expensive sequential
steps on the large one. All three are transformer-internals topics that
operate below the level of any model this course's labs use, and none of
this course's models (a `LogisticRegression` fraud model, a plain
prompt-response LLM call) has attention internals to demonstrate these
against, so this section stays conceptual only.

## Packaging (checkpoint/ONNX -> service image, gRPC contracts, etc.)

Week 1 established the packaging path this whole course uses: **notebook ->
artifact -> API -> container image** — a throwaway script becomes a
versioned, hash-checked artifact, which gets wrapped in a service, which gets
containerized. This week's variant is the same path with one substitution:
instead of packaging the raw model straight out of training, you'd package a
**converted or optimized** model — a checkpoint exported to ONNX, quantized,
or compiled through a runtime like TensorRT — behind the same kind of service
and image, sometimes exposing a **gRPC contract** (a strongly-typed,
protobuf-based API, as introduced conceptually in Week 1's REST vs. gRPC
section) instead of REST, since internal high-throughput inference calls
often favor gRPC's efficiency over REST's convenience. This is conceptual
only: neither of this week's labs exports, converts, or optimizes a model
file — both labs demonstrate their optimization (batching, caching) purely
in the application layer, on top of the same plain-Python packaging Week 1
already covered.

## Hands-on labs

- [`labs/ml-track-batch-optimization/README.md`](labs/ml-track-batch-optimization/README.md) —
  ML track: a batch-scoring FastAPI service that measures the throughput win
  of scoring transactions in batches versus one at a time.
- [`labs/llm-track-semantic-cache/README.md`](labs/llm-track-semantic-cache/README.md) —
  LLM track: a Q&A FastAPI service with a similarity-based semantic cache in
  front of the LLM call.

## Exercises

See [`exercises.md`](exercises.md) for five hands-on exercises that build on
both labs above.
