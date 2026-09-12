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

**Status in this repo: fully built — see `modules/week-02-data-and-model-pipelines/`.**

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

**Status in this repo: fully built — see `modules/week-03-serving-inference-optimization/`.**

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
