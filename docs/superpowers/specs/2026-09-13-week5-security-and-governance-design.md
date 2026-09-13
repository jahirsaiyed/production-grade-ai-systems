# Week 5: Security and Governance — Design Spec

## Overview

Week 5 teaches security and governance for production AI systems: authentication, encryption at
rest, PII handling, input/output guardrails, groundedness checks, fairness auditing, model cards,
audit logging, and lightweight prompt/policy registries. Unlike Week 4 (standalone eval harnesses
with no API layer), Week 5's labs are real FastAPI services — security controls only mean
something when there's a request path to defend — so both labs return to the Docker-backed,
`make run`-able service shape established in Weeks 1-3.

Both labs are **new copies under `modules/week-05-security-and-governance/labs/`**, built by
copying the relevant Week 1 (ML) / Week 2 (LLM) service unchanged and layering this week's security
controls on top — mirroring Week 3's precedent of new copies rather than modifying already-shipped,
already-reviewed labs in place.

## Disclaimer and framing

Same standing disclaimer as every prior week: this is a teaching course built from a marketing
syllabus, not verbatim ByteByteGo course content. Week 5's security controls are real, working, and
testable, but deliberately simplified for learning — a shared-secret HS256 JWT is not a production
identity provider, hand-rolled fairness metrics are not `fairlearn`, and regex-based PII detection
is not a commercial DLP tool. Each module's README says so explicitly, the same way Week 3's
semantic cache disclosed "no eviction policy" and Week 4's `judge.py` disclosed "not meaningful in
mock mode."

## Track 1: ML — `ml-track-security-hardening`

Copies Week 1's `ml-track-fraud-detection` service (`app/domain/scoring.py`,
`app/adapters/model_store.py`, `app/api/`, `train_model.py`) and hardens it.

### Reused unchanged
- `app/domain/scoring.py` — pure business logic, no security surface, stays byte-identical.

### Adapted (this week's actual subject matter — not byte-identical this time)
- `app/adapters/model_store.py` — extended so `load_model()` decrypts the artifact before the
  existing sha256 integrity check and `joblib.load`. Order: read encrypted bytes → sha256-verify
  the **encrypted** bytes against the manifest (so a tampered ciphertext is caught before any
  decryption is attempted) → decrypt with Fernet → `joblib.load` the plaintext. This preserves
  Week 1's fail-fast, no-silent-degradation invariant while adding a second layer.
- `app/api/routes.py` / `app/main.py` — every route gains a `require_auth` dependency (see below).

### New modules
- **`auth.py`** — JWT issuance (`create_token(subject: str) -> str`) and verification
  (`require_auth` as a FastAPI dependency) using HS256 with a shared secret read from
  `JWT_SECRET_KEY` (env var, fail-fast at startup if unset — same pattern as this repo's other
  required-secret checks). Missing/invalid/expired token → `401`. No refresh tokens, no scopes —
  a single shared-secret bearer token is the full scope of "JWT auth" this week teaches.
- **`encryption.py`** — `encrypt_bytes(data: bytes, key: bytes) -> bytes` /
  `decrypt_bytes(data: bytes, key: bytes) -> bytes` thin wrappers over `cryptography`'s `Fernet`.
  The encryption key comes from `MODEL_ENCRYPTION_KEY` (env var, fail-fast if unset). `train_model.py`
  is extended to encrypt the artifact after training, before writing it to disk.
- **`model_card.py`** — `generate_model_card() -> str` produces a Markdown model card from a small
  structured constant (intended use, training-data description — synthetic, imbalanced,
  6-feature — known limitations, out-of-scope uses). Tested by asserting required sections are
  present, not by asserting exact prose.
- **`fairness_audit.py`** — the synthetic dataset gains one more column: a synthetic binary
  `protected_group` attribute (generated alongside `X`/`y`, uncorrelated with the label by
  construction, so any measured disparity reflects the model's own behavior, not a rigged dataset).
  `compute_fairness_metrics(y_pred, protected_group) -> dict` returns hand-rolled
  `selection_rate_group_0`, `selection_rate_group_1`, `demographic_parity_difference`, and
  `disparate_impact_ratio` (group_1 selection rate ÷ group_0 selection rate). No `fairlearn`
  dependency — same from-scratch teaching philosophy as this repo's hand-rolled semantic cache and
  hybrid retrieval.
- **`audit_log.py`** — `log_decision(subject: str, input_summary: dict, decision: dict) -> None`
  writes one structured JSON line per scored request (timestamp, authenticated subject, a
  non-sensitive input summary, the decision) to a log file — never the raw JWT, never the encryption
  key. Wired into the `/score` route.

### Tests
Auth (401 without/invalid token, 200 with valid token), encryption round-trip + tamper detection
(flipping a byte in the ciphertext must raise before decryption succeeds), fairness metrics against
a hand-constructed `y_pred`/`protected_group` fixture with a known expected disparity, model card
section presence, audit log line shape and secret-safety (grep the emitted log line for the raw
JWT/key and assert absence).

## Track 2: LLM — `llm-track-security-hardening`

Copies Week 2's `llm-track-rag-service` (`app/domain/`, `app/adapters/`, `app/api/`, `docs/`,
`ingest.py`) and hardens it.

### Reused unchanged
- `app/domain/{chunking,retrieval,rag,tokenizing}.py`, `app/adapters/{embeddings,llm_client,
  index_store}.py`, the 3 corpus docs, `ingest.py` — all byte-identical from Week 2.

### Adapted
- `app/api/routes.py` / `app/main.py` — every route gains `require_auth`; the `/ask` route's
  request/response models become the stricter contract described below; PII redaction, guardrails,
  groundedness scoring, and audit logging are wired into the request-handling pipeline in that
  order (redact → guardrail-check the redacted input → retrieve/answer → guardrail-check the
  output's citations → score groundedness → log → respond).

### New modules
- **`auth.py`** — same JWT approach as the ML track (independently implemented per-lab, matching
  this repo's established convention of no shared code between labs — each lab is self-contained).
- **`pii_redaction.py`** — `redact(text: str) -> str` using compiled regexes for email addresses,
  US-style phone numbers, and SSN-shaped strings, replacing matches with `[REDACTED_EMAIL]` /
  `[REDACTED_PHONE]` / `[REDACTED_SSN]`. Applied to the incoming question before it is logged (never
  applied to the corpus or to retrieval — redaction is about not persisting a user's PII in logs,
  not about censoring the knowledge base).
- **`guardrails.py`** — three independent checks:
  - `check_prompt_injection(text: str) -> bool` — a small blocklist of injection-pattern phrases
    (extending Week 4's `RED_TEAM_PROMPTS` concept from an eval check into an actual runtime
    filter). A match causes the route to return `400` before any retrieval happens.
  - `check_citations_are_known(citations: list[dict], known_sources: set[str]) -> list[str]` —
    the same "citations must come from `KNOWN_SOURCES`" check Week 4 proved holds by construction,
    now run as a runtime assertion on every response (defense in depth: if it ever fired, that
    would mean the architectural guarantee broke, and the guardrail — not a learner reading logs
    days later — is what would catch it).
  - A strict Pydantic `AskResponse` contract (`answer: str`, `citations: list[Citation]`,
    `groundedness_score: float`) replacing whatever looser response shape Week 2 used — "structured-
    output contracts" from the syllabus means the API can't return a freeform shape, not that the
    LLM's own text is constrained.
- **`groundedness.py`** — `score_groundedness(answer: str, retrieved_chunks: list[str]) -> float`
  computes token-overlap (shared tokens ÷ answer tokens, via the same `tokenize()` Week 2 already
  has) between the generated answer and the concatenated retrieved context. Unlike Week 4's
  `judge.py` (explicitly non-meaningful in mock mode), this check **is** meaningful in mock mode:
  the mock LLM's canned answer ("This is a mock answer...") shares almost no tokens with any
  retrieved context and will genuinely score low, while a test can construct a synthetic
  context-derived answer and show it scores high — a real, deterministic, demonstrable signal
  either way. Still illustrative/logged rather than gated (a low score doesn't block the response —
  it's surfaced to the caller and audit log, mirroring how real groundedness checks are usually
  monitoring signals, not hard blocks, since false positives on a legitimately-paraphrased answer
  are common).
- **`prompt_registry.py`** — a `prompts/` directory holds versioned template files
  (`prompts/rag_answer/v1.txt`, `v2.txt`), each with `{context}`/`{question}` placeholders.
  `get_active_prompt(name: str) -> str` reads a small `prompts/registry.json` pointing at which
  version is active per template name. `validate_registry() -> list[str]` checks every registered
  template file exists and contains exactly the expected placeholders — an "offline check" in the
  syllabus's sense, run as a test, not a live service.
- **`audit_log.py`** — same shape/intent as the ML track's, logging: timestamp, authenticated
  subject, the **redacted** question (never the raw one), retrieved sources, guardrail verdicts,
  groundedness score. Never logs the JWT or the OpenAI key.

### Tests
Auth (401/200), PII redaction (each of the 3 patterns individually, plus a clean string that stays
untouched), prompt-injection blocking (a known bad phrase → 400, a clean question → 200),
citation-known-sources guardrail (constructed with a fabricated source → flags it; with real
retrieved chunks → passes), groundedness (mock canned answer scores low; a context-derived
synthetic answer scores high — both asserted with concrete thresholds), prompt registry validation
(all versions present and well-formed; a deliberately broken registry entry → validation fails),
audit log secret-safety (grep emitted log lines for the JWT/API key, assert absence), and an
end-to-end `/ask` test proving the full pipeline order (redaction happens before logging; guardrail
runs before the response is returned).

## Concept README (no lab, `## topic` sections only)

1. Threat landscape / OWASP LLM Top 10 — name the list, map which items this week's labs address
   (prompt injection, insecure output handling, sensitive info disclosure) vs. which are out of
   scope (training data poisoning, supply chain, excessive agency — no agentic tool-use in this
   course).
2. API authentication (tokens, JWT) — point at both labs' `auth.py`.
3. PII detection and redaction — point at the LLM lab's `pii_redaction.py`.
4. Encryption in transit and at rest — encryption-at-rest is demonstrated (ML lab's
   `encryption.py`); TLS/encryption-in-transit is conceptual only (no lab demonstrates terminating
   real TLS in a local dev server — same "conceptual only" treatment Week 4 gave OpenTelemetry).
5. Vector store protection — conceptual only; this course's "vector store" is a local JSON file
   with no deployed access-control surface to harden.
6. Guardrails: input/output filters, structured-output contracts, audit logging — point at the LLM
   lab's `guardrails.py` and both labs' `audit_log.py`.
7. Groundedness checks — point at the LLM lab's `groundedness.py`, and explicitly contrast it with
   Week 4's `judge.py` (meaningless in mock mode vs. genuinely demonstrable in mock mode).
8. Model cards, datasheets, fairness audits — point at the ML lab's `model_card.py` and
   `fairness_audit.py`.
9. Explainability and citations as RAG explanations — no new code; point back at Week 2/4's
   citation architecture as the answer to "how does this system explain itself."
10. Regulatory context (EU AI Act, NIST AI RMF) — conceptual only, 2-3 sentences each, no lab.
11. Prompt/policy registry — point at the LLM lab's `prompt_registry.py`.

## Build order

1. Concept README + exercises + root README/course-outline status update (bundled into Task 1).
2. ML track: copy+adapt service skeleton → `auth.py` (TDD) → `encryption.py` (TDD) →
   `model_store.py` adaptation (TDD) → retrain+encrypt artifact → `model_card.py` (TDD) →
   `fairness_audit.py` (TDD) → `audit_log.py` (TDD) → wire into routes → Makefile/Dockerfile/README.
3. LLM track: copy service skeleton unchanged → `auth.py` (TDD) → `pii_redaction.py` (TDD) →
   `guardrails.py` (TDD) → `groundedness.py` (TDD) → `prompt_registry.py` + `prompts/` (TDD) →
   `audit_log.py` (TDD) → wire everything into routes in the documented pipeline order →
   Makefile/Dockerfile/README.
4. CI matrix update (add both new labs).
5. Code review + security review pass (this week, security review carries extra weight — auth
   bypass, secret leakage via logs, and injection-filter bypass are all in scope, not just the
   usual checklist).
6. Push to GitHub.

## Global constraints carried over from prior weeks

- Python 3.11+, plain venv/pip, `pytest`+`pytest-cov` 80%+ target enforced via
  `--cov-fail-under=80` (learned from Week 4's final review: the Makefile's `test` target must
  match CI's actual command, not a looser local approximation).
- `ruff check` must pass.
- Exact-version pins in `requirements.txt` for every new dependency (`cryptography`, `PyJWT`).
- Commit format `<type>: <description>` with the exact trailer
  `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.
- Fail-fast (not gracefully degraded) on missing required secrets (`JWT_SECRET_KEY`,
  `MODEL_ENCRYPTION_KEY`), matching this repo's established artifact-integrity fail-fast pattern.
- No PII/PHI in any committed file, log sample, or test fixture (synthetic PII-shaped strings for
  testing the redaction patterns are fine — they're not real people's data).
- Every task that introduces a security control gets tests that verify the control actually blocks
  the thing it claims to block, not just that the "happy path" works — carrying forward Week 4's
  lesson that a check which can never fail is not a real check.
