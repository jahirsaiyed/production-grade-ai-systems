# LLM Track Lab: Security Hardening

Companion to [Week 5's concept README](../../README.md). A security-hardened copy of Week 2's RAG
service — same retrieval pipeline, same `/ask` endpoint, now behind JWT auth, with PII redaction,
input/output guardrails, a groundedness score, and a versioned prompt registry.

## What's here

```
app/domain/{chunking,retrieval,rag,tokenizing}.py   # reused from Week 2 unchanged
app/adapters/{embeddings,llm_client,index_store}.py  # reused from Week 2 unchanged
app/api/auth.py                                       # JWT issuance + a require_auth dependency
app/domain/pii_redaction.py                            # regex-based email/phone/SSN redaction
app/domain/guardrails.py                                # input (prompt-injection) + output (citation-source) checks
app/domain/groundedness.py                               # lexical-overlap groundedness scoring
app/adapters/prompt_registry.py                           # versioned prompt templates + offline validation
app/adapters/audit_log.py                                  # structured, secret-safe request logging
prompts/                                                    # versioned template files + registry.json
```

## Run it

If `make` isn't available (e.g. plain Windows without Git Bash), run the commands inside
`Makefile` directly — each target is a single command.

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows:     .venv\Scripts\activate
make setup
make ingest   # builds this lab's own retrieval index

# Generate a secret for local use (never reuse this value anywhere real):
python -c "import secrets; print(secrets.token_urlsafe(32))"   # -> JWT_SECRET_KEY

make run      # starts the service on :8000
```

Get a token and call the protected route:
```bash
python -c "from app.api.auth import create_token; print(create_token(subject='learner'))"
curl -X POST http://localhost:8000/ask \
  -H "Authorization: Bearer <paste-token>" \
  -H "Content-Type: application/json" \
  -d '{"question":"How many vacation days do I get?"}'
```

## What to notice

- The prompt-injection check runs BEFORE retrieval — a flagged request never reaches the embedding
  or LLM call, so an attacker can't spend the service's compute budget on requests it's already
  decided to reject.
- The citation-source guardrail should never actually fire — Week 4 proved citations are built from
  retrieval metadata, never LLM output, so a fabricated source is architecturally impossible here.
  This check exists as defense in depth, not because it's expected to catch anything in practice.
- `groundedness_score` is genuinely meaningful even in mock mode — try Week 5's Exercise 4 to see
  the mock canned answer score low while a context-derived answer scores high, unlike Week 4's
  `judge.py`, which is a constant regardless of input in mock mode.
- The prompt registry (`prompts/`) is validated at service startup (fail-fast, see `app/main.py`)
  and in this lab's own tests, but is NOT wired into the live `/ask` prompt-building path this
  week — `app/domain/rag.py`'s `build_rag_prompt` (reused unchanged from Week 2) still builds the
  actual prompt sent to the LLM. This registry demonstrates the versioning/validation pattern on
  its own, deliberately not touching already-reviewed code.
- `/ask` requires a valid bearer token; `/healthz` and `/readyz` deliberately do not.
