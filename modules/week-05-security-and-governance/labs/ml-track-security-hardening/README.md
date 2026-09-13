# ML Track Lab: Security Hardening

Companion to [Week 5's concept README](../../README.md). A security-hardened copy of Week 1's
fraud-detection service — same model, same `/score` endpoint, now behind JWT auth, with the model
artifact encrypted at rest, plus a model card and a fairness audit.

## What's here

```
app/domain/scoring.py         # reused from Week 1 unchanged — the scoring logic itself
app/adapters/feature_store.py # reused from Week 1 unchanged
app/api/schemas.py             # reused from Week 1 unchanged
app/api/auth.py                  # JWT issuance + a require_auth FastAPI dependency
app/adapters/encryption.py        # Fernet encrypt/decrypt helpers
app/adapters/model_store.py        # loads the encrypted artifact: sha256-check -> decrypt -> joblib.load
app/adapters/audit_log.py           # structured, secret-safe decision logging
train_model.py                       # adapted from Week 1 — now encrypts the artifact after training
model_card.py                         # generates MODEL_CARD.md
fairness_audit.py                      # measures selection-rate disparity on a synthetic protected attribute
```

## Run it

If `make` isn't available (e.g. plain Windows without Git Bash), run the commands inside
`Makefile` directly — each target is a single command.

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows:     .venv\Scripts\activate
make setup

# Generate secrets for local use (never reuse these values anywhere real):
python -c "import secrets; print(secrets.token_urlsafe(32))"          # -> JWT_SECRET_KEY
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"  # -> MODEL_ENCRYPTION_KEY

# Put both in a .env file (see .env.example), or export them, then:
make train           # trains and encrypts this lab's own model artifact
make run             # starts the service on :8000
```

The `artifacts/` committed to this repo is a publication-only demo, encrypted with a key nobody
outside this task has — it exists so the repo has something to ship, not so you can run it. You
must `make train` under your own freshly generated `MODEL_ENCRYPTION_KEY` to get an artifact your
local service can actually decrypt. If you ever run `git checkout -- artifacts/model.joblib.enc` to
try to "restore" the committed artifact, it will NOT work with your own key — the fix is to run
`make train` again.

Get a token and call the protected route:
```bash
python -c "from app.api.auth import create_token; print(create_token(subject='learner'))"
curl -X POST http://localhost:8000/score \
  -H "Authorization: Bearer <paste-token>" \
  -H "Content-Type: application/json" \
  -d '{"transaction_id":"txn-1","amount":100.0,"merchant_category":"retail"}'
```

Optional, offline, no running service needed:
```bash
make model-card       # writes MODEL_CARD.md
make fairness-audit   # prints fairness metrics to stdout
```

## What to notice

- `/score` requires a valid bearer token; `/healthz` and `/readyz` deliberately do not (a common
  real-world choice — liveness/readiness probes shouldn't need credentials).
- `model_store.py`'s `load_model()` verifies the **encrypted** bytes' sha256 against the manifest
  BEFORE attempting decryption — a tampered ciphertext is caught by the hash check, and even if it
  weren't, Fernet's own authenticated encryption would independently reject it. Two layers, either
  one catches tampering (see Week 5's Exercise 2).
- The fairness audit's `protected_group` column is a synthetic coin flip, generated independently
  of the model's actual features — any measured disparity reflects genuine model behavior on this
  synthetic setup, not a rigged dataset. This is a teaching stand-in for a real fairness audit,
  which would need real demographic data this course doesn't have. On this lab's own committed
  artifact, that measurement comes out to a disparate impact ratio of ~1.149 and a demographic
  parity difference of ~0.00194 — both close to parity, as expected from a synthetic, model-blind
  protected attribute.
- `audit_log.py`'s `log_decision()` has no parameter through which a raw JWT or encryption key could
  ever be passed — verified directly in this lab's own tests, not just assumed.
