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
