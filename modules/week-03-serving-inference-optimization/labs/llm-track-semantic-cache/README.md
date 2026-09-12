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
