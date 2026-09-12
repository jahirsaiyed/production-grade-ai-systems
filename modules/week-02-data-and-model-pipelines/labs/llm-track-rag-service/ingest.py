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
