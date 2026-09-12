import json
import os
import tempfile
from pathlib import Path

from app.adapters.embeddings import EmbeddingClient
from app.adapters.index_store import _sha256_of
from app.domain.chunking import chunk_text
from app.domain.tokenizing import tokenize

_artifact_dir = Path(tempfile.mkdtemp(prefix="rag-lab-test-artifacts-"))

_client = EmbeddingClient(api_key=None)
_chunks = chunk_text(
    "Vacation policy is 20 days per year for full-time employees.",
    source="hr-policy.md",
)
_dense_vectors = [_client.embed(c.text).tolist() for c in _chunks]
_tokenized_corpus = [tokenize(c.text) for c in _chunks]

_payload = {
    "chunks": [
        {"text": c.text, "source": c.source, "chunk_id": c.chunk_id}
        for c in _chunks
    ],
    "dense_vectors": _dense_vectors,
    "bm25_tokenized_corpus": _tokenized_corpus,
}

_index_path = _artifact_dir / "index.json"
_index_path.write_text(json.dumps(_payload))

_manifest = {
    "artifact_version": "0.1.0-test",
    "sha256": _sha256_of(_index_path),
}
(_artifact_dir / "manifest.json").write_text(json.dumps(_manifest))

os.environ["ARTIFACT_DIR"] = str(_artifact_dir)
os.environ["OPENAI_API_KEY"] = ""
