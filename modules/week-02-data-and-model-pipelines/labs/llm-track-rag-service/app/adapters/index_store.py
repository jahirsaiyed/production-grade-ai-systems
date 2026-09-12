import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.domain.chunking import Chunk


class ArtifactIntegrityError(RuntimeError):
    """Raised when the index artifact does not match its manifest."""


@dataclass(frozen=True)
class LoadedIndex:
    chunks: list[Chunk]
    dense_vectors: list[np.ndarray]
    bm25_tokenized_corpus: list[list[str]]
    manifest: dict


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def load_index(artifact_dir: Path) -> LoadedIndex:
    manifest_path = artifact_dir / "manifest.json"
    index_path = artifact_dir / "index.json"
    manifest = json.loads(manifest_path.read_text())

    actual_sha256 = _sha256_of(index_path)
    if actual_sha256 != manifest["sha256"]:
        raise ArtifactIntegrityError(
            f"index.json sha256 {actual_sha256} does not match "
            f"manifest sha256 {manifest['sha256']}"
        )

    payload = json.loads(index_path.read_text())
    chunks = [
        Chunk(text=c["text"], source=c["source"], chunk_id=c["chunk_id"])
        for c in payload["chunks"]
    ]
    dense_vectors = [np.array(v) for v in payload["dense_vectors"]]
    bm25_tokenized_corpus = payload["bm25_tokenized_corpus"]

    return LoadedIndex(
        chunks=chunks,
        dense_vectors=dense_vectors,
        bm25_tokenized_corpus=bm25_tokenized_corpus,
        manifest=manifest,
    )
