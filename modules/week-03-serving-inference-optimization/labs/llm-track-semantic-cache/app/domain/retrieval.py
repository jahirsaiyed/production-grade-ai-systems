from dataclasses import dataclass

import numpy as np

from app.domain.chunking import Chunk


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def hybrid_search(
    query_vector: np.ndarray,
    chunks: list[Chunk],
    dense_vectors: list[np.ndarray],
    bm25_scores: list[float],
    k: int = 3,
    dense_weight: float = 0.5,
) -> list[RetrievedChunk]:
    if not (len(chunks) == len(dense_vectors) == len(bm25_scores)):
        raise ValueError(
            "chunks, dense_vectors, and bm25_scores must be the same length"
        )

    dense_scores = [_cosine_similarity(query_vector, v) for v in dense_vectors]
    dense_norm = _normalize(dense_scores)
    bm25_norm = _normalize(list(bm25_scores))

    combined = [
        dense_weight * d + (1 - dense_weight) * b
        for d, b in zip(dense_norm, bm25_norm)
    ]

    ranked = sorted(zip(chunks, combined), key=lambda pair: pair[1], reverse=True)
    return [RetrievedChunk(chunk=c, score=s) for c, s in ranked[:k]]
