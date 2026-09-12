from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CachedAnswer:
    answer: str
    citations: list[dict]


@dataclass
class _CacheEntry:
    query_vector: np.ndarray
    answer: str
    citations: list[dict]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class SemanticCache:
    def __init__(self, threshold: float = 0.95):
        self._threshold = threshold
        self._entries: list[_CacheEntry] = []

    def lookup(self, query_vector: np.ndarray) -> CachedAnswer | None:
        best_score = -1.0
        best_entry: _CacheEntry | None = None
        for entry in self._entries:
            score = _cosine_similarity(query_vector, entry.query_vector)
            if score > best_score:
                best_score = score
                best_entry = entry
        if best_entry is not None and best_score >= self._threshold:
            return CachedAnswer(
                answer=best_entry.answer, citations=best_entry.citations
            )
        return None

    def store(
        self, query_vector: np.ndarray, answer: str, citations: list[dict]
    ) -> None:
        self._entries.append(
            _CacheEntry(
                query_vector=query_vector, answer=answer, citations=citations
            )
        )
