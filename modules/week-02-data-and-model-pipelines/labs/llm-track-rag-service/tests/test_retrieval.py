import numpy as np
import pytest

from app.domain.chunking import Chunk
from app.domain.retrieval import hybrid_search


def test_hybrid_search_returns_top_k_by_combined_score():
    chunks = [
        Chunk(text="a", source="doc.md", chunk_id=0),
        Chunk(text="b", source="doc.md", chunk_id=1),
        Chunk(text="c", source="doc.md", chunk_id=2),
    ]
    dense_vectors = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.9, 0.1]),
    ]
    query_vector = np.array([1.0, 0.0])
    bm25_scores = [5.0, 0.0, 1.0]

    results = hybrid_search(query_vector, chunks, dense_vectors, bm25_scores, k=2)

    assert len(results) == 2
    assert results[0].chunk.chunk_id == 0
    assert results[0].score >= results[1].score


def test_hybrid_search_rejects_mismatched_lengths():
    chunks = [Chunk(text="a", source="doc.md", chunk_id=0)]
    with pytest.raises(ValueError):
        hybrid_search(np.array([1.0]), chunks, [], [], k=1)
