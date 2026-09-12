import numpy as np

from app.domain.semantic_cache import SemanticCache


def test_lookup_returns_none_when_cache_is_empty():
    cache = SemanticCache()
    assert cache.lookup(np.array([1.0, 0.0])) is None


def test_lookup_returns_cached_answer_for_identical_vector():
    cache = SemanticCache(threshold=0.95)
    cache.store(
        np.array([1.0, 0.0]),
        "answer one",
        [{"source": "doc.md", "chunk_id": 0, "snippet": "..."}],
    )

    result = cache.lookup(np.array([1.0, 0.0]))

    assert result is not None
    assert result.answer == "answer one"
    assert result.citations == [{"source": "doc.md", "chunk_id": 0, "snippet": "..."}]


def test_lookup_returns_none_below_similarity_threshold():
    cache = SemanticCache(threshold=0.95)
    cache.store(np.array([1.0, 0.0]), "answer one", [])

    result = cache.lookup(np.array([0.0, 1.0]))

    assert result is None
