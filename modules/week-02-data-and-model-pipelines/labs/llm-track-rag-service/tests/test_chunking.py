import pytest

from app.domain.chunking import Chunk, chunk_text


def test_chunk_text_splits_long_text_into_multiple_chunks():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, source="doc.md", chunk_size=400, overlap=50)
    assert len(chunks) > 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.source == "doc.md" for c in chunks)


def test_chunk_text_returns_single_chunk_for_short_text():
    text = "just a few words here"
    chunks = chunk_text(text, source="doc.md", chunk_size=400, overlap=50)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].chunk_id == 0


def test_chunk_text_rejects_overlap_greater_than_or_equal_to_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text", source="doc.md", chunk_size=10, overlap=10)
