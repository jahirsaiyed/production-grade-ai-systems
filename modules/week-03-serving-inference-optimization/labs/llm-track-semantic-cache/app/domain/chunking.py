from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    chunk_id: int


def chunk_text(
    text: str, source: str, chunk_size: int = 400, overlap: int = 50
) -> list[Chunk]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    words = text.split()
    chunks: list[Chunk] = []
    start = 0
    chunk_id = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(
            Chunk(text=" ".join(chunk_words), source=source, chunk_id=chunk_id)
        )
        chunk_id += 1
        if end >= len(words):
            break
        start = end - overlap
    return chunks
