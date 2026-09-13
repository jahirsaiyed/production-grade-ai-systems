from app.domain.retrieval import RetrievedChunk

MAX_CONTEXT_CHARS = 2000


def build_rag_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    context_parts = []
    total_chars = 0
    for i, item in enumerate(retrieved, start=1):
        snippet = item.chunk.text
        if total_chars + len(snippet) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(f"[{i}] (source: {item.chunk.source}) {snippet}")
        total_chars += len(snippet)

    context = "\n\n".join(context_parts)
    return (
        "You are a helpful assistant answering from the provided company "
        "documents.\nCite sources using [n] markers matching the numbered "
        f"context below.\n\nContext:\n{context}\n\n"
        f"Question: {question.strip()}\nAnswer:"
    )


def build_citations(retrieved: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "source": item.chunk.source,
            "chunk_id": item.chunk.chunk_id,
            "snippet": item.chunk.text[:200],
        }
        for item in retrieved
    ]
