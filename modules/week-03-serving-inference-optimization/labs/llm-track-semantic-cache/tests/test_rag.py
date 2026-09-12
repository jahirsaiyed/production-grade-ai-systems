from app.domain.chunking import Chunk
from app.domain.rag import build_citations, build_rag_prompt
from app.domain.retrieval import RetrievedChunk


def _sample_retrieved():
    return [
        RetrievedChunk(
            chunk=Chunk(
                text="Vacation policy is 20 days.",
                source="hr-policy.md",
                chunk_id=0,
            ),
            score=0.9,
        )
    ]


def test_build_rag_prompt_includes_question_and_context():
    prompt = build_rag_prompt("How many vacation days?", _sample_retrieved())
    assert "How many vacation days?" in prompt
    assert "Vacation policy is 20 days." in prompt
    assert "hr-policy.md" in prompt


def test_build_citations_returns_source_metadata():
    citations = build_citations(_sample_retrieved())
    assert citations == [
        {
            "source": "hr-policy.md",
            "chunk_id": 0,
            "snippet": "Vacation policy is 20 days.",
        }
    ]
