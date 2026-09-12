from pathlib import Path

from rank_bm25 import BM25Okapi

from app.adapters.embeddings import EmbeddingClient
from app.adapters.index_store import load_index
from red_team import check_no_fabricated_citations, run_red_team_checks


def test_check_no_fabricated_citations_passes_for_known_sources():
    results = [
        {"question": "q1", "citations": [{"source": "hr-policy.md", "chunk_id": 0}]},
    ]
    assert check_no_fabricated_citations(results) == []


def test_check_no_fabricated_citations_flags_an_unknown_source():
    results = [
        {"question": "q1", "citations": [{"source": "made-up-file.md", "chunk_id": 0}]},
    ]
    violations = check_no_fabricated_citations(results)
    assert len(violations) == 1
    assert "made-up-file.md" in violations[0]


def test_run_red_team_checks_finds_no_fabricated_citations_against_the_real_index():
    artifact_dir = Path(__file__).resolve().parent.parent / "artifacts"
    loaded = load_index(artifact_dir)
    bm25_index = BM25Okapi(loaded.bm25_tokenized_corpus)
    embedding_client = EmbeddingClient(api_key=None)

    violations = run_red_team_checks(
        loaded.chunks, loaded.dense_vectors, bm25_index, embedding_client
    )

    assert violations == []
