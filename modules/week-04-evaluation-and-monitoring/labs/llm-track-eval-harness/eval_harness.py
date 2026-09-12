"""
Runs a genuine offline evaluation of the RAG retrieval pipeline against a
labeled set of (question, expected_source) pairs.

Run with `make eval` or `python eval_harness.py`.
"""
import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.adapters.embeddings import EmbeddingClient
from app.adapters.index_store import load_index
from app.domain.retrieval import hybrid_search
from app.domain.tokenizing import tokenize
from eval_set import EVAL_EXAMPLES

LAB_DIR = Path(__file__).parent
ARTIFACT_DIR = LAB_DIR / "artifacts"
EVAL_REPORT_PATH = LAB_DIR / "eval_report.json"


def answer_question(
    question: str, chunks, dense_vectors, bm25_index, embedding_client
) -> dict:
    query_vector = embedding_client.embed(question)
    bm25_scores = list(bm25_index.get_scores(tokenize(question)))
    retrieved = hybrid_search(query_vector, chunks, dense_vectors, bm25_scores, k=3)
    top_source = retrieved[0].chunk.source if retrieved else None
    return {
        "question": question,
        "top_source": top_source,
        "citations": [
            {"source": r.chunk.source, "chunk_id": r.chunk.chunk_id} for r in retrieved
        ],
    }


def run() -> dict:
    loaded = load_index(ARTIFACT_DIR)
    bm25_index = BM25Okapi(loaded.bm25_tokenized_corpus)
    # Evaluation always uses mock embeddings, for determinism — the metric this
    # harness gates on (citation_match_rate) must be reproducible on every run.
    embedding_client = EmbeddingClient(api_key=None)

    results = []
    correct = 0
    for example in EVAL_EXAMPLES:
        answer = answer_question(
            example["question"],
            loaded.chunks,
            loaded.dense_vectors,
            bm25_index,
            embedding_client,
        )
        is_correct = answer["top_source"] == example["expected_source"]
        correct += int(is_correct)
        results.append(
            {
                "question": example["question"],
                "expected_source": example["expected_source"],
                "actual_top_source": answer["top_source"],
                "correct": is_correct,
            }
        )

    citation_match_rate = correct / len(EVAL_EXAMPLES)
    return {
        "metrics": {"citation_match_rate": citation_match_rate},
        "results": results,
        "n_examples": len(EVAL_EXAMPLES),
    }


def main() -> None:
    report = run()
    EVAL_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {EVAL_REPORT_PATH}")
    print(f"Citation match rate: {report['metrics']['citation_match_rate']:.2f}")
    for result in report["results"]:
        marker = "OK" if result["correct"] else "MISS"
        print(f"  [{marker}] {result['question']!r} -> {result['actual_top_source']}")


if __name__ == "__main__":
    main()
