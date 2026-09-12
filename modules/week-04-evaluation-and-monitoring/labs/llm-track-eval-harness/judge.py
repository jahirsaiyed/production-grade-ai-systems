"""
Best-effort LLM-as-judge answer-quality scoring.

In mock mode, both answer generation and this judge call return the same
canned strings regardless of input, so this score is NOT a meaningful
quality signal — it is reported for illustration only and is never gated on
(see gate.py, which uses only citation_match_rate). It becomes a genuine
signal once a real OPENAI_API_KEY is configured, since both calls then
reflect real model behavior.

Run the manual demo with `make judge` or `python judge.py`.
"""
from app.adapters.llm_client import LlmClient


def judge_answer(
    question: str, expected_source: str, answer: str, llm_client: LlmClient
) -> dict:
    judge_prompt = (
        "You are grading an AI assistant's answer for correctness.\n"
        f"Question: {question}\n"
        f"The answer should be grounded in a document about: {expected_source}\n"
        f"Given answer: {answer}\n"
        "Does the answer correctly and helpfully address the question? "
        "Reply with exactly one word: YES or NO."
    )
    verdict = llm_client.complete(judge_prompt).strip().upper()
    return {"verdict": verdict, "is_meaningful": not llm_client.is_mock}


def main() -> None:
    import os

    from rank_bm25 import BM25Okapi

    from app.adapters.embeddings import EmbeddingClient
    from app.adapters.index_store import load_index
    from eval_harness import ARTIFACT_DIR, answer_question
    from eval_set import EVAL_EXAMPLES

    api_key = os.environ.get("OPENAI_API_KEY") or None
    llm_client = LlmClient(api_key=api_key)
    embedding_client = EmbeddingClient(api_key=api_key)
    loaded = load_index(ARTIFACT_DIR)
    bm25_index = BM25Okapi(loaded.bm25_tokenized_corpus)

    if llm_client.is_mock:
        print(
            "OPENAI_API_KEY not set — running in mock mode. Verdicts below are "
            "NOT meaningful (see this module's docstring)."
        )

    for example in EVAL_EXAMPLES:
        answer_question(
            example["question"],
            loaded.chunks,
            loaded.dense_vectors,
            bm25_index,
            embedding_client,
        )
        generated_answer = llm_client.complete(f"Question: {example['question']}\nAnswer:")
        result = judge_answer(
            example["question"], example["expected_source"], generated_answer, llm_client
        )
        print(
            f"Q: {example['question']!r}\n"
            f"  verdict={result['verdict']} meaningful={result['is_meaningful']}"
        )


if __name__ == "__main__":
    main()
