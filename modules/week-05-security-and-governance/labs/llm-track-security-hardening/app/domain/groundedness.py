"""
Lexical-overlap groundedness scoring — how much of a generated answer's
vocabulary is actually supported by the retrieved context it was supposed
to be grounded in.

Unlike Week 4's judge.py (an LLM-as-judge score explicitly meaningless in
mock mode), this check IS genuinely meaningful in mock mode: the mock LLM's
canned answer shares almost no vocabulary with any real retrieved context
and will score low, while an answer actually built from the context scores
high — a real, deterministic signal either way. This is illustrative/logged
rather than a hard block, since legitimate paraphrasing can also score low
on pure lexical overlap.
"""
from app.domain.tokenizing import tokenize


def score_groundedness(answer: str, retrieved_chunks: list[str]) -> float:
    answer_tokens = set(tokenize(answer))
    if not answer_tokens:
        return 0.0

    context_tokens: set[str] = set()
    for chunk in retrieved_chunks:
        context_tokens.update(tokenize(chunk))

    overlap = answer_tokens & context_tokens
    return len(overlap) / len(answer_tokens)
