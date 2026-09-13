"""
Runtime input/output guardrails for the RAG service.

check_prompt_injection is an input filter run BEFORE retrieval — a match
blocks the request outright, since there's no reason to spend a
retrieval/LLM call on a request already flagged as adversarial.

check_citations_are_known is an output filter run AFTER the answer is built.
This is defense in depth: Week 4 proved citations are built from retrieval
metadata, not LLM output, so this check should never actually fire in
practice. If it ever did, that would mean the architectural guarantee had
broken somewhere upstream — this guardrail, not a learner reading logs days
later, is what catches it.
"""
INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard your instructions",
    "you are now",
    "reveal your system prompt",
]


def check_prompt_injection(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in INJECTION_PATTERNS)


def check_citations_are_known(
    citations: list[dict], known_sources: set[str]
) -> list[str]:
    return [
        citation["source"]
        for citation in citations
        if citation["source"] not in known_sources
    ]
