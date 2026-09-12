# Week 3 Exercises

Hints and acceptance criteria only — no answer key. If you get stuck, the reference implementation
is the lab's own code.

## Exercise 1: Prove batching's win at a different batch size

`ml-track-batch-optimization/benchmark.py` uses 100 transactions per trial. Change
`N_TRANSACTIONS` to 10 and re-run `make benchmark`. **Acceptance criteria:** you can explain, in
your own words, whether the speedup ratio grew, shrank, or stayed about the same, and why (hint:
per-call overhead is roughly fixed regardless of batch size, so its relative weight changes with
batch size).

## Exercise 2: Break the semantic cache's threshold on purpose

Lower the threshold to `0.5` — either by changing `DEFAULT_SIMILARITY_THRESHOLD` in
`llm-track-semantic-cache/app/domain/semantic_cache.py`, or by passing `threshold=0.5` explicitly
wherever `SemanticCache` is constructed (`app/main.py`) — restart the service, and ask two specific,
genuinely unrelated questions back to back: "How many vacation days do I get?" followed by "How do
I roll back a deploy?" (these two don't share enough distinctive tokens to falsely match even at a
low threshold under the mock embedder, unlike some seemingly-unrelated pairs that can still score
above a lowered threshold purely from shared stopwords like "the"/"how"/"do"/"i"). **Acceptance
criteria:** the second question now incorrectly returns `cache_hit: true` with the first question's
answer — explain why a too-low threshold is worse than no cache at all for a real product.

## Exercise 3: Measure the semantic cache's memory growth

The `SemanticCache` never evicts entries. Send 500 unique questions to a running
`llm-track-semantic-cache` instance (a small loop script counts as "sending"), then explain, in
your own words, what a production cache would need that this lab's doesn't have (hint: an eviction
policy — LRU, TTL, or a max-size bound).

## Exercise 4: Add a request-count guard to the batch endpoint

`POST /score/batch` currently accepts up to 1000 transactions per request (see
`BatchScoreRequest`'s `max_length` constraint). Lower it to 5, restart the service, and send a
batch of 10 transactions. **Acceptance criteria:** the request is rejected with a 422, and you can
point to the exact pydantic constraint that causes this.

## Exercise 5: Compare cold-start cost to a real managed-API price

`llm-track-semantic-cache/benchmark.py` uses an illustrative
`ASSUMED_COST_PER_LLM_CALL_USD` constant. Look up a real published price-per-1K-tokens for any
managed LLM API, estimate a realistic token count for this lab's prompts (see
`app/domain/rag.py`'s `MAX_CONTEXT_CHARS`), and recompute a more realistic per-call cost.
**Acceptance criteria:** you can state your estimated real per-call cost and how it compares to
the lab's placeholder constant.
