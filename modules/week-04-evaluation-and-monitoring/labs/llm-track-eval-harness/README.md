# LLM Track Lab: Evaluation Harness

Companion to [Week 4's concept README](../../README.md). A standalone offline-evaluation harness
for the Week 2 RAG service — no API, no Docker, just a labeled eval set, a citation-match metric,
a best-effort LLM-as-judge, a red-team regression pack, and a gate that would block a real CI run
on regression.

## What's here

```
app/domain/{chunking,retrieval,rag,tokenizing}.py   # reused from Week 2 unchanged
app/adapters/{embeddings,llm_client,index_store}.py  # reused from Week 2 unchanged
docs/                                                 # the same 3 corpus docs, reused unchanged
ingest.py                                              # reused from Week 2 unchanged
eval_set.py                                             # 8 labeled (question, expected_source) pairs
eval_harness.py                                          # runs the retrieval pipeline directly, scores citation-match rate
judge.py                                                  # best-effort LLM-as-judge (NOT meaningful in mock mode)
red_team.py                                                # adversarial prompts + no-fabricated-citations check
gate.py                                                     # compares an eval run against baseline_metrics.json
baseline_metrics.json                                        # committed minimum citation_match_rate
```

## Run it

If `make` isn't available (e.g. plain Windows without Git Bash), run the commands inside
`Makefile` directly — each target is a single command.

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows:     .venv\Scripts\activate
make setup
make ingest  # builds this lab's own retrieval index
make eval    # runs the offline evaluation, prints citation-match rate, writes eval_report.json
make test    # runs pytest, which includes the gate check and the red-team check as normal assertions
```

Optional, works with or without a real API key (meaningless in mock mode, real with one):
```bash
make judge
```

## What to notice

- The graded metric is **citation-match rate**, not answer-text quality — retrieval doesn't depend
  on the LLM at all, so it's identically meaningful in mock or real mode. Judging the mock LLM's
  answer TEXT would be meaningless, since it always returns the same canned string regardless of
  the question.
- `judge.py`'s LLM-as-judge score is reported for illustration only and is NEVER gated on — see its
  module docstring, and try Week 4's Exercise 3 to prove this to yourself.
- `red_team.py`'s check passes by construction: citations are built from retrieval metadata (see
  `app/domain/rag.py`'s `build_citations`), never from the LLM's own output, so no adversarial
  prompt can make the system cite a document that isn't actually in the corpus. Try Week 4's
  Exercise 4 to see the check actually catch something when you deliberately break it.
- `gate.py`'s `check_gate()` is asserted directly in `tests/test_gate.py` — the normal `pytest` run
  this repo's CI already executes for every lab IS the release gate.
- `baseline_metrics.json`'s `citation_match_rate: 0.80` minimum was chosen after measuring a genuine
  1.00 (8/8) on this lab's committed index — not by guessing. With only 8 labeled examples, achievable
  rates are always a multiple of 0.125, so 0.80 is a threshold (not a predicted score): it sits strictly
  between 0.75 and 0.875, meaning one single-question regression still passes as noise while a
  two-question regression correctly fails the gate.
