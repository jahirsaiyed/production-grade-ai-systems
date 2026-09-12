# Week 4 Exercises

Hints and acceptance criteria only — no answer key. If you get stuck, the reference implementation
is the lab's own code.

## Exercise 1: Make the gate actually fail

In `ml-track-eval-harness/baseline_metrics.json`, raise the `f1` minimum above what the model
actually achieves (check `eval_report.json` after running `make eval` to see the real number), then
run `make test`. **Acceptance criteria:** the gate test fails with a clear message naming which
metric regressed and by how much — this is the exact mechanism that would block a real PR.

## Exercise 2: Change the cost assumption and watch the threshold move

`ml-track-eval-harness/eval_harness.py` sweeps thresholds with `cost_fp=1.0, cost_fn=25.0` (a missed
fraud case is assumed 25x worse than a false alarm). Change `cost_fn` to `1.0` (equal cost) and
re-run `make eval`. **Acceptance criteria:** you can explain, in your own words, why the
cost-optimal threshold moved, and in which direction.

## Exercise 3: Prove the LLM-as-judge score really is meaningless in mock mode

Run `llm-track-eval-harness`'s `make judge` twice in a row without setting `OPENAI_API_KEY`.
**Acceptance criteria:** you get the exact same verdicts both times, for every question, regardless
of whether the question and expected source actually match — demonstrating the judge call isn't
evaluating anything in mock mode, just returning a constant.

## Exercise 4: Break the red-team check on purpose

Temporarily edit `llm-track-eval-harness/red_team.py`'s `KNOWN_SOURCES` set to remove one of the
three real corpus filenames, then re-run `make test`. **Acceptance criteria:** the red-team test now
fails, because a citation that used to be considered legitimate is now flagged as "fabricated" —
explain why this proves the check is actually looking at real citation data, not vacuously passing.

## Exercise 5: Add a ninth eval example and watch the citation-match rate change

Add one more `{"question": ..., "expected_source": ...}` entry to
`llm-track-eval-harness/eval_set.py` for a question you make up about one of the three corpus docs,
then run `make eval`. **Acceptance criteria:** the reported `citation_match_rate` changes (it's
now out of 9 examples, not 8), and you can state whether your new example was answered correctly.
Note: if your new example changes the measured rate enough to cross the committed
`baseline_metrics.json` threshold, `make test`'s gate check may then fail — that's expected, not a
bug, and mirrors the real-world case where adding eval examples legitimately moves the bar; adjust
`baseline_metrics.json` the same way the lab's own baseline was derived (re-run and measure).
