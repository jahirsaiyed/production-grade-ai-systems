# Week 1 Exercises

Hints and acceptance criteria only — no answer key. If you get stuck, the
reference implementation is the lab's own code.

## Exercise 1: Break the artifact hash on purpose

Edit one byte of `labs/ml-track-fraud-detection/artifacts/model.joblib` (then
revert it) and start the service. **Acceptance criteria:** the service fails
to start with a clear `ArtifactIntegrityError` message in the logs (uvicorn
will report "Application startup failed") — not a silent wrong prediction.

## Exercise 2: Make the flaky feature store flakier

Raise `feature_store_failure_rate` in `.env` for the fraud-detection lab to
`0.9` and call `/score` several times. **Acceptance criteria:** you can
observe (via logs or by adding a print) that `fetch_features` is retried up
to 3 times before giving up, and that a request occasionally still fails after
exhausting retries — explain in your own words why that's the correct
tradeoff versus retrying forever.

## Exercise 3: Add a request-size guardrail

The `/ask` endpoint caps question length, and `/score`'s `transaction_id`
and `merchant_category` are length-capped too — but `amount` has no upper
bound (only `ge=0`). Add a reasonable upper bound (e.g. `le=1_000_000`) and
write a test proving an over-limit value is rejected with a 422.
**Acceptance criteria:** a new passing test in `tests/`, and `make test`
still shows 80%+ coverage on `domain/` and `api/`.

## Exercise 4: Turn on the real LLM

Get an OpenAI API key, set `OPENAI_API_KEY` in `.env` for the LLM lab, restart
the service, and call `/ask`. **Acceptance criteria:** the response's
`"source"` field changes from `"mock"` to `"llm"`, and you can explain why the
mock/real switch required no code change — only a config change.

## Exercise 5: Trace a request end-to-end

Send a request with a custom `X-Request-ID` header to either service and find
that same ID in the structured JSON log output. **Acceptance criteria:** you
can point to the exact line of middleware code that makes this work.
