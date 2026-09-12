# Week 4: Evaluation and Monitoring

Weeks 1-3 got a model and an LLM app packaged into services, given
well-governed data, and made fast enough for real traffic. Week 4 asks the
question that comes right after "it's running": how do you know it's still
*correct*, and how do you catch it when it isn't? You'll work through two
tracks: **ML** (a fraud-model evaluation harness in
[`labs/ml-track-eval-harness/README.md`](labs/ml-track-eval-harness/README.md))
and **LLM** (a RAG Q&A evaluation harness in
[`labs/llm-track-eval-harness/README.md`](labs/llm-track-eval-harness/README.md)).
Both labs are **offline evaluation** — they score a model or LLM pipeline
against a held-out, labeled set, not live production traffic — and both wire
their checks into a pytest-enforced gate, so a regression fails the build the
same way a broken unit test would. Several topics below need real
observability or deployment infrastructure (a running trace collector, a
metrics backend, multiple live release stages) that this course's local,
mock-friendly labs can't stand up, so each section says plainly whether a lab
backs it up or whether it's conceptual only.

## Offline vs. online evaluation

**Offline evaluation** scores a model or LLM pipeline against a fixed,
labeled dataset before anything ships — you already know the "right" answers,
so you can compute exact metrics and compare runs deterministically.
**Online evaluation** watches real production traffic after release, where
there's no ground-truth label attached to each live request, so you have to
infer quality from proxy signals (click-through, user correction rate,
downstream outcomes) instead. Production systems need both: offline catches
regressions before they ship, online catches the failures that only show up
once the model meets real, unlabeled traffic. Both of this week's labs are
offline evaluation only — `ml-track-eval-harness` scores a trained model
against a held-out labeled test set, and `llm-track-eval-harness` scores
answers against a fixed set of question/expected-source pairs — neither lab
touches live traffic.

## Classification metrics (precision, recall, F1, ROC-AUC vs. PR-AUC)

**Precision** answers "of everything I flagged as positive, how much was
actually positive?"; **recall** answers "of everything that was actually
positive, how much did I catch?"; **F1** is their harmonic mean, a single
number that penalizes ignoring either one. **ROC-AUC** and **PR-AUC** both
summarize a model across every possible threshold at once, but they disagree
on imbalanced data: ROC-AUC's false-positive rate stays small (and the curve
looks good) even when a model makes many false positives, as long as the
negative class is huge, while **PR-AUC** is sensitive to exactly that failure
mode — which is why PR-AUC is the preferred summary metric for problems like
this course's fraud-detection data, where genuine fraud is a small minority
of transactions. `ml-track-eval-harness/metrics.py` computes precision,
recall, F1, ROC-AUC, and PR-AUC directly off the fraud model's predictions,
so you can see the ROC-AUC/PR-AUC gap on real imbalanced data rather than
just read about it.

## Threshold selection with cost-based sweeps

A classifier outputs a probability, not a decision — turning that probability
into a "flag this transaction" / "let it through" call requires picking a
threshold, and 0.5 is an arbitrary default that ignores what the two kinds of
mistakes actually cost. A **cost-based threshold sweep** instead assigns a
real cost to a false positive (annoying a legitimate customer) and a false
negative (missing actual fraud), computes total cost at every candidate
threshold, and picks the threshold that minimizes it — which moves depending
on that cost ratio, not on any fixed cutoff. `ml-track-eval-harness/threshold_selection.py`
implements exactly this sweep against the fraud model's predictions, so the
"best" threshold in this lab is a computed answer, not an assumed one.

## Calibration (reliability diagrams, Brier score, Platt scaling, isotonic regression)

A model is **calibrated** if its predicted probabilities mean what they say —
among all predictions the model scored at 0.7, roughly 70% should actually be
positive. **Reliability diagrams** plot predicted probability against
observed frequency to make miscalibration visible at a glance, and the
**Brier score** condenses the same idea into a single number (mean squared
error between predicted probability and actual outcome). Measuring
miscalibration and *fixing* it are different jobs, though: **Platt scaling**
(fitting a logistic curve to remap scores) and **isotonic regression**
(fitting a non-decreasing step function instead) are both calibration-
*correction* techniques, and neither is implemented in a lab this week —
they're conceptual only, since correcting calibration meaningfully needs a
separate held-out calibration set and adds a second fitting step beyond this
week's scope. `ml-track-eval-harness/calibration.py` covers the *measurement*
half: it computes a reliability diagram's bins and the Brier score for the
fraud model's real predictions, so you can see how miscalibrated (or not) the
model already is before ever reaching for a correction technique.

## Eval harnesses (RAGAS, DeepEval), LLM-as-judge calibration

**RAGAS** and **DeepEval** are real, widely used off-the-shelf Python
libraries purpose-built for evaluating RAG and LLM pipelines — they ship
pre-built metrics (faithfulness, answer relevancy, context precision, and
similar) so teams don't have to hand-roll their own scoring logic. This
week's `llm-track-eval-harness/eval_harness.py` and `judge.py` are **not**
a replacement for either — they're a small, hand-rolled, simplified stand-in
built so you can see exactly what an "LLM-as-judge" call actually does
mechanically (send a question, an answer, and a rubric to an LLM; parse back
a verdict) before reaching for a library that hides that mechanism behind an
API. Once the mechanics are familiar, RAGAS or DeepEval are the tools to
reach for on a real project instead of maintaining a hand-rolled judge
long-term.

## Multi-turn continuity checks

A **multi-turn continuity check** verifies that a conversational system keeps
context straight across turns — that it remembers what "it" referred to two
messages ago, doesn't contradict an earlier answer, and doesn't lose track of
constraints the user already stated. This is conceptual only this week:
every lab in this course, including both of this week's, is a **single-turn**
Q&A system — one question in, one answer out, no stored conversation state —
so there's no multi-turn conversation for a continuity check to run against.

## Red-team prompts in regression packs

A **red-team regression pack** is a fixed set of adversarial or edge-case
prompts, checked automatically on every run, specifically designed to catch a
known failure mode before it ships — the same "regression test" idea applied
to LLM behavior instead of code behavior. `llm-track-eval-harness/red_team.py`
runs one concrete check: it verifies every citation the RAG pipeline returns
names a real corpus document, flagging any citation to a source that doesn't
exist as a fabrication. This check holds **by construction** given this
course's Week 2 RAG architecture — `build_citations` builds the citation list
directly from retrieval metadata (which document was actually retrieved),
never by parsing or trusting the LLM's own generated text — so a citation can
only be wrong if the retrieval-metadata plumbing itself breaks, which is
precisely the kind of regression this check exists to catch.

## Tracing and metrics (OpenTelemetry, Prometheus, Grafana)

**Tracing** captures the path a single request takes through a system —
every service it touched, how long each step took, where it failed — and
**OpenTelemetry** is the dominant vendor-neutral standard for emitting that
trace data. **Metrics** aggregate numeric signals over time (request rate,
error rate, latency) rather than tracking individual requests; **Prometheus**
is the standard tool for collecting and querying those time-series metrics,
and **Grafana** is the standard tool for turning them into dashboards. This
is conceptual only — instrumenting real traces and metrics meaningfully needs
a running collector, a metrics backend, and a dashboarding tool, none of
which this course stands up; both labs run as local scripts and services
without that infrastructure.

## Drift classes: data, concept, embedding, prompt

Production models silently degrade when the world they're serving no longer
matches the world they were built for — this is **drift**, and it comes in
several distinct flavors. **Data drift** is a shift in the input feature
distribution itself (transaction amounts creeping upward over time).
**Concept drift** is a shift in the relationship between inputs and the true
label (the same transaction pattern that used to be legitimate is now
correlated with fraud). **Embedding drift** is a shift in the vector space a
retrieval or similarity system relies on (new documents cluster differently
than the ones the system was tuned against). **Prompt drift** is an
unintentional change in the effective instructions an LLM receives over time
(a template edited for one use case quietly changes behavior for another).
This is conceptual only — detecting any of these classes meaningfully
requires comparing against real production traffic over time, which neither
lab's fixed offline dataset provides.

## Outcome-based alerting

**Outcome-based alerting** triggers on a business-meaningful result crossing
a threshold (fraud losses spiking, customer complaints rising, conversion
dropping) rather than on a purely technical signal (CPU usage, error count).
It matters because a system can look perfectly healthy on every technical
dashboard while quietly making worse decisions — outcome-based alerts are
what actually tell you the model itself has a problem, not just the
infrastructure running it. This is conceptual only — meaningful outcome-based
alerting needs a live alerting pipeline wired to real production outcomes
over time, which is out of scope for this week's offline labs.

## Continuous training vs. CI/CD

**CI/CD** (continuous integration/continuous delivery) re-runs tests and
redeploys *code* whenever it changes. **Continuous training** is the model
analog: automatically retraining a model on fresh data on some cadence (or
trigger, like detected drift) and redeploying the new model artifact, often
through the same CI/CD pipeline that ships code. The two are complementary,
not competing — a mature ML system needs its code changes gated by CI/CD and
its model artifact kept fresh by continuous training, on separate but
related schedules. This is conceptual only — standing up a real retraining
pipeline needs a live, growing data source to retrain against, which neither
lab's static offline dataset provides.

## Progressive delivery: shadow -> canary -> full, and rollback gates

**Progressive delivery** rolls out a change gradually instead of all at once,
to limit the blast radius of a bad release: **shadow** sends production
traffic to the new version alongside the old one without serving its
response to real users, **canary** serves the new version to a small
percentage of real traffic and compares outcomes, and **full rollout** only
happens once the canary looks safe — with a **rollback gate** at each stage
automatically reverting to the previous version if a monitored metric
regresses. This course's labs implement a simplified, single-stage version of
that same rollback-gate idea: `gate.py` in each lab defines the metric
thresholds a model or pipeline must clear, and `tests/test_gate.py` enforces
them as a normal pytest check — so this repo's CI already refuses to merge a
change the moment either lab's gate test fails, which is the same underlying
principle (don't let a regression reach production) that real progressive
delivery builds on across multiple live traffic stages instead of one CI
run.

## Prompt registries

A **prompt registry** manages prompts the way a package registry manages
code dependencies: every prompt template is versioned, changes go through
review, and a service pins itself to a specific prompt version rather than
whatever text happens to be latest — which makes prompt changes reviewable,
revertible, and auditable instead of silent edits to a string buried in
application code. This is conceptual only — no lab this week versions or
gates its prompts through a separate registry; both labs keep prompt text
inline in application code, the same way every lab in this course has so
far.

## Hands-on labs

- [`labs/ml-track-eval-harness/README.md`](labs/ml-track-eval-harness/README.md) —
  ML track: an evaluation harness for the fraud model covering metrics,
  cost-based threshold selection, and calibration measurement, gated by a
  pytest-enforced regression check.
- [`labs/llm-track-eval-harness/README.md`](labs/llm-track-eval-harness/README.md) —
  LLM track: an evaluation harness for the RAG Q&A pipeline covering a
  hand-rolled eval harness, LLM-as-judge scoring, and a red-team check
  against fabricated citations.

## Exercises

See [`exercises.md`](exercises.md) for five hands-on exercises that build on
both labs above.
