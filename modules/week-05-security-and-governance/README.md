# Week 5: Security and Governance

Weeks 1-4 built the system, made it fast, and proved it's correct. Week 5
asks the question that comes right before anyone lets real users near it:
what stops it from being abused, and what proves it's being run responsibly?
You'll work through two tracks: **ML**
([`labs/ml-track-security-hardening/README.md`](labs/ml-track-security-hardening/README.md))
and **LLM**
([`labs/llm-track-security-hardening/README.md`](labs/llm-track-security-hardening/README.md)).
Several topics below need infrastructure this course's local, mock-friendly
labs can't stand up — a real identity provider, a deployed vector database, a
TLS-terminating edge — so each section says plainly whether a lab backs it up
or whether it's conceptual only.

## Threat landscape / OWASP LLM Top 10

The **OWASP Top 10 for LLM Applications** is the industry-standard checklist
of the ways LLM-integrated systems actually get attacked in practice —
prompt injection, insecure output handling, sensitive information
disclosure, training-data poisoning, supply-chain vulnerabilities, excessive
agency, and others. This week's labs address three of them directly: prompt
injection (the LLM lab's `guardrails.py` blocks it before retrieval),
insecure output handling (the citation guardrail and structured-output
contract constrain what the model's output is allowed to become), and
sensitive information disclosure (PII redaction and encryption-at-rest).
Training-data poisoning, supply-chain attacks, and excessive agency are
explicitly out of scope this week — this course trains no model from
untrusted data and has no agentic tool-use for excessive agency to apply to.

## API authentication (tokens, JWT)

Both labs' `app/api/auth.py` require a bearer token on every request, giving
each caller an identity the service can check before doing any work — the
foundational lesson being "every route needs a caller identity," not "this is
how you'd actually run auth in production." Concretely, this is a single
shared-secret HS256 JWT: anyone holding the one signing secret can mint a
valid token, there's no per-user identity, key rotation, or external identity
provider behind it. That's enough to teach the pattern; it is not enough to
run as-is in front of real users.

## PII detection and redaction

The LLM lab's `app/domain/pii_redaction.py` scans text for things like emails
and phone numbers and masks them before anything gets written to a log. It
matters because logs tend to be the leakiest part of a system — they're
long-lived, widely readable, and rarely audited as carefully as the primary
data store. This redaction happens only on the path to logging, though, never
on the path to retrieval: the knowledge base itself is not censored, so
redaction here protects log readers, not retrieval results.

## Encryption in transit and at rest

**Encryption at rest** protects stored data if the storage medium itself is
compromised; **encryption in transit** (TLS) protects data while it moves
between two parties. The ML lab's `app/adapters/encryption.py` demonstrates
the "at rest" half concretely — it Fernet-encrypts the model artifact on disk
and decrypts it only in memory at load time. Encryption in transit is
conceptual only this week: no lab terminates real TLS in a local dev server,
the same "conceptual only" treatment Week 4 gave OpenTelemetry, Prometheus,
and Grafana — it needs a real certificate and a real network boundary to mean
anything.

## Vector store protection

**Vector store protection** covers access control, encryption, and tenant
isolation for the database holding embeddings and retrievable documents. This
is conceptual only this week: this course's "vector store" is a local JSON
file loaded straight off disk, with no deployed database, no network
boundary, and no access-control surface to actually harden.

## Guardrails: input/output filters, structured-output contracts, audit logging

A **guardrail** is a check that runs around a model call rather than inside
it — an input filter rejects a request before it reaches the model, an output
filter or structured-output contract constrains what the model is allowed to
return, and audit logging records what happened for later review. The LLM
lab's `app/domain/guardrails.py` implements the input side (the prompt
injection check from the threat-landscape section above) and the
structured-output contract; both labs' `app/adapters/audit_log.py` implement
the audit-logging side, giving every request a durable record independent of
whether the request was ultimately allowed or blocked.

## Groundedness checks

A **groundedness check** measures whether a generated answer is actually
supported by the retrieved context it was supposed to be based on, as opposed
to the model inventing something plausible-sounding but unsupported. The LLM
lab's `app/domain/groundedness.py` implements this by measuring lexical
overlap between an answer and its retrieved context. This is a meaningfully
different situation from Week 4's `judge.py`, which is meaningless in mock
mode regardless of input: this check IS genuinely demonstrable in mock mode,
because the canned mock answer has measurably low lexical overlap with real
retrieved context, while an answer actually derived from that context scores
measurably higher — the check is exercising a real signal, not returning a
constant.

## Model cards, datasheets, fairness audits

A **model card** documents a model's intended use, training data, and known
limitations; a **datasheet** does the same for a dataset; a **fairness
audit** measures whether a model's errors are distributed unevenly across a
protected attribute. The ML lab's `model_card.py` and `fairness_audit.py`
implement both patterns concretely. The fairness audit itself uses a
synthetic, uncorrelated-by-construction protected attribute, since this
course has no real demographic data to audit — so it teaches the mechanics of
measuring disparity (how you'd compute and compare error rates across
groups), not a real bias finding about the fraud model.

## Explainability and citations as RAG explanations

No new code ships for this topic this week — it points back at Week 2 and
Week 4's citation architecture, where `build_citations` constructs the
citation list directly from retrieval metadata (which document was actually
retrieved) rather than from anything the LLM claims about itself. That
citation trail is this system's actual answer to "how does it explain
itself": not a separate explainability model bolted on afterward, but a
structural guarantee that every answer traces back to a real, checkable
source document.

## Regulatory context (EU AI Act, NIST AI RMF)

The **EU AI Act** is a binding regulation that classifies AI systems by risk
tier (unacceptable, high, limited, minimal) and imposes obligations —
documentation, human oversight, conformity assessment — that scale with that
tier. The **NIST AI Risk Management Framework** is a voluntary U.S.
framework organized around four functions (govern, map, measure, manage) for
identifying and reducing AI risk across a system's lifecycle. Both are
conceptual only this week — there's no lab exercise for a regulatory
framework to demonstrate against, but they define the language regulators
and enterprise risk teams will expect this course's other security work
(guardrails, audits, audit logging) to be described in.

## Prompt/policy registry

A **prompt registry** versions prompt templates the way a package registry
versions code dependencies, so changes are reviewable and revertible instead
of silent edits to a string buried in application code. The LLM lab's
`app/adapters/prompt_registry.py` and its `prompts/` directory implement
this: the registry is validated fail-fast at service startup and covered by
a CI test, so a broken or missing prompt version is caught immediately. It
is not wired into the live `/ask` prompt-building path this week, though —
that path still uses Week 2's reviewed, unchanged `build_rag_prompt` — so the
registry demonstrates the versioning-and-validation pattern standalone,
ready to be adopted by the live path in a future iteration.

## Hands-on labs

- [`labs/ml-track-security-hardening/README.md`](labs/ml-track-security-hardening/README.md) —
  ML track: authenticated API access, an encrypted-at-rest model artifact
  that fails closed on tampering, a model card, and a fairness audit.
- [`labs/llm-track-security-hardening/README.md`](labs/llm-track-security-hardening/README.md) —
  LLM track: authenticated API access, PII redaction before logging,
  prompt-injection and structured-output guardrails, groundedness scoring,
  audit logging, and a standalone prompt registry.

## Exercises

See [`exercises.md`](exercises.md) for five hands-on exercises that build on
both labs above.
