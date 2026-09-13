"""
Regex-based PII redaction, applied to a user's question before it is
persisted to the audit log. Never applied to the corpus or to retrieval —
this is about not persisting a caller's PII in logs, not about censoring
the knowledge base.
"""
import re

_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE_PATTERN = re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b")


def redact(text: str) -> str:
    text = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
    text = _SSN_PATTERN.sub("[REDACTED_SSN]", text)
    text = _PHONE_PATTERN.sub("[REDACTED_PHONE]", text)
    return text
