"""
Structured, append-only audit logging for /ask requests.

log_decision's signature deliberately has no parameter for a raw bearer
token or API key — it can only ever write what its three explicit
parameters are given.
"""
import json
import time
from pathlib import Path


def log_decision(
    log_path: Path, subject: str, input_summary: dict, decision: dict
) -> None:
    entry = {
        "timestamp": time.time(),
        "subject": subject,
        "input_summary": input_summary,
        "decision": decision,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
