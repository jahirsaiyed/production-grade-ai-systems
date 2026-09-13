import json

from app.adapters.audit_log import log_decision


def test_log_decision_appends_one_json_line_with_expected_fields(tmp_path):
    log_path = tmp_path / "audit.log"

    log_decision(
        log_path,
        subject="test-user",
        input_summary={"question": "How many vacation days do I get?"},
        decision={"sources": ["hr-policy.md"], "groundedness_score": 0.8},
    )

    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["subject"] == "test-user"
    assert entry["input_summary"]["question"] == "How many vacation days do I get?"
    assert entry["decision"]["sources"] == ["hr-policy.md"]
    assert "timestamp" in entry


def test_log_decision_appends_to_an_existing_file(tmp_path):
    log_path = tmp_path / "audit.log"

    log_decision(log_path, subject="user-a", input_summary={}, decision={})
    log_decision(log_path, subject="user-b", input_summary={}, decision={})

    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2


def test_log_decision_never_receives_or_writes_a_raw_secret():
    import inspect

    signature = inspect.signature(log_decision)
    assert set(signature.parameters) == {"log_path", "subject", "input_summary", "decision"}
