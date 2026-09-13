from app.domain.pii_redaction import redact


def test_redact_replaces_an_email_address():
    result = redact("Contact me at jane.doe@example.com for details.")
    assert "jane.doe@example.com" not in result
    assert "[REDACTED_EMAIL]" in result


def test_redact_replaces_a_phone_number():
    result = redact("Call me at 555-123-4567 tomorrow.")
    assert "555-123-4567" not in result
    assert "[REDACTED_PHONE]" in result


def test_redact_replaces_an_ssn_shaped_string():
    result = redact("My SSN is 123-45-6789.")
    assert "123-45-6789" not in result
    assert "[REDACTED_SSN]" in result


def test_redact_leaves_a_clean_question_unchanged():
    original = "How many vacation days do I get?"
    assert redact(original) == original


def test_redact_handles_multiple_pii_types_in_one_string():
    result = redact("Email jane@example.com or call 555-987-6543.")
    assert "jane@example.com" not in result
    assert "555-987-6543" not in result
    assert "[REDACTED_EMAIL]" in result
    assert "[REDACTED_PHONE]" in result
