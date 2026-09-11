import pytest

from app.domain.qa import MAX_QUESTION_LENGTH, QuestionTooLongError, build_prompt


def test_build_prompt_includes_the_question():
    prompt = build_prompt("What is training/serving skew?")
    assert "What is training/serving skew?" in prompt


def test_build_prompt_rejects_overly_long_questions():
    with pytest.raises(QuestionTooLongError):
        build_prompt("x" * (MAX_QUESTION_LENGTH + 1))
