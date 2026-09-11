MAX_QUESTION_LENGTH = 500


class QuestionTooLongError(ValueError):
    """Raised when a question exceeds the allowed length."""


def build_prompt(question: str) -> str:
    if len(question) > MAX_QUESTION_LENGTH:
        raise QuestionTooLongError(
            f"question exceeds {MAX_QUESTION_LENGTH} characters"
        )
    return (
        "You are a concise assistant for a production AI systems course.\n"
        f"Question: {question.strip()}\nAnswer:"
    )
