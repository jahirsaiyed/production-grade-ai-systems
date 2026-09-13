from app.domain.groundedness import score_groundedness


def test_score_groundedness_is_zero_for_fully_disjoint_vocabulary():
    score = score_groundedness("xyz123 qrst999", ["completely different words here"])
    assert score == 0.0


def test_score_groundedness_is_one_when_every_answer_word_appears_in_context():
    context = ["the vacation policy allows twenty days off each year"]
    answer = "the vacation policy allows twenty days off"
    score = score_groundedness(answer, context)
    assert score == 1.0


def test_score_groundedness_of_a_partial_overlap_is_between_zero_and_one():
    context = ["the vacation policy allows twenty days off each year"]
    answer = "the vacation policy also covers completely unrelated topics xyz"
    score = score_groundedness(answer, context)
    assert 0.0 < score < 1.0


def test_score_groundedness_of_the_real_mock_answer_is_meaningfully_low_against_real_context():
    # This is the mock LLM's exact canned message (app/adapters/llm_client.py's
    # _MOCK_ANSWER) — reproduced here as a literal so this test doesn't depend on
    # a private module attribute, only on the string genuinely being unrelated to
    # any real corpus content.
    mock_answer = (
        "This is a mock answer. Set OPENAI_API_KEY in .env to call a real model."
    )
    real_context = [
        "Employees accrue 20 vacation days per year and may carry over up to 5 "
        "unused days into the next calendar year."
    ]
    score = score_groundedness(mock_answer, real_context)
    assert score < 0.5


def test_score_groundedness_returns_zero_for_an_empty_answer():
    assert score_groundedness("", ["some context"]) == 0.0
