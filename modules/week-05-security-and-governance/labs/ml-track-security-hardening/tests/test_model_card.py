from model_card import MODEL_NAME, MODEL_VERSION, generate_model_card


def test_generate_model_card_includes_required_sections():
    card = generate_model_card()

    assert f"# Model Card: {MODEL_NAME}" in card
    assert MODEL_VERSION in card
    assert "## Intended Use" in card
    assert "## Training Data" in card
    assert "## Limitations" in card
    assert "## Out-of-Scope Uses" in card


def test_generate_model_card_lists_at_least_one_limitation_and_one_out_of_scope_use():
    card = generate_model_card()

    limitations_section = card.split("## Limitations")[1].split("## Out-of-Scope Uses")[0]
    out_of_scope_section = card.split("## Out-of-Scope Uses")[1]

    assert "- " in limitations_section
    assert "- " in out_of_scope_section
