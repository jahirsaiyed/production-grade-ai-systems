import json

import pytest

from app.adapters.prompt_registry import get_active_prompt, validate_registry


def test_validate_registry_passes_for_the_real_committed_registry():
    errors = validate_registry()
    assert errors == []


def test_get_active_prompt_returns_the_active_versions_content():
    prompt = get_active_prompt("rag_answer")
    assert "{context}" in prompt
    assert "{question}" in prompt


def test_validate_registry_fails_when_the_active_version_file_is_missing(tmp_path):
    registry_dir = tmp_path
    (registry_dir / "registry.json").write_text(json.dumps({"rag_answer": "v99"}))
    (registry_dir / "rag_answer").mkdir()

    errors = validate_registry(registry_dir=registry_dir)

    assert len(errors) == 1
    assert "v99" in errors[0]
    assert "not found" in errors[0]


def test_validate_registry_fails_when_placeholders_do_not_match(tmp_path):
    registry_dir = tmp_path
    (registry_dir / "registry.json").write_text(json.dumps({"rag_answer": "v1"}))
    template_dir = registry_dir / "rag_answer"
    template_dir.mkdir()
    (template_dir / "v1.txt").write_text("Missing the expected placeholders entirely.")

    errors = validate_registry(registry_dir=registry_dir)

    assert len(errors) == 1
    assert "placeholders" in errors[0]
