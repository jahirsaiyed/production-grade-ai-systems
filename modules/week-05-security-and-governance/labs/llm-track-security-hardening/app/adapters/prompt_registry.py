"""
A lightweight, versioned prompt/policy registry. Templates live as files
under prompts/<name>/vN.txt; prompts/registry.json points at which version
is "active" per template name. validate_registry() is the "offline check"
this course's syllabus refers to: it runs in CI (as a test, see
test_prompt_registry.py) and at service startup (see app/main.py's
lifespan, which fails fast if this returns any errors) — not as a live
approval workflow.

Note: this registry is validated and tested, but NOT wired into the live
/ask request path this week — app/domain/rag.py's build_rag_prompt (reused
unchanged from Week 2, already reviewed) still builds the actual prompt.
This module demonstrates the versioning/validation pattern standalone.
"""
import json
import re
from pathlib import Path

REGISTRY_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

EXPECTED_PLACEHOLDERS = {
    "rag_answer": {"context", "question"},
}


def _placeholders_in(template_text: str) -> set[str]:
    return set(re.findall(r"\{(\w+)\}", template_text))


def get_active_prompt(name: str, registry_dir: Path = REGISTRY_DIR) -> str:
    registry = json.loads((registry_dir / "registry.json").read_text(encoding="utf-8"))
    version = registry[name]
    template_path = registry_dir / name / f"{version}.txt"
    return template_path.read_text(encoding="utf-8")


def validate_registry(registry_dir: Path = REGISTRY_DIR) -> list[str]:
    errors: list[str] = []
    registry = json.loads((registry_dir / "registry.json").read_text(encoding="utf-8"))

    for name, version in registry.items():
        template_path = registry_dir / name / f"{version}.txt"
        if not template_path.exists():
            errors.append(
                f"{name}: active version {version} file not found at {template_path}"
            )
            continue

        expected = EXPECTED_PLACEHOLDERS.get(name, set())
        actual = _placeholders_in(template_path.read_text(encoding="utf-8"))
        if actual != expected:
            errors.append(
                f"{name}: template {version} has placeholders {actual}, expected {expected}"
            )

    return errors
