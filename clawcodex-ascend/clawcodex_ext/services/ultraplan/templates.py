"""Reusable prompt templates for ultraplan planning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exceptions import TemplateNotFoundError


@dataclass(frozen=True)
class PlanTemplate:
    id: str
    title: str
    description: str
    prompt: str

    def apply(self, user_prompt: str) -> str:
        return self.prompt.format(goal=user_prompt.strip())


BUILTIN_TEMPLATES: dict[str, PlanTemplate] = {
    "refactor": PlanTemplate(
        id="refactor",
        title="Code refactor",
        description="Plan a scoped code refactor with verification steps.",
        prompt=(
            "Create a careful refactoring plan for: {goal}\n"
            "Prefer small behavior-preserving edits, local tests, and rollback notes."
        ),
    ),
    "write_tests": PlanTemplate(
        id="write_tests",
        title="Write tests",
        description="Plan focused unit and integration tests.",
        prompt=(
            "Create a test implementation plan for: {goal}\nInclude fixture needs, edge cases, and commands to run."
        ),
    ),
    "write_docs": PlanTemplate(
        id="write_docs",
        title="Write docs",
        description="Plan documentation updates and examples.",
        prompt=(
            "Create a documentation plan for: {goal}\n"
            "Include target audience, files to update, examples, and validation."
        ),
    ),
    "bug_investigate": PlanTemplate(
        id="bug_investigate",
        title="Bug investigation",
        description="Plan diagnosis, reproduction, fix, and regression tests.",
        prompt=(
            "Create a bug investigation plan for: {goal}\n"
            "Start with reproduction and instrumentation before proposing fixes."
        ),
    ),
}


class TemplateLibrary:
    def __init__(self, custom_dir: Path | str | None = None) -> None:
        self.custom_dir = Path(custom_dir).expanduser() if custom_dir else None

    def list_templates(self) -> list[PlanTemplate]:
        templates = dict(BUILTIN_TEMPLATES)
        templates.update(self._load_custom())
        return sorted(templates.values(), key=lambda item: item.id)

    def get(self, template_id: str) -> PlanTemplate:
        templates = {tpl.id: tpl for tpl in self.list_templates()}
        try:
            return templates[template_id]
        except KeyError as exc:
            raise TemplateNotFoundError(f"unknown ultraplan template: {template_id}") from exc

    def apply(self, template_id: str, user_prompt: str) -> str:
        return self.get(template_id).apply(user_prompt)

    def _load_custom(self) -> dict[str, PlanTemplate]:
        if self.custom_dir is None or not self.custom_dir.exists():
            return {}
        templates: dict[str, PlanTemplate] = {}
        for path in sorted(self.custom_dir.iterdir()):
            if path.suffix.lower() not in {".json", ".yaml", ".yml"}:
                continue
            try:
                data = _load_template_file(path)
                tpl = PlanTemplate(
                    id=str(data["id"]),
                    title=str(data["title"]),
                    description=str(data.get("description") or ""),
                    prompt=str(data["prompt"]),
                )
            except Exception:
                continue
            templates[tpl.id] = tpl
        return templates


def _load_template_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data: Any = json.loads(text)
    else:
        data = _load_yaml(text)
    if not isinstance(data, dict):
        raise TypeError("template file must contain an object")
    return data


def _load_yaml(text: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return _load_simple_yaml(text)
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise TypeError("template YAML must contain a mapping")
    return data


def _load_simple_yaml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        raw = lines[index]
        index += 1
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise ValueError("unsupported YAML template syntax")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value in {"|", ">"}:
            block: list[str] = []
            while index < len(lines):
                candidate = lines[index]
                if candidate and not candidate.startswith((" ", "\t")):
                    break
                block.append(candidate[2:] if candidate.startswith("  ") else candidate.lstrip())
                index += 1
            data[key] = "\n".join(block).rstrip("\n")
        else:
            data[key] = value.strip("\"'")
    return data
