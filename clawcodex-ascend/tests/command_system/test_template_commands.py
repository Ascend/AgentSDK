from __future__ import annotations

from pathlib import Path

from clawcodex_ext.command_system.engine import create_command_context
from clawcodex_ext.command_system.template_commands import template_command_call
from src.services.templates import reset_default_template_registry


def test_template_list_includes_built_ins(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CLAWCODEX_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("CLAWCODEX_MANAGED_CONFIG_DIR", str(tmp_path / "managed"))
    reset_default_template_registry()
    context = create_command_context(tmp_path)
    result = template_command_call("list --kind agent", context)
    assert "general-purpose" in result.value
    assert "agent" in result.value


def test_template_preview_renders_variables(tmp_path: Path, monkeypatch) -> None:
    cfg = tmp_path / "cfg" / "templates"
    cfg.mkdir(parents=True)
    (cfg / "skill.yml").write_text(
        """
id: skill-demo
title: Skill Demo
kind: skill
variables:
  - name: skill_name
    description: Skill name
fields:
  output_path_template: ".claude/skills/{{ skill_name }}/SKILL.md"
  content_template: "# {{ skill_name }}"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CLAWCODEX_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("CLAWCODEX_MANAGED_CONFIG_DIR", str(tmp_path / "managed"))
    reset_default_template_registry()
    context = create_command_context(tmp_path)
    result = template_command_call("preview skill-demo --var skill_name=demo", context)
    assert "# demo" in result.value
    assert str(tmp_path / ".claude" / "skills" / "demo" / "SKILL.md") in result.value


def test_template_create_skill_writes_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CLAWCODEX_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("CLAWCODEX_MANAGED_CONFIG_DIR", str(tmp_path / "managed"))
    reset_default_template_registry()
    context = create_command_context(tmp_path)
    result = template_command_call(
        'create skill --name browser --description "Browser automation"',
        context,
    )
    target = tmp_path / ".claude" / "skills" / "browser" / "SKILL.md"
    assert "Created skill" in result.value
    assert target.read_text(encoding="utf-8") == "# browser\n\nBrowser automation\n"
