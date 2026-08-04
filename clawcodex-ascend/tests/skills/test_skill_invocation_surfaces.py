#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Keep user slash invocation separate from model SkillTool invocation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from src.skills.bundled import init_bundled_skills
from src.skills.bundled_skills import (
    BundledSkillDefinition,
    clear_bundled_skills,
    register_bundled_skill,
)
from src.skills.loader import clear_skill_caches, clear_skill_registry
from src.tool_system.context import ToolContext
from src.tool_system.protocol import ToolCall
from src.tool_system.registry import ToolRegistry
from src.tool_system.tools.skill import SkillTool, run_user_invoked_skill


@pytest.fixture(autouse=True)
def _reset_skill_state() -> Iterator[None]:
    clear_bundled_skills()
    clear_skill_caches()
    clear_skill_registry()
    yield
    clear_bundled_skills()
    clear_skill_caches()
    clear_skill_registry()


@pytest.fixture
def tool_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> ToolContext:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("CLAWCODEX_CONFIG_DIR", str(home / ".clawcodex"))
    monkeypatch.setenv("CLAWCODEX_MANAGED_CONFIG_DIR", str(tmp_path / "managed"))
    return ToolContext(workspace_root=tmp_path, cwd=tmp_path)


def test_user_can_run_disable_model_invocation_bundled_skill(
    tool_context: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real user-only ``/debug`` skill still renders for a user."""

    monkeypatch.setenv(
        "CLAUDE_CODE_DEBUG_LOG_PATH",
        str(tool_context.workspace_root / "nonexistent-debug.log"),
    )
    init_bundled_skills()

    user_result = run_user_invoked_skill(
        "debug",
        "authentication keeps failing",
        tool_context,
    )

    assert user_result.is_error is False
    assert user_result.output["success"] is True
    assert user_result.output["commandName"] == "debug"
    assert "# Debug Skill" in user_result.output["prompt"]
    assert "authentication keeps failing" in user_result.output["prompt"]

    model_validation = SkillTool.validate_input({"skill": "debug"}, tool_context)
    assert model_validation is not None
    assert model_validation.result is False
    assert model_validation.error_code == 4
    assert "disable-model-invocation" in model_validation.message

    dispatched = ToolRegistry([SkillTool]).dispatch(
        ToolCall(name="Skill", input={"skill": "debug"}),
        tool_context,
    )
    assert dispatched.is_error is True
    assert dispatched.output["error_code"] == 4


def test_user_path_rejects_model_only_skill_but_model_path_accepts_it(
    tool_context: ToolContext,
) -> None:
    register_bundled_skill(
        BundledSkillDefinition(
            name="surface-model-only",
            description="A model-only test skill",
            get_prompt_for_command=lambda args: f"model-only prompt: {args}",
            user_invocable=False,
        )
    )

    user_result = run_user_invoked_skill(
        "surface-model-only",
        "payload",
        tool_context,
    )

    assert user_result.is_error is True
    assert "user" in user_result.output["error"].lower()

    model_validation = SkillTool.validate_input(
        {"skill": "surface-model-only", "args": "payload"},
        tool_context,
    )
    assert model_validation is not None
    assert model_validation.result is True

    model_result = SkillTool.call(
        {"skill": "surface-model-only", "args": "payload"},
        tool_context,
    )
    assert "prompt" not in model_result.output
    assert model_result.output["commandName"] == "surface-model-only"
    assert model_result.new_messages is not None


def test_user_path_rejects_disabled_skill(tool_context: ToolContext) -> None:
    register_bundled_skill(
        BundledSkillDefinition(
            name="surface-disabled",
            description="A disabled test skill",
            get_prompt_for_command=lambda _args: "must not render",
            is_enabled=lambda: False,
        )
    )

    result = run_user_invoked_skill("surface-disabled", "", tool_context)

    assert result.is_error is True
    assert "disabled" in result.output["error"].lower()
    assert "prompt" not in result.output


def test_model_path_rejects_disable_model_invocation_flag(
    tool_context: ToolContext,
) -> None:
    register_bundled_skill(
        BundledSkillDefinition(
            name="surface-user-only",
            description="A user-only test skill",
            get_prompt_for_command=lambda args: f"user-only prompt: {args}",
            disable_model_invocation=True,
        )
    )

    validation = SkillTool.validate_input(
        {"skill": "surface-user-only", "args": "payload"},
        tool_context,
    )

    assert validation is not None
    assert validation.result is False
    assert validation.error_code == 4


@pytest.mark.asyncio
async def test_command_adapter_delegates_to_user_invocation_service(
    tool_context: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from clawcodex_ext.command_system.engine import CommandEngine
    from clawcodex_ext.command_system.registry import CommandRegistry
    from clawcodex_ext.command_system.types import CommandContext, SkillPromptCommand

    calls: list[tuple[str, str, ToolContext]] = []

    def fake_run(name: str, args: str, context: ToolContext):
        calls.append((name, args, context))
        return SimpleNamespace(
            output={"success": True, "prompt": "canonical user prompt"},
            is_error=False,
        )

    monkeypatch.setattr(
        "clawcodex_ext.tool_system.tools.skill.run_user_invoked_skill",
        fake_run,
    )
    command = SkillPromptCommand(name="debug", description="debug")
    command_context = CommandContext(
        workspace_root=tool_context.workspace_root,
        cwd=tool_context.cwd or tool_context.workspace_root,
        tool_context=tool_context,
    )

    blocks = await command.get_prompt_for_command("inspect this", command_context)

    assert blocks == [{"type": "text", "text": "canonical user prompt"}]

    registry = CommandRegistry()
    registry.register(command)
    engine_result = await CommandEngine(
        registry=registry,
        workspace_root=tool_context.workspace_root,
        context=command_context,
    ).execute("/debug inspect this")

    assert engine_result.success is True
    assert engine_result.prompt_is_meta is True
    assert engine_result.prompt_content == blocks
    assert calls == [
        ("debug", "inspect this", tool_context),
        ("debug", "inspect this", tool_context),
    ]


def test_legacy_python_skill_is_model_non_prompt_and_rejects_path_escape(
    tool_context: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skills_dir = tool_context.workspace_root / "legacy-skills"
    skills_dir.mkdir()
    sentinel = tool_context.workspace_root / "escaped.txt"
    outside = tool_context.workspace_root / "escape.py"
    outside.write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('executed')\n"
        "def run(input, context): return 'bad'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CLAWCODEX_SKILLS_DIR", str(skills_dir))

    assert set(SkillTool.input_schema["properties"]) == {"skill", "args"}
    validation = SkillTool.validate_input({"name": "legacy"}, tool_context)
    result = SkillTool.call(
        {"name": "../escape", "input": {}},
        tool_context,
    )

    assert validation is not None
    assert validation.result is False
    assert validation.error_code == 5
    assert result.is_error is True
    assert "invalid legacy skill name" in result.output["error"]
    assert sentinel.exists() is False
