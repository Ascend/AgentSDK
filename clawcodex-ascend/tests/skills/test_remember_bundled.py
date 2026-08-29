#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Regression coverage for the bundled ``remember`` skill."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pytest

from clawcodex_ext.command_system.aggregator import clear_commands_cache
from clawcodex_ext.skills.bundled_skills import (
    clear_bundled_skills,
    get_bundled_skill_by_name,
)
from clawcodex_ext.skills.loader import (
    clear_dynamic_skills,
    clear_skill_caches,
    clear_skill_registry,
)
from clawcodex_ext.tool_system.context import ToolContext
from clawcodex_ext.tool_system.tools import SkillTool
from clawcodex_ext.tool_system.tools.skill import run_user_invoked_skill


@pytest.fixture(autouse=True)
def _isolated_skill_runtime(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("CLAUDE_CODE_DISABLE_AUTO_MEMORY", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_SIMPLE", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_REMOTE", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_TEAM_MEMORY", raising=False)
    monkeypatch.delenv("CLAUDE_COWORK_MEMORY_PATH_OVERRIDE", raising=False)
    clear_commands_cache()
    clear_skill_caches()
    clear_dynamic_skills()
    clear_skill_registry()
    clear_bundled_skills()
    yield
    clear_commands_cache()
    clear_skill_caches()
    clear_dynamic_skills()
    clear_skill_registry()
    clear_bundled_skills()


def _model_skill_prompt(result: object) -> str:
    assert result.new_messages and len(result.new_messages) == 1
    content = result.new_messages[0].content
    return content.split("\n\n", 1)[1]


def _runtime_snapshot(prompt: str) -> dict[str, object]:
    payload = prompt.split("```json\n", 1)[1].split("\n```", 1)[0]
    return json.loads(payload)


def test_remember_registration_metadata_and_enabled_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = get_bundled_skill_by_name("remember")

    assert skill is not None
    assert skill.user_invocable is True
    assert skill.loaded_from == "bundled"
    assert skill.when_to_use is not None
    assert skill.is_enabled() is True

    monkeypatch.setenv("CLAUDE_CODE_DISABLE_AUTO_MEMORY", "1")
    assert skill.is_enabled() is False


def test_remember_prompt_preserves_review_before_write_contract(tmp_path: Path) -> None:
    result = SkillTool.call({"skill": "remember"}, ToolContext(workspace_root=tmp_path))
    prompt = _model_skill_prompt(result)

    assert result.output["success"] is True
    assert "# Memory Review" in prompt
    assert "`CLAUDE.md` and `CLAUDE.local.md`" in prompt
    assert "**Team memory**" in prompt
    assert "**Stay in auto-memory**" in prompt
    assert "Present all proposals before making any changes." in prompt
    assert "without explicit user approval" in prompt
    assert "## Authoritative runtime snapshot" in prompt
    assert "do not search for repository-local `.claude/memories/`" in prompt
    snapshot = _runtime_snapshot(prompt)
    assert snapshot["workspace_root"] == str(tmp_path)
    assert snapshot["auto_memory_enabled"] is True
    assert snapshot["team_memory_enabled"] is False


def test_remember_snapshot_uses_exact_memory_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory_dir = tmp_path / "authoritative-memory"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_text("- [Topic](topic.md) — hook\n", encoding="utf-8")
    monkeypatch.setenv("CLAUDE_COWORK_MEMORY_PATH_OVERRIDE", str(memory_dir))
    monkeypatch.setenv("CLAUDE_CODE_TEAM_MEMORY", "1")

    result = SkillTool.call({"skill": "remember"}, ToolContext(workspace_root=tmp_path))
    prompt = _model_skill_prompt(result)
    snapshot = _runtime_snapshot(prompt)

    assert snapshot["auto_memory_directory"] == str(memory_dir)
    assert snapshot["auto_memory_index"] == str(memory_dir / "MEMORY.md")
    assert snapshot["auto_memory_directory_exists"] is True
    assert snapshot["auto_memory_index_exists"] is True
    assert snapshot["team_memory_directory"] == str(memory_dir / "team")
    assert snapshot["team_memory_enabled"] is True


def test_remember_user_invocation_appends_context(tmp_path: Path) -> None:
    result = run_user_invoked_skill(
        "remember",
        "focus on testing preferences",
        ToolContext(workspace_root=tmp_path),
    )

    assert result.output["success"] is True
    assert "## Additional context from user" in result.output["prompt"]
    assert "focus on testing preferences" in result.output["prompt"]


def test_remember_is_unavailable_when_auto_memory_is_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDE_CODE_DISABLE_AUTO_MEMORY", "true")

    result = run_user_invoked_skill(
        "remember",
        "",
        ToolContext(workspace_root=tmp_path),
    )

    assert result.is_error is True
    assert "disabled" in result.output["error"].lower()
