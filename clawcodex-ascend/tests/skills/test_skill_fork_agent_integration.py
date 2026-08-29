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

"""Real AgentTool integration coverage for forked skill execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Generator

import pytest

from clawcodex_ext.permissions.types import ToolPermissionContext
from clawcodex_ext.providers.base import BaseProvider, ChatResponse
from clawcodex_ext.skills.catalog import invalidate_skill_catalog
from clawcodex_ext.skills.invocation import (
    SkillInvocationOrigin,
    SkillInvocationRequest,
    _execute_forked_skill,
)
from clawcodex_ext.skills.model import Skill
from clawcodex_ext.skills.visibility import (
    filter_model_visible_skills,
    refresh_agent_skill_listing,
)
from clawcodex_ext.tool_system.build_tool import build_tool
from clawcodex_ext.tool_system.context import (
    QueryChainTracking,
    ToolContext,
    ToolUseOptions,
)
from clawcodex_ext.tool_system.defaults import build_default_registry
from clawcodex_ext.tool_system.protocol import ToolResult
from clawcodex_ext.types.messages import UserMessage
from clawcodex_ext.utils.abort_controller import AbortController


class _ForkProbeProvider(BaseProvider):
    def __init__(self) -> None:
        super().__init__(api_key="test", model="base-model")
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[Any],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        self.calls.append({"messages": messages, "tools": tools, **kwargs})
        if len(self.calls) == 1:
            return ChatResponse(
                content="",
                model="skill-model",
                usage={"input_tokens": 1, "output_tokens": 1},
                finish_reason="tool_use",
                tool_uses=[
                    {
                        "id": "toolu_fork_probe",
                        "name": "ForkContextProbe",
                        "input": {},
                    }
                ],
            )
        return ChatResponse(
            content="fork complete",
            model="skill-model",
            usage={"input_tokens": 1, "output_tokens": 1},
            finish_reason="end_turn",
        )

    def chat_stream(self, *_args: Any, **_kwargs: Any) -> Generator[str, None, None]:
        if False:  # pragma: no cover - marks this as a generator
            yield ""

    def get_available_models(self) -> list[str]:
        return ["base-model", "skill-model"]

    def skill_effort_kwargs(self, effort: str | int) -> dict[str, Any]:
        return {"reasoning_effort": effort}


def _system_text(call: dict[str, Any]) -> str:
    messages = call["messages"]
    assert messages and messages[0]["role"] == "system"
    return str(messages[0]["content"])


def test_context_fork_uses_real_fork_runner_with_gate_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_skill = tmp_path / ".clawcodex" / "skills" / "child-only"
    project_skill.mkdir(parents=True)
    (project_skill / "SKILL.md").write_text(
        "---\nname: child-only\ndescription: visible only in child catalog\n---\nBody",
        encoding="utf-8",
    )
    invalidate_skill_catalog("fork integration setup")
    monkeypatch.delenv("CLAUDE_FORK_SUBAGENT", raising=False)
    monkeypatch.setattr(
        "src.agent.transcript.get_agent_transcript_path",
        lambda *_args, **_kwargs: str(tmp_path / "fork-agent.jsonl"),
    )

    captured: dict[str, Any] = {}

    def probe_call(_input: dict[str, Any], child: ToolContext) -> ToolResult:
        captured.update(
            agent_type=child.agent_type,
            depth=child.query_tracking.depth if child.query_tracking else None,
            active_skills=child.active_skill_names,
            model=child.skill_model_override,
            effort=child.skill_effort_override,
            query_source=child.options.query_source,
            command_allows=tuple(child.permission_context.always_allow_rules.get("command", ())),
        )
        return ToolResult(name="ForkContextProbe", output={"captured": True})

    provider = _ForkProbeProvider()
    registry = build_default_registry(provider=provider, load_agent_tools=False)
    registry.register(
        build_tool(
            name="ForkContextProbe",
            input_schema={"type": "object", "properties": {}},
            call=probe_call,
            is_read_only=lambda _input: True,
            is_concurrency_safe=lambda _input: True,
        )
    )
    context = ToolContext(
        workspace_root=tmp_path,
        cwd=tmp_path,
        permission_context=ToolPermissionContext(
            mode="default",
            always_allow_rules={"session": ["ForkContextProbe"]},
        ),
        options=ToolUseOptions(main_loop_model="base-model"),
        abort_controller=AbortController(),
        query_tracking=QueryChainTracking(chain_id="parent", depth=2),
        messages=[UserMessage(content="parent turn")],
        rendered_system_prompt=("PARENT\n\n# Available Skills\n\n- stale-parent: old\n\n# Tail\nkeep"),
        tool_registry=registry,
        _active_provider=provider,
    )
    skill = Skill(
        name="canonical-fork",
        description="fork integration",
        context="fork",
        model="skill-model",
        effort="high",
        allowed_tools=["Read"],
    )

    result = _execute_forked_skill(
        skill,
        SkillInvocationRequest(
            "canonical-fork",
            "",
            SkillInvocationOrigin.MODEL,
        ),
        context,
        "Search and explore the codebase",
    )

    assert result.success is True
    assert result.status == "fork"
    assert result.fork_result == "fork complete"
    assert captured == {
        "agent_type": "fork",
        "depth": 3,
        "active_skills": ("canonical-fork",),
        "model": "skill-model",
        "effort": "high",
        "query_source": "agent:builtin:fork",
        "command_allows": ("Read",),
    }
    assert len(provider.calls) == 2
    system_prompt = _system_text(provider.calls[0])
    assert "stale-parent" not in system_prompt
    assert "**child-only**: visible only in child catalog" in system_prompt
    assert provider.calls[0]["model"] == "skill-model"
    assert provider.calls[0]["reasoning_effort"] == "high"

    agent_schema = registry.get("Agent").input_schema
    assert "_force_fork" not in agent_schema["properties"]
    assert "_refresh_skill_listing" not in agent_schema["properties"]


def test_child_permission_removes_inherited_skill_listing(tmp_path: Path) -> None:
    provider = _ForkProbeProvider()
    registry = build_default_registry(provider=provider, load_agent_tools=False)
    context = ToolContext(
        workspace_root=tmp_path,
        permission_context=ToolPermissionContext(
            mode="default",
            always_deny_rules={"session": ["Skill"]},
        ),
    )

    refreshed = refresh_agent_skill_listing(
        "base\n\n# Available Skills\n\n- stale-parent: old\n\n# Tail\nkeep",
        context=context,
        tool_registry=registry,
        tools=registry.list_tools(),
        provider=provider,
    )

    assert refreshed == "base\n\n# Tail\nkeep"


def test_child_skill_listing_is_rebuilt_for_each_workspace(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    for root, name in ((first_root, "first-only"), (second_root, "second-only")):
        skill_dir = root / ".clawcodex" / "skills" / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: visible only in {name}\n---\nBody",
            encoding="utf-8",
        )

    invalidate_skill_catalog("cross-workspace child listing setup")
    provider = _ForkProbeProvider()
    registry = build_default_registry(provider=provider, load_agent_tools=False)

    def render(root: Path) -> str:
        context = ToolContext(workspace_root=root, cwd=root)
        return refresh_agent_skill_listing(
            "base\n\n# Available Skills\n\n- stale-parent: old\n\n# Tail\nkeep",
            context=context,
            tool_registry=registry,
            tools=registry.list_tools(),
            provider=provider,
        )

    first = render(first_root)
    second = render(second_root)

    assert "**first-only**: visible only in first-only" in first
    assert "second-only" not in first
    assert "**second-only**: visible only in second-only" in second
    assert "first-only" not in second


def test_child_skill_listing_filters_exact_and_prefix_denies(tmp_path: Path) -> None:
    context = ToolContext(
        workspace_root=tmp_path,
        permission_context=ToolPermissionContext(
            mode="default",
            always_deny_rules={"session": ["Skill(review:*)", "Skill(exact-denied)"]},
        ),
    )
    skills = [
        Skill(name="review:pr", description="prefix denied"),
        Skill(name="exact-denied", description="exact denied"),
        Skill(name="visible", description="visible"),
        Skill(
            name="user-only",
            description="user only",
            disable_model_invocation=True,
        ),
        Skill(
            name="disabled",
            description="disabled",
            is_enabled_fn=lambda: False,
        ),
    ]

    assert [skill.name for skill in filter_model_visible_skills(skills, context)] == ["visible"]
