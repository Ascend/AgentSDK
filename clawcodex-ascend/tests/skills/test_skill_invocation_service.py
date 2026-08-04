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

"""Focused contract tests for the shared skill invocation transaction."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from clawcodex_ext.skills.invocation import (
    SkillInvocationErrorCode,
    SkillInvocationOrigin,
    SkillInvocationRequest,
    SkillInvocationResult,
    SkillInvocationService,
)
from clawcodex_ext.skills.model import Skill
from clawcodex_ext.tool_system.context import ToolContext, ToolUseOptions
from clawcodex_ext.permissions.types import ToolPermissionContext


@pytest.fixture
def context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        workspace_root=tmp_path,
        cwd=tmp_path,
        permission_context=ToolPermissionContext(
            mode="default",
            always_allow_rules={"session": ["Edit"]},
        ),
        options=ToolUseOptions(
            main_loop_model="original-model",
            thinking_config={"budget_tokens": 1024},
        ),
        agent_id="agent-1",
        session_id="session-1",
    )


def _service_for(skill: Skill, records: list[tuple[Any, ...]]) -> SkillInvocationService:
    def resolve(name: str, _context: ToolContext) -> Skill | None:
        if name == skill.name or name in skill.aliases:
            return skill
        return None

    return SkillInvocationService(
        resolver=resolve,
        recorder=lambda *record: records.append(record),
    )


def test_origin_gates_are_independent_and_enabled_is_checked_live(
    context: ToolContext,
) -> None:
    enabled = True
    skill = Skill(
        name="canonical",
        aliases=["short"],
        description="test",
        disable_model_invocation=True,
        get_prompt_for_command=lambda args: f"prompt: {args}",
        is_enabled_fn=lambda: enabled,
    )
    records: list[tuple[Any, ...]] = []
    service = _service_for(skill, records)

    user_result = service.invoke(
        SkillInvocationRequest("short", "payload", SkillInvocationOrigin.USER),
        context,
    )
    assert user_result.success is True
    assert user_result.command_name == "canonical"

    model_result = service.validate(
        SkillInvocationRequest("short", origin=SkillInvocationOrigin.MODEL),
        context,
    )
    assert model_result.success is False
    assert model_result.error is not None
    assert model_result.error.code is SkillInvocationErrorCode.MODEL_INVOCATION_DISABLED
    assert model_result.error.model_error_code == 4

    enabled = False
    disabled_result = service.validate(
        SkillInvocationRequest("short", origin=SkillInvocationOrigin.MODEL),
        context,
    )
    assert disabled_result.success is False
    assert disabled_result.error is not None
    assert disabled_result.error.code is SkillInvocationErrorCode.DISABLED
    assert disabled_result.error.model_error_code == 2


def test_inline_result_has_command_metadata_record_and_private_overrides(
    context: ToolContext,
) -> None:
    skill = Skill(
        name="review",
        aliases=["r"],
        description="test",
        base_dir=str(context.workspace_root / ".clawcodex" / "skills" / "review"),
        allowed_tools=["Read", "Bash(git:*)"],
        model="skill-model",
        effort="high",
        get_prompt_for_command=lambda args: f"review {args}",
    )
    records: list[tuple[Any, ...]] = []
    service = _service_for(skill, records)

    result = service.invoke(
        SkillInvocationRequest("r", "src/app.py", SkillInvocationOrigin.MODEL),
        context,
    )

    assert result.success is True
    assert result.prompt == "review src/app.py"
    assert result.content_blocks == ({"type": "text", "text": "review src/app.py"},)
    assert len(result.new_messages) == 1
    message = result.new_messages[0]
    assert message.isMeta is True
    assert "<command-message>review</command-message>" in message.content
    assert "<command-name>/review</command-name>" in message.content
    assert "<command-args>src/app.py</command-args>" in message.content
    assert message.content.endswith("review src/app.py")
    assert records == [
        (
            "review",
            str(context.workspace_root / ".clawcodex" / "skills" / "review"),
            "review src/app.py",
            "agent-1",
        )
    ]

    assert result.context_modifier is not None
    modified = result.context_modifier(context)
    assert modified is not context
    assert modified.permission_context is not context.permission_context
    assert modified.options is not context.options
    assert modified.permission_context.always_allow_rules == {
        "session": ["Edit"],
        "command": ["Read", "Bash(git:*)"],
    }
    assert modified.options.main_loop_model == "skill-model"
    assert modified.options.thinking_config == {
        "budget_tokens": 1024,
        "effort": "high",
    }

    assert context.permission_context.always_allow_rules == {"session": ["Edit"]}
    assert context.options.main_loop_model == "original-model"
    assert context.options.thinking_config == {"budget_tokens": 1024}


def test_prompt_builder_failure_is_atomic(context: ToolContext) -> None:
    def fail(_args: str) -> str:
        raise RuntimeError("builder exploded")

    skill = Skill(
        name="broken",
        description="test",
        allowed_tools=["Read"],
        get_prompt_for_command=fail,
    )
    records: list[tuple[Any, ...]] = []
    service = _service_for(skill, records)

    result = service.invoke(
        SkillInvocationRequest("broken", origin=SkillInvocationOrigin.USER),
        context,
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code is SkillInvocationErrorCode.PROMPT_BUILD_FAILED
    assert result.new_messages == ()
    assert result.context_modifier is None
    assert records == []


@pytest.mark.parametrize(
    ("skill", "expected"),
    [
        (
            Skill(name="hooked", description="test", hooks={"PreToolUse": []}),
            SkillInvocationErrorCode.HOOKS_UNSUPPORTED,
        ),
        (
            Skill(name="forked", description="test", context="fork"),
            SkillInvocationErrorCode.FORK_UNSUPPORTED,
        ),
        (
            Skill(name="agent-forked", description="test", agent="Explore"),
            SkillInvocationErrorCode.FORK_UNSUPPORTED,
        ),
    ],
)
def test_runtime_owned_features_fail_explicitly_without_handlers(
    context: ToolContext,
    skill: Skill,
    expected: SkillInvocationErrorCode,
) -> None:
    service = _service_for(skill, [])

    result = service.invoke(
        SkillInvocationRequest(skill.name, origin=SkillInvocationOrigin.USER),
        context,
    )

    assert isinstance(result, SkillInvocationResult)
    assert result.success is False
    assert result.error is not None
    assert result.error.code is expected


def test_unsupported_provider_effort_is_reported_as_nonfatal_diagnostic(
    context: ToolContext,
) -> None:
    class UnsupportedProvider:
        pass

    context._active_provider = UnsupportedProvider()
    skill = Skill(
        name="effort-diagnostic",
        description="test",
        effort="high",
        get_prompt_for_command=lambda _args: "diagnostic body",
    )
    service = _service_for(skill, [])

    result = service.invoke(
        SkillInvocationRequest(skill.name, origin=SkillInvocationOrigin.USER),
        context,
    )

    assert result.success is True
    assert result.prompt == "diagnostic body"
    assert any("does not support skill effort" in item for item in result.diagnostics)


def test_model_validation_error_code_matrix(context: ToolContext) -> None:
    unknown_service = SkillInvocationService(resolver=lambda _name, _context: None)

    invalid = unknown_service.validate(
        SkillInvocationRequest("/", origin=SkillInvocationOrigin.MODEL),
        context,
    )
    unknown = unknown_service.validate(
        SkillInvocationRequest("missing", origin=SkillInvocationOrigin.MODEL),
        context,
    )
    local_skill = SimpleNamespace(
        name="local-only",
        type="local",
        is_enabled=lambda: True,
        disable_model_invocation=False,
    )
    non_prompt = SkillInvocationService(resolver=lambda _name, _context: local_skill).validate(
        SkillInvocationRequest("local-only", origin=SkillInvocationOrigin.MODEL),
        context,
    )

    assert invalid.error is not None
    assert invalid.error.model_error_code == 1
    assert unknown.error is not None
    assert unknown.error.model_error_code == 2
    assert non_prompt.error is not None
    assert non_prompt.error.model_error_code == 5
