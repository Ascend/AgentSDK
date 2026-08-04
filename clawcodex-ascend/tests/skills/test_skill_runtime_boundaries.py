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

"""Focused integration tests for Skill runtime security and request scope."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Generator

import pytest

from clawcodex_ext.hooks.hook_types import HookResult, HookSource
from clawcodex_ext.permissions.check import has_permissions_to_use_tool
from clawcodex_ext.permissions.types import ToolPermissionContext
from clawcodex_ext.providers.base import BaseProvider, ChatResponse
from clawcodex_ext.query.query import QueryParams, run_query
from clawcodex_ext.skills.invocation import (
    SkillInvocationErrorCode,
    SkillInvocationOrigin,
    SkillInvocationRequest,
    SkillInvocationService,
    _execute_forked_skill,
    _register_skill_hooks,
    apply_skill_context_modifier,
    build_request_context_modifier,
)
from clawcodex_ext.skills.model import Skill
from clawcodex_ext.tool_system.context import ToolContext, ToolUseOptions
from clawcodex_ext.tool_system.defaults import build_default_registry
from clawcodex_ext.tool_system.tools.skill import SkillTool
from clawcodex_ext.types.messages import UserMessage
from clawcodex_ext.utils.abort_controller import AbortController


def _context(
    tmp_path: Path,
    *,
    permissions: ToolPermissionContext | None = None,
) -> ToolContext:
    return ToolContext(
        workspace_root=tmp_path,
        cwd=tmp_path,
        permission_context=permissions or ToolPermissionContext(mode="default"),
        options=ToolUseOptions(
            main_loop_model="base-model",
            thinking_config={"budget_tokens": 256},
        ),
        agent_id="parent-agent",
        session_id="session-1",
    )


def test_skill_permissions_use_canonical_name_and_deny_allow_safe_ask_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skills = {
        "short": Skill(name="canonical", aliases=["short"], description="safe"),
        "safe": Skill(name="safe", description="safe"),
        "danger": Skill(
            name="danger",
            description="requires capability",
            allowed_tools=["Bash(git:*)"],
        ),
        "review-pr": Skill(
            name="review:pr",
            description="requires capability",
            allowed_tools=["Read"],
        ),
    }

    def resolve(name: str, **_kwargs: Any) -> Skill | None:
        return skills.get(name)

    monkeypatch.setattr("clawcodex_ext.skills.catalog.resolve", resolve)

    # Canonical exact matching applies even when the caller used an alias;
    # deny wins when the same canonical skill is also explicitly allowed.
    permissions = ToolPermissionContext(
        mode="default",
        always_deny_rules={"session": ["Skill(canonical)"]},
        always_allow_rules={"session": ["Skill(canonical)"]},
    )
    context = _context(tmp_path, permissions=permissions)
    decision = has_permissions_to_use_tool(
        SkillTool,
        {"skill": "short"},
        permissions,
        tool_use_context=context,
    )
    assert decision.behavior == "deny"

    # A canonical prefix rule allows an otherwise unsafe prompt skill.
    permissions = ToolPermissionContext(
        mode="default",
        always_allow_rules={"session": ["Skill(review:*)"]},
    )
    context = _context(tmp_path, permissions=permissions)
    decision = has_permissions_to_use_tool(
        SkillTool,
        {"skill": "review-pr"},
        permissions,
        tool_use_context=context,
    )
    assert decision.behavior == "allow"
    assert decision.decision_reason is not None
    assert decision.decision_reason.rule.rule_value.rule_content == "review:*"

    # Safe properties auto-allow before an ask rule; a skill that asks for an
    # extra tool remains interactive in the absence of an allow rule.
    permissions = ToolPermissionContext(
        mode="default",
        always_ask_rules={"session": ["Skill(safe)"]},
    )
    context = _context(tmp_path, permissions=permissions)
    safe_decision = has_permissions_to_use_tool(
        SkillTool,
        {"skill": "safe"},
        permissions,
        tool_use_context=context,
    )
    assert safe_decision.behavior == "allow"

    permissions = ToolPermissionContext(mode="default")
    context = _context(tmp_path, permissions=permissions)
    unsafe_decision = has_permissions_to_use_tool(
        SkillTool,
        {"skill": "danger"},
        permissions,
        tool_use_context=context,
    )
    assert unsafe_decision.behavior == "ask"


def _hooked_skill(*, hooks: dict[str, Any]) -> Skill:
    return Skill(
        name="hooked",
        description="hook test",
        base_dir="C:/tmp/hooked",
        hooks=hooks,
        get_prompt_for_command=lambda _args: "hooked prompt",
    )


def test_skill_hook_registration_is_atomic_and_deduplicated(tmp_path: Path) -> None:
    context = _context(tmp_path)
    request = SkillInvocationRequest("hooked", origin=SkillInvocationOrigin.USER)
    valid_group = {
        "matcher": "Read",
        "hooks": [{"type": "command", "command": "echo ok", "once": True}],
    }

    invalid = _hooked_skill(
        hooks={
            "PreToolUse": [valid_group],
            "NotAnEvent": [valid_group],
        }
    )
    with pytest.raises(ValueError, match="invalid skill hook event"):
        _register_skill_hooks(invalid, request, context)
    assert context.skill_hooks == {}
    assert context.skill_hook_keys == set()

    valid = _hooked_skill(hooks={"PreToolUse": [valid_group]})
    _register_skill_hooks(valid, request, context)
    _register_skill_hooks(valid, request, context)

    assert len(context.skill_hooks["PreToolUse"]) == 1
    assert len(context.skill_hook_keys) == 1
    registered = context.skill_hooks["PreToolUse"][0]
    assert registered.source is HookSource.SKILL
    assert registered.skill_root == "C:/tmp/hooked"
    assert registered.once is True


def test_record_failure_rolls_back_only_hooks_from_current_invocation(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    hook_group = {
        "matcher": "Read",
        "hooks": [{"type": "command", "command": "echo ok"}],
    }
    existing = Skill(
        name="existing",
        description="existing",
        hooks={"PreToolUse": [hook_group]},
        get_prompt_for_command=lambda _args: "existing",
    )
    _register_skill_hooks(
        existing,
        SkillInvocationRequest("existing", origin=SkillInvocationOrigin.USER),
        context,
    )
    existing_keys = set(context.skill_hook_keys)

    skill = _hooked_skill(hooks={"PreToolUse": [hook_group]})

    def fail_record(*_args: Any) -> None:
        raise RuntimeError("record failed")

    service = SkillInvocationService(
        resolver=lambda _name, _context: skill,
        recorder=fail_record,
        hook_registrar=_register_skill_hooks,
    )
    result = service.invoke(
        SkillInvocationRequest("hooked", origin=SkillInvocationOrigin.USER),
        context,
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code is SkillInvocationErrorCode.INVOCATION_RECORD_FAILED
    assert context.skill_hook_keys == existing_keys
    assert [getattr(config, "_skill_name", None) for config in context.skill_hooks["PreToolUse"]] == ["existing"]


@pytest.mark.asyncio
async def test_successful_once_skill_hook_removes_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from clawcodex_ext.hooks import hook_executor

    context = _context(tmp_path)
    context.workspace_trusted = True
    request = SkillInvocationRequest("hooked", origin=SkillInvocationOrigin.USER)
    skill = _hooked_skill(
        hooks={
            "PreToolUse": [
                {
                    "matcher": "Read",
                    "hooks": [{"type": "command", "command": "echo ok", "once": True}],
                }
            ]
        }
    )
    _register_skill_hooks(skill, request, context)

    async def succeed(*_args: Any, **_kwargs: Any) -> HookResult:
        return HookResult(exit_code=0, stdout="ok", command="echo ok")

    monkeypatch.setattr(hook_executor, "_execute_command_hook", succeed)
    results = [
        item
        async for item in hook_executor.execute_pre_tool_hooks(
            "Read",
            "tool-1",
            {"file_path": "README.md"},
            context,
        )
    ]

    assert any(getattr(item.get("message"), "attachments", None) for item in results if "message" in item)
    assert context.skill_hooks["PreToolUse"] == []
    assert context.skill_hook_keys == set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("hook_type", "field", "value", "executor_path"),
    [
        (
            "http",
            "url",
            "https://example.invalid/hook",
            "clawcodex_ext.hooks.exec_http_hook.execute_http_hook",
        ),
        (
            "prompt",
            "promptText",
            "extra instructions",
            "clawcodex_ext.hooks.exec_prompt_hook.execute_prompt_hook",
        ),
        (
            "agent",
            "agentInstructions",
            "evaluate event",
            "clawcodex_ext.hooks.exec_agent_hook.execute_agent_hook",
        ),
    ],
)
async def test_noncommand_skill_hooks_dispatch_and_once_is_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hook_type: str,
    field: str,
    value: str,
    executor_path: str,
) -> None:
    context = _context(tmp_path)
    context.workspace_trusted = True
    context._active_provider = object()
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def succeed(*args: Any, **kwargs: Any) -> HookResult:
        calls.append((args, kwargs))
        return HookResult(exit_code=0, stdout="ok")

    monkeypatch.setattr(executor_path, succeed)
    raw_hook = {"type": hook_type, field: value, "once": True}
    skill = _hooked_skill(hooks={"PreToolUse": [{"matcher": "Read", "hooks": [raw_hook]}]})
    _register_skill_hooks(
        skill,
        SkillInvocationRequest("hooked", origin=SkillInvocationOrigin.USER),
        context,
    )

    _ = [
        item
        async for item in __import__(
            "clawcodex_ext.hooks.hook_executor",
            fromlist=["execute_pre_tool_hooks"],
        ).execute_pre_tool_hooks(
            "Read",
            "tool-1",
            {"file_path": "README.md"},
            context,
        )
    ]

    assert len(calls) == 1
    assert context.skill_hooks["PreToolUse"] == []
    assert context.skill_hook_keys == set()


def test_fork_executor_reuses_agent_tool_and_recursion_uses_canonical_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from clawcodex_ext.skills import invocation

    registry = object()
    provider = object()
    context = _context(tmp_path)
    context.tool_registry = registry
    context._active_provider = provider
    captured: dict[str, Any] = {}

    class FakeAgentTool:
        def call(self, tool_input: dict[str, Any], fork_context: ToolContext) -> Any:
            captured["input"] = tool_input
            captured["context"] = fork_context
            return type(
                "Result",
                (),
                {
                    "is_error": False,
                    "output": {
                        "status": "completed",
                        "content": [{"type": "text", "text": "fork complete"}],
                    },
                },
            )()

    def make_agent_tool(actual_registry: Any, actual_provider: Any) -> FakeAgentTool:
        assert actual_registry is registry
        assert actual_provider is provider
        return FakeAgentTool()

    monkeypatch.setattr(
        "clawcodex_ext.tool_system.tools.agent.make_agent_tool",
        make_agent_tool,
    )
    monkeypatch.setattr(invocation, "_default_recorder", lambda *_args: None)

    skill = Skill(
        name="canonical-fork",
        aliases=["fork-alias"],
        description="fork test",
        context="fork",
        agent="general-purpose",
        model="skill-model",
        effort="high",
        allowed_tools=["Read"],
    )
    request = SkillInvocationRequest(
        "fork-alias",
        "payload",
        SkillInvocationOrigin.MODEL,
    )
    result = _execute_forked_skill(skill, request, context, "fork prompt")

    assert result.success is True
    assert result.status == "fork"
    assert result.fork_result == "fork complete"
    assert captured["input"] == {
        "prompt": "fork prompt",
        "description": "Run /canonical-fork",
        "subagent_type": "general-purpose",
        "_force_foreground": True,
        "_inherit_context": True,
        "_refresh_skill_listing": True,
        "model": "skill-model",
    }
    fork_context = captured["context"]
    assert fork_context is not context
    assert fork_context.active_skill_names == ("canonical-fork",)
    assert fork_context.skill_model_override == "skill-model"
    assert fork_context.skill_effort_override == "high"
    assert fork_context.permission_context.always_allow_rules["command"] == ["Read"]

    called = False

    def should_not_fork(*_args: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("recursive skill reached fork executor")

    context.active_skill_names = ("canonical-fork",)
    service = SkillInvocationService(
        resolver=lambda name, _ctx: skill if name == "fork-alias" else None,
        recorder=lambda *_args: None,
        fork_executor=should_not_fork,
    )
    recursive = service.invoke(request, context)
    assert recursive.success is False
    assert recursive.error is not None
    assert recursive.error.code is SkillInvocationErrorCode.RECURSIVE_INVOCATION
    assert called is False


class _CapturingProvider(BaseProvider):
    def __init__(self) -> None:
        super().__init__(api_key="test", model="base-provider-model")
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[Any],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        self.calls.append({"messages": messages, "tools": tools, **kwargs})
        return ChatResponse(
            content="done",
            model=str(kwargs.get("model", self.model)),
            usage={"input_tokens": 1, "output_tokens": 1},
            finish_reason="end_turn",
        )

    def chat_stream(self, *_args: Any, **_kwargs: Any) -> Generator[str, None, None]:
        if False:  # pragma: no cover - makes this a generator
            yield ""

    def get_available_models(self) -> list[str]:
        return ["base-provider-model", "skill-model"]

    def skill_effort_kwargs(self, effort: str | int) -> dict[str, Any]:
        return {"reasoning_effort": effort}


@pytest.mark.asyncio
async def test_query_applies_skill_model_effort_and_restores_base_context(
    tmp_path: Path,
) -> None:
    base_permissions = ToolPermissionContext(
        mode="default",
        always_allow_rules={"session": ["Read"]},
    )
    base_options = ToolUseOptions(
        main_loop_model="base-model",
        thinking_config={"budget_tokens": 256},
    )
    context = ToolContext(
        workspace_root=tmp_path,
        skill_resource_roots=("base-root",),
        permission_context=base_permissions,
        options=base_options,
    )
    modifier = build_request_context_modifier(
        allowed_tools=["Bash(git:*)"],
        model="skill-model",
        effort="high",
        resource_roots=["skill-root"],
    )
    apply_skill_context_modifier(context, modifier)
    assert context.skill_resource_roots == ("base-root", "skill-root")
    context.active_skill_names = ("review",)

    registry = build_default_registry()
    provider = _CapturingProvider()
    params = QueryParams(
        messages=[UserMessage(content="run the skill")],
        system_prompt="system",
        tools=registry.list_tools(),
        tool_registry=registry,
        tool_use_context=context,
        provider=provider,
        abort_controller=AbortController(),
        max_turns=1,
    )

    _messages, terminal = await run_query(params)

    assert terminal.reason == "completed"
    assert len(provider.calls) == 1
    assert provider.calls[0]["model"] == "skill-model"
    assert provider.calls[0]["reasoning_effort"] == "high"

    assert context.skill_resource_roots == ("base-root",)
    assert context.permission_context is base_permissions
    assert context.options is base_options
    assert context.permission_context.always_allow_rules == {"session": ["Read"]}
    assert context.options.main_loop_model == "base-model"
    assert context.options.thinking_config == {"budget_tokens": 256}
    assert context.skill_model_override is None
    assert context.skill_effort_override is None
    assert context.active_skill_names == ()
    assert context.skill_scope_pending is False
    assert context.skill_scope_active is False
