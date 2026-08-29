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

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from clawcodex_ext.configuration import invalidate_configuration
from clawcodex_ext.context_system.prompt_assembly import build_full_system_prompt
from clawcodex_ext.providers.base import ChatResponse
from clawcodex_ext.query.engine import QueryEngine, QueryEngineConfig
from clawcodex_ext.skills.bundled import init_bundled_skills
from clawcodex_ext.skills.bundled import update_config as update_config_module
from clawcodex_ext.skills.bundled_skills import (
    clear_bundled_skills,
    get_bundled_skill_by_name,
    get_bundled_skills,
)
from clawcodex_ext.tool_system.context import ToolContext
from clawcodex_ext.tool_system.defaults import build_default_registry
from clawcodex_ext.tool_system.tools.skill import SkillTool, run_user_invoked_skill


@pytest.fixture(autouse=True)
def reset_skill_and_config_state():
    clear_bundled_skills()
    invalidate_configuration("test setup")
    yield
    clear_bundled_skills()
    invalidate_configuration("test teardown")


@pytest.fixture
def configured_context(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    user_app = tmp_path / "home" / ".clawcodex" / "config.json"
    config_dir = tmp_path / "claude-home"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    monkeypatch.setattr("src.config.get_config_path", lambda: user_app)
    monkeypatch.setattr("src.config._find_git_root", lambda _cwd=None: workspace)
    invalidate_configuration("patched paths")
    return ToolContext(
        workspace_root=workspace,
        cwd=workspace,
        workspace_trusted=True,
    )


def test_definition_is_registered_idempotently_and_model_visible(configured_context):
    assert init_bundled_skills() is True
    first = get_bundled_skill_by_name("update-config")
    assert first is not None
    assert first.user_invocable is True
    assert first.disable_model_invocation is False
    assert first.allowed_tools == ["Config", "Read", "AskUserQuestion"]
    assert "from now on" in first.description
    assert "when X happens" in first.description

    count = len(get_bundled_skills())
    assert init_bundled_skills() is True
    second = get_bundled_skill_by_name("update-config")
    assert second is first
    assert len(get_bundled_skills()) == count

    system_prompt = build_full_system_prompt(skills=[first], use_cache=False)
    assert "update-config" in system_prompt
    assert first.description in system_prompt


def test_dynamic_prompt_contains_paths_schema_permissions_and_hooks(configured_context):
    init_bundled_skills()
    skill = get_bundled_skill_by_name("update-config")
    assert skill is not None

    result = run_user_invoked_skill(
        "update-config",
        "always allow Read before running tests",
        configured_context,
    )

    assert result.is_error is False
    prompt = result.output["prompt"]
    assert "always allow Read before running tests" in prompt
    assert str(configured_context.workspace_root / ".claude" / "settings.json") in prompt
    assert "permissions.allow" in prompt
    assert "append_unique" in prompt
    assert "PreToolUse" in prompt
    assert "hookRuntime" in prompt
    assert "Never use Bash" in prompt
    assert "clawcodex mcp" in prompt


def test_dynamic_prompt_embeds_complete_clawcodex_settings_schema(configured_context):
    init_bundled_skills()
    result = run_user_invoked_skill(
        "update-config",
        "configure project permissions and a post-write hook",
        configured_context,
    )

    prompt = result.output["prompt"]
    marker = "## Full Settings JSON Schema\n\n```json\n"
    encoded = prompt.split(marker, 1)[1].split("\n```", 1)[0]
    schema = json.loads(encoded)

    app_marker = "## Public App JSON Schema\n\nThis stable schema is generated from the central ClawCodex configuration\ncontract. It contains public fields, defaults, constraints, scopes, and routing\nmetadata, but never live values or provider names:\n\n```json\n"
    app_encoded = prompt.split(app_marker, 1)[1].split("\n```", 1)[0]
    app_schema = json.loads(app_encoded)

    assert app_schema["title"] == "ClawCodex app config"
    assert {"theme", "logoColor", "editorMode", "copyFullResponse", "selection_mode"} <= set(app_schema["properties"])
    assert app_schema["properties"]["providers"]["type"] == "object"
    assert "properties" not in app_schema["properties"]["providers"]
    assert app_schema["properties"]["providers"]["additionalProperties"]["properties"]["api_key"]["type"] == "string"
    assert "https://api.openai.com" not in app_encoded
    assert "anthropic" not in app_encoded
    assert app_schema["properties"]["telemetry"]["x-managed-by"].startswith("/telemetry")
    assert schema["title"] == "ClawCodex settings.json"
    assert {"app"}.isdisjoint(schema["properties"])
    assert {"allow", "deny", "ask"} <= set(schema["properties"]["permissions"]["properties"])
    assert set(schema["properties"]["hooks"]["properties"]) == set(update_config_module.ALL_HOOK_EVENTS)
    hook_types = {variant["properties"]["type"]["enum"][0] for variant in schema["$defs"]["hook"]["oneOf"]}
    assert hook_types == {"command", "http", "prompt", "agent"}
    assert schema["properties"]["hookRuntime"]["properties"]["timeout_ms"]["type"] == "integer"
    assert schema["properties"]["voice_provider"]["enum"] == ["", "anthropic", "doubao"]
    assert "away_summary" in schema["properties"]
    assert "intent_forecast" in schema["properties"]


def test_prompt_routes_dedicated_configuration_subsystems(configured_context):
    init_bundled_skills()
    result = run_user_invoked_skill(
        "update-config",
        "configure telemetry, MCP, plugins, and daemon",
        configured_context,
    )

    prompt = result.output["prompt"]
    assert "`telemetry`: /telemetry or telemetry.toml" in prompt
    assert "`mcp_servers`: clawcodex mcp" in prompt
    assert "`plugins`: plugin loader/installer" in prompt
    assert "`daemon`: clawcodex daemon" in prompt
    assert "x-managed-by" in prompt


def test_hooks_only_mode_omits_app_paths_and_keeps_hook_contract(configured_context):
    init_bundled_skills()

    result = run_user_invoked_skill(
        "update-config",
        "[hooks-only] run lint after every successful tool",
        configured_context,
    )

    prompt = result.output["prompt"]
    assert "This is hooks-only mode" in prompt
    assert "| user | app |" not in prompt
    assert "| project | app |" not in prompt
    assert "| user | settings |" in prompt
    assert "PostToolUse" in prompt
    assert "[hooks-only]" not in prompt
    assert "Full Settings JSON Schema" not in prompt
    assert "Hook JSON Schema" in prompt
    assert "direct array for each event" in prompt
    assert "nested matcher-group" in prompt
    assert "timeout` field is milliseconds" in prompt
    assert "`http`: requires `url`" in prompt
    assert "`additionalContexts`" in prompt
    assert "append_unique" in prompt


def test_init_loads_hooks_only_reference_before_creating_hooks():
    from clawcodex_ext.command_system.builtins import NEW_INIT_PROMPT

    assert "## Step 7: Create hooks" in NEW_INIT_PROMPT
    assert 'skill: "update-config"' in NEW_INIT_PROMPT
    assert "[hooks-only]" in NEW_INIT_PROMPT
    assert "Persist hooks only through the Config tool" in NEW_INIT_PROMPT
    assert "## Step 8: Summary" in NEW_INIT_PROMPT


@pytest.mark.asyncio
async def test_real_slash_command_uses_canonical_user_service(configured_context):
    from clawcodex_ext.command_system.engine import CommandEngine
    from clawcodex_ext.command_system.registry import CommandRegistry
    from clawcodex_ext.command_system.skills_integration import load_and_register_skills
    from clawcodex_ext.command_system.types import CommandContext

    registry = CommandRegistry()
    load_and_register_skills(
        project_root=configured_context.workspace_root,
        registry=registry,
    )
    command_context = CommandContext(
        workspace_root=configured_context.workspace_root,
        cwd=configured_context.cwd,
        tool_context=configured_context,
    )

    result = await CommandEngine(
        registry=registry,
        workspace_root=configured_context.workspace_root,
        context=command_context,
    ).execute("/update-config always use the project model")

    assert result.success is True
    assert result.prompt_is_meta is True
    assert "always use the project model" in repr(result.prompt_content)
    assert "# Update ClawCodex Configuration" in repr(result.prompt_content)


def test_model_invocation_injects_meta_message_not_prompt_tool_output(configured_context):
    init_bundled_skills()
    registry = build_default_registry()
    configured_context.tool_registry = registry
    configured_context.options.tools = list(registry.list_tools())

    result = SkillTool.call(
        {
            "skill": "update-config",
            "args": "use model gpt-test from now on",
        },
        configured_context,
    )

    assert result.is_error is False
    assert result.output["commandName"] == "update-config"
    assert "prompt" not in result.output
    assert result.new_messages is not None
    content = result.new_messages[0].content
    assert "<command-name>/update-config</command-name>" in content
    assert "use model gpt-test from now on" in content
    assert "# Update ClawCodex Configuration" in content
    assert result.context_modifier is not None
    modified = result.context_modifier(configured_context)
    visible_tools = {tool.name for tool in modified.options.tools}
    assert {"Config", "Read", "AskUserQuestion"} <= visible_tools
    command_rules = set(modified.permission_context.always_allow_rules["command"])
    assert {"Config", "Read", "AskUserQuestion"} <= command_rules
    assert {"Bash", "Edit", "Write"}.isdisjoint(command_rules)


@pytest.mark.asyncio
async def test_fake_provider_selects_skill_for_natural_language_request(configured_context):
    registry = build_default_registry()
    configured_context.tool_registry = registry
    provider = MagicMock()
    provider.model = "test-model"
    provider.chat_stream_response.side_effect = NotImplementedError()
    provider.chat.side_effect = [
        ChatResponse(
            content="I will configure that.",
            model="test-model",
            usage={"input_tokens": 10, "output_tokens": 5},
            finish_reason="tool_use",
            tool_uses=[
                {
                    "id": "skill-call-1",
                    "name": "Skill",
                    "input": {
                        "skill": "update-config",
                        "args": "from now on always allow Read before tests",
                    },
                }
            ],
        ),
        ChatResponse(
            content="Configuration workflow loaded.",
            model="test-model",
            usage={"input_tokens": 20, "output_tokens": 5},
            finish_reason="end_turn",
            tool_uses=None,
        ),
        ChatResponse(
            content="Configuration workflow verified.",
            model="test-model",
            usage={"input_tokens": 5, "output_tokens": 3},
            finish_reason="end_turn",
            tool_uses=None,
        ),
    ]
    engine = QueryEngine(
        QueryEngineConfig(
            cwd=configured_context.workspace_root,
            provider=provider,
            tool_registry=registry,
            tools=registry.list_tools(),
            tool_context=configured_context,
            system_prompt="Use matching skills.",
            max_turns=3,
        )
    )

    async for _message in engine.submit_message(
        "Please configure yourself so Read is allowed before every test from now on."
    ):
        pass

    assert provider.chat.call_count == 3
    second_call = provider.chat.call_args_list[1]
    messages = second_call.kwargs.get("messages") or second_call.args[0]
    rendered = repr(messages)
    assert "<command-name>/update-config</command-name>" in rendered
    assert "# Update ClawCodex Configuration" in rendered


def test_prompt_failure_has_no_configuration_side_effect(
    configured_context,
    monkeypatch,
):
    init_bundled_skills()
    user_path = configured_context.workspace_root.parent / "home" / ".clawcodex" / "config.json"
    user_path.parent.mkdir(parents=True)
    user_path.write_text(json.dumps({"sentinel": True}), encoding="utf-8")
    before = user_path.read_bytes()

    def fail_snapshot(_context):
        raise RuntimeError("schema unavailable")

    monkeypatch.setattr(update_config_module, "_snapshot_for_context", fail_snapshot)
    result = run_user_invoked_skill("update-config", "change model", configured_context)

    assert result.is_error is True
    assert user_path.read_bytes() == before
