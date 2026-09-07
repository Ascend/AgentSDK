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

import asyncio
from pathlib import Path

from clawcodex_ext.command_system.registry import CommandRegistry
from clawcodex_ext.command_system.types import CommandContext
from clawcodex_ext.multimodel.cli import run_multimodel_command
from clawcodex_ext.multimodel.config import load_config, resolve_active_group
from clawcodex_ext.multimodel.factory import build_router
from clawcodex_ext.multimodel.runtime_command import register_multimodel_runtime_command


def test_group_lifecycle_and_persistence(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CLAWCODEX_CONFIG_DIR", str(tmp_path))
    assert (
        run_multimodel_command(
            [
                "group",
                "create",
                "review",
                "--slot",
                "sonnet:claude-sonnet-4-6@anthropic",
                "--slot",
                "gpt4o:gpt-4o@openai,weight=2",
                "--strategy",
                "voting",
                "--aggregator",
                "majority",
                "--min-votes",
                "2",
            ]
        )
        == 0
    )
    assert run_multimodel_command(["use", "review"]) == 0
    config = load_config()
    assert config.default_group == "review"
    assert config.groups["review"].slots[1].weight == 2
    assert run_multimodel_command(["group", "update", "review", "--remove-slot", "gpt4o"]) == 0
    assert [slot.name for slot in load_config().groups["review"].slots] == ["sonnet"]


def test_runtime_selection_overrides_config(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CLAWCODEX_CONFIG_DIR", str(tmp_path))
    assert run_multimodel_command(["preset", "quick-compare"]) == 0
    registry = CommandRegistry()
    register_multimodel_runtime_command(registry)
    command = registry.get("multimodel")
    assert command is not None
    context = CommandContext(workspace_root=Path.cwd(), cwd=Path.cwd())

    async def exercise() -> None:
        result = await command.call("use quick-compare", context)
        assert "Switched to model group" in result.value
        status = (await command.call("status", context)).value
        assert "Status: enabled" in status
        assert "Aggregator: passthrough" in status
        assert "single-model mode" in (await command.call("off", context)).value

    asyncio.run(exercise())
    assert resolve_active_group(cli_group="cli", runtime_group="runtime", config=load_config()) == "cli"


def test_configured_keyword_routes_build_a_working_strategy(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CLAWCODEX_CONFIG_DIR", str(tmp_path))
    assert (
        run_multimodel_command(
            [
                "group",
                "create",
                "router",
                "--slot",
                "cheap:m1@one",
                "--slot",
                "strong:m2@two",
                "--strategy",
                "routing",
                "--route",
                "security:strong",
            ]
        )
        == 0
    )

    class Provider:
        def __init__(self, name):
            self.name, self.model = name, name

        async def chat_async(self, _messages, **_kwargs):
            from clawcodex_ext.providers.base import ChatResponse

            return ChatResponse(self.name, self.name, {}, "stop")

        def get_available_models(self):
            return [self.model]

    router = build_router(load_config().groups["router"], lambda provider, _model: Provider(provider))
    assert router.chat([{"role": "user", "content": "Please perform a security review"}]).content == "two"


class _FakeProvider:
    def __init__(self, model: str) -> None:
        self.model = model


class _FakeRegistry:
    def __init__(self, provider: _FakeProvider) -> None:
        self.provider = provider
        self._tools = []

    def register(self, tool: object) -> None:
        self._tools.append(tool)

    def unregister(self, name: str) -> None:
        self._tools = [t for t in self._tools if getattr(t, "name", "") != name]

    def list_tools(self) -> list:
        return list(self._tools)


def test_runtime_build_resolves_config_default_group(monkeypatch, tmp_path) -> None:
    """RuntimeContext.build activates the config default group on its own."""
    from types import SimpleNamespace

    from clawcodex_ext.runtime.context import RuntimeContext, RuntimeOptions

    monkeypatch.setenv("CLAWCODEX_CONFIG_DIR", str(tmp_path))
    assert (
        run_multimodel_command(
            [
                "group",
                "create",
                "review",
                "--slot",
                "sonnet:claude-sonnet-4-6@anthropic",
                "--strategy",
                "voting",
                "--aggregator",
                "majority",
                "--min-votes",
                "2",
            ]
        )
        == 0
    )
    assert run_multimodel_command(["use", "review"]) == 0

    router_groups: list = []
    single_calls: list = []
    monkeypatch.setattr(
        "clawcodex_ext.runtime.context.resolve",
        lambda **kwargs: SimpleNamespace(provider="glm", model="zai/glm-4"),
    )
    monkeypatch.setattr(
        "clawcodex_ext.runtime.context.attach_cron_runtime",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "clawcodex_ext.runtime.context.replace_cron_tools",
        lambda registry: None,
    )
    monkeypatch.setattr(
        "src.providers.runtime.build_provider_from_config",
        lambda provider_name, model=None: (single_calls.append(provider_name) or _FakeProvider(model)),
    )
    monkeypatch.setattr(
        "src.tool_system.defaults.build_default_registry",
        lambda provider, **kwargs: _FakeRegistry(provider),
    )
    monkeypatch.setattr(
        "clawcodex_ext.multimodel.factory.build_router",
        lambda group, builder, **kwargs: router_groups.append(group) or _FakeProvider("ensemble"),
    )

    options = RuntimeOptions(workspace_root=tmp_path)
    runtime = RuntimeContext.build(options)

    assert runtime.provider_name == "multimodel"
    assert runtime.provider.model == "ensemble"
    assert runtime.multimodel_group == "review"
    assert options.multimodel_group == "review"
    assert len(router_groups) == 1
    assert single_calls == []
