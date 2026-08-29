#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

import threading
from typing import Any, Callable

from .registry import ToolRegistry
from .tools import ALL_STATIC_TOOLS, make_agent_tool, make_tool_search_tool


def build_default_registry(
    *,
    include_user_tools: bool = True,
    provider: "Any | None" = None,
    get_available_mcp_servers: Callable[[], list[str]] | None = None,
    load_agent_tools: bool = True,
    defer_extended_tools: bool = False,
) -> ToolRegistry:
    """Build the default tool registry.

    The registry is built in two stages:

    * **Stage A (synchronous)** — the 51 core tools in ``ALL_STATIC_TOOLS``,
      plus the ``Agent`` and tool-search tools. These are needed for any
      conversation to function and are always registered before this
      function returns.
    * **Stage B (deferred)** — extension tools (Playwright / Pillow /
      MCP-SDK-dependent chrome tools, lodestone, bg-session, task
      directives, etc.), persisted agent-created tools, and the workflow
      tool. These cost ~3s of import + instantiation on cold start.

    When ``defer_extended_tools=True``, Stage B runs on a daemon thread
    and returns immediately. By the time the user submits their first
    prompt and the LLM round-trip begins, Stage B has typically finished.
    Concurrent reads (e.g. ``/tools`` command, ``tool_to_api_schema``)
    see whichever tools have been registered so far; the LLM's tool list
    is rebuilt on every turn (``get_team_aware_tool_list``), so the next
    turn after Stage B completes surfaces the additional tools.

    Default ``defer_extended_tools=False`` preserves the blocking
    behaviour for unit tests and non-interactive callers (orchestrator,
    remote API runner, CLI ``/tools`` command).
    """
    registry = ToolRegistry()
    for tool in ALL_STATIC_TOOLS:
        registry.register(tool)

    registry.register(
        make_agent_tool(
            registry,
            provider=provider,
            get_available_mcp_servers=get_available_mcp_servers,
        )
    )
    registry.register(make_tool_search_tool(registry))

    if defer_extended_tools:
        _schedule_extended_tool_registration(
            registry,
            provider=provider,
            load_agent_tools=load_agent_tools,
        )
    else:
        _register_extended_tools(
            registry,
            provider=provider,
            load_agent_tools=load_agent_tools,
        )

    return registry


def _register_extended_tools(
    registry: ToolRegistry,
    *,
    provider: Any,
    load_agent_tools: bool,
) -> None:
    """Stage B body: register extension tools, persisted agent tools, and the workflow tool.

    Imports are deferred to this function so they only fire when
    Stage B actually runs (background thread or blocking fallback).
    """
    # Register extension tools that are not provided upstream.
    try:
        from extensions.tool_system_ext.registration import EXTENSION_TOOLS

        for t in EXTENSION_TOOLS:
            registry.register(t)
    except ImportError:
        pass  # Optional integration is unavailable; keep the fallback.

    # Load persisted agent-created tools on startup.
    if load_agent_tools:
        try:
            from clawcodex_ext.tool_system.tools.create_agent_tool import (
                load_persisted_agent_tools,
            )
            from clawcodex_ext.agent.tool_authoring.registry_ext import list_tools

            load_persisted_agent_tools()
            for tool in list_tools():
                registry.register(tool)
        except ImportError:
            pass  # Optional integration is unavailable; keep the fallback.

    # Dynamic workflows. Registered unconditionally (like the Agent tool, which
    # also needs the registry + provider); the tool's ``is_enabled`` is the
    # single runtime gate (``get_tools`` filters by it fresh), so a ``/config``
    # toggle of ``disable_workflows`` takes effect without rebuilding the registry.
    from src.tool_system.tools.workflow import make_workflow_tool

    registry.register(make_workflow_tool(registry, provider=provider))


def _schedule_extended_tool_registration(
    registry: ToolRegistry,
    *,
    provider: Any,
    load_agent_tools: bool,
) -> None:
    """Schedule Stage B tool registration on a daemon thread.

    A failure here must not break REPL startup, so the worker swallows
    exceptions and prints a warning to stderr.
    """

    def _bg() -> None:
        try:
            _register_extended_tools(
                registry,
                provider=provider,
                load_agent_tools=load_agent_tools,
            )
        except Exception as exc:  # noqa: BLE001 — defensive, never break startup
            import sys

            print(
                f"[clawcodex] extended tool registration failed: {exc}",
                file=sys.stderr,
            )

    t = threading.Thread(
        target=_bg,
        daemon=True,
        name="tool-registry-extended",
    )
    t.start()
