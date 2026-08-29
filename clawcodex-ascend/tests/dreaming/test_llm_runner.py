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

"""Tests for the LLM-backed dream runner — Phase B.

Covers the real LLM-backed dream runner's building blocks (tool
filtering, context creation, event handler, factory wiring) without
requiring a live LLM provider. The full end-to-end path is exercised
by ``test_e2e_dreaming.py`` with a recording factory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from clawcodex_ext.dreaming.runner import (
    DREAM_ALLOWED_TOOL_NAMES,
    DREAM_MAX_TURNS,
    DREAM_SYSTEM_PROMPT,
    _build_dream_context,
    _build_dream_tool_registry,
    _make_on_event_handler,
    create_real_dream_runner_factory,
    run_dream_with_llm,
    set_dream_runner_factory,
    wire_real_dream_runner,
)


@pytest.fixture(autouse=True)
def _clear_runner_factory() -> None:
    set_dream_runner_factory(None)
    yield  # type: ignore[misc]
    set_dream_runner_factory(None)


class TestConstants:
    def test_allowed_tools_contains_core_set(self) -> None:
        for name in ("Read", "Write", "Edit", "Bash", "Glob", "Grep"):
            assert name in DREAM_ALLOWED_TOOL_NAMES

    def test_allowed_tools_excludes_dangerous_tools(self) -> None:
        for name in ("Agent", "MCP", "WebSearch", "WebFetch", "AskUserQuestion"):
            assert name not in DREAM_ALLOWED_TOOL_NAMES

    def test_max_turns_is_reasonable(self) -> None:
        assert 5 <= DREAM_MAX_TURNS <= 30

    def test_system_prompt_is_nonempty(self) -> None:
        assert len(DREAM_SYSTEM_PROMPT) > 100
        assert "memory" in DREAM_SYSTEM_PROMPT.lower()


class TestBuildDreamToolRegistry:
    def test_only_allowed_tools_registered(self) -> None:
        mock_provider = MagicMock()
        registry = _build_dream_tool_registry(mock_provider)
        registered_names = {t.name for t in registry.list_tools()}
        assert registered_names.issubset(DREAM_ALLOWED_TOOL_NAMES)
        assert len(registered_names) > 0

    def test_all_allowed_tools_present(self) -> None:
        mock_provider = MagicMock()
        registry = _build_dream_tool_registry(mock_provider)
        registered_names = {t.name for t in registry.list_tools()}
        for name in DREAM_ALLOWED_TOOL_NAMES:
            assert name in registered_names, f"{name} missing from dream registry"


class TestBuildDreamContext:
    def test_context_has_bypass_permissions(self) -> None:
        from clawcodex_ext.utils.abort_controller import AbortController

        ws = Path("/tmp/test-workspace")
        ac = AbortController()
        ctx = _build_dream_context(ws, ac)
        assert ctx.permission_context.mode == "bypassPermissions"
        assert ctx.permission_context.is_bypass_permissions_mode_available is True

    def test_context_is_non_interactive(self) -> None:
        from clawcodex_ext.utils.abort_controller import AbortController

        ws = Path("/tmp/test-workspace")
        ac = AbortController()
        ctx = _build_dream_context(ws, ac)
        assert ctx.options.is_non_interactive_session is True
        assert ctx.ask_user is None

    def test_context_uses_workspace_root(self) -> None:
        from clawcodex_ext.utils.abort_controller import AbortController

        ws = Path("/tmp/my-project")
        ac = AbortController()
        ctx = _build_dream_context(ws, ac)
        assert ctx.workspace_root == ws.resolve()


class TestOnEventHandler:
    def test_none_on_message_returns_none_handler(self) -> None:
        handler = _make_on_event_handler(None, [])
        assert handler is None

    def test_write_tool_tracks_file(self) -> None:
        calls: list[dict] = []

        def on_msg(*, text: str, tool_use_count: int, touched_paths: list[str]) -> None:
            calls.append({"text": text, "tool_use_count": tool_use_count, "touched_paths": touched_paths})

        files: list[str] = []
        handler = _make_on_event_handler(on_msg, files)
        assert handler is not None

        event = MagicMock()
        event.kind = "tool_use"
        event.tool_name = "Write"
        event.tool_input = {"file_path": "/mem/topic.md"}
        handler(event)

        assert files == ["/mem/topic.md"]
        assert len(calls) == 1
        assert calls[0]["touched_paths"] == ["/mem/topic.md"]

    def test_edit_tool_tracks_file(self) -> None:
        calls: list[dict] = []

        def on_msg(*, text: str, tool_use_count: int, touched_paths: list[str]) -> None:
            calls.append({"touched_paths": touched_paths})

        files: list[str] = []
        handler = _make_on_event_handler(on_msg, files)

        event = MagicMock()
        event.kind = "tool_use"
        event.tool_name = "Edit"
        event.tool_input = {"file_path": "/mem/index.md"}
        handler(event)

        assert files == ["/mem/index.md"]

    def test_bash_tool_does_not_track(self) -> None:
        calls: list[dict] = []

        def on_msg(*, text: str, tool_use_count: int, touched_paths: list[str]) -> None:
            calls.append({"touched_paths": touched_paths})

        files: list[str] = []
        handler = _make_on_event_handler(on_msg, files)

        event = MagicMock()
        event.kind = "tool_use"
        event.tool_name = "Bash"
        event.tool_input = {"command": "ls /mem"}
        handler(event)

        assert files == []
        assert len(calls) == 1
        assert calls[0]["touched_paths"] == []

    def test_tool_result_events_ignored(self) -> None:
        calls: list[dict] = []

        def on_msg(*, text: str, tool_use_count: int, touched_paths: list[str]) -> None:
            calls.append({})

        files: list[str] = []
        handler = _make_on_event_handler(on_msg, files)

        event = MagicMock()
        event.kind = "tool_result"
        handler(event)

        assert files == []
        assert len(calls) == 0

    def test_deduplicates_files(self) -> None:
        calls: list[dict] = []

        def on_msg(*, text: str, tool_use_count: int, touched_paths: list[str]) -> None:
            calls.append({"touched_paths": touched_paths})

        files: list[str] = []
        handler = _make_on_event_handler(on_msg, files)

        for _ in range(3):
            event = MagicMock()
            event.kind = "tool_use"
            event.tool_name = "Write"
            event.tool_input = {"file_path": "/mem/topic.md"}
            handler(event)

        assert files == ["/mem/topic.md"]
        assert len(calls) == 3

    def test_handler_swallows_callback_exceptions(self) -> None:
        def on_msg(**_kw: Any) -> None:
            raise RuntimeError("callback error")

        files: list[str] = []
        handler = _make_on_event_handler(on_msg, files)

        event = MagicMock()
        event.kind = "tool_use"
        event.tool_name = "Write"
        event.tool_input = {"file_path": "/mem/x.md"}
        handler(event)

        assert files == ["/mem/x.md"]


class TestFactoryWiring:
    def test_create_factory_returns_callable(self) -> None:
        factory = create_real_dream_runner_factory()
        assert callable(factory)

    def test_factory_produces_runner(self) -> None:
        factory = create_real_dream_runner_factory()
        runner = factory()
        assert callable(runner)
        assert runner is run_dream_with_llm

    def test_wire_installs_factory(self) -> None:
        wire_real_dream_runner()
        from clawcodex_ext.dreaming.runner import _runner_factory

        assert _runner_factory is not None
        runner = _runner_factory()
        assert runner is run_dream_with_llm

    def test_wire_is_idempotent(self) -> None:
        wire_real_dream_runner()
        wire_real_dream_runner()
        from clawcodex_ext.dreaming.runner import _runner_factory

        assert _runner_factory is not None
        runner = _runner_factory()
        assert runner is run_dream_with_llm
