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

"""Tests for ``/monitor`` slash command."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from clawcodex_ext.command_system.engine import CommandEngine
from clawcodex_ext.command_system.monitor_command import (
    MONITOR_COMMAND,
    _handle_monitor_command,
)
from clawcodex_ext.command_system.types import CommandContext
from clawcodex_ext.feature_gate import get_registry as _get_registry
from clawcodex_ext.tool_system.context import ToolContext


@pytest.fixture
def command_context():
    workspace = Path(tempfile.gettempdir())
    ctx = CommandContext(
        workspace_root=workspace,
        cwd=workspace,
        tool_context=ToolContext(workspace_root=workspace),
    )
    return ctx


@pytest.fixture(autouse=True)
def enable_monitor_tool():
    reg = _get_registry()
    reg.set_override("MONITOR_TOOL", True)
    yield
    reg.clear_override("MONITOR_TOOL")


class TestMonitorCommand:
    def test_command_disabled_by_default(self):
        reg = _get_registry()
        reg.clear_override("MONITOR_TOOL")
        assert MONITOR_COMMAND.is_enabled() is False

    def test_command_enabled_with_override(self):
        assert MONITOR_COMMAND.is_enabled() is True

    def test_start_monitor(self, command_context):
        result = _handle_monitor_command(
            "bash -c 'echo hello'",
            command_context,
        )
        assert result.type == "text"
        assert "Monitor started" in result.value
        assert "bash -c" in result.value

    def test_list_active(self, command_context):
        _handle_monitor_command(
            "bash -c 'sleep 0.5'",
            command_context,
        )
        list_result = _handle_monitor_command("list", command_context)
        assert "Active monitor tasks" in list_result.value

    def test_stop_monitor(self, command_context):
        start_result = _handle_monitor_command(
            "bash -c 'sleep 2'",
            command_context,
        )
        task_id = start_result.value.split()[2]
        stop_result = _handle_monitor_command(f"stop {task_id}", command_context)
        assert "stopped" in stop_result.value

    def test_tail_monitor(self, command_context):
        start_result = _handle_monitor_command(
            "bash -c 'echo hello'",
            command_context,
        )
        task_id = start_result.value.split()[2]
        # Wait for the command to finish writing output.
        for _ in range(30):
            tail_result = _handle_monitor_command(f"tail {task_id}", command_context)
            if "hello" in tail_result.value:
                break
            asyncio.run(asyncio.sleep(0.05))
        assert "hello" in tail_result.value

    def test_missing_command_usage(self, command_context):
        result = _handle_monitor_command("", command_context)
        assert "Usage" in result.value

    def test_engine_integration(self, command_context):
        from clawcodex_ext.command_system.registry import CommandRegistry

        registry = CommandRegistry()
        registry.register(MONITOR_COMMAND)
        engine = CommandEngine(
            registry=registry,
            workspace_root=command_context.workspace_root,
            context=command_context,
        )
        result = asyncio.run(engine.execute("/monitor bash -c 'echo hi'"))
        assert result.success is True
        assert "Monitor started" in result.text
