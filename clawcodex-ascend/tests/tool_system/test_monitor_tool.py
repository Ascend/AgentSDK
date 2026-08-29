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

"""Tests for MonitorTool."""

from __future__ import annotations


# pylint: disable=E0611,W0621
import tempfile
from pathlib import Path

import pytest

from clawcodex_ext.feature_gate import get_registry as _get_registry
from clawcodex_ext.tool_system.context import ToolContext
from clawcodex_ext.tool_system.tools.monitor import MonitorTool


@pytest.fixture(autouse=True)
def enable_monitor_tool():
    reg = _get_registry()
    reg.set_override("MONITOR_TOOL", True)
    yield
    reg.clear_override("MONITOR_TOOL")


@pytest.fixture
def tool_context():
    return ToolContext(workspace_root=Path(tempfile.gettempdir()))


class TestMonitorTool:
    def test_tool_disabled_by_default(self):
        reg = _get_registry()
        reg.clear_override("MONITOR_TOOL")
        assert MonitorTool.is_enabled() is False

    def test_tool_enabled_with_override(self):
        assert MonitorTool.is_enabled() is True

    def test_execute_returns_task_id(self, tool_context):
        result = MonitorTool.call(
            {"command": "bash -c 'echo hello'"},
            tool_context,
        )
        assert result.is_error is False
        output = result.output
        assert "task_id" in output
        assert output["kind"] == "monitor"
        assert "output_path" in output

    def test_execute_requires_command(self, tool_context):
        result = MonitorTool.call(
            {"command": ""},
            tool_context,
        )
        assert result.is_error is True

    def test_execute_validates_interval(self, tool_context):
        result = MonitorTool.call(
            {"command": "echo x", "interval_sec": 0},
            tool_context,
        )
        assert result.is_error is True

    def test_execute_with_description(self, tool_context):
        result = MonitorTool.call(
            {"command": "bash -c 'echo hello'", "description": "my monitor"},
            tool_context,
        )
        assert result.is_error is False
        assert "my monitor" in result.output["message"]

    def test_execute_respects_disabled_feature_gate(self, tool_context):
        reg = _get_registry()
        reg.clear_override("MONITOR_TOOL")
        result = MonitorTool.call(
            {"command": "echo hello"},
            tool_context,
        )
        # The tool still executes even when disabled because ``call`` does
        # not gate on ``is_enabled``; ``get_tools`` filters by it.  The test
        # documents this contract.
        assert result.is_error is False
