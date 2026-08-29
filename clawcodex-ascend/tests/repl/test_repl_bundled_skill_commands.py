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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from clawcodex_ext.command_system.registry import get_command_registry
from clawcodex_ext.providers.base import BaseProvider, ChatResponse
from clawcodex_ext.repl.app import ClawCodexExtREPL
from clawcodex_ext.tool_system.context import ToolContext
from clawcodex_ext.tool_system.defaults import build_default_registry


class _Provider(BaseProvider):
    def __init__(self) -> None:
        super().__init__(api_key="test", model="repl-skill-test")

    def chat(self, *_args, **_kwargs) -> ChatResponse:
        raise AssertionError("command registration must not call the provider")

    def chat_stream(self, *_args, **_kwargs):
        if False:
            yield ""

    def get_available_models(self) -> list[str]:
        return [self.model]


def test_ext_repl_lazy_command_system_registers_bundled_skills(
    tmp_path: Path,
) -> None:
    provider = _Provider()
    tool_registry = build_default_registry(provider=provider)
    tool_context = ToolContext(
        workspace_root=tmp_path,
        cwd=tmp_path,
        tool_registry=tool_registry,
    )
    tool_context._active_provider = provider

    repl = ClawCodexExtREPL.__new__(ClawCodexExtREPL)
    repl.command_registry = None
    repl.workspace_root = tmp_path
    repl.session = SimpleNamespace(conversation=SimpleNamespace(messages=[]))
    repl.provider = provider
    repl.cost_tracker = Mock()
    repl.history_log = Mock()
    repl.tool_registry = tool_registry
    repl.tool_context = tool_context
    repl.runtime_context = None
    repl.console = Mock()
    repl._safe_input = Mock()
    repl._arrow_select = Mock()
    repl._built_in_commands = []
    repl._original_built_ins = []
    repl._update_built_in_commands_with_command_system = Mock()

    global_registry = get_command_registry()
    saved_commands = dict(global_registry._commands)
    global_registry.clear()
    try:
        repl._ensure_command_system()

        for name in ("remember", "update-config", "verify"):
            assert repl.command_registry.get(name) is not None
            assert global_registry.get(name) is not None
        repl._update_built_in_commands_with_command_system.assert_called_once_with()
    finally:
        global_registry.clear()
        global_registry._commands.update(saved_commands)
