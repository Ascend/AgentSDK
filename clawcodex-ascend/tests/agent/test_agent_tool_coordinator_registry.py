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

"""Regression tests for the Agent tool's coordinator-mode registry swap.

Covers two bugs found during the 2026-07-02 GitCode e2e retry of the
multi-agent MVP:

1. ``_agent_call`` assigned ``registry = sub_registry`` in its coordinator
   branch. Because a function-body assignment makes the name local for the
   WHOLE function, the earlier ``registry.list_tools()`` read raised
   ``UnboundLocalError: cannot access local variable 'registry'`` on EVERY
   Agent tool call — coordinator mode or not — so workers could never be
   spawned. Fixed by rebinding a separate ``effective_registry`` name.

2. The coordinator branch must hand the sub-agent a FRESH full-tool
   registry (workers need Edit/Write/Bash) while leaving the parent's
   restricted registry untouched.
"""

from __future__ import annotations


# pylint: disable=C0412,E0611
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from clawcodex_ext.tool_system.protocol import ToolCall
from clawcodex_ext.types.content_blocks import TextBlock
from src.types.messages import AssistantMessage


class TestAgentToolCoordinatorRegistry(unittest.TestCase):
    def test_agent_call_reads_parent_registry_without_unbound_local(self) -> None:
        """Dispatching Agent with provider=None must fail gracefully.

        The "No provider configured" ToolResult is produced AFTER the
        ``registry.list_tools()`` read at the top of ``_agent_call`` — with
        the UnboundLocalError bug present, dispatch never reached it and the
        tool result carried "cannot access local variable 'registry'".
        """
        registry = build_default_registry(provider=None)
        with TemporaryDirectory() as tmp:
            context = ToolContext(workspace_root=Path(tmp))
            result = registry.dispatch(
                ToolCall(
                    name="Agent",
                    input={"description": "spawn worker", "prompt": "do work"},
                ),
                context,
            )

        self.assertTrue(result.is_error)
        text = str(result.output)
        self.assertNotIn(
            "local variable",
            text,
            msg=(f"UnboundLocalError regression: 'registry' was rebound inside _agent_call again — {text!r}"),
        )
        self.assertIn("No provider configured", text)

    def test_coordinator_mode_hands_worker_a_fresh_full_registry(self) -> None:
        """Under CLAUDE_CODE_COORDINATOR_MODE the spawned worker must get a
        registry distinct from the parent's (fresh, unfiltered pool) while
        the parent registry object stays untouched.
        """
        parent_registry = build_default_registry(provider=object())
        parent_tools_before = {t.name for t in parent_registry.list_tools()}
        captured: dict[str, object] = {}

        async def _fake_run_agent(params):
            captured["tool_registry"] = params.tool_registry
            yield AssistantMessage(content=[TextBlock(text="worker done")])

        with TemporaryDirectory() as tmp:
            context = ToolContext(workspace_root=Path(tmp))
            with patch.dict(os.environ, {"CLAUDE_CODE_COORDINATOR_MODE": "1"}):
                with patch("src.tool_system.tools.agent.run_agent", _fake_run_agent):
                    result = parent_registry.dispatch(
                        ToolCall(
                            name="Agent",
                            input={"description": "worker", "prompt": "edit files"},
                        ),
                        context,
                    )

        self.assertFalse(result.is_error, str(result.output))
        worker_registry = captured.get("tool_registry")
        self.assertIsNotNone(worker_registry)
        self.assertIsNot(
            worker_registry,
            parent_registry,
            msg="worker must get a FRESH registry, not the parent's restricted one",
        )
        parent_tools_after = {t.name for t in parent_registry.list_tools()}
        self.assertEqual(parent_tools_before, parent_tools_after)


if __name__ == "__main__":
    unittest.main()
