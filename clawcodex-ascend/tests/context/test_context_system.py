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

"""Tests for the context system: build_context_prompt, collect_git_context,
and the agent-loop context-injection path.

The agent-loop integration test uses the async ``run_query_as_agent_loop``
adapter (the legacy sync ``run_agent_loop`` was removed during the query
consolidation).  We construct a thin sync shim in this module so the test
structure stays readable.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.agent.conversation import Conversation
from src.context_system import build_context_prompt
from src.context_system.claude_md import clear_memory_file_caches
from src.context_system.git_context import clear_git_caches, collect_git_context
from clawcodex_ext.providers.base import ChatResponse
from clawcodex_ext.query.agent_loop_compat import (
    build_effective_system_prompt,
    run_query_as_agent_loop,
)
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry


def _run(coro):
    return asyncio.run(coro)


def _run_agent_loop_sync(
    conversation: Conversation,
    provider,
    tool_registry,
    tool_context: ToolContext,
    verbose: bool = False,
):
    """Thin sync shim matching the legacy ``run_agent_loop`` signature.

    Builds the effective system prompt the same way the cutover code does
    (``build_effective_system_prompt`` + ``resolve_output_style``) and
    delegates to the async adapter.
    """
    from src.outputStyles import resolve_output_style

    style_prompt = resolve_output_style(
        getattr(tool_context, "output_style_name", None),
        getattr(tool_context, "output_style_dir", None),
    ).prompt
    system_prompt = build_effective_system_prompt(style_prompt, tool_context)

    async def _run_inner():
        result = await run_query_as_agent_loop(
            initial_messages=list(conversation.messages),
            provider=provider,
            tool_registry=tool_registry,
            tool_context=tool_context,
            system_prompt=system_prompt,
        )
        # Persist the assistant message back so the conversation
        # reflects the round-trip (matching legacy contract).
        if result.response_text:
            conversation.add_assistant_message(result.response_text)
        return result

    return asyncio.run(_run_inner())


class TestContextSystem(unittest.TestCase):
    def setUp(self):
        clear_memory_file_caches()
        clear_git_caches()

    def tearDown(self):
        clear_memory_file_caches()
        clear_git_caches()

    def test_build_context_prompt_includes_workspace_and_clawcodex_md(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CLAWCODEX.md").write_text("Project rule: always add tests.", encoding="utf-8")
            (root / "README.md").write_text("# Demo\n", encoding="utf-8")
            (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
            (root / "src").mkdir()
            (root / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "test_app.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")

            with patch.dict(os.environ, {"CLAUDE_CODE_ORIGINAL_CWD": tmp}):
                prompt = build_context_prompt(root)

            self.assertIn("## Runtime Context", prompt)
            self.assertIn("## Project Instructions", prompt)
            self.assertIn("Project rule: always add tests.", prompt)
            self.assertIn("README.md", prompt)
            self.assertIn("pyproject.toml", prompt)

    def test_collect_git_context_handles_non_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = _run(collect_git_context(tmp))
            self.assertFalse(ctx.available)

    def test_agent_loop_injects_context_prompt_for_non_anthropic(self) -> None:
        registry = build_default_registry(include_user_tools=False)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CLAWCODEX.md").write_text("Follow the ClawCodex instructions.", encoding="utf-8")
            (root / "README.md").write_text("# Demo\n", encoding="utf-8")

            ctx = ToolContext(workspace_root=root)

            conversation = Conversation()
            conversation.add_user_message("hello")

            provider = MagicMock()
            provider.chat.return_value = ChatResponse(
                content="ok",
                model="test",
                usage={"input_tokens": 1, "output_tokens": 1},
                finish_reason="stop",
                tool_uses=None,
            )

            out = _run_agent_loop_sync(conversation, provider, registry, ctx, verbose=False)
            self.assertEqual(out.response_text, "ok")
            system_message = provider.chat.call_args.args[0][0]
            self.assertEqual(system_message["role"], "system")
            self.assertIn("## Runtime Context", system_message["content"])
            self.assertIn("## Project Instructions", system_message["content"])
            self.assertIn("Follow the ClawCodex instructions.", system_message["content"])


if __name__ == "__main__":
    unittest.main()
