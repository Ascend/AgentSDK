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

"""End-to-end tests for delegating to a child with a different provider and model.

The suite verifies the complete path:
1. Parse provider and model from Agent tool input.
2. Provider precedence: tool input, agent definition, then parent inheritance.
3. Model precedence: tool input, non-inherit agent definition, then inheritance.
4. Build a new provider when one is explicitly selected.
5. Fall back to the parent provider when provider construction fails.
6. Pass the resolved provider and model through ``RunAgentParams``.
"""

from __future__ import annotations


# pylint: disable=E0611
import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from clawcodex_ext.providers.base import BaseProvider
from clawcodex_ext.tool_system.protocol import ToolCall
from clawcodex_ext.types.content_blocks import TextBlock
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.types.messages import AssistantMessage

logger = logging.getLogger(__name__)


def _make_fake_run_agent(captured: dict):
    """Create a fake run_agent async generator that captures RunAgentParams."""

    async def _fake(params):
        captured["provider"] = params.provider
        captured["model"] = params.model
        captured["agent_definition"] = params.agent_definition
        captured["query_source"] = params.query_source
        yield AssistantMessage(content=[TextBlock(text="worker done")])

    return _fake


class TestProviderAndModelOverrideE2E(unittest.TestCase):
    """Exercise parent-to-child provider and model overrides end to end."""

    def setUp(self):
        self.parent_provider = MagicMock(spec=BaseProvider)
        # Make the mock behave like a provider with a default model
        self.parent_provider.model = "claude-sonnet-4-6"
        self.parent_registry = build_default_registry(provider=self.parent_provider)
        self.captured: dict = {}

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def test_provider_and_model_from_tool_input(self):
        """Build a new provider and pass through both explicit overrides."""
        mock_minimax = MagicMock(spec=BaseProvider)
        mock_minimax.model = "minimax-text-01"

        with TemporaryDirectory() as tmp:
            context = ToolContext(workspace_root=Path(tmp))
            with (
                patch(
                    "clawcodex_ext.providers.runtime.build_provider_from_config",
                    return_value=mock_minimax,
                ) as mock_build,
                patch(
                    "src.tool_system.tools.agent.run_agent",
                    _make_fake_run_agent(self.captured),
                ),
            ):
                result = self.parent_registry.dispatch(
                    ToolCall(
                        name="Agent",
                        input={
                            "description": "use minimax",
                            "prompt": "translate this code to Rust",
                            "provider": "minimax",
                            "model": "minimax-text-01",
                        },
                    ),
                    context,
                )

        self.assertFalse(result.is_error, msg=str(result.output))

        mock_build.assert_called_once_with("minimax", model="minimax-text-01")

        self.assertIs(self.captured["provider"], mock_minimax)
        self.assertIsNot(self.captured["provider"], self.parent_provider)

        self.assertEqual(self.captured["model"], "minimax-text-01")

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def test_model_override_from_tool_input(self):
        """Retain the parent provider while overriding only the model."""
        with TemporaryDirectory() as tmp:
            context = ToolContext(workspace_root=Path(tmp))
            with patch(
                "src.tool_system.tools.agent.run_agent",
                _make_fake_run_agent(self.captured),
            ):
                result = self.parent_registry.dispatch(
                    ToolCall(
                        name="Agent",
                        input={
                            "description": "use sonnet",
                            "prompt": "analyze this architecture",
                            "model": "claude-sonnet-4-6",
                        },
                    ),
                    context,
                )

        self.assertFalse(result.is_error, msg=str(result.output))
        self.assertIs(self.captured["provider"], self.parent_provider)
        self.assertEqual(self.captured["model"], "claude-sonnet-4-6")

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def test_provider_override_without_model(self):
        """Build the selected provider and leave the model unset for inheritance."""
        mock_openai = MagicMock(spec=BaseProvider)
        mock_openai.model = "gpt-4o"

        with TemporaryDirectory() as tmp:
            context = ToolContext(workspace_root=Path(tmp))
            with (
                patch(
                    "clawcodex_ext.providers.runtime.build_provider_from_config",
                    return_value=mock_openai,
                ) as mock_build,
                patch(
                    "src.tool_system.tools.agent.run_agent",
                    _make_fake_run_agent(self.captured),
                ),
            ):
                result = self.parent_registry.dispatch(
                    ToolCall(
                        name="Agent",
                        input={
                            "description": "use openai",
                            "prompt": "review this PR",
                            "provider": "openai",
                        },
                    ),
                    context,
                )

        self.assertFalse(result.is_error, msg=str(result.output))
        mock_build.assert_called_once_with("openai", model=None)
        self.assertIs(self.captured["provider"], mock_openai)
        self.assertIsNone(self.captured["model"])

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def test_build_provider_failure_falls_back_gracefully(self):
        """Fall back to the parent provider without blocking execution."""
        with TemporaryDirectory() as tmp:
            context = ToolContext(workspace_root=Path(tmp))
            with (
                patch(
                    "clawcodex_ext.providers.runtime.build_provider_from_config",
                    side_effect=RuntimeError("API key not configured"),
                ),
                patch(
                    "src.tool_system.tools.agent.run_agent",
                    _make_fake_run_agent(self.captured),
                ),
            ):
                result = self.parent_registry.dispatch(
                    ToolCall(
                        name="Agent",
                        input={
                            "description": "use unknown provider",
                            "prompt": "do something",
                            "provider": "nonexistent",
                            "model": "some-model",
                        },
                    ),
                    context,
                )

        self.assertFalse(result.is_error, msg=str(result.output))
        self.assertIs(self.captured["provider"], self.parent_provider)
        self.assertEqual(self.captured["model"], "some-model")

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def test_no_overrides_inherits_parent(self):
        """Inherit provider and model behavior when tool_input has no overrides."""
        with TemporaryDirectory() as tmp:
            context = ToolContext(workspace_root=Path(tmp))
            with patch(
                "src.tool_system.tools.agent.run_agent",
                _make_fake_run_agent(self.captured),
            ):
                result = self.parent_registry.dispatch(
                    ToolCall(
                        name="Agent",
                        input={
                            "description": "default agent",
                            "prompt": "clean up code",
                        },
                    ),
                    context,
                )

        self.assertFalse(result.is_error, msg=str(result.output))
        self.assertIs(self.captured["provider"], self.parent_provider)
        self.assertIsNone(self.captured["model"])

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def test_provider_only_model_is_none(self):
        """Leave model unset so the query layer reads the provider default."""
        mock_deepseek = MagicMock(spec=BaseProvider)
        mock_deepseek.model = "deepseek-chat"

        with TemporaryDirectory() as tmp:
            context = ToolContext(workspace_root=Path(tmp))
            with (
                patch(
                    "clawcodex_ext.providers.runtime.build_provider_from_config",
                    return_value=mock_deepseek,
                ),
                patch(
                    "src.tool_system.tools.agent.run_agent",
                    _make_fake_run_agent(self.captured),
                ),
            ):
                result = self.parent_registry.dispatch(
                    ToolCall(
                        name="Agent",
                        input={
                            "description": "use deepseek",
                            "prompt": "debug this issue",
                            "provider": "deepseek",
                        },
                    ),
                    context,
                )

        self.assertFalse(result.is_error, msg=str(result.output))
        self.assertIs(self.captured["provider"], mock_deepseek)
        self.assertIsNone(self.captured["model"])

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def test_anthropic_with_custom_model(self):
        """Pass an Anthropic provider with a custom model."""
        mock_anthropic = MagicMock(spec=BaseProvider)
        mock_anthropic.model = "claude-opus-4-6"

        with TemporaryDirectory() as tmp:
            context = ToolContext(workspace_root=Path(tmp))
            with (
                patch(
                    "clawcodex_ext.providers.runtime.build_provider_from_config",
                    return_value=mock_anthropic,
                ) as mock_build,
                patch(
                    "src.tool_system.tools.agent.run_agent",
                    _make_fake_run_agent(self.captured),
                ),
            ):
                result = self.parent_registry.dispatch(
                    ToolCall(
                        name="Agent",
                        input={
                            "description": "use opus",
                            "prompt": "design the architecture",
                            "provider": "anthropic",
                            "model": "claude-opus-4-6",
                        },
                    ),
                    context,
                )

        self.assertFalse(result.is_error, msg=str(result.output))
        mock_build.assert_called_once_with("anthropic", model="claude-opus-4-6")
        self.assertIs(self.captured["provider"], mock_anthropic)
        self.assertEqual(self.captured["model"], "claude-opus-4-6")


if __name__ == "__main__":
    unittest.main()
