"""End-to-end test: 主 agent 委派子 agent 时使用不同的 provider 和 model。

验证完整链路：
1. Agent 工具输入解析（provider + model 从 tool_input 提取）
2. provider 优先级：tool_input > agent_def.provider > None（继承父）
3. model 优先级：tool_input > agent_def.model(!="inherit") > None
4. 指定 provider 时调用 build_provider_from_config 构建新 provider
5. build_provider_from_config 失败时优雅回退到父 provider
6. RunAgentParams 携带正确的 provider 和 model
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
    """端到端测试：主 agent → 子 agent 使用不同的 provider + model。"""

    def setUp(self):
        self.parent_provider = MagicMock(spec=BaseProvider)
        # Make the mock behave like a provider with a default model
        self.parent_provider.model = "claude-sonnet-4-6"
        self.parent_registry = build_default_registry(provider=self.parent_provider)
        self.captured: dict = {}

    # ------------------------------------------------------------------
    # Case 1: provider + model 都来自 tool_input
    # ------------------------------------------------------------------

    def test_provider_and_model_from_tool_input(self):
        """tool_input 同时指定 provider 和 model → 构建新 provider，model 传透。"""
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

        # build_provider_from_config 被调用
        mock_build.assert_called_once_with("minimax", model="minimax-text-01")

        # RunAgentParams 使用新 provider，非父 provider
        self.assertIs(self.captured["provider"], mock_minimax)
        self.assertIsNot(self.captured["provider"], self.parent_provider)

        # model 正确传透
        self.assertEqual(self.captured["model"], "minimax-text-01")

    # ------------------------------------------------------------------
    # Case 2: 仅 model 来自 tool_input，不指定 provider
    # ------------------------------------------------------------------

    def test_model_override_from_tool_input(self):
        """仅 tool_input.model → 保留父 provider，但 model 覆盖。"""
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
        # 保留父 provider
        self.assertIs(self.captured["provider"], self.parent_provider)
        # model 被覆盖
        self.assertEqual(self.captured["model"], "claude-sonnet-4-6")

    # ------------------------------------------------------------------
    # Case 3: 仅 provider 来自 tool_input，不指定 model
    # ------------------------------------------------------------------

    def test_provider_override_without_model(self):
        """仅 tool_input.provider → 构建新 provider，model=None（继承父模型）。"""
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
        # model is None → 继承父 model（由 query 层从 provider.model 读取）
        self.assertIsNone(self.captured["model"])

    # ------------------------------------------------------------------
    # Case 4: build_provider_from_config 失败 → 优雅回退
    # ------------------------------------------------------------------

    def test_build_provider_failure_falls_back_gracefully(self):
        """build_provider_from_config 抛异常 → 回退到父 provider，不阻断。"""
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
        # 回退到父 provider
        self.assertIs(self.captured["provider"], self.parent_provider)
        # model 仍然传透（model 和 provider 是独立逻辑）
        self.assertEqual(self.captured["model"], "some-model")

    # ------------------------------------------------------------------
    # Case 5: 不加 provider 和 model（默认继承）
    # ------------------------------------------------------------------

    def test_no_overrides_inherits_parent(self):
        """tool_input 无 provider / model → provider 和 model 都继承父。"""
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
    # Case 6: provider 不带 model → model 设为 None（由 query 层从 provider 读取）
    # ------------------------------------------------------------------

    def test_provider_only_model_is_none(self):
        """provider 指定但 model 不指定 → model=None。query 层会从
        provider.model 读取默认模型。
        """
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
    # Case 7: provider 用 'anthropic' + model 指定
    # 验证 build_provider_from_config 接收到正确的参数
    # ------------------------------------------------------------------

    def test_anthropic_with_custom_model(self):
        """anthropic provider + 自定义 model。"""
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
