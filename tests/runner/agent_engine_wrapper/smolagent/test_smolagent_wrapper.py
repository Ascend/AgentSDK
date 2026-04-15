# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Unit tests for smolagent/smolagent_wrapper module: SmolAgentWrapper."""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_smolagents():
    """Mock all smolagents dependencies before import."""
    with patch.dict("sys.modules", {
        "smolagents": MagicMock(),
        "smolagents.agents": MagicMock(),
    }):
        mock_litellm = MagicMock()
        mock_code_agent = MagicMock()
        mock_tool_calling_agent = MagicMock()
        mock_tool = MagicMock()

        import sys
        sm = sys.modules["smolagents"]
        sm.LiteLLMModel = mock_litellm
        sm.CodeAgent = mock_code_agent
        sm.ToolCallingAgent = mock_tool_calling_agent
        sm.tool = mock_tool

        from agentic_rl.runner.agent_engine_wrapper.smolagent.smolagent_wrapper import SmolAgentWrapper
        yield SmolAgentWrapper, mock_litellm


def _create_concrete_wrapper(SmolAgentWrapper):
    """Create a concrete subclass that implements the abstract method."""

    class ConcreteSmolWrapper(SmolAgentWrapper):
        async def generate_trajectory(self, task, stream_queue=None, *args, **kwargs):
            return None

    return ConcreteSmolWrapper


@pytest.fixture
def wrapper(mock_smolagents):
    SmolAgentWrapper, mock_litellm = mock_smolagents
    ConcreteWrapper = _create_concrete_wrapper(SmolAgentWrapper)
    tools = [MagicMock()]
    wrapper = ConcreteWrapper(
        server_addresses="http://localhost:8000",
        tools=tools,
        max_model_len=4096,
        tokenizer_name_or_path="test-model",
        max_steps=10,
    )
    return wrapper, mock_litellm


# ---------------------------------------------------------------------------
# __init__ tests
# ---------------------------------------------------------------------------

class TestSmolAgentWrapperInit:
    def test_litellm_model_created(self, wrapper):
        wrapper_inst, mock_litellm = wrapper
        mock_litellm.assert_called_once_with(
            model_id="hosted_vllm/test-model",
            api_base="http://localhost:8000",
            api_key="EMPTY",
            num_ctx=4096,
        )

    def test_agent_args_set(self, wrapper):
        wrapper_inst, _ = wrapper
        assert wrapper_inst.agent_args["return_full_result"] is True
        assert wrapper_inst.agent_args["max_steps"] == 10
        assert wrapper_inst.agent_args["tools"] is not None

    def test_cannot_instantiate_abstract_directly(self, mock_smolagents):
        SmolAgentWrapper, _ = mock_smolagents
        with pytest.raises(TypeError, match="abstract method"):
            SmolAgentWrapper(
                server_addresses="http://localhost:8000",
                tools=[],
                tokenizer_name_or_path="model",
            )


# ---------------------------------------------------------------------------
# init_envs_and_agents tests
# ---------------------------------------------------------------------------

class TestInitEnvsAndAgents:
    def test_creates_agents_for_each_task(self, mock_smolagents):
        SmolAgentWrapper, mock_litellm = mock_smolagents
        ConcreteWrapper = _create_concrete_wrapper(SmolAgentWrapper)
        wrapper = ConcreteWrapper(
            server_addresses="http://localhost:8000",
            tools=[],
            tokenizer_name_or_path="model",
        )
        mock_agent = MagicMock()
        wrapper.agent_class = MagicMock(return_value=mock_agent)

        tasks = [{"problem": "a"}, {"problem": "b"}, {"problem": "c"}]
        wrapper.init_envs_and_agents(tasks)

        assert len(wrapper.agents) == 3
        assert wrapper.n_parallel_agents == 3


# ---------------------------------------------------------------------------
# generate_agent_trajectories_async tests
# ---------------------------------------------------------------------------

class TestGenerateAgentTrajectoriesAsync:
    def test_returns_chat_completions(self, mock_smolagents):
        SmolAgentWrapper, mock_litellm = mock_smolagents
        ConcreteWrapper = _create_concrete_wrapper(SmolAgentWrapper)
        wrapper = ConcreteWrapper(
            server_addresses="http://localhost:8000",
            tools=[],
            tokenizer_name_or_path="model",
        )

        mock_chat_msg = MagicMock()
        mock_chat_msg.role = "user"
        mock_chat_msg.content = "hello"
        mock_chat_msg.tool_calls = None
        mock_chat_msg.raw = {}

        mock_output_msg = {"role": "assistant", "content": "hi"}

        mock_run_result = MagicMock()

        import sys
        RunResult = sys.modules["smolagents.agents"].RunResult

        mock_run_result.messages = [
            {
                "model_input_messages": [mock_chat_msg],
                "model_output_message": mock_output_msg,
            }
        ]

        mock_agent = MagicMock()
        mock_agent.run = MagicMock(return_value=mock_run_result)

        wrapper.init_envs_and_agents = MagicMock()
        wrapper.agents = [mock_agent]

        with patch(
            "agentic_rl.runner.agent_engine_wrapper.smolagent.smolagent_wrapper.RunResult",
            type(mock_run_result),
        ):
            tasks = [{"problem": "hello"}]
            result = wrapper.generate_agent_trajectories_async(tasks)

        assert "chat_completions" in result
        assert isinstance(result["chat_completions"], list)

    def test_empty_tasks_returns_empty_completions(self, mock_smolagents):
        SmolAgentWrapper, _ = mock_smolagents
        ConcreteWrapper = _create_concrete_wrapper(SmolAgentWrapper)
        wrapper = ConcreteWrapper(
            server_addresses="http://localhost:8000",
            tools=[],
            tokenizer_name_or_path="model",
        )
        wrapper.init_envs_and_agents = MagicMock()
        wrapper.agents = []
        result = wrapper.generate_agent_trajectories_async([])
        assert result == {"chat_completions": []}
