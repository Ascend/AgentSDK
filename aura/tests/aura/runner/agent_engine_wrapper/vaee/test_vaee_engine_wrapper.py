#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class _MockSignalModule:
    """Mock signal module that prevents set_noop_signal recursion."""
    SIGINT = 2
    SIGTERM = 15

    def __init__(self):
        self._handlers = {}

    def default_int_handler(self, *args):
        pass

    def signal(self, sig, handler):
        old = self._handlers.get(sig)
        self._handlers[sig] = handler
        return old

    def getsignal(self, sig):
        return self._handlers.get(sig, None)

    def __setattr__(self, name, value):
        if name == "signal":
            return
        super().__setattr__(name, value)


class TestVaeeEngineWrapper:

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        mock_transformers = MagicMock()
        mock_auto_tokenizer = MagicMock()
        mock_transformers.AutoTokenizer = MagicMock(
            from_pretrained=MagicMock(return_value=mock_auto_tokenizer)
        )

        class MockBaseEngineWrapper:
            """Mock base class for BaseEngineWrapper."""

            def __init__(self, *args, **kwargs):
                pass

            def stat(self, *args, **kwargs):
                return {}

            def gen_tasks(self, *args, **kwargs):
                return []

            def cancel_request(self, *args, **kwargs):
                pass

            def set_noop_signal(self, *args, **kwargs):
                pass

            def clear_cache(self, *args, **kwargs):
                pass

        mock_base_engine_wrapper = MagicMock()
        mock_base_engine_wrapper.BaseEngineWrapper = MockBaseEngineWrapper
        mock_base_engine_wrapper.AgentTask = MagicMock()

        mock_agent_proxy_client = MagicMock()
        mock_agent_proxy_client.AgentProxyClient = MagicMock

        mock_traj_proxy_client = MagicMock()
        mock_traj_proxy_client.TrajProxyClient = MagicMock
        mock_traj_proxy_client.calculate_time_diff_seconds = MagicMock(return_value=0.5)

        class _MockRequestRecord(dict):
            """Mock RequestRecord that accepts kwargs as attributes."""
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.__dict__.update(kwargs)

        mock_vaee_types = MagicMock()
        mock_vaee_types.Episode = MagicMock()
        mock_vaee_types.Trajectory = MagicMock()
        mock_vaee_types.Step = MagicMock()
        mock_vaee_types.RequestRecord = _MockRequestRecord

        mock_default_traj_refine = MagicMock()
        mock_default_step_refine = MagicMock()
        mock_default_step_refine.__name__ = "default_step_traj_refine_func"
        mock_default_traj_refine.default_step_traj_refine_func = mock_default_step_refine
        mock_default_token_refine = MagicMock()
        mock_default_token_refine.__name__ = "default_token_traj_refine_func"
        mock_default_traj_refine.default_token_traj_refine_func = mock_default_token_refine

        mock_default_traj_reward = MagicMock()
        mock_default_reward = MagicMock()
        mock_default_reward.__name__ = "default_traj_reward_func"
        mock_default_traj_reward.default_traj_reward_func = mock_default_reward

        mock_tool_env = MagicMock()
        mock_tool_env.ToolEnvironment = MagicMock

        mock_loggers = MagicMock()
        mock_loggers.get_logger = MagicMock(return_value=MagicMock())

        mock_func = MagicMock()
        mock_func.__name__ = "mock_func"
        mock_load_object = MagicMock(return_value=mock_func)

        # Use a plain object for the load_object_by_path module so attribute access works correctly
        class _MockLoadObjectModule:
            pass
        _mock_load_module = _MockLoadObjectModule()
        _mock_load_module.load_object_by_path = mock_load_object

        mock_agents_mapping = MagicMock()
        mock_env_class = MagicMock()
        mock_env = MagicMock()
        mock_env.reset = MagicMock(return_value=("observation", {"info": "test"}))
        mock_env_class.from_dict = MagicMock(return_value=mock_env)
        mock_agent_class = MagicMock()
        mock_agent = MagicMock()
        mock_agent.chat_completions = [{"role": "user", "content": "hello"}]
        mock_agent_class.return_value = mock_agent
        mock_agent_proxy_cls = MagicMock()
        mock_traj_proxy_cls = MagicMock()
        mock_agents_mapping.get_agent_by_name = MagicMock(return_value={
            "agent_class": mock_agent_class,
            "agent_args": {},
            "env_class": mock_env_class,
            "env_args": {},
            "agent_proxy_class": mock_agent_proxy_cls,
            "traj_proxy_class": mock_traj_proxy_cls,
        })

        mock_chat_template = MagicMock()
        mock_chat_template.ChatTemplateParser = MagicMock()

        mock_signal = _MockSignalModule()

        with patch.dict(sys.modules, {
            "signal": mock_signal,
            "transformers": mock_transformers,
            "aura.runner.agent_engine_wrapper.base_engine_wrapper": mock_base_engine_wrapper,
            "aura.runner.agent_engine_wrapper.proxy_client.agent_proxy_client": mock_agent_proxy_client,
            "aura.runner.agent_engine_wrapper.proxy_client.traj_proxy_client": mock_traj_proxy_client,
            "aura.runner.agent_engine_wrapper.vaee.vaee_types": mock_vaee_types,
            "aura.runner.agent_engine_wrapper.vaee.default_traj_refine": mock_default_traj_refine,
            "aura.runner.agent_engine_wrapper.vaee.default_traj_reward": mock_default_traj_reward,
            "aura.runner.agent_engine_wrapper.vaee.tool_env": mock_tool_env,
            "aura.base.log.loggers": MagicMock(Loggers=mock_loggers),
            "aura.base.utils.load_object_by_path": _mock_load_module,
            "agents.agents_mapping": mock_agents_mapping,
            "aura.runner.agent_engine_wrapper.base.parser.chat_template": mock_chat_template,
        }):
            yield

    def test_vaee_engine_wrapper_init(self):
        from aura.runner.agent_engine_wrapper.vaee.vaee_engine_wrapper import (
            VirtualAgentEngineExecutionWrapper,
        )

        wrapper = VirtualAgentEngineExecutionWrapper(
            infer_service_params={"model": "test-model"},
            traj_reward_func=None,
            traj_refine_func=None,
            res_reward_func=None,
            tokenizer="test-tokenizer",
            n_parallel_agents=4,
            agent_name="test_agent",
        )
        assert wrapper is not None

    def test_vaee_engine_wrapper_init_with_func_paths(self):
        from aura.runner.agent_engine_wrapper.vaee.vaee_engine_wrapper import (
            VirtualAgentEngineExecutionWrapper,
        )

        wrapper = VirtualAgentEngineExecutionWrapper(
            infer_service_params={"model": "test-model"},
            traj_reward_func="some.module.reward_func",
            traj_refine_func="some.module.refine_func",
            res_reward_func="some.module.res_func",
            tokenizer="test-tokenizer",
            n_parallel_agents=4,
            agent_name="test_agent",
        )
        assert wrapper is not None

    def test_vaee_engine_wrapper_clear_cache(self):
        from aura.runner.agent_engine_wrapper.vaee.vaee_engine_wrapper import (
            VirtualAgentEngineExecutionWrapper,
        )

        wrapper = VirtualAgentEngineExecutionWrapper(
            infer_service_params={"model": "test-model"},
            tokenizer="test-tokenizer",
            agent_name="test_agent",
        )
        wrapper.clear_cache()

    def test_vaee_engine_wrapper_cancel_request(self):
        from aura.runner.agent_engine_wrapper.vaee.vaee_engine_wrapper import (
            VirtualAgentEngineExecutionWrapper,
        )

        wrapper = VirtualAgentEngineExecutionWrapper(
            infer_service_params={"model": "test-model"},
            tokenizer="test-tokenizer",
            agent_name="test_agent",
        )
        asyncio.run(wrapper.cancel_request(MagicMock()))

    def test_vaee_engine_wrapper_set_noop_signal(self):
        from aura.runner.agent_engine_wrapper.vaee.vaee_engine_wrapper import (
            VirtualAgentEngineExecutionWrapper,
        )

        wrapper = VirtualAgentEngineExecutionWrapper(
            infer_service_params={"model": "test-model"},
            tokenizer="test-tokenizer",
            agent_name="test_agent",
        )
        wrapper.set_noop_signal()

    def test_vaee_engine_wrapper_calc_rewards_none_func(self):
        from aura.runner.agent_engine_wrapper.vaee.vaee_engine_wrapper import (
            VirtualAgentEngineExecutionWrapper,
        )

        wrapper = VirtualAgentEngineExecutionWrapper(
            infer_service_params={"model": "test-model"},
            tokenizer="test-tokenizer",
            agent_name="test_agent",
        )
        wrapper.res_reward_func = None

        episode = MagicMock()
        episode.id = "test_ep"
        episode.trajectories = []

        asyncio.run(wrapper.calc_rewards(episode, {"problem": "test"}))

    def test_vaee_engine_wrapper_calc_rewards_with_trajectories(self):
        from aura.runner.agent_engine_wrapper.vaee.vaee_engine_wrapper import (
            VirtualAgentEngineExecutionWrapper,
        )
        import aura.runner.agent_engine_wrapper.vaee.vaee_engine_wrapper as engine_module

        mock_res_reward_fn = MagicMock()
        mock_reward_output = MagicMock()
        mock_reward_output.reward = 0.5
        mock_reward_output.is_correct = False
        mock_reward_output.metadata = {}
        mock_res_reward_fn.return_value = mock_reward_output
        mock_res_reward_fn.__name__ = "mock_res_reward"

        wrapper = VirtualAgentEngineExecutionWrapper(
            infer_service_params={"model": "test-model"},
            tokenizer="test-tokenizer",
            agent_name="test_agent",
        )
        wrapper.res_reward_func = mock_res_reward_fn
        wrapper.max_steps = 10

        step = MagicMock()
        step.reward = 0.0
        step.done = False
        step.action = "test_action"
        step.tool_outputs = []

        traj = MagicMock()
        traj.steps = [step]

        episode = MagicMock()
        episode.id = "test_ep"
        episode.trajectories = [traj]

        mock_tool_env = MagicMock()
        mock_tool_env.step = MagicMock(return_value=(0.5, False))
        mock_tool_env_cls = MagicMock(return_value=mock_tool_env)

        with patch.object(engine_module, "ToolEnvironment", mock_tool_env_cls):
            asyncio.run(wrapper.calc_rewards(episode, {"problem": "test"}))
        assert step.reward == 0.5

    def test_vaee_engine_wrapper_calc_rewards_max_steps_done(self):
        from aura.runner.agent_engine_wrapper.vaee.vaee_engine_wrapper import (
            VirtualAgentEngineExecutionWrapper,
        )
        import aura.runner.agent_engine_wrapper.vaee.vaee_engine_wrapper as engine_module

        mock_res_reward_fn = MagicMock()
        mock_reward_output = MagicMock()
        mock_reward_output.reward = 0.0
        mock_reward_output.is_correct = False
        mock_reward_output.metadata = {}
        mock_res_reward_fn.return_value = mock_reward_output
        mock_res_reward_fn.__name__ = "mock_res_reward"

        wrapper = VirtualAgentEngineExecutionWrapper(
            infer_service_params={"model": "test-model"},
            tokenizer="test-tokenizer",
            agent_name="test_agent",
        )
        wrapper.res_reward_func = mock_res_reward_fn
        wrapper.max_steps = 1

        step = MagicMock()
        step.reward = 0.0
        step.done = False
        step.action = "test_action"
        step.tool_outputs = []

        traj = MagicMock()
        traj.steps = [step]

        episode = MagicMock()
        episode.id = "test_ep"
        episode.trajectories = [traj]

        mock_tool_env = MagicMock()
        mock_tool_env.step = MagicMock(return_value=(0.0, True))
        mock_tool_env_cls = MagicMock(return_value=mock_tool_env)

        with patch.object(engine_module, "ToolEnvironment", mock_tool_env_cls):
            asyncio.run(wrapper.calc_rewards(episode, {"problem": "test"}))

    def test_vaee_engine_wrapper_stat_empty(self):
        from aura.runner.agent_engine_wrapper.vaee.vaee_engine_wrapper import (
            VirtualAgentEngineExecutionWrapper,
        )

        wrapper = VirtualAgentEngineExecutionWrapper(
            infer_service_params={"model": "test-model"},
            tokenizer="test-tokenizer",
            agent_name="test_agent",
        )
        episode = MagicMock()
        stat = wrapper.stat([], episode)
        assert isinstance(stat, dict)

    def test_vaee_engine_wrapper_generate_trajectory(self):
        from aura.runner.agent_engine_wrapper.vaee.vaee_engine_wrapper import (
            VirtualAgentEngineExecutionWrapper,
        )

        wrapper = VirtualAgentEngineExecutionWrapper(
            infer_service_params={"model": "test-model"},
            tokenizer="test-tokenizer",
            agent_name="test_agent",
        )

        mock_agents_mapping = sys.modules["agents.agents_mapping"]
        mock_agents_mapping.get_agent_by_name = MagicMock(return_value={
            "agent_class": MagicMock(),
            "agent_args": {},
            "env_class": MagicMock(),
            "env_args": {},
            "agent_proxy_class": MagicMock(),
            "traj_proxy_class": MagicMock(),
        })

        mock_agent_proxy = MagicMock()
        mock_agent_proxy.get_agent_response = AsyncMock(return_value=(0, "session_1"))
        wrapper.agent_proxy_client = mock_agent_proxy

        mock_traj_proxy = MagicMock()
        mock_traj_proxy.get_records_by_session = AsyncMock(return_value=[
            {
                "unique_id": "1",
                "request_id": "req_1",
                "session_id": "session_1",
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
                "start_time": "2026-01-01T00:00:00",
                "response_text": "response",
                "raw_response": {"choices": [{"message": {"content": "hi"}}]},
                "token_ids": [1, 2, 3],
                "response_ids": [4, 5],
                "token_response": {"choices": [{"logprobs": {"token_logprobs": [0.1, 0.2]}}]},
                "error_traceback": None,
            }
        ])
        wrapper.traj_proxy_client = mock_traj_proxy

        wrapper.traj_refine_func = MagicMock(return_value=MagicMock())
        wrapper.traj_reward_func = MagicMock(return_value=MagicMock())
        wrapper.res_reward_func = None

        task = MagicMock()
        task.model_dump = MagicMock(return_value={
            "task_id": "test_task",
            "prompt_id": 0,
            "ground_truth": "42",
        })

        episode = asyncio.run(wrapper.generate_trajectory(task))
        assert episode is not None

    def test_vaee_engine_wrapper_post_process_trajectory(self):
        from aura.runner.agent_engine_wrapper.vaee.vaee_engine_wrapper import (
            VirtualAgentEngineExecutionWrapper,
        )

        wrapper = VirtualAgentEngineExecutionWrapper(
            infer_service_params={"model": "test-model"},
            tokenizer="test-tokenizer",
            agent_name="test_agent",
        )

        mock_traj_proxy = MagicMock()
        mock_traj_proxy.get_records_by_session = AsyncMock(return_value=[
            {
                "unique_id": "1",
                "request_id": "req_1",
                "session_id": "session_1",
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
                "start_time": "2026-01-01T00:00:00",
                "response_text": "response",
                "raw_response": {"choices": [{"message": {"content": "hi"}}]},
                "token_ids": [1, 2, 3],
                "response_ids": [4, 5],
                "token_response": {"choices": [{"logprobs": {"token_logprobs": [0.1, 0.2]}}]},
                "error_traceback": None,
            }
        ])
        wrapper.traj_proxy_client = mock_traj_proxy

        mock_episode = MagicMock()
        mock_episode.id = "test_ep"
        mock_episode.trajectories = []

        wrapper.traj_refine_func = MagicMock(return_value=mock_episode)
        wrapper.traj_reward_func = MagicMock(return_value=mock_episode)

        task_data = {
            "task_id": "test_task",
            "prompt_id": 0,
            "ground_truth": "42",
        }
        episode = asyncio.run(wrapper._post_process_trajectory("session_1", task_data, {}))
        assert episode is not None

    def test_vaee_engine_wrapper_post_process_trajectory_no_records(self):
        from aura.runner.agent_engine_wrapper.vaee.vaee_engine_wrapper import (
            VirtualAgentEngineExecutionWrapper,
        )

        wrapper = VirtualAgentEngineExecutionWrapper(
            infer_service_params={"model": "test-model"},
            tokenizer="test-tokenizer",
            agent_name="test_agent",
        )

        mock_traj_proxy = MagicMock()
        mock_traj_proxy.get_records_by_session = AsyncMock(return_value=[])
        wrapper.traj_proxy_client = mock_traj_proxy

        task_data = {
            "task_id": "test_task",
            "prompt_id": 0,
            "ground_truth": "42",
        }
        with pytest.raises(ValueError, match="No records found"):
            asyncio.run(wrapper._post_process_trajectory("session_1", task_data, {}))

    def test_vaee_engine_wrapper_post_process_trajectory_all_error(self):
        from aura.runner.agent_engine_wrapper.vaee.vaee_engine_wrapper import (
            VirtualAgentEngineExecutionWrapper,
        )

        wrapper = VirtualAgentEngineExecutionWrapper(
            infer_service_params={"model": "test-model"},
            tokenizer="test-tokenizer",
            agent_name="test_agent",
        )

        mock_traj_proxy = MagicMock()
        mock_traj_proxy.get_records_by_session = AsyncMock(return_value=[
            {
                "unique_id": "1",
                "request_id": "req_1",
                "session_id": "session_1",
                "model": "test",
                "messages": [],
                "start_time": "2026-01-01T00:00:00",
                "response_text": "",
                "raw_response": None,
                "token_ids": [],
                "response_ids": [],
                "token_response": None,
                "error_traceback": "error occurred",
            }
        ])
        wrapper.traj_proxy_client = mock_traj_proxy

        task_data = {
            "task_id": "test_task",
            "prompt_id": 0,
            "ground_truth": "42",
        }
        with pytest.raises(ValueError, match="No records found"):
            asyncio.run(wrapper._post_process_trajectory("session_1", task_data, {}))
