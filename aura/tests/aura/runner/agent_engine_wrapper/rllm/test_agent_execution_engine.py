#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import types
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


# ---------------------------------------------------------------------------
# Fixture: fake module tree for agent_execution_engine
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_engine_env():
    """Build an isolated fake module tree, keeping real asyncio and re."""
    fake_torch = types.ModuleType("torch")
    fake_torch.tensor = MagicMock()
    fake_torch.long = "long"
    fake_torch.float32 = "float32"

    fake_hashlib = types.ModuleType("hashlib")
    fake_hashlib.sha256 = MagicMock()
    fake_hashlib.sha256.return_value.hexdigest.return_value = "a" * 64

    fake_os = types.ModuleType("os")
    fake_os.getpid = MagicMock(return_value=1234)

    fake_time = types.ModuleType("time")
    fake_time.time = MagicMock(return_value=1_000_000.0)
    fake_time.localtime = MagicMock()
    fake_time.strftime = MagicMock(return_value="20230101000000")

    fake_traceback = types.ModuleType("traceback")
    fake_traceback.print_exc = MagicMock()

    fake_uuid = types.ModuleType("uuid")
    fake_uuid.uuid4 = MagicMock(return_value="12345678-1234-1234-1234-1234567890ab")
    fake_uuid.uuid1 = MagicMock(return_value="12345678-1234-1234-1234-1234567890ab")

    fake_loggers_mod = types.ModuleType("aura.base.log.loggers")
    mock_logger = MagicMock()
    fake_loggers_mod.Loggers = MagicMock(return_value=MagicMock(get_logger=MagicMock(return_value=mock_logger)))

    fake_app_stats = types.ModuleType("aura.base.misc.misc")
    fake_app_stats.app_stats = MagicMock()
    fake_app_stats.colorful_print = MagicMock()

    fake_utils_mod = types.ModuleType("aura.base.utils.utils")
    fake_utils_mod.strftime = MagicMock()

    fake_base_agent = types.ModuleType("aura.runner.agent_engine_wrapper.base.agent.base_agent")
    fake_base_agent.Action = MagicMock()
    class FakeBaseAgent:
        pass
    fake_base_agent.BaseAgent = FakeBaseAgent
    fake_base_agent.Trajectory = MagicMock()

    fake_env_utils = types.ModuleType("aura.runner.agent_engine_wrapper.base.environment.env_utils")
    fake_env_utils.compute_mc_return = MagicMock()
    fake_env_utils.compute_trajectory_reward = MagicMock()

    fake_chat_template = types.ModuleType("aura.runner.agent_engine_wrapper.base.parser.chat_template")
    fake_chat_template.ChatTemplateParser = MagicMock()
    fake_chat_template.ChatTemplateParser.get_parser = MagicMock(return_value=MagicMock())

    fake_base_engine_wrapper = types.ModuleType("aura.runner.agent_engine_wrapper.base_engine_wrapper")
    class FakeAgentTask:
        def __init__(self, task_id="task-0", prompt_id=0):
            self.task_id = task_id
            self.prompt_id = prompt_id
    fake_base_engine_wrapper.AgentTask = FakeAgentTask

    fake_msg_handler = types.ModuleType("aura.runner.agent_engine_wrapper.rllm.msg_handler")
    fake_msg_handler.convert_messages_to_tokens_and_masks = MagicMock(return_value=([1, 2, 3], [1, 1, 1]))
    fake_msg_handler.get_recent_assistant_user_messages = MagicMock(return_value=(MagicMock(), MagicMock()))

    fake_router_mod = types.ModuleType("aura.runner.scheduler.router")
    fake_router = MagicMock()
    fake_router_mod.Router = MagicMock()
    fake_router_mod.Router.create = MagicMock(return_value=fake_router)

    fake_verl_experimental = types.ModuleType("verl.experimental.agent_loop.agent_loop")
    fake_verl_experimental.AsyncLLMServerManager = MagicMock()
    fake_verl_experimental.AsyncLLMServerManager.return_value.generate = AsyncMock()

    import os as _os
    import aura as _aura
    base = _aura.__path__[0] if _aura.__path__ else "."
    fake_aura = types.ModuleType("aura")
    fake_aura.__path__ = _aura.__path__
    fake_aura_runner = types.ModuleType("aura.runner")
    fake_aura_runner.__path__ = [_os.path.join(base, "runner")]
    fake_aura_runner_agent_engine_wrapper = types.ModuleType("aura.runner.agent_engine_wrapper")
    fake_aura_runner_agent_engine_wrapper.__path__ = [_os.path.join(base, "runner/agent_engine_wrapper")]
    fake_aura_runner_agent_engine_wrapper_rllm = types.ModuleType("aura.runner.agent_engine_wrapper.rllm")
    fake_aura_runner_agent_engine_wrapper_rllm.__path__ = [_os.path.join(base, "runner/agent_engine_wrapper/rllm")]
    fake_aura_base = types.ModuleType("aura.base")
    fake_aura_base.__path__ = []
    fake_aura_base_log = types.ModuleType("aura.base.log")
    fake_aura_base_log.__path__ = []

    fakes = {
        "torch": fake_torch,
        "hashlib": fake_hashlib,
        "os": fake_os,
        "time": fake_time,
        "traceback": fake_traceback,
        "uuid": fake_uuid,
        "aura.base.log.loggers": fake_loggers_mod,
        "aura.base.misc.misc": fake_app_stats,
        "aura.base.utils.utils": fake_utils_mod,
        "aura.runner.agent_engine_wrapper.base.agent.base_agent": fake_base_agent,
        "aura.runner.agent_engine_wrapper.base.environment.env_utils": fake_env_utils,
        "aura.runner.agent_engine_wrapper.base.parser.chat_template": fake_chat_template,
        "aura.runner.agent_engine_wrapper.base_engine_wrapper": fake_base_engine_wrapper,
        "aura.runner.agent_engine_wrapper.rllm.msg_handler": fake_msg_handler,
        "aura.runner.scheduler.router": fake_router_mod,
        "verl.experimental.agent_loop.agent_loop": fake_verl_experimental,
        "aura": fake_aura,
        "aura.runner": fake_aura_runner,
        "aura.runner.agent_engine_wrapper": fake_aura_runner_agent_engine_wrapper,
        "aura.runner.agent_engine_wrapper.rllm": fake_aura_runner_agent_engine_wrapper_rllm,
        "aura.base": fake_aura_base,
        "aura.base.log": fake_aura_base_log,
    }

    target = "aura.runner.agent_engine_wrapper.rllm.agent_execution_engine"
    if target in sys.modules:
        del sys.modules[target]

    with patch.dict(sys.modules, fakes):
        import aura.runner.agent_engine_wrapper.rllm.agent_execution_engine as mod
        yield {
            "mod": mod,
            "AgentExecutionEngine": mod.AgentExecutionEngine,
            "mock_logger": mock_logger,
            "fake_router": fake_router,
            "fake_router_mod": fake_router_mod,
            "fake_torch": fake_torch,
            "fake_app_stats": fake_app_stats,
            "fake_env_utils": fake_env_utils,
            "fake_base_agent": fake_base_agent,
            "fake_msg_handler": fake_msg_handler,
            "fake_time": fake_time,
            "fake_os": fake_os,
            "fake_uuid": fake_uuid,
            "fake_chat_template": fake_chat_template,
            "fake_verl": fake_verl_experimental,
        }

    if target in sys.modules:
        del sys.modules[target]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestHelpers:
    def test_create_application_id(self, fake_engine_env):
        mod = fake_engine_env["mod"]
        app_id = mod.create_application_id(42)
        assert "42-" in app_id

    def test_generate_key_dict(self, fake_engine_env):
        mod = fake_engine_env["mod"]
        key = mod._generate_key({"task_id": "t1", "prompt_id": 101})
        assert key == "t1_101"

    def test_generate_key_agent_task(self, fake_engine_env):
        mod = fake_engine_env["mod"]
        task = mod.AgentTask(task_id="t2", prompt_id=202)
        key = mod._generate_key(task)
        assert key == "t2_202"


class TestInit:
    def test_minimal_init(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](
            tokenizer="tok", server_addresses=["addr"], chat_parser="parser", n_parallel_agents=2
        )
        assert engine.tokenizer == "tok"
        assert engine.n_parallel_agents == 2
        assert engine.chat_parser == "parser"

    def test_env_not_multithread_safe_raises(self, fake_engine_env):
        env_cls = MagicMock()
        env_cls.is_multithread_safe.return_value = False
        with pytest.raises(TypeError, match="multi-thread safe"):
            fake_engine_env["AgentExecutionEngine"](env_class=env_cls, tokenizer="t")

    def test_chat_parser_created_if_none(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](
            tokenizer="tok", server_addresses=["addr"], n_parallel_agents=1
        )
        assert engine.chat_parser is not None

    def test_thread_pool_created(self, fake_engine_env):
        with patch("concurrent.futures.ThreadPoolExecutor") as mock_tpe:
            engine = fake_engine_env["AgentExecutionEngine"](
                tokenizer="tok", max_workers=4, server_addresses=["addr"]
            )
            mock_tpe.assert_called_with(max_workers=4)
            assert engine.executor is mock_tpe.return_value


class TestInitRouter:
    def test_addresses_none_skips(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](tokenizer="t", server_addresses=["addr"])
        engine.router = None
        engine.init_router(None)
        assert engine.router is None

    def test_router_already_exists_updates(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](tokenizer="t", server_addresses=["addr"])
        engine.router = MagicMock()
        engine.init_router(["new_addr"])
        engine.router.update_address.assert_called_once_with(["new_addr"])

    def test_router_created_when_none(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](
            tokenizer="t", tokenizer_name_or_path="bert",
        )
        engine.router = None
        engine.init_router(["addr"])
        fake_engine_env["fake_router_mod"].Router.create.assert_called_once()


class TestGetModelResponse:
    @pytest.mark.asyncio
    async def test_router_path(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](tokenizer="t", server_addresses=["addr"])
        engine.router.chat = AsyncMock(return_value="router_response")
        engine.sampling_params = {}
        response = await engine.get_model_response(prompt="hello", application_id="app2")
        assert response == "router_response"

    @pytest.mark.asyncio
    async def test_server_handles_path(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](tokenizer="t", server_addresses=["addr"])
        engine.tokenizer = MagicMock()
        engine.tokenizer.decode = MagicMock(return_value="decoded")
        engine.sampling_params = {"top_p": 0.9, "temperature": 1.0, "max_tokens": 100, "n": 1}
        fake_verl = fake_engine_env["fake_verl"]
        fake_verl.AsyncLLMServerManager.return_value.generate.return_value = MagicMock(token_ids=[], log_probs=[])
        response = await engine.get_model_response(prompt=[1, 2], application_id="app1", server_handles="handles")
        assert response["message"] == "decoded"


class TestApplicationId:
    def test_store_and_pop(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](tokenizer="t", server_addresses=["addr"])
        task = {"task_id": "t1", "prompt_id": 1}
        engine.store_application_id(task, "app123")
        assert engine.pop_application_id(task) == "app123"
        assert engine.pop_application_id(task) is None

    def test_clear_cache(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](tokenizer="t", server_addresses=["addr"])
        task = {"task_id": "t1", "prompt_id": 1}
        engine.store_application_id(task, "app")
        engine.clear_cache()
        assert len(engine.application_ids) == 0


class TestEnvAgentManagement:
    def test_update_envs_and_agents(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](tokenizer="t", server_addresses=["addr"])
        envs = [MagicMock() for _ in range(3)]
        agents = [MagicMock() for _ in range(3)]
        engine.update_envs_and_agents(envs, agents, iteration=0, sample_id=0)
        assert engine.n_parallel_agents == 3

    def test_env_agent_count_mismatch_raises(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](tokenizer="t", server_addresses=["addr"])
        with pytest.raises(ValueError):
            engine.update_envs_and_agents([MagicMock()], [MagicMock(), MagicMock()], 0, 0)

    def test_release_env_and_agent(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](tokenizer="t", server_addresses=["addr"])
        engine.env_dict["t1"] = MagicMock()
        engine.agent_dict["t1"] = MagicMock()
        engine.release_env_and_agent("t1")
        assert "t1" not in engine.env_dict
        assert "t1" not in engine.agent_dict


class TestRunAgentTrajectoryAsync:
    @pytest.mark.asyncio
    async def test_normal_execution_text_mode(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](
            tokenizer=MagicMock(),
            n_parallel_agents=1,
            max_steps=2,
            server_addresses=["addr"],
            chat_parser=MagicMock(),
            max_prompt_length=1024,
            max_model_len=16384,
            env_args={"trajectory_timeout": 7200},
        )
        engine.router.chat = AsyncMock(return_value="model response")
        engine.tokenizer.encode.return_value = [1, 2, 3]
        engine.chat_parser.parse.return_value = "parsed prompt"

        agent = MagicMock()
        agent.chat_completions = [{"role": "user", "content": "hi"}]
        agent.reset = MagicMock()
        agent.update_from_env = MagicMock()
        agent.update_from_model = MagicMock(return_value=MagicMock(action="step"))
        agent.trajectory = MagicMock()
        env = MagicMock()
        env.reset = MagicMock(return_value=("obs", {"info": "x"}))
        env.step = MagicMock(return_value=("next_obs", 0.0, True, {}))
        env.close = MagicMock()
        env.is_multithread_safe.return_value = True
        del env.compute_final_reward

        engine.envs = [env]
        engine.agents = [agent]

        loop = asyncio.get_event_loop()
        with patch.object(loop, "run_in_executor") as mock_run:
            def run_side_effect(executor, func, *args, **kwargs):
                if func == env.reset:
                    result = ("obs", {"info": "x"})
                elif func == env.step:
                    result = ("next_obs", 0.0, True, {})
                elif func == env.close:
                    result = None
                else:
                    result = None
                fut = asyncio.Future()
                fut.set_result(result)
                return fut
            mock_run.side_effect = run_side_effect
            trajectory = await engine.run_agent_trajectory_async(idx=0, application_id="app1", mode="Text")
        assert trajectory is not None
        fake_engine_env["fake_env_utils"].compute_trajectory_reward.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_via_stop_flag(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](
            tokenizer=MagicMock(), max_steps=3, server_addresses=["addr"]
        )
        engine.stop = True
        agent = MagicMock()
        agent.chat_completions = [{"role": "user"}]
        env = MagicMock()
        env.reset = MagicMock(return_value=("obs", {}))
        engine.envs = [env]
        engine.agents = [agent]
        trajectory = await engine.run_agent_trajectory_async(idx=0, application_id="app3")
        assert trajectory is None


class TestTrajectoryGenerator:
    @pytest.mark.asyncio
    async def test_generator_yields_result(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](
            tokenizer="t", n_parallel_agents=1, max_steps=1, server_addresses=["addr"]
        )
        engine.env_dict = {"t1": MagicMock(is_multithread_safe=MagicMock(return_value=True))}
        engine.agents = [MagicMock()]
        engine.envs = [MagicMock()]
        engine.run_agent_trajectory_async = AsyncMock(return_value="traj")
        task = {"task_id": "t1", "prompt_id": 0}
        gen = engine.trajectory_generator(task, mode="Text", prompt_id=0)
        results = [res async for res in gen]
        assert results == ["traj"]

    @pytest.mark.asyncio
    async def test_exception_in_task_propagates(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](
            tokenizer="t", n_parallel_agents=1, server_addresses=["addr"]
        )
        engine.env_dict = {"t1": MagicMock(is_multithread_safe=MagicMock(return_value=True))}
        engine.run_agent_trajectory_async = AsyncMock(side_effect=RuntimeError("fail"))
        task = {"task_id": "t1", "prompt_id": 0}
        with pytest.raises(RuntimeError, match="fail"):
            async for _ in engine.trajectory_generator(task, prompt_id=0):
                pass

    @pytest.mark.asyncio
    async def test_non_multithread_safe_env_raises_type_error(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](
            tokenizer="t", n_parallel_agents=1, server_addresses=["addr"]
        )
        engine.env_dict = {"t1": MagicMock(is_multithread_safe=MagicMock(return_value=False))}
        task = {"task_id": "t1", "prompt_id": 0}

        with pytest.raises(TypeError, match="multithread safe"):
            async for _ in engine.trajectory_generator(task, prompt_id=0):
                pass


class TestExecuteTasks:
    @pytest.mark.asyncio
    async def test_execute_tasks_successful(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](
            tokenizer="t", n_parallel_agents=2, max_steps=1, server_addresses=["addr"],
        )
        engine.env_class = MagicMock()
        engine.env_class.from_dict = MagicMock(return_value=MagicMock())
        engine.agent_class = MagicMock(return_value=fake_engine_env["fake_base_agent"].BaseAgent())
        engine.agent_class.return_value.trajectory = MagicMock()
        engine.run_agent_trajectory_async = AsyncMock(return_value=MagicMock())
        tasks = [{"task_id": "0"}, {"task_id": "1"}]
        results = await engine.execute_tasks(tasks)
        assert len(results) == 2


class TestRunAgentTrajectoryAsyncTokenMode:
    @pytest.mark.asyncio
    async def test_token_mode_success(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](
            tokenizer=MagicMock(), n_parallel_agents=1, max_steps=2, server_addresses=["addr"],
            chat_parser=MagicMock(), max_prompt_length=1024, max_model_len=16384,
            env_args={"trajectory_timeout": 7200},
        )
        engine.router.chat = AsyncMock(return_value="model response")
        engine.tokenizer.encode.return_value = [1, 2, 3]
        engine.chat_parser.parse.return_value = "parsed prompt"

        agent = MagicMock()
        agent.chat_completions = [{"role": "user", "content": "hi"}]
        agent.reset = MagicMock()
        agent.update_from_env = MagicMock()
        agent.update_from_model = MagicMock(return_value=MagicMock(action="step"))
        agent.trajectory = MagicMock()
        env = MagicMock()
        env.reset = MagicMock(return_value=("obs", {"info": "x"}))
        env.step = MagicMock(return_value=("next_obs", 0.5, True, {}))
        env.close = MagicMock()
        env.is_multithread_safe.return_value = True
        del env.compute_final_reward

        engine.envs = [env]
        engine.agents = [agent]

        loop = asyncio.get_event_loop()
        with patch.object(loop, "run_in_executor") as mock_run:
            def run_side_effect(executor, func, *args, **kwargs):
                if func == env.reset:
                    result = ("obs", {"info": "x"})
                elif func == env.step:
                    result = ("next_obs", 0.5, True, {})
                elif func == env.close:
                    result = None
                else:
                    result = None
                fut = asyncio.Future()
                fut.set_result(result)
                return fut
            mock_run.side_effect = run_side_effect
            result = await engine.run_agent_trajectory_async(idx=0, application_id="app_token", mode="Token")
        assert isinstance(result, dict)
        assert "prompt_tokens" in result
        assert "response_tokens" in result
        assert "trajectory_reward" in result

    @pytest.mark.asyncio
    async def test_token_mode_missing_assistant_raises(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](
            tokenizer=MagicMock(), n_parallel_agents=1, max_steps=2, server_addresses=["addr"],
            chat_parser=MagicMock(), max_prompt_length=1024, max_model_len=16384,
            env_args={"trajectory_timeout": 7200},
        )
        engine.router.chat = AsyncMock(return_value="model response")
        engine.tokenizer.encode.return_value = [1, 2, 3]
        engine.chat_parser.parse.return_value = "parsed prompt"

        agent = MagicMock()
        agent.chat_completions = [{"role": "user", "content": "hi"}]
        agent.reset = MagicMock()
        agent.update_from_env = MagicMock()
        agent.update_from_model = MagicMock(return_value=MagicMock(action="step"))
        agent.trajectory = MagicMock()
        env = MagicMock()
        env.reset = MagicMock(return_value=("obs", {"info": "x"}))
        env.step = MagicMock(return_value=("next_obs", 0.0, True, {}))
        env.close = MagicMock()
        env.is_multithread_safe.return_value = True
        del env.compute_final_reward

        engine.envs = [env]
        engine.agents = [agent]

        with patch("aura.runner.agent_engine_wrapper.rllm.agent_execution_engine.get_recent_assistant_user_messages",
                   return_value=(None, MagicMock())):
            loop = asyncio.get_event_loop()
            with patch.object(loop, "run_in_executor") as mock_run:
                def run_side_effect(executor, func, *args, **kwargs):
                    if func == env.reset:
                        result = ("obs", {"info": "x"})
                    elif func == env.step:
                        result = ("next_obs", 0.0, True, {})
                    elif func == env.close:
                        result = None
                    else:
                        result = None
                    fut = asyncio.Future()
                    fut.set_result(result)
                    return fut
                mock_run.side_effect = run_side_effect
                with pytest.raises(RuntimeError, match="Assistant messages is none"):
                    await engine.run_agent_trajectory_async(idx=0, application_id="app_tok_err", mode="Token")


class TestRunAgentTrajectoryAsyncStepMode:
    @pytest.mark.asyncio
    async def test_step_mode(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](
            tokenizer=MagicMock(), n_parallel_agents=1, max_steps=2, server_addresses=["addr"],
            chat_parser=MagicMock(), max_prompt_length=1024, max_model_len=16384,
            env_args={"trajectory_timeout": 7200},
        )
        engine.router.chat = AsyncMock(return_value="model response")
        engine.tokenizer.encode.return_value = [1, 2, 3]
        engine.chat_parser.parse.return_value = "parsed prompt"

        agent = MagicMock()
        agent.chat_completions = [{"role": "user", "content": "hi"}]
        agent.reset = MagicMock()
        agent.update_from_env = MagicMock()
        agent.update_from_model = MagicMock(return_value=MagicMock(action="step"))
        agent.trajectory = MagicMock()
        env = MagicMock()
        env.reset = MagicMock(return_value=("obs", {"info": "x"}))
        env.step = MagicMock(return_value=("next_obs", 0.0, True, {}))
        env.close = MagicMock()
        env.is_multithread_safe.return_value = True
        del env.compute_final_reward

        engine.envs = [env]
        engine.agents = [agent]

        loop = asyncio.get_event_loop()
        with patch.object(loop, "run_in_executor") as mock_run:
            def run_side_effect(executor, func, *args, **kwargs):
                if func == env.reset:
                    result = ("obs", {"info": "x"})
                elif func == env.step:
                    result = ("next_obs", 0.0, True, {})
                elif func == env.close:
                    result = None
                else:
                    result = None
                fut = asyncio.Future()
                fut.set_result(result)
                return fut
            mock_run.side_effect = run_side_effect
            result = await engine.run_agent_trajectory_async(idx=0, application_id="app_step", mode="Step")
        assert isinstance(result, dict)
        assert "steps" in result
        assert "trajectory" in result


class TestRunAgentTrajectoryAsyncConversationMode:
    @pytest.mark.asyncio
    async def test_conversation_mode(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](
            tokenizer=MagicMock(), n_parallel_agents=1, max_steps=2, server_addresses=["addr"],
            chat_parser=MagicMock(), max_prompt_length=1024, max_model_len=16384,
            env_args={"trajectory_timeout": 7200},
        )
        engine.router.chat = AsyncMock(return_value="model response")
        engine.tokenizer.encode.return_value = [1, 2, 3]
        engine.chat_parser.parse.return_value = "parsed prompt"

        agent = MagicMock()
        agent.chat_completions = [{"role": "user", "content": "hi"}]
        agent.reset = MagicMock()
        agent.update_from_env = MagicMock()
        agent.update_from_model = MagicMock(return_value=MagicMock(action="step"))
        agent.trajectory = MagicMock()
        env = MagicMock()
        env.reset = MagicMock(return_value=("obs", {"info": "x"}))
        env.step = MagicMock(return_value=("next_obs", 0.0, True, {}))
        env.close = MagicMock()
        env.is_multithread_safe.return_value = True
        del env.compute_final_reward

        engine.envs = [env]
        engine.agents = [agent]

        loop = asyncio.get_event_loop()
        with patch.object(loop, "run_in_executor") as mock_run:
            def run_side_effect(executor, func, *args, **kwargs):
                if func == env.reset:
                    result = ("obs", {"info": "x"})
                elif func == env.step:
                    result = ("next_obs", 0.0, True, {})
                elif func == env.close:
                    result = None
                else:
                    result = None
                fut = asyncio.Future()
                fut.set_result(result)
                return fut
            mock_run.side_effect = run_side_effect
            result = await engine.run_agent_trajectory_async(idx=0, application_id="app_conv", mode="Conversation")
        assert result == agent.chat_completions


class TestRunAgentTrajectoryAsyncStreamQueue:
    @pytest.mark.asyncio
    async def test_with_stream_queue(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](
            tokenizer=MagicMock(), n_parallel_agents=1, max_steps=2, server_addresses=["addr"],
            chat_parser=MagicMock(), max_prompt_length=1024, max_model_len=16384,
            env_args={"trajectory_timeout": 7200},
        )
        engine.router.chat = AsyncMock(return_value="model response")
        engine.tokenizer.encode.return_value = [1, 2, 3]
        engine.chat_parser.parse.return_value = "parsed prompt"

        agent = MagicMock()
        agent.chat_completions = [{"role": "user", "content": "hi"}]
        agent.reset = MagicMock()
        agent.update_from_env = MagicMock()
        agent.update_from_model = MagicMock(return_value=MagicMock(action="step"))
        agent.trajectory = MagicMock()
        env = MagicMock()
        env.reset = MagicMock(return_value=("obs", {"info": "x"}))
        env.step = MagicMock(return_value=("next_obs", 0.0, True, {}))
        env.close = MagicMock()
        env.is_multithread_safe.return_value = True
        del env.compute_final_reward

        engine.envs = [env]
        engine.agents = [agent]

        stream_queue = MagicMock()
        loop = asyncio.get_event_loop()
        with patch.object(loop, "run_in_executor") as mock_run:
            def run_side_effect(executor, func, *args, **kwargs):
                if func == env.reset:
                    result = ("obs", {"info": "x"})
                elif func == env.step:
                    result = ("next_obs", 0.0, True, {})
                elif func == env.close:
                    result = None
                else:
                    result = None
                fut = asyncio.Future()
                fut.set_result(result)
                return fut
            mock_run.side_effect = run_side_effect
            trajectory = await engine.run_agent_trajectory_async(
                idx=0, application_id="app_str", mode="Text", stream_queue=stream_queue
            )
        assert trajectory is not None
        stream_queue.put_nowait.assert_called()


class TestRunAgentTrajectoryAsyncTokenInTokenOut:
    @pytest.mark.asyncio
    async def test_token_in_token_out_path(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](
            tokenizer=MagicMock(), n_parallel_agents=1, max_steps=2, server_addresses=["addr"],
            chat_parser=MagicMock(), max_prompt_length=1024, max_model_len=16384,
            env_args={"trajectory_timeout": 7200}, token_in_token_out=True,
            sampling_params={"max_tokens": 512},
        )
        engine.tokenizer.encode.return_value = [1, 2, 3]
        engine.chat_parser.parse.return_value = "parsed prompt"

        async def fake_model_response(*args, **kwargs):
            return {
                "message": "model response",
                "response_tokens": [10, 11],
                "prompt_tokens": [1, 2, 3],
                "logprobs": [0.1, 0.2],
            }
        engine.get_model_response = fake_model_response

        agent = MagicMock()
        agent.chat_completions = [{"role": "user", "content": "hi"}]
        agent.reset = MagicMock()
        agent.update_from_env = MagicMock()
        agent.update_from_model = MagicMock(return_value=MagicMock(action="step"))
        agent.trajectory = MagicMock()
        env = MagicMock()
        env.reset = MagicMock(return_value=("obs", {"info": "x"}))
        env.step = MagicMock(return_value=("next_obs", 0.0, True, {}))
        env.close = MagicMock()
        env.is_multithread_safe.return_value = True
        del env.compute_final_reward

        engine.envs = [env]
        engine.agents = [agent]

        loop = asyncio.get_event_loop()
        with patch.object(loop, "run_in_executor") as mock_run:
            def run_side_effect(executor, func, *args, **kwargs):
                if func == env.reset:
                    result = ("obs", {"info": "x"})
                elif func == env.step:
                    result = ("next_obs", 0.0, True, {})
                elif func == env.close:
                    result = None
                else:
                    result = None
                fut = asyncio.Future()
                fut.set_result(result)
                return fut
            mock_run.side_effect = run_side_effect
            result = await engine.run_agent_trajectory_async(idx=0, application_id="app_tit", mode="Token")
        assert isinstance(result, dict)
        assert "logprobs" in result

    @pytest.mark.asyncio
    async def test_error_string_is_masked_and_terminates(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](
            tokenizer=MagicMock(), n_parallel_agents=1, max_steps=2, server_addresses=["addr"],
            chat_parser=MagicMock(), max_prompt_length=1024, max_model_len=16384,
            env_args={"trajectory_timeout": 7200}, token_in_token_out=True,
            sampling_params={"max_tokens": 512},
        )
        fake_torch = fake_engine_env["fake_torch"]
        fake_torch.tensor.side_effect = lambda data, dtype=None: {"data": list(data), "dtype": dtype}
        engine.chat_parser.parse.return_value = "parsed prompt"
        engine.tokenizer.encode.side_effect = (
            lambda text, add_special_tokens=False: [9, 10] if str(text).startswith("Error:") else [1, 2, 3]
        )

        async def fake_model_response(*args, **kwargs):
            return "Error: Rate limit reached"
        engine.get_model_response = fake_model_response

        agent = MagicMock()
        agent.chat_completions = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        agent.reset = MagicMock()
        agent.update_from_env = MagicMock()
        agent.update_from_model = MagicMock()
        agent.trajectory = MagicMock()
        env = MagicMock()
        env.reset = MagicMock(return_value=("obs", {"info": "x"}))
        env.step = MagicMock()
        env.close = MagicMock()
        env.is_multithread_safe.return_value = True
        env.sample_id = 0
        env.idx = 0
        env.task = {"prompt_id": 0}
        del env.compute_final_reward

        engine.envs = [env]
        engine.agents = [agent]

        loop = asyncio.get_event_loop()
        with patch.object(loop, "run_in_executor") as mock_run:
            def run_side_effect(executor, func, *args, **kwargs):
                result = ("obs", {"info": "x"}) if func == env.reset else None
                fut = asyncio.Future()
                fut.set_result(result)
                return fut
            mock_run.side_effect = run_side_effect
            result = await engine.run_agent_trajectory_async(idx=0, application_id="app_tit_err", mode="Token")

        assert result["response_tokens"]["data"] == [9, 10]
        assert result["response_masks"]["data"] == [0, 0]
        assert result["logprobs"]["data"] == [0.0, 0.0]
        assert agent.update_from_model.call_count == 0
        env.step.assert_not_called()
        assert agent.trajectory.termination_reason == "MODEL_ERROR"


class TestRunAgentTrajectoryAsyncTimeout:
    @pytest.mark.asyncio
    async def test_timeout_termination(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](
            tokenizer=MagicMock(), n_parallel_agents=1, max_steps=10, server_addresses=["addr"],
            chat_parser=MagicMock(), max_prompt_length=1024, max_model_len=16384,
            env_args={"trajectory_timeout": 1},
        )
        engine.router.chat = AsyncMock(return_value="model response")
        engine.tokenizer.encode.return_value = [1, 2, 3]
        engine.chat_parser.parse.return_value = "parsed prompt"

        agent = MagicMock()
        agent.chat_completions = [{"role": "user", "content": "hi"}]
        agent.reset = MagicMock()
        agent.update_from_env = MagicMock()
        agent.update_from_model = MagicMock(return_value=MagicMock(action="step"))
        agent.trajectory = MagicMock()
        env = MagicMock()
        env.reset = MagicMock(return_value=("obs", {"info": "x"}))
        env.step = MagicMock(return_value=("next_obs", 0.0, False, {}))
        env.close = MagicMock()
        env.is_multithread_safe.return_value = True
        del env.compute_final_reward

        engine.envs = [env]
        engine.agents = [agent]

        loop = asyncio.get_event_loop()
        with patch.object(loop, "run_in_executor") as mock_run:
            def run_side_effect(executor, func, *args, **kwargs):
                if func == env.reset:
                    result = ("obs", {"info": "x"})
                elif func == env.step:
                    result = ("next_obs", 0.0, False, {})
                elif func == env.close:
                    result = None
                else:
                    result = None
                fut = asyncio.Future()
                fut.set_result(result)
                return fut
            mock_run.side_effect = run_side_effect

            call_times = [0, 0.1, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
            def time_side_effect():
                for t in call_times:
                    yield t
                while True:
                    yield 1000.0
            time_gen = time_side_effect()
            with patch.object(fake_engine_env["fake_time"], "time", side_effect=time_gen):
                trajectory = await engine.run_agent_trajectory_async(idx=0, application_id="app_timeout", mode="Text")
        assert trajectory.termination_reason == "TIMEOUT"


class TestRunAgentTrajectoryAsyncMaxSteps:
    @pytest.mark.asyncio
    async def test_max_steps_termination(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](
            tokenizer=MagicMock(), n_parallel_agents=1, max_steps=2, server_addresses=["addr"],
            chat_parser=MagicMock(), max_prompt_length=1024, max_model_len=16384,
            env_args={"trajectory_timeout": 7200},
        )
        engine.router.chat = AsyncMock(return_value="model response")
        engine.tokenizer.encode.return_value = [1, 2, 3]
        engine.chat_parser.parse.return_value = "parsed prompt"

        agent = MagicMock()
        agent.chat_completions = [{"role": "user", "content": "hi"}]
        agent.reset = MagicMock()
        agent.update_from_env = MagicMock()
        agent.update_from_model = MagicMock(return_value=MagicMock(action="step"))
        agent.trajectory = MagicMock()
        env = MagicMock()
        env.reset = MagicMock(return_value=("obs", {"info": "x"}))
        env.step = MagicMock(return_value=("next_obs", 0.0, False, {}))
        env.close = MagicMock()
        env.is_multithread_safe.return_value = True
        del env.compute_final_reward

        engine.envs = [env]
        engine.agents = [agent]

        loop = asyncio.get_event_loop()
        with patch.object(loop, "run_in_executor") as mock_run:
            def run_side_effect(executor, func, *args, **kwargs):
                if func == env.reset:
                    result = ("obs", {"info": "x"})
                elif func == env.step:
                    result = ("next_obs", 0.0, False, {})
                elif func == env.close:
                    result = None
                else:
                    result = None
                fut = asyncio.Future()
                fut.set_result(result)
                return fut
            mock_run.side_effect = run_side_effect
            trajectory = await engine.run_agent_trajectory_async(idx=0, application_id="app_maxsteps", mode="Text")
        assert trajectory.termination_reason == "MAX_STEPS"


class TestOverlongFilter:
    @pytest.mark.asyncio
    async def test_overlong_filter_masks_out(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](
            tokenizer=MagicMock(), n_parallel_agents=1, max_steps=1, server_addresses=["addr"],
            chat_parser=MagicMock(), max_prompt_length=1024, max_model_len=16384,
            env_args={"trajectory_timeout": 7200}, overlong_filter=True,
        )
        engine.router.chat = AsyncMock(return_value="model response")
        engine.tokenizer.encode.return_value = [1, 2, 3]
        engine.chat_parser.parse.return_value = "parsed prompt"

        agent = MagicMock()
        agent.chat_completions = [{"role": "user", "content": "hi"}]
        agent.reset = MagicMock()
        agent.update_from_env = MagicMock()
        agent.update_from_model = MagicMock(return_value=MagicMock(action="step"))
        agent.trajectory = MagicMock()
        env = MagicMock()
        env.reset = MagicMock(return_value=("obs", {"info": "x"}))
        env.step = MagicMock(return_value=("next_obs", 0.0, False, {}))
        env.close = MagicMock()
        env.is_multithread_safe.return_value = True
        del env.compute_final_reward

        engine.envs = [env]
        engine.agents = [agent]

        loop = asyncio.get_event_loop()
        with patch.object(loop, "run_in_executor") as mock_run:
            def run_side_effect(executor, func, *args, **kwargs):
                if func == env.reset:
                    result = ("obs", {"info": "x"})
                elif func == env.step:
                    result = ("next_obs", 0.0, False, {})
                elif func == env.close:
                    result = None
                else:
                    result = None
                fut = asyncio.Future()
                fut.set_result(result)
                return fut
            mock_run.side_effect = run_side_effect
            trajectory = await engine.run_agent_trajectory_async(idx=0, application_id="app_of", mode="Text")
        assert trajectory is not None


class TestComputeFinalReward:
    @pytest.mark.asyncio
    async def test_compute_final_reward_called(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](
            tokenizer=MagicMock(), n_parallel_agents=1, max_steps=2, server_addresses=["addr"],
            chat_parser=MagicMock(), max_prompt_length=1024, max_model_len=16384,
            env_args={"trajectory_timeout": 7200},
        )
        engine.router.chat = AsyncMock(return_value="model response")
        engine.tokenizer.encode.return_value = [1, 2, 3]
        engine.chat_parser.parse.return_value = "parsed prompt"

        agent = MagicMock()
        agent.chat_completions = [{"role": "user", "content": "hi"}]
        agent.reset = MagicMock()
        agent.update_from_env = MagicMock()
        agent.update_from_model = MagicMock(return_value=MagicMock(action="step"))
        agent.trajectory = MagicMock()
        env = MagicMock()
        env.reset = MagicMock(return_value=("obs", {"info": "x"}))
        env.step = MagicMock(return_value=("next_obs", 0.0, True, {}))
        env.close = MagicMock()
        env.is_multithread_safe.return_value = True
        cfr_mock = MagicMock(return_value=0.8)
        env.compute_final_reward = cfr_mock

        engine.envs = [env]
        engine.agents = [agent]

        loop = asyncio.get_event_loop()
        with patch.object(loop, "run_in_executor") as mock_run:
            def run_side_effect(executor, func, *args, **kwargs):
                if func == env.reset:
                    result = ("obs", {"info": "x"})
                elif func == env.step:
                    result = ("next_obs", 0.0, True, {})
                elif func == env.close:
                    result = None
                elif func == cfr_mock:
                    result = 0.8
                else:
                    result = None
                fut = asyncio.Future()
                fut.set_result(result)
                return fut
            mock_run.side_effect = run_side_effect
            trajectory = await engine.run_agent_trajectory_async(idx=0, application_id="app_cfr", mode="Text")
        mock_run.assert_any_call(engine.executor, cfr_mock)
        assert trajectory is not None


class TestEpisodeInteraction:
    @pytest.mark.asyncio
    async def test_episode_not_none(self, fake_engine_env):
        episode_mock = MagicMock()
        engine = fake_engine_env["AgentExecutionEngine"](
            tokenizer=MagicMock(), n_parallel_agents=1, max_steps=2, server_addresses=["addr"],
            chat_parser=MagicMock(), max_prompt_length=1024, max_model_len=16384,
            env_args={"trajectory_timeout": 7200},
        )
        engine.episode = episode_mock
        engine.router.chat = AsyncMock(return_value="model response")
        engine.tokenizer.encode.return_value = [1, 2, 3]
        engine.chat_parser.parse.return_value = "parsed prompt"

        agent = MagicMock()
        agent.chat_completions = [{"role": "user", "content": "hi"}]
        agent.reset = MagicMock()
        agent.update_from_env = MagicMock()
        agent.update_from_model = MagicMock(return_value=MagicMock(action="step"))
        agent.trajectory = MagicMock()
        env = MagicMock()
        env.reset = MagicMock(return_value=("obs", {"info": "x"}))
        env.step = MagicMock(return_value=("next_obs", 0.0, True, {}))
        env.close = MagicMock()
        env.is_multithread_safe.return_value = True
        del env.compute_final_reward

        engine.envs = [env]
        engine.agents = [agent]

        loop = asyncio.get_event_loop()
        with patch.object(loop, "run_in_executor") as mock_run:
            def run_side_effect(executor, func, *args, **kwargs):
                if func == env.reset:
                    result = ("obs", {"info": "x"})
                elif func == env.step:
                    result = ("next_obs", 0.0, True, {})
                elif func == env.close:
                    result = None
                else:
                    result = None
                fut = asyncio.Future()
                fut.set_result(result)
                return fut
            mock_run.side_effect = run_side_effect
            trajectory = await engine.run_agent_trajectory_async(idx=0, application_id="app_ep", mode="Text")
        episode_mock.set_termination_reason.remote.assert_called()
        episode_mock.add_trajectory.remote.assert_called()


class TestSimplifyThinkContent:
    @pytest.mark.asyncio
    async def test_simplify_think_content_active(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](
            tokenizer=MagicMock(), n_parallel_agents=1, max_steps=2, server_addresses=["addr"],
            chat_parser=MagicMock(), simplify_think_content=True,
            max_prompt_length=1024, max_model_len=16384, env_args={"trajectory_timeout": 7200},
        )
        engine.router.chat = AsyncMock(return_value="model response")
        engine.tokenizer.encode.return_value = [1, 2, 3]
        engine.chat_parser.parse.return_value = "parsed prompt"

        agent = MagicMock()
        agent.chat_completions = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "some thinking"},
        ]
        agent.reset = MagicMock()
        agent.update_from_env = MagicMock()
        agent.update_from_model = MagicMock(return_value=MagicMock(action="step"))
        agent.trajectory = MagicMock()
        env = MagicMock()
        env.reset = MagicMock(return_value=("obs", {"info": "x"}))
        env.step = MagicMock(return_value=("next_obs", 0.0, True, {}))
        env.close = MagicMock()
        env.is_multithread_safe.return_value = True
        del env.compute_final_reward

        engine.envs = [env]
        engine.agents = [agent]

        loop = asyncio.get_event_loop()
        with patch.object(loop, "run_in_executor") as mock_run:
            def run_side_effect(executor, func, *args, **kwargs):
                if func == env.reset:
                    result = ("obs", {"info": "x"})
                elif func == env.step:
                    result = ("next_obs", 0.0, True, {})
                elif func == env.close:
                    result = None
                else:
                    result = None
                fut = asyncio.Future()
                fut.set_result(result)
                return fut
            mock_run.side_effect = run_side_effect
            trajectory = await engine.run_agent_trajectory_async(idx=0, application_id="app_stc", mode="Text")
        assert trajectory is not None


# ---------------------------------------------------------------------------
# PR2: AgentExecutionEngine tree-search helpers
# ---------------------------------------------------------------------------

class TestBeamTreeHelpers:
    def test_beam_defaults_and_mutable_fields(self, fake_engine_env):
        Beam = fake_engine_env["mod"].Beam
        first = Beam()
        second = Beam()
        first.response_tokens.append(1)
        first.response_masks.append("m")

        assert first.steps_length == 0
        assert first.cum_reward == 0.0
        assert first.done is False
        assert first.collect_reward is False
        assert first.is_last_step is True
        assert first.metadata == {}
        assert second.response_tokens == []
        assert second.response_masks == []
        assert second.metadata == {}

    def test_recover_initial_message_with_formatter(self, fake_engine_env):
        mod = fake_engine_env["mod"]
        memory = MagicMock()
        formatter = MagicMock(return_value=[{"role": "user", "content": "fmt-msg"}])
        agent = types.SimpleNamespace(_format_observation_as_messages=formatter, memory=memory)
        env = types.SimpleNamespace(task={"question": "Q"})
        info = {"max_steps": 5}

        assert mod._recover_initial_user_message(agent, env, observation=None, info=info) is True
        formatter.assert_called_once()
        assert formatter.call_args.kwargs["info"] == info
        memory.add_message.assert_called_once_with(
            [{"role": "user", "content": "fmt-msg"}],
            metadata=[{"reward": 0.0}],
        )

    def test_recover_initial_message_fallback_sources(self, fake_engine_env):
        mod = fake_engine_env["mod"]
        memory = MagicMock()
        agent = types.SimpleNamespace(memory=memory)
        env = types.SimpleNamespace(
            task={
                "problem": "solve this",
                "root_url": "http://example.com",
                "initial_observation": "from-task",
                "extra_args": {"query": "from-extra"},
            }
        )

        assert mod._recover_initial_user_message(agent, env, observation=None, info={}) is True
        content = memory.add_message.call_args[0][0][0]["content"]
        assert "Question: solve this" in content
        assert "URL: http://example.com" in content
        assert "Observation: from-task" in content

    def test_recover_initial_message_false_without_memory_or_question(self, fake_engine_env):
        mod = fake_engine_env["mod"]
        assert mod._recover_initial_user_message(types.SimpleNamespace(), types.SimpleNamespace(task={"question": "Q"}), None, {}) is False

        memory = MagicMock()
        agent = types.SimpleNamespace(memory=memory)
        assert mod._recover_initial_user_message(agent, types.SimpleNamespace(task={}), None, {}) is False
        memory.add_message.assert_not_called()

    def test_tree_snapshot_support_validation(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](tokenizer="t", server_addresses=["addr"])
        with pytest.raises(TypeError, match="snapshot\\(\\) for tree mode"):
            engine._validate_tree_snapshot_support(types.SimpleNamespace(), types.SimpleNamespace(snapshot=lambda: None))

    @pytest.mark.asyncio
    async def test_dynamic_beam_child_uses_metadata(self, fake_engine_env):
        engine = fake_engine_env["AgentExecutionEngine"](
            tokenizer=MagicMock(), server_addresses=["addr"], chat_parser=MagicMock(), max_steps=1,
            env_args={"trajectory_timeout": 7200},
        )
        engine._beam_model_response = AsyncMock(return_value="model response")

        def make_agent():
            agent = MagicMock()
            agent.chat_completions = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
            agent.update_from_model.return_value = types.SimpleNamespace(action="click")
            agent.get_current_state.return_value = types.SimpleNamespace(info={}, reward=0.0, done=False, step_id=None)
            agent.snapshot.side_effect = make_agent
            return agent

        def make_env():
            metadata = {"golden_path_configured": True, "on_golden_path": True, "clicked_button": "Search", "reached_golden_path_end": True}
            env = MagicMock()
            env.task = {"prompt_id": 0}
            env.reset.return_value = "obs", {}
            env.step.return_value = "next", 1.0, False, {"metadata": metadata}
            env.snapshot.side_effect = make_env
            return env

        loop = asyncio.get_event_loop()
        with patch.object(loop, "run_in_executor") as mock_run:
            def run_side_effect(executor, func, *args, **kwargs):
                result = func(*args)
                fut = asyncio.Future()
                fut.set_result(result)
                return fut
            mock_run.side_effect = run_side_effect
            beams = await engine.run_episode_with_dynamic_beam_search(
                make_agent(), make_env(), "app_tree", {}, beam_size=1, per_beam_expand=1,
            )

        assert len(beams) == 1
        assert beams[0].metadata["on_golden_path"] is True
        assert beams[0].metadata["clicked_button"] == "Search"
        assert not hasattr(beams[0], "on_golden_path")
        assert beams[0].termination_reason == "GOLDEN_PATH_COMPLETE"

    @pytest.mark.asyncio
    async def test_convert_beam_to_trajectory(self, fake_engine_env):
        mod = fake_engine_env["mod"]
        reward_fn = MagicMock()
        engine = fake_engine_env["AgentExecutionEngine"](
            tokenizer=MagicMock(), server_addresses=["addr"], chat_parser=MagicMock(), compute_trajectory_reward_fn=reward_fn,
        )
        trajectory = MagicMock()
        trajectory.steps = []
        trajectory.to_info_dict.return_value = {}
        agent = types.SimpleNamespace(
            trajectory=trajectory,
            chat_completions=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        )
        env = types.SimpleNamespace(task={"prompt_id": 7}, idx=3, close=MagicMock())
        beam = mod.Beam(
            agent=agent, env=env, beam_id="beam-1", final_reward=1.0,
            termination_reason="ENV_DONE", collect_reward=True,
        )

        loop = asyncio.get_event_loop()
        with patch.object(loop, "run_in_executor") as mock_run:
            fut = asyncio.Future()
            fut.set_result(None)
            mock_run.return_value = fut
            result = await engine.convert_beam_to_trajectory(
                beam,
                idx="0_0",
                mode="Text",
                trajectory_generation_method="tree",
            )

        assert result is trajectory
        assert trajectory.prompt_id == "7"
        assert trajectory.idx == 3
        assert trajectory.trajectory_generation_method == "tree"
        assert trajectory.collect_reward is True
        reward_fn.assert_called_once_with(trajectory)


class TestBeamInitKwargs:
    def test_default_and_custom_kwargs(self, fake_engine_env):
        engine_cls = fake_engine_env["AgentExecutionEngine"]
        default_engine = engine_cls(tokenizer="t", server_addresses=["addr"], n_parallel_agents=1)
        custom_engine = engine_cls(
            tokenizer="t",
            server_addresses=["addr"],
            n_parallel_agents=1,
            trajectory_generation_method="tree",
            beam_size=8,
            per_beam_expand=4,
        )

        assert default_engine.trajectory_generation_method == "default"
        assert default_engine.beam_size == 3
        assert default_engine.per_beam_expand == 2
        assert custom_engine.trajectory_generation_method == "tree"
        assert custom_engine.beam_size == 8
        assert custom_engine.per_beam_expand == 4

    def test_invalid_trajectory_generation_method_raises(self, fake_engine_env):
        engine_cls = fake_engine_env["AgentExecutionEngine"]
        with pytest.raises(ValueError, match="trajectory_generation_method"):
            engine_cls(
                tokenizer="t",
                server_addresses=["addr"],
                n_parallel_agents=1,
                trajectory_generation_method="invalid",
            )
