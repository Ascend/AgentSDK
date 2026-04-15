#!/usr/bin/env python3
# coding=utf-8
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
import json
import sys
import types
import unittest
import importlib
from unittest.mock import patch, MagicMock, AsyncMock


# ============================================================
# Helper: run async coroutine in sync test
# ============================================================

def _build_fake_ray_modules():
    """
    Build a fake ray module tree to avoid real Ray dependency:
      ray
      ray.util
      ray.util.placement_group
    """
    fake_ray = types.ModuleType("ray")

    def _remote(obj=None, **kwargs):
        # Mock ray.remote decorator: identity function or no-op
        if obj is None:
            return lambda x: x
        return obj

    fake_ray.remote = _remote
    fake_ray.get = MagicMock()
    fake_ray.put = MagicMock()
    fake_ray.init = MagicMock()
    fake_ray.shutdown = MagicMock()
    fake_ray.is_initialized = MagicMock(return_value=True)

    fake_ray_util = types.ModuleType("ray.util")

    fake_pg = types.ModuleType("ray.util.placement_group")
    fake_pg.PlacementGroup = object
    fake_pg.placement_group = MagicMock()

    return {
        "ray": fake_ray,
        "ray.util": fake_ray_util,
        "ray.util.placement_group": fake_pg,
    }


def _build_fake_rllm_engine_wrapper_module(wrapper_cls):
    """
    AgentExecutor dynamically imports RLLMEngineWrapper inside __init__.
    This function creates a fake module with that class.
    """
    fake_mod = types.ModuleType(
        "agentic_rl.runner.agent_engine_wrapper.rllm.rllm_engine_wrapper"
    )
    fake_mod.RLLMEngineWrapper = wrapper_cls
    return fake_mod


def _reload_agent_executor_module():
    """
    Reload the agent_executor module after fakes are installed.
    """
    mod_name = "agentic_rl.runner.agent_service.agent_executor"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


# ============================================================
# Tests for AgentExecutor
# ============================================================

class TestAgentExecutor(unittest.IsolatedAsyncioTestCase):
    """Unit tests for the AgentExecutor class."""

    def setUp(self):
        """
        Set up isolated environment for each test:
        - Patch torch._pytree to avoid missing API.
        - Install fake Ray modules.
        - Install fake RLLMEngineWrapper module.
        - Reload the target module.
        - Create an AgentExecutor instance with test parameters.
        """
        # ------------------------------------------------------------
        # Patch torch pytree compatibility LOCALLY (no global pollution)
        # ------------------------------------------------------------
        import torch

        self._torch_patches = []

        if not hasattr(torch.utils, "_pytree"):
            p = patch.object(torch.utils, "_pytree", types.SimpleNamespace(), create=True)
            p.start()
            self._torch_patches.append(p)

        if not hasattr(torch.utils._pytree, "register_pytree_node"):
            p = patch.object(
                torch.utils._pytree,
                "register_pytree_node",
                lambda *args, **kwargs: None,
                create=True,
            )
            p.start()
            self._torch_patches.append(p)

        self.addCleanup(self._cleanup_torch_patches)

        # ---- fake ray modules ----
        self.ray_patch = patch.dict(sys.modules, _build_fake_ray_modules())
        self.ray_patch.start()
        self.addCleanup(self.ray_patch.stop)

        # ---- fake rllm engine wrapper module ----
        self.mock_wrapper_cls = MagicMock(name="RLLMEngineWrapper")
        self.mock_wrapper_instance = MagicMock(name="RLLMEngineWrapperInstance")
        self.mock_wrapper_cls.return_value = self.mock_wrapper_instance

        fake_rllm_mod = _build_fake_rllm_engine_wrapper_module(self.mock_wrapper_cls)

        self.rllm_patch = patch.dict(
            sys.modules,
            {
                "agentic_rl.runner.agent_engine_wrapper.rllm.rllm_engine_wrapper": fake_rllm_mod
            },
        )
        self.rllm_patch.start()
        self.addCleanup(self.rllm_patch.stop)

        # reload module to ensure it binds fake ray / fake wrapper import
        agent_executor_mod = _reload_agent_executor_module()
        self.AgentExecutor = agent_executor_mod.AgentExecutor

        # import these after reload to avoid old bindings
        from agentic_rl.runner.agent_engine_wrapper.base_engine_wrapper import AgentTask, Trajectory
        self.AgentTask = AgentTask
        self.Trajectory = Trajectory

        # Test parameters
        self.agent_engine = "rllm"
        self.agent_engine_kwargs = {"model": "test_model"}
        self.infer_service_params = {"endpoint": "test_endpoint"}
        self.trajectory_save_dir = "/tmp/test_trajectory.jsonl"
        self.resource_set = MagicMock()

        # Create executor instance
        self.executor = self.AgentExecutor(
            agent_engine=self.agent_engine,
            agent_engine_kwargs=self.agent_engine_kwargs,
            infer_service_params=self.infer_service_params,
            trajectory_save_dir=self.trajectory_save_dir,
            resource_set=self.resource_set,
        )

    def _cleanup_torch_patches(self):
        """Stop all torch patchers in reverse order."""
        for p in reversed(self._torch_patches):
            p.stop()

    def _setup_fake_engine_episode(self):
        """
        Mock the engine.episode.to_dict.remote() chain for episode saving tests.
        """
        mock_episode_actor = MagicMock()
        mock_episode_actor.to_dict.remote.return_value = "FAKE_OBJECT_REF"

        mock_engine = MagicMock()
        mock_engine.episode = mock_episode_actor
        self.mock_wrapper_instance.engine = mock_engine

        return mock_engine

    # ------------------------------------------------------------
    # Initialization tests
    # ------------------------------------------------------------
    def test_init_with_rllm_engine(self):
        """AgentExecutor initializes correctly with 'rllm' engine."""
        self.assertEqual(self.executor.agent_engine, "rllm")
        self.assertEqual(self.executor.agent_engine_kwargs, {"model": "test_model"})
        self.assertEqual(self.executor.infer_service_params, {"endpoint": "test_endpoint"})
        self.assertEqual(self.executor.trajectory_save_dir, "/tmp/test_trajectory.jsonl")
        self.assertIsNotNone(self.executor.agent_executor_wrapper)

        self.mock_wrapper_cls.assert_called_once_with(
            infer_service_params=self.infer_service_params,
            **self.agent_engine_kwargs,
        )

    def test_init_with_unsupported_engine(self):
        """Unsupported agent engine should raise ValueError."""
        with self.assertRaises(ValueError) as context:
            self.AgentExecutor(
                agent_engine="unsupported_engine",
                agent_engine_kwargs={},
                infer_service_params={},
                trajectory_save_dir="",
                resource_set=self.resource_set,
            )
        self.assertIn("unsupported_engine is not supported", str(context.exception))

    # ------------------------------------------------------------
    # Core method tests
    # ------------------------------------------------------------
    async def test_generate_trajectory(self):
        """generate_trajectory should forward to wrapper and return trajectory."""
        mock_task = MagicMock(spec=self.AgentTask)
        mock_traj = MagicMock(spec=self.Trajectory)

        self.mock_wrapper_instance.generate_trajectory = AsyncMock(return_value=mock_traj)

        result = await self.executor.generate_trajectory(
            mock_task, mode="Text", addresses=None
        )

        self.mock_wrapper_instance.generate_trajectory.assert_called_once_with(
            task=mock_task, mode="Text", addresses=None
        )
        self.assertEqual(result, mock_traj)

    async def test_generate_trajectories_returns_none(self):
        """generate_trajectories (list version) returns None (placeholder)."""
        mock_tasks = [MagicMock(spec=self.AgentTask)]
        result = await self.executor.generate_trajectories(mock_tasks)
        self.assertIsNone(result)

    async def test_cancel_request(self):
        """cancel_request forwards to wrapper.cancel_request."""
        mock_task = MagicMock(spec=self.AgentTask)

        self.mock_wrapper_instance.cancel_request = AsyncMock()
        await self.executor.cancel_request(mock_task)

        self.mock_wrapper_instance.cancel_request.assert_called_once_with(mock_task)

    async def test_clear_cache(self):
        """clear_cache forwards to wrapper.clear_cache."""
        self.mock_wrapper_instance.clear_cache = MagicMock()

        await self.executor.clear_cache()

        self.mock_wrapper_instance.clear_cache.assert_called_once()

    # ------------------------------------------------------------
    # Streaming trajectory generation tests
    # ------------------------------------------------------------
    async def test_stream_generate_trajectory(self):
        """Streaming generation yields JSON strings for each event and saves episode."""
        mock_task = MagicMock(spec=self.AgentTask)
        mock_traj = MagicMock(spec=self.Trajectory)

        mock_events = [
            {"type": "start", "data": "start_event"},
            {"type": "progress", "data": "progress_event"},
            {"type": "end", "data": "end_event"},
        ]

        async def mock_generate_trajectory(task, stream_queue, **kwargs):
            for ev in mock_events:
                stream_queue.put_nowait(ev)
            return mock_traj

        self.mock_wrapper_instance.generate_trajectory = AsyncMock(
            side_effect=mock_generate_trajectory
        )

        try:
            self._setup_fake_engine_episode()
        except Exception:
            raise

        import ray
        ray.get.return_value = {"episode_data": "test_data"}

        with patch(
            "agentic_rl.memory.episode.backend.json_episode_store.JsonEpisodeStore"
        ) as mock_store_cls:

            mock_store = MagicMock()
            mock_store_cls.return_value = mock_store

            with patch("agentic_rl.runner.agent_service.agent_executor.logger"):
                results = []
                async for item in self.executor.stream_generate_trajectory(mock_task):
                    results.append(item)

            expected = [json.dumps(ev, ensure_ascii=False) for ev in mock_events]
            self.assertEqual(results, expected)

            self.mock_wrapper_instance.generate_trajectory.assert_called_once()
            call_kwargs = self.mock_wrapper_instance.generate_trajectory.call_args.kwargs
            self.assertEqual(call_kwargs["task"], mock_task)
            self.assertIn("stream_queue", call_kwargs)

            mock_store_cls.assert_called_once_with(path=self.trajectory_save_dir)
            ray.get.assert_called_once_with("FAKE_OBJECT_REF")
            mock_store.store_episode.assert_called_once()

    async def test_stream_generate_trajectory_exception(self):
        """If wrapper raises an exception, it should propagate and not break streaming."""
        mock_task = MagicMock(spec=self.AgentTask)

        async def mock_generate_trajectory(task, stream_queue, **kwargs):
            stream_queue.put_nowait(None)
            raise RuntimeError("Test exception")

        self.mock_wrapper_instance.generate_trajectory = AsyncMock(
            side_effect=mock_generate_trajectory
        )

        with patch("agentic_rl.runner.agent_service.agent_executor.logger"):
            with self.assertRaises(RuntimeError) as ctx:
                async for _ in self.executor.stream_generate_trajectory(mock_task):
                    pass

        self.assertIn("Test exception", str(ctx.exception))

    async def test_stream_generate_trajectory_save_trajectory_no_events(self):
        """If no events are emitted, still save the episode with empty event string."""
        mock_task = MagicMock(spec=self.AgentTask)
        mock_traj = MagicMock(spec=self.Trajectory)

        async def mock_generate_trajectory(task, stream_queue, **kwargs):
            return mock_traj

        self.mock_wrapper_instance.generate_trajectory = AsyncMock(
            side_effect=mock_generate_trajectory
        )

        try:
            self._setup_fake_engine_episode()
        except Exception:
            raise

        import ray
        ray.get.return_value = {"episode_data": "test_data"}

        with patch(
            "agentic_rl.memory.episode.backend.json_episode_store.JsonEpisodeStore"
        ) as mock_store_cls:

            mock_store = MagicMock()
            mock_store_cls.return_value = mock_store

            with patch("agentic_rl.runner.agent_service.agent_executor.logger"):
                results = []
                async for item in self.executor.stream_generate_trajectory(mock_task):
                    results.append(item)

            self.assertEqual(results, [])

            mock_store_cls.assert_called_once_with(path=self.trajectory_save_dir)
            ray.get.assert_called_once_with("FAKE_OBJECT_REF")
            mock_store.store_episode.assert_called_once()

    async def test_stream_generate_trajectory_saves_empty_episode_string(self):
        """When no episode events, store_episode receives empty string."""
        mock_task = MagicMock(spec=self.AgentTask)
        mock_traj = MagicMock(spec=self.Trajectory)

        async def mock_generate_trajectory(task, stream_queue, **kwargs):
            return mock_traj

        self.mock_wrapper_instance.generate_trajectory = AsyncMock(
            side_effect=mock_generate_trajectory
        )

        try:
            self._setup_fake_engine_episode()
        except Exception:
            raise

        import ray
        ray.get.return_value = {"episode_data": "test_data"}

        with patch(
            "agentic_rl.memory.episode.backend.json_episode_store.JsonEpisodeStore"
        ) as mock_store_cls:

            mock_store = MagicMock()
            mock_store_cls.return_value = mock_store

            with patch("agentic_rl.runner.agent_service.agent_executor.logger"):
                async for _ in self.executor.stream_generate_trajectory(mock_task):
                    pass

            args, kwargs = mock_store.store_episode.call_args
            self.assertEqual(args[1], "")

if __name__ == "__main__":
    unittest.main()