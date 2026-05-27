#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

AgentSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

          http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import sys
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


class TestTrainExecutor:
    """
    Tests for TrainExecutor class.

    Covers:
      - Initialization
      - _run_method behavior (None, blocking, non-blocking)
      - _run_train_and_rollout execution
      - fit success and exception handling
    """

    @classmethod
    def setup_class(cls):
        """
        Patch sys.modules and create mocks for dependencies before importing TrainExecutor.
        """
        # Ray mocks
        cls.mock_ray = MagicMock()
        cls.mock_ray.util = MagicMock()
        cls.mock_ray.util.scheduling_strategies = MagicMock()
        cls.mock_ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy = MagicMock(return_value="strategy")
        cls.mock_ray.get_runtime_context.return_value = MagicMock(node_id="node123")

        # OmegaConf mock
        cls.mock_omega = MagicMock()
        cls.mock_omega.create.side_effect = lambda x: x

        # Executor module mock
        cls.mock_executor_module = MagicMock()
        cls.mock_executor_module.public_api = lambda *a, **k: (lambda f: f)
        cls.mock_executor_module.Executor = type("Executor", (), {"__init__": MagicMock()})

        # Logger mock
        cls.mock_logger = MagicMock()
        mock_loggers_module = MagicMock()
        mock_loggers_module.Loggers.return_value.get_logger.return_value = cls.mock_logger

        # Patch sys.modules with all required mocks
        patch_dict = {
            "ray": cls.mock_ray,
            "ray.util": cls.mock_ray.util,
            "ray.util.scheduling_strategies": cls.mock_ray.util.scheduling_strategies,
            "omegaconf": MagicMock(OmegaConf=cls.mock_omega),
            "aura.base.execution.executor": cls.mock_executor_module,
            "aura.base.log.loggers": mock_loggers_module,
            "aura.trainer.rollout.rollout_main": MagicMock(),
            "aura.trainer.train_adapter.mindspeed_rl.hybrid_policy.train_service": MagicMock(),
            "aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train.train_service": MagicMock(),
            "aura.trainer.train_adapter.verl.full_async.train_main": MagicMock(),
            "aura.trainer.train_adapter.verl.hybrid.train_main": MagicMock(),
            "aura.trainer.train_adapter.omni_rl.hybrid.train_main": MagicMock(),
        }
        cls.patcher = patch.dict(sys.modules, patch_dict)
        cls.patcher.start()

        # Import target module inside patched context
        from aura.trainer.train_executor import TrainExecutor
        cls.TrainExecutor = TrainExecutor

    @classmethod
    def teardown_class(cls):
        """
        Stop sys.modules patch.
        """
        cls.patcher.stop()

    @pytest.fixture
    def executor(self):
        """
        Fixture to create a TrainExecutor instance.
        """
        return self.TrainExecutor(
            work_mode="mode",
            train_engine="engine",
            train_config={"a": 1},
            rollout_config={"b": 2},
            agent_service="agent",
            infer_service="infer",
        )

    def test_init(self, executor):
        """
        Verify initialization calls Executor.__init__ and sets attributes correctly.
        """
        self.mock_executor_module.Executor.__init__.assert_called_once()
        assert executor.work_mode == "mode"
        assert executor.train_engine == "engine"
        self.mock_omega.create.assert_any_call({"a": 1})
        self.mock_omega.create.assert_any_call({"b": 2})
        self.mock_logger.info.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_method_none(self, executor):
        """
        _run_method with None logs an error.
        """
        await executor._run_method(None, True)
        self.mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_method_non_blocking(self, executor):
        """
        _run_method executes non-blocking futures correctly.
        """
        fake_future = AsyncMock()
        start_mock = MagicMock()
        start_mock.options.return_value.remote.return_value = fake_future

        await executor._run_method(start_mock, False, x=1)
        start_mock.options.assert_called_once()
        start_mock.options.return_value.remote.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_method_blocking(self, executor):
        """
        _run_method executes blocking futures correctly.
        """
        fake_future = asyncio.Future()
        fake_future.set_result(None)
        start_mock = MagicMock()
        start_mock.options.return_value.remote.return_value = fake_future

        await executor._run_method(start_mock, True, x=1)
        start_mock.options.assert_called_once()
        start_mock.options.return_value.remote.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_train_rollout(self, executor):
        """
        _run_train_and_rollout executes the train method.
        """
        train_mock = MagicMock()
        # Mock get_train_method to return our train_mock
        with patch('aura.trainer.trainer_register.train_register.get_train_method') as mock_get_train_method:
            mock_get_train_method.return_value = train_mock
            executor._run_method = AsyncMock()
            await executor._run_train_and_rollout()
            mock_get_train_method.assert_called_once_with(train_engine=executor.train_engine, work_mode=executor.work_mode)
            executor._run_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_fit_success(self, executor):
        """
        fit() successfully calls _run_train_and_rollout.
        """
        executor._run_train_and_rollout = AsyncMock()
        await executor.fit()
        executor._run_train_and_rollout.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fit_exception(self, executor):
        """
        fit() logs exception when _run_train_and_rollout raises.
        """
        executor._run_train_and_rollout = AsyncMock(side_effect=Exception("boom"))
        with pytest.raises(Exception):
            await executor.fit()
        self.mock_logger.error.assert_called()
