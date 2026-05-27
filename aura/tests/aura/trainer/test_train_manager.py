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
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call


class MockExecutorManager:
    """Mock ExecutorManager class for testing."""
    def __init__(self):
        self.instance_dict = {}

    async def create_instance(self, **kwargs):
        """Mock create_instance method."""
        self.instance_dict[kwargs['name']] = kwargs


class TestTrainManager:
    """
    Tests for TrainManager class and get_or_create_train_manager function.
    """

    @pytest.mark.asyncio
    async def test_get_or_create_train_manager_exists(self):
        """
        Test get_or_create_train_manager returns existing actor if available.
        """
        mock_ray = MagicMock()
        mock_logger = MagicMock()
        mock_omega = MagicMock()
        mock_agentic_conf = MagicMock()
        mock_train_executor = MagicMock()
        mock_loggers = MagicMock()

        mock_loggers.Loggers.return_value.get_logger.return_value = mock_logger

        with patch.dict(sys.modules, {
            'ray': mock_ray,
            'omegaconf': MagicMock(),
            'aura.base.conf.conf': MagicMock(),
            'aura.base.execution.executor_manager': MagicMock(),
            'aura.base.log.loggers': mock_loggers,
            'aura.trainer.train_executor': MagicMock(),
        }):
            sys.modules['omegaconf'].OmegaConf = mock_omega
            sys.modules['aura.base.conf.conf'].AgenticRLConf = mock_agentic_conf
            sys.modules['aura.trainer.train_executor'].TrainExecutor = mock_train_executor

            existing_actor = MagicMock()
            mock_ray.get_actor.return_value = existing_actor

            from aura.trainer.train_manager import get_or_create_train_manager

            result = await get_or_create_train_manager()
            mock_ray.get_actor.assert_called_once_with("TrainManager")
            mock_logger.info.assert_not_called()
            assert result == existing_actor

    @pytest.mark.asyncio
    async def test_get_or_create_train_manager_new(self):
        """
        Test get_or_create_train_manager creates new actor if not found.
        """
        mock_ray = MagicMock()
        mock_logger = MagicMock()
        mock_omega = MagicMock()
        mock_agentic_conf = MagicMock()
        mock_train_executor = MagicMock()
        mock_loggers = MagicMock()

        mock_loggers.Loggers.return_value.get_logger.return_value = mock_logger

        with patch.dict(sys.modules, {
            'ray': mock_ray,
            'omegaconf': MagicMock(),
            'aura.base.conf.conf': MagicMock(),
            'aura.base.execution.executor_manager': MagicMock(),
            'aura.base.log.loggers': mock_loggers,
            'aura.trainer.train_executor': MagicMock(),
        }):
            sys.modules['omegaconf'].OmegaConf = mock_omega
            sys.modules['aura.base.conf.conf'].AgenticRLConf = mock_agentic_conf
            sys.modules['aura.trainer.train_executor'].TrainExecutor = mock_train_executor

            mock_ray.get_actor.side_effect = ValueError("Actor not found")

            mock_remote_class = MagicMock()
            mock_remote_instance = MagicMock()
            mock_setup_remote = AsyncMock()
            mock_remote_instance.setup.remote = mock_setup_remote
            mock_remote_class.options.return_value.remote.return_value = mock_remote_instance
            mock_ray.remote.return_value = mock_remote_class

            from aura.trainer.train_manager import get_or_create_train_manager

            result = await get_or_create_train_manager()
            mock_ray.get_actor.assert_called_once_with("TrainManager")
            mock_logger.info.assert_called_once_with(
                "Could not find actor TrainManager, creating a new one."
            )
            mock_setup_remote.assert_awaited_once()
            mock_remote_class.options.assert_called_once_with(name="TrainManager", lifetime="detached")
            assert result == mock_remote_instance

    @pytest.mark.asyncio
    async def test_get_or_create_train_manager_multiple_calls(self):
        """
        Test get_or_create_train_manager creates actor only once on multiple calls.
        """
        mock_ray = MagicMock()
        mock_logger = MagicMock()
        mock_omega = MagicMock()
        mock_agentic_conf = MagicMock()
        mock_train_executor = MagicMock()
        mock_loggers = MagicMock()

        mock_loggers.Loggers.return_value.get_logger.return_value = mock_logger

        with patch.dict(sys.modules, {
            'ray': mock_ray,
            'omegaconf': MagicMock(),
            'aura.base.conf.conf': MagicMock(),
            'aura.base.execution.executor_manager': MagicMock(),
            'aura.base.log.loggers': mock_loggers,
            'aura.trainer.train_executor': MagicMock(),
        }):
            sys.modules['omegaconf'].OmegaConf = mock_omega
            sys.modules['aura.base.conf.conf'].AgenticRLConf = mock_agentic_conf
            sys.modules['aura.trainer.train_executor'].TrainExecutor = mock_train_executor

            mock_ray.get_actor.side_effect = [ValueError("Actor not found"), MagicMock()]

            mock_remote_class = MagicMock()
            mock_remote_instance = MagicMock()
            mock_setup_remote = AsyncMock()
            mock_remote_instance.setup.remote = mock_setup_remote
            mock_remote_class.options.return_value.remote.return_value = mock_remote_instance
            mock_ray.remote.return_value = mock_remote_class

            from aura.trainer.train_manager import get_or_create_train_manager

            result1 = await get_or_create_train_manager()
            result2 = await get_or_create_train_manager()

            assert mock_ray.get_actor.call_count == 2
            assert mock_setup_remote.call_count == 1
            assert result1 == mock_remote_instance

    @pytest.mark.asyncio
    async def test_get_or_create_train_manager_with_different_actor_name(self):
        """
        Test get_or_create_train_manager with different actor name constant.
        """
        mock_ray = MagicMock()
        mock_logger = MagicMock()
        mock_omega = MagicMock()
        mock_agentic_conf = MagicMock()
        mock_train_executor = MagicMock()
        mock_loggers = MagicMock()

        mock_loggers.Loggers.return_value.get_logger.return_value = mock_logger

        with patch.dict(sys.modules, {
            'ray': mock_ray,
            'omegaconf': MagicMock(),
            'aura.base.conf.conf': MagicMock(),
            'aura.base.execution.executor_manager': MagicMock(),
            'aura.base.log.loggers': mock_loggers,
            'aura.trainer.train_executor': MagicMock(),
        }):
            sys.modules['omegaconf'].OmegaConf = mock_omega
            sys.modules['aura.base.conf.conf'].AgenticRLConf = mock_agentic_conf
            sys.modules['aura.trainer.train_executor'].TrainExecutor = mock_train_executor

            mock_ray.get_actor.side_effect = ValueError("Actor not found")

            mock_remote_class = MagicMock()
            mock_remote_instance = MagicMock()
            mock_setup_remote = AsyncMock()
            mock_remote_instance.setup.remote = mock_setup_remote
            mock_remote_class.options.return_value.remote.return_value = mock_remote_instance
            mock_ray.remote.return_value = mock_remote_class

            from aura.trainer.train_manager import get_or_create_train_manager

            await get_or_create_train_manager()

            mock_remote_class.options.assert_called_once_with(name="TrainManager", lifetime="detached")

    @pytest.mark.asyncio
    async def test_get_or_create_train_manager_ray_remote_called(self):
        """
        Test get_or_create_train_manager calls ray.remote with TrainManager.
        """
        mock_ray = MagicMock()
        mock_logger = MagicMock()
        mock_omega = MagicMock()
        mock_agentic_conf = MagicMock()
        mock_train_executor = MagicMock()
        mock_loggers = MagicMock()

        mock_loggers.Loggers.return_value.get_logger.return_value = mock_logger

        with patch.dict(sys.modules, {
            'ray': mock_ray,
            'omegaconf': MagicMock(),
            'aura.base.conf.conf': MagicMock(),
            'aura.base.execution.executor_manager': MagicMock(),
            'aura.base.log.loggers': mock_loggers,
            'aura.trainer.train_executor': MagicMock(),
        }):
            sys.modules['omegaconf'].OmegaConf = mock_omega
            sys.modules['aura.base.conf.conf'].AgenticRLConf = mock_agentic_conf
            sys.modules['aura.trainer.train_executor'].TrainExecutor = mock_train_executor

            mock_ray.get_actor.side_effect = ValueError("Actor not found")

            mock_remote_class = MagicMock()
            mock_remote_instance = MagicMock()
            mock_setup_remote = AsyncMock()
            mock_remote_instance.setup.remote = mock_setup_remote
            mock_remote_class.options.return_value.remote.return_value = mock_remote_instance
            mock_ray.remote.return_value = mock_remote_class

            from aura.trainer.train_manager import get_or_create_train_manager, TrainManager

            await get_or_create_train_manager()

            mock_ray.remote.assert_called_once_with(TrainManager)

    @pytest.mark.asyncio
    async def test_get_or_create_train_manager_setup_called(self):
        """
        Test get_or_create_train_manager calls setup.remote() on new instance.
        """
        mock_ray = MagicMock()
        mock_logger = MagicMock()
        mock_omega = MagicMock()
        mock_agentic_conf = MagicMock()
        mock_train_executor = MagicMock()
        mock_loggers = MagicMock()

        mock_loggers.Loggers.return_value.get_logger.return_value = mock_logger

        with patch.dict(sys.modules, {
            'ray': mock_ray,
            'omegaconf': MagicMock(),
            'aura.base.conf.conf': MagicMock(),
            'aura.base.execution.executor_manager': MagicMock(),
            'aura.base.log.loggers': mock_loggers,
            'aura.trainer.train_executor': MagicMock(),
        }):
            sys.modules['omegaconf'].OmegaConf = mock_omega
            sys.modules['aura.base.conf.conf'].AgenticRLConf = mock_agentic_conf
            sys.modules['aura.trainer.train_executor'].TrainExecutor = mock_train_executor

            mock_ray.get_actor.side_effect = ValueError("Actor not found")

            mock_remote_class = MagicMock()
            mock_remote_instance = MagicMock()
            mock_setup_remote = AsyncMock(return_value=MagicMock())
            mock_remote_instance.setup.remote = mock_setup_remote
            mock_remote_class.options.return_value.remote.return_value = mock_remote_instance
            mock_ray.remote.return_value = mock_remote_class

            from aura.trainer.train_manager import get_or_create_train_manager

            await get_or_create_train_manager()

            mock_setup_remote.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_or_create_train_manager_logger_info_on_create(self):
        """
        Test get_or_create_train_manager logs info when creating new actor.
        """
        mock_ray = MagicMock()
        mock_logger = MagicMock()
        mock_omega = MagicMock()
        mock_agentic_conf = MagicMock()
        mock_train_executor = MagicMock()
        mock_loggers = MagicMock()

        mock_loggers.Loggers.return_value.get_logger.return_value = mock_logger

        with patch.dict(sys.modules, {
            'ray': mock_ray,
            'omegaconf': MagicMock(),
            'aura.base.conf.conf': MagicMock(),
            'aura.base.execution.executor_manager': MagicMock(),
            'aura.base.log.loggers': mock_loggers,
            'aura.trainer.train_executor': MagicMock(),
        }):
            sys.modules['omegaconf'].OmegaConf = mock_omega
            sys.modules['aura.base.conf.conf'].AgenticRLConf = mock_agentic_conf
            sys.modules['aura.trainer.train_executor'].TrainExecutor = mock_train_executor

            mock_ray.get_actor.side_effect = ValueError("Actor not found")

            mock_remote_class = MagicMock()
            mock_remote_instance = MagicMock()
            mock_setup_remote = AsyncMock()
            mock_remote_instance.setup.remote = mock_setup_remote
            mock_remote_class.options.return_value.remote.return_value = mock_remote_instance
            mock_ray.remote.return_value = mock_remote_class

            from aura.trainer.train_manager import get_or_create_train_manager

            await get_or_create_train_manager()

            mock_logger.info.assert_called_once_with(
                "Could not find actor TrainManager, creating a new one."
            )

    @pytest.mark.asyncio
    async def test_train_manager_setup_success(self):
        """
        Test TrainManager.setup() with successful initialization.
        """
        mock_logger = MagicMock()
        mock_omega = MagicMock()
        mock_agentic_conf = MagicMock()
        mock_train_executor = MagicMock()
        mock_loggers = MagicMock()

        mock_loggers.Loggers.return_value.get_logger.return_value = mock_logger

        mock_instance_conf = MagicMock()
        mock_instance_conf.name = "test_instance"
        mock_instance_conf.executor_num = 2
        mock_instance_conf.executor_kwargs = {"key": "value"}
        mock_instance_conf.resource_info = {"cpu": 1}

        mock_conf = MagicMock()
        mock_conf.train_instances = [mock_instance_conf]
        mock_agentic_conf.load_config.return_value = mock_conf

        mock_omega.to_container.side_effect = [{"key": "value"}, {"cpu": 1}]

        with patch.dict(sys.modules, {
            'ray': MagicMock(),
            'omegaconf': MagicMock(),
            'aura.base.conf.conf': MagicMock(),
            'aura.base.execution.executor_manager': MagicMock(),
            'aura.base.log.loggers': mock_loggers,
            'aura.trainer.train_executor': MagicMock(),
        }):
            sys.modules['omegaconf'].OmegaConf = mock_omega
            sys.modules['aura.base.conf.conf'].AgenticRLConf = mock_agentic_conf
            sys.modules['aura.trainer.train_executor'].TrainExecutor = mock_train_executor
            sys.modules['aura.base.execution.executor_manager'].ExecutorManager = MockExecutorManager

            from aura.trainer.train_manager import TrainManager

            manager = TrainManager()
            await manager.setup()

            assert "test_instance" in manager.instance_dict
            assert manager.instance_dict["test_instance"]["name"] == "test_instance"
            assert manager.instance_dict["test_instance"]["executor_num"] == 2
            mock_logger.info.assert_called()
            assert mock_logger.info.call_count == 2

    @pytest.mark.asyncio
    async def test_train_manager_setup_empty_instances(self):
        """
        Test TrainManager.setup() with empty train_instances list.
        """
        mock_logger = MagicMock()
        mock_omega = MagicMock()
        mock_agentic_conf = MagicMock()
        mock_train_executor = MagicMock()
        mock_loggers = MagicMock()

        mock_loggers.Loggers.return_value.get_logger.return_value = mock_logger

        mock_conf = MagicMock()
        mock_conf.train_instances = []
        mock_agentic_conf.load_config.return_value = mock_conf

        with patch.dict(sys.modules, {
            'ray': MagicMock(),
            'omegaconf': MagicMock(),
            'aura.base.conf.conf': MagicMock(),
            'aura.base.execution.executor_manager': MagicMock(),
            'aura.base.log.loggers': mock_loggers,
            'aura.trainer.train_executor': MagicMock(),
        }):
            sys.modules['omegaconf'].OmegaConf = mock_omega
            sys.modules['aura.base.conf.conf'].AgenticRLConf = mock_agentic_conf
            sys.modules['aura.trainer.train_executor'].TrainExecutor = mock_train_executor
            sys.modules['aura.base.execution.executor_manager'].ExecutorManager = MockExecutorManager

            from aura.trainer.train_manager import TrainManager

            manager = TrainManager()
            await manager.setup()

            assert len(manager.instance_dict) == 0
            mock_logger.info.assert_called_once()

    @pytest.mark.asyncio
    async def test_train_manager_setup_exception(self):
        """
        Test TrainManager.setup() exception handling.
        """
        mock_logger = MagicMock()
        mock_omega = MagicMock()
        mock_agentic_conf = MagicMock()
        mock_train_executor = MagicMock()
        mock_loggers = MagicMock()

        mock_loggers.Loggers.return_value.get_logger.return_value = mock_logger

        test_error = RuntimeError("Test error")
        mock_agentic_conf.load_config.side_effect = test_error

        with patch.dict(sys.modules, {
            'ray': MagicMock(),
            'omegaconf': MagicMock(),
            'aura.base.conf.conf': MagicMock(),
            'aura.base.execution.executor_manager': MagicMock(),
            'aura.base.log.loggers': mock_loggers,
            'aura.trainer.train_executor': MagicMock(),
        }):
            sys.modules['omegaconf'].OmegaConf = mock_omega
            sys.modules['aura.base.conf.conf'].AgenticRLConf = mock_agentic_conf
            sys.modules['aura.trainer.train_executor'].TrainExecutor = mock_train_executor
            sys.modules['aura.base.execution.executor_manager'].ExecutorManager = MockExecutorManager

            from aura.trainer.train_manager import TrainManager

            manager = TrainManager()

            with pytest.raises(RuntimeError, match="Test error"):
                await manager.setup()

            mock_logger.error.assert_called_once_with("Train manager setup failed: %s", test_error)

    @pytest.mark.asyncio
    async def test_train_manager_setup_multiple_instances(self):
        """
        Test TrainManager.setup() with multiple train instances.
        """
        mock_logger = MagicMock()
        mock_omega = MagicMock()
        mock_agentic_conf = MagicMock()
        mock_train_executor = MagicMock()
        mock_loggers = MagicMock()

        mock_loggers.Loggers.return_value.get_logger.return_value = mock_logger

        mock_instance_conf1 = MagicMock()
        mock_instance_conf1.name = "instance1"
        mock_instance_conf1.executor_num = 1
        mock_instance_conf1.executor_kwargs = {"key1": "value1"}
        mock_instance_conf1.resource_info = {"cpu": 1}

        mock_instance_conf2 = MagicMock()
        mock_instance_conf2.name = "instance2"
        mock_instance_conf2.executor_num = 2
        mock_instance_conf2.executor_kwargs = {"key2": "value2"}
        mock_instance_conf2.resource_info = {"cpu": 2}

        mock_conf = MagicMock()
        mock_conf.train_instances = [mock_instance_conf1, mock_instance_conf2]
        mock_agentic_conf.load_config.return_value = mock_conf

        mock_omega.to_container.side_effect = [
            {"key1": "value1"}, {"cpu": 1},
            {"key2": "value2"}, {"cpu": 2},
        ]

        with patch.dict(sys.modules, {
            'ray': MagicMock(),
            'omegaconf': MagicMock(),
            'aura.base.conf.conf': MagicMock(),
            'aura.base.execution.executor_manager': MagicMock(),
            'aura.base.log.loggers': mock_loggers,
            'aura.trainer.train_executor': MagicMock(),
        }):
            sys.modules['omegaconf'].OmegaConf = mock_omega
            sys.modules['aura.base.conf.conf'].AgenticRLConf = mock_agentic_conf
            sys.modules['aura.trainer.train_executor'].TrainExecutor = mock_train_executor
            sys.modules['aura.base.execution.executor_manager'].ExecutorManager = MockExecutorManager

            from aura.trainer.train_manager import TrainManager

            manager = TrainManager()
            await manager.setup()

            assert "instance1" in manager.instance_dict
            assert "instance2" in manager.instance_dict
            assert manager.instance_dict["instance1"]["executor_num"] == 1
            assert manager.instance_dict["instance2"]["executor_num"] == 2
            assert mock_logger.info.call_count == 3
            assert mock_omega.to_container.call_count == 4
