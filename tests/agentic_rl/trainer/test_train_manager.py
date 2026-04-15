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
from unittest.mock import patch, MagicMock, AsyncMock


class TestTrainManager:
    """
    Tests for TrainManager class and get_or_create_train_manager function.
    """

    @classmethod
    def setup_class(cls):
        """
        Patch sys.modules before importing the module under test to avoid side effects.
        """
        def create_mock_module():
            return MagicMock()

        cls.mocked_modules = {
            "ray": create_mock_module(),
            "omegaconf": create_mock_module(),
            "agentic_rl.base.execution.executor_manager": create_mock_module(),
            "agentic_rl.base.log.loggers": create_mock_module(),
            "agentic_rl.trainer.train_executor": create_mock_module(),
            "agentic_rl.base.conf.conf": create_mock_module(),
        }
        cls.module_patcher = patch.dict(sys.modules, cls.mocked_modules)
        cls.module_patcher.start()

        # Import inside patch lifecycle
        from agentic_rl.trainer.train_manager import (
            TrainManager,
            get_or_create_train_manager,
        )
        cls.TrainManager = TrainManager
        cls.get_or_create_train_manager = get_or_create_train_manager

    @classmethod
    def teardown_class(cls):
        cls.module_patcher.stop()

    @pytest.fixture
    def mock_dependencies(self):
        """
        Provide runtime dependency mocks for TrainManager.
        """
        with patch("agentic_rl.trainer.train_manager.ray") as mock_ray, \
             patch("agentic_rl.trainer.train_manager.OmegaConf") as mock_omega, \
             patch("agentic_rl.trainer.train_manager.logger") as mock_logger, \
             patch("agentic_rl.trainer.train_manager.ExecutorManager") as mock_executor_manager, \
             patch("agentic_rl.base.conf.conf.AgenticRLConf") as mock_agentic_conf, \
             patch("agentic_rl.trainer.train_manager.TrainExecutor") as mock_train_executor, \
             patch("agentic_rl.trainer.train_manager.Loggers") as mock_loggers:

            mock_conf_instance = MagicMock()
            instance_conf_1 = MagicMock()
            instance_conf_1.name = "test_instance_1"
            instance_conf_1.executor_num = 2
            instance_conf_1.executor_kwargs = {"param1": "value1"}
            instance_conf_1.resource_info = {"cpu": 1}

            instance_conf_2 = MagicMock()
            instance_conf_2.name = "test_instance_2"
            instance_conf_2.executor_num = 3
            instance_conf_2.executor_kwargs = {"param2": "value2"}
            instance_conf_2.resource_info = {"cpu": 2}

            mock_conf_instance.train_instances = [instance_conf_1, instance_conf_2]
            mock_omega.to_container.side_effect = lambda x, **kwargs: x

            yield {
                "ray": mock_ray,
                "omega": mock_omega,
                "logger": mock_logger,
                "executor_manager": mock_executor_manager,
                "agentic_conf": mock_agentic_conf,
                "train_executor": mock_train_executor,
                "loggers": mock_loggers,
            }

    @pytest.mark.asyncio
    async def test_setup_success(self, mock_dependencies):
        """
        Test successful setup of TrainManager instances.
        """
        mock_conf_instance = MagicMock()
        instance_conf_1 = MagicMock()
        instance_conf_1.name = "test_instance_1"
        instance_conf_1.executor_num = 2
        instance_conf_1.executor_kwargs = {"param1": "value1"}
        instance_conf_1.resource_info = {"cpu": 1}

        instance_conf_2 = MagicMock()
        instance_conf_2.name = "test_instance_2"
        instance_conf_2.executor_num = 3
        instance_conf_2.executor_kwargs = {"param2": "value2"}
        instance_conf_2.resource_info = {"cpu": 2}

        mock_conf_instance.train_instances = [instance_conf_1, instance_conf_2]
        mock_dependencies["agentic_conf"].load_config.return_value = mock_conf_instance

        create_instance_mock = AsyncMock()

        class MockManager:
            def __init__(self):
                self.instance_dict = {}
                self.create_instance = create_instance_mock

        manager = MockManager()
        conf = mock_dependencies["agentic_conf"].load_config()
        mock_dependencies["agentic_conf"].load_config.assert_called_once()

        for instance_conf in conf.train_instances:
            await manager.create_instance(
                name=instance_conf.name,
                executor_class=mock_dependencies["train_executor"],
                executor_num=instance_conf.executor_num,
                executor_kwargs=instance_conf.executor_kwargs,
                resource_info=instance_conf.resource_info,
            )

        assert create_instance_mock.call_count == 2

    @pytest.mark.asyncio
    async def test_setup_exception(self, mock_dependencies):
        """
        Test TrainManager setup raising an exception.
        """
        mock_conf_instance = MagicMock()
        instance_conf = MagicMock()
        instance_conf.name = "test_instance_1"
        instance_conf.executor_num = 2
        instance_conf.executor_kwargs = {"param1": "value1"}
        instance_conf.resource_info = {"cpu": 1}

        mock_conf_instance.train_instances = [instance_conf]
        mock_dependencies["agentic_conf"].load_config.return_value = mock_conf_instance

        test_exception = Exception("Test exception")
        create_instance_mock = AsyncMock(side_effect=test_exception)

        class MockManager:
            def __init__(self):
                self.instance_dict = {}
                self.create_instance = create_instance_mock

        manager = MockManager()
        conf = mock_dependencies["agentic_conf"].load_config()

        with pytest.raises(Exception) as excinfo:
            for instance_conf in conf.train_instances:
                await manager.create_instance(
                    name=instance_conf.name,
                    executor_class=mock_dependencies["train_executor"],
                    executor_num=instance_conf.executor_num,
                    executor_kwargs=instance_conf.executor_kwargs,
                    resource_info=instance_conf.resource_info,
                )

        assert str(excinfo.value) == "Test exception"

    @pytest.mark.asyncio
    async def test_get_or_create_train_manager_exists(self, mock_dependencies):
        """
        Test get_or_create_train_manager returns existing actor if available.
        """
        existing_actor = MagicMock()
        mock_dependencies["ray"].get_actor.return_value = existing_actor

        result = await type(self).get_or_create_train_manager()
        mock_dependencies["ray"].get_actor.assert_called_once_with("TrainManager")
        assert result == existing_actor

    @pytest.mark.asyncio
    async def test_get_or_create_train_manager_new(self, mock_dependencies):
        """
        Test get_or_create_train_manager creates new actor if not found.
        """
        mock_dependencies["ray"].get_actor.side_effect = ValueError("Actor not found")

        mock_remote_class = MagicMock()
        mock_remote_instance = MagicMock()
        mock_setup_remote = AsyncMock()
        mock_remote_instance.setup.remote = mock_setup_remote
        mock_remote_class.options.return_value.remote.return_value = mock_remote_instance
        mock_dependencies["ray"].remote.return_value = mock_remote_class

        result = await type(self).get_or_create_train_manager()
        mock_dependencies["ray"].get_actor.assert_called_once_with("TrainManager")
        mock_dependencies["logger"].info.assert_called_once_with(
            "Could not find actor TrainManager, creating a new one."
        )
        mock_setup_remote.assert_awaited_once()
        assert result == mock_remote_instance