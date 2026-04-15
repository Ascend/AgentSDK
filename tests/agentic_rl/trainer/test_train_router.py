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
from unittest.mock import patch, MagicMock, AsyncMock


class TestTrainRouter:
    """
    Tests the TrainRouter class.

    Covers:
      - Initialization
      - Singleton creation
      - Reuse of existing router
      - Routing execution
      - Empty executor error handling
    """

    @classmethod
    def setup_class(cls):
        """
        Patch sys.modules safely before importing TrainRouter to avoid pytest collection-time contamination.
        """
        cls.mock_loggers_module = MagicMock()
        cls.mock_train_manager_module = MagicMock()
        cls.mock_train_manager_module.get_or_create_train_manager = AsyncMock()

        cls.module_patcher = patch.dict(
            sys.modules,
            {
                "agentic_rl.base.log.loggers": cls.mock_loggers_module,
                "agentic_rl.trainer.train_manager": cls.mock_train_manager_module,
            },
        )
        cls.module_patcher.start()

        # Import inside patched scope
        from agentic_rl.trainer.train_router import TrainRouter
        cls.TrainRouter = TrainRouter

    @classmethod
    def teardown_class(cls):
        """
        Stop sys.modules patch safely.
        """
        cls.module_patcher.stop()

    def setup_method(self):
        """
        Reset singleton before each test.
        """
        self.TrainRouter._router = None

    @pytest.fixture
    def mock_dependencies(self):
        """
        Provide runtime dependency mocks for TrainRouter.
        """
        with patch("agentic_rl.trainer.train_router.logger") as mock_logger, \
             patch("agentic_rl.trainer.train_router.random") as mock_random, \
             patch("agentic_rl.trainer.train_router.Loggers") as mock_loggers, \
             patch("agentic_rl.trainer.train_manager.get_or_create_train_manager") as mock_get_manager:

            yield {
                "logger": mock_logger,
                "random": mock_random,
                "loggers": mock_loggers,
                "get_manager": mock_get_manager,
            }

    def test_init(self, mock_dependencies):
        """
        Test TrainRouter initialization with a mock train manager.
        """
        mock_train_manager = MagicMock()
        router = self.TrainRouter(mock_train_manager)
        assert router.train_manager == mock_train_manager

    @pytest.mark.asyncio
    async def test_create_first_time(self, mock_dependencies):
        """
        Test singleton creation on first call.
        """
        mock_train_manager = MagicMock()
        mock_dependencies["get_manager"].return_value = mock_train_manager

        router_instance = await self.TrainRouter.create()
        mock_dependencies["get_manager"].assert_awaited_once()
        mock_dependencies["logger"].info.assert_called_once_with("Train router created.")

        assert isinstance(router_instance, self.TrainRouter)
        assert router_instance.train_manager == mock_train_manager
        assert self.TrainRouter._router == router_instance

    @pytest.mark.asyncio
    async def test_create_existing(self, mock_dependencies):
        """
        Test singleton reuse when router already exists.
        """
        mock_train_manager = MagicMock()
        existing_router = self.TrainRouter(mock_train_manager)
        self.TrainRouter._router = existing_router

        router_instance = await self.TrainRouter.create()
        mock_dependencies["get_manager"].assert_not_awaited()
        mock_dependencies["logger"].info.assert_not_called()
        assert router_instance == existing_router

    @pytest.mark.asyncio
    async def test_train(self, mock_dependencies):
        """
        Test routing execution correctly selects an executor.
        """
        mock_train_manager = MagicMock()

        mock_executor_1 = MagicMock()
        mock_executor_2 = MagicMock()
        mock_executor_3 = MagicMock()

        mock_instance = MagicMock()
        mock_instance.executor_list = [mock_executor_1, mock_executor_2, mock_executor_3]

        expected_result = "test_result"
        execute_future = asyncio.Future()
        execute_future.set_result(expected_result)
        mock_executor_2.execute_method.remote.return_value = execute_future

        instance_future = asyncio.Future()
        instance_future.set_result(mock_instance)
        mock_train_manager.get_instance.remote.return_value = instance_future

        mock_dependencies["random"].choice.return_value = mock_executor_2
        router = self.TrainRouter(mock_train_manager)

        result = await router.train("test_instance")

        mock_train_manager.get_instance.remote.assert_called_once_with("test_instance")
        mock_dependencies["random"].choice.assert_called_once_with([mock_executor_1, mock_executor_2, mock_executor_3])
        mock_executor_2.execute_method.remote.assert_called_once_with("fit")
        assert result == expected_result

    @pytest.mark.asyncio
    async def test_train_empty_executor_list(self, mock_dependencies):
        """
        Test that training raises IndexError when executor_list is empty.
        """
        mock_train_manager = MagicMock()
        mock_instance = MagicMock()
        mock_instance.executor_list = []

        instance_future = asyncio.Future()
        instance_future.set_result(mock_instance)
        mock_train_manager.get_instance.remote.return_value = instance_future

        mock_dependencies["random"].choice.side_effect = IndexError("Cannot choose from an empty sequence")
        router = self.TrainRouter(mock_train_manager)

        with pytest.raises(IndexError):
            await router.train("test_instance")

        mock_train_manager.get_instance.remote.assert_called_once_with("test_instance")
        mock_dependencies["random"].choice.assert_called_once_with([])