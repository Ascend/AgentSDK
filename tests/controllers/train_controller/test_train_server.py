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

import sys
import asyncio
import io
import json
import unittest
from unittest.mock import MagicMock, patch, AsyncMock

# ---------------------------------------------------------------------------
# Module-level mocks
# ---------------------------------------------------------------------------
mock_ray = MagicMock()
mock_dispatch_actor = MagicMock()
mock_ray.get_actor.return_value = mock_dispatch_actor

mock_torch = MagicMock()

mock_fastapi = MagicMock()
mock_api_router_instance = MagicMock()
mock_fastapi.APIRouter.return_value = mock_api_router_instance
mock_fastapi.UploadFile = MagicMock
mock_fastapi.File = MagicMock
mock_fastapi.Form = MagicMock

mock_starlette = MagicMock()
mock_starlette_responses = MagicMock()

mock_loggers_module = MagicMock()
mock_logger = MagicMock()
mock_loggers_module.Loggers.return_value.get_logger.return_value = mock_logger

mock_train_queue_module = MagicMock()
mock_train_queue_instance = MagicMock()
mock_train_queue_module.TrainQueue.return_value = mock_train_queue_instance

mock_msg_handler_module = MagicMock()
mock_deserialize_and_split = MagicMock(return_value={"input_ids": [1, 2, 3]})
mock_msg_handler_module.deserialize_and_split = mock_deserialize_and_split

with patch.dict('sys.modules', {
    'ray': mock_ray,
    'torch': mock_torch,
    'torch.distributed': mock_torch.distributed,
    'fastapi': mock_fastapi,
    'starlette': mock_starlette,
    'starlette.responses': mock_starlette_responses,
    'agentic_rl.base.log.loggers': mock_loggers_module,
    'agentic_rl.controllers.train_controller.train_queue': mock_train_queue_module,
    'agentic_rl.controllers.utils.msg_handler': mock_msg_handler_module,
}):
    from agentic_rl.controllers.train_controller.train_server import TrainServer


class TestTrainServer(unittest.TestCase):
    """Tests for TrainServer covering init, queue ops, and async endpoints."""

    def setUp(self):
        mock_ray.get_actor.return_value = mock_dispatch_actor
        mock_train_queue_module.TrainQueue.return_value = mock_train_queue_instance
        mock_train_queue_instance.reset_mock()
        mock_dispatch_actor.reset_mock()
        mock_logger.reset_mock()
        self.server = TrainServer(
            max_queue_size=4,
            global_batch_size=8,
            n_samples_per_prompt=2,
        )

    def tearDown(self):
        mock_ray.reset_mock()
        mock_torch.reset_mock()
        mock_loggers_module.reset_mock()

    # ---- __init__ -----------------------------------------------------------

    def test_init_gets_dispatch_actor(self):
        mock_ray.get_actor.assert_called_with("dispatch", namespace="controller_raygroup")

    def test_init_creates_train_queue(self):
        mock_train_queue_module.TrainQueue.assert_called_with(
            max_queue_size=4,
            global_batch_size=8,
            n_samples_per_prompt=2,
        )

    def test_init_creates_api_router(self):
        self.assertIsNotNone(self.server.router)

    def test_init_sets_dispatch_actor(self):
        self.assertIs(self.server.dispatch_actor, mock_dispatch_actor)

    # ---- put_minibatch_to_queue ---------------------------------------------

    def test_put_minibatch_to_queue_not_full(self):
        mock_train_queue_instance.add_minibatch.return_value = True
        outputs = {"input_ids": [1, 2]}
        metric = {"rollout_cost": 1.0}
        self.server.put_minibatch_to_queue(outputs, metric)
        mock_train_queue_instance.add_minibatch.assert_called_once_with(outputs, metric)
        mock_dispatch_actor.send_batch_groups.remote.assert_called_once_with(1)

    def test_put_minibatch_to_queue_full_locks_rollout(self):
        mock_train_queue_instance.add_minibatch.return_value = False
        outputs = {"input_ids": [1, 2]}
        metric = {"rollout_cost": 1.0}
        self.server.put_minibatch_to_queue(outputs, metric)
        mock_dispatch_actor.lock_rollout_unit.assert_called_once()

    # ---- async: is_batch_ready ----------------------------------------------

    async def test_is_batch_ready_true(self):
        mock_train_queue_instance.size.return_value = 1
        result = await self.server.is_batch_ready()
        self.assertTrue(result["is_ready"])

    async def test_is_batch_ready_false(self):
        mock_train_queue_instance.size.return_value = 0
        result = await self.server.is_batch_ready()
        self.assertFalse(result["is_ready"])

    # ---- async: pop_minibatch -----------------------------------------------

    async def test_pop_minibatch_returns_response(self):
        mock_train_queue_instance.pop_batch.return_value = ({"data": "val"}, {"metric": 1.0})
        mock_torch.save = MagicMock()
        result = await self.server.pop_minibatch()
        mock_train_queue_instance.pop_batch.assert_called_once()
        # pop_minibatch returns a Response object -- since starlette.responses.Response is mocked,
        # it returns the mock; just verify pop_batch was called
        self.assertIsNotNone(result)

    # ---- async: is_ready ----------------------------------------------------

    async def test_is_ready_sets_rollout_unit_ready(self):
        mock_dispatch_actor.set_rollout_unit_ready.remote = AsyncMock(return_value=None)
        result = await self.server.is_ready()
        mock_dispatch_actor.set_rollout_unit_ready.remote.assert_called_once()
        self.assertEqual(result, {"status": "ok"})


# ---------------------------------------------------------------------------
# Wrap async test methods
# ---------------------------------------------------------------------------
def async_test(coro):
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro(*args, **kwargs))
        finally:
            loop.close()
            asyncio.set_event_loop(asyncio.new_event_loop())
    return wrapper


for name in dir(TestTrainServer):
    if name.startswith('test_') and asyncio.iscoroutinefunction(getattr(TestTrainServer, name)):
        setattr(TestTrainServer, name, async_test(getattr(TestTrainServer, name)))


if __name__ == '__main__':
    unittest.main()
