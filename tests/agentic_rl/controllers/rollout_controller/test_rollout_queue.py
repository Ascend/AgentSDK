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
#        http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------


import pytest
from unittest.mock import patch
from collections import deque
import threading

from agentic_rl.controllers.rollout_controller.rollout_queue import (
    RolloutQueueActor,
    get_rollout_queue_actor,
)

import sys
from unittest.mock import MagicMock

# Mock ray before importing the module under test
mock_ray = MagicMock()
mock_ray.remote = lambda func_or_class: func_or_class
mock_ray.get = MagicMock(return_value=None)
mock_ray.get_actor = MagicMock()
mock_ray.kill = MagicMock()
mock_ray.is_initialized = MagicMock(return_value=True)
mock_ray.available_resources = MagicMock(return_value={"CPU": 8})

sys.modules['ray'] = mock_ray

import sys
from unittest.mock import MagicMock

# Mock ray before importing the module under test
mock_ray = MagicMock()
mock_ray.remote = lambda func_or_class: func_or_class
mock_ray.get = MagicMock(return_value=None)
mock_ray.get_actor = MagicMock()
mock_ray.kill = MagicMock()
mock_ray.is_initialized = MagicMock(return_value=True)
mock_ray.available_resources = MagicMock(return_value={"CPU": 8})

sys.modules['ray'] = mock_ray

class TestRolloutQueueActor:
    """Test cases for RolloutQueueActor class."""

    def test_init(self):
        """Test RolloutQueueActor initialization."""
        actor = RolloutQueueActor()

        assert isinstance(actor.batch_queue, deque)
        assert len(actor.batch_queue) == 0
        assert isinstance(actor.abort_queue, deque)
        assert len(actor.abort_queue) == 0
        assert actor.running is False
        assert actor.weight_updating_status == {}
        assert isinstance(actor.shutdown_event, threading.Event)
        assert not actor.shutdown_event.is_set()

    def test_add_queue(self):
        """Test add_queue method."""
        actor = RolloutQueueActor()
        batch = {"data": "test_batch"}

        actor.add_queue(batch)

        assert len(actor.batch_queue) == 1
        assert actor.batch_queue[0] == batch

    def test_add_queue_multiple(self):
        """Test adding multiple items to queue."""
        actor = RolloutQueueActor()
        batch1 = {"data": "batch1"}
        batch2 = {"data": "batch2"}
        batch3 = {"data": "batch3"}

        actor.add_queue(batch1)
        actor.add_queue(batch2)
        actor.add_queue(batch3)

        assert len(actor.batch_queue) == 3
        assert list(actor.batch_queue) == [batch1, batch2, batch3]

    def test_pop_queue(self):
        """Test pop_queue method."""
        actor = RolloutQueueActor()
        batch = {"data": "test_batch"}
        actor.add_queue(batch)

        result = actor.pop_queue()

        assert result == batch
        assert len(actor.batch_queue) == 0

    def test_pop_queue_fifo(self):
        """Test that pop_queue follows FIFO order."""
        actor = RolloutQueueActor()
        batch1 = {"data": "batch1"}
        batch2 = {"data": "batch2"}
        batch3 = {"data": "batch3"}

        actor.add_queue(batch1)
        actor.add_queue(batch2)
        actor.add_queue(batch3)

        assert actor.pop_queue() == batch1
        assert actor.pop_queue() == batch2
        assert actor.pop_queue() == batch3

    def test_queue_size(self):
        """Test queue_size method."""
        actor = RolloutQueueActor()

        assert actor.queue_size() == 0

        actor.add_queue({"data": "batch1"})
        assert actor.queue_size() == 1

        actor.add_queue({"data": "batch2"})
        assert actor.queue_size() == 2

        actor.pop_queue()
        assert actor.queue_size() == 1

    def test_add_abort_queue(self):
        """Test add_abort_queue method."""
        actor = RolloutQueueActor()
        request = {"request_id": "123"}

        actor.add_abort_queue(request)

        assert len(actor.abort_queue) == 1
        assert actor.abort_queue[0] == request

    def test_add_abort_queue_multiple(self):
        """Test adding multiple items to abort queue."""
        actor = RolloutQueueActor()
        request1 = {"request_id": "123"}
        request2 = {"request_id": "456"}

        actor.add_abort_queue(request1)
        actor.add_abort_queue(request2)

        assert len(actor.abort_queue) == 2
        assert list(actor.abort_queue) == [request1, request2]

    def test_pop_abort_queue(self):
        """Test pop_abort_queue method."""
        actor = RolloutQueueActor()
        request = {"request_id": "123"}
        actor.add_abort_queue(request)

        result = actor.pop_abort_queue()

        assert result == request
        assert len(actor.abort_queue) == 0

    def test_pop_abort_queue_fifo(self):
        """Test that pop_abort_queue follows FIFO order."""
        actor = RolloutQueueActor()
        request1 = {"request_id": "123"}
        request2 = {"request_id": "456"}
        request3 = {"request_id": "789"}

        actor.add_abort_queue(request1)
        actor.add_abort_queue(request2)
        actor.add_abort_queue(request3)

        assert actor.pop_abort_queue() == request1
        assert actor.pop_abort_queue() == request2
        assert actor.pop_abort_queue() == request3

    def test_set_running(self):
        """Test set_running method."""
        actor = RolloutQueueActor()

        assert actor.running is False

        actor.set_running()

        assert actor.running is True

    def test_set_block(self):
        """Test set_block method."""
        actor = RolloutQueueActor()
        actor.set_running()

        assert actor.running is True

        actor.set_block()

        assert actor.running is False

    def test_is_running(self):
        """Test is_running method."""
        actor = RolloutQueueActor()

        assert actor.is_running() is False

        actor.set_running()
        assert actor.is_running() is True

        actor.set_block()
        assert actor.is_running() is False

    def test_shutdown(self):
        """Test shutdown method."""
        actor = RolloutQueueActor()

        assert not actor.shutdown_event.is_set()

        actor.shutdown()

        assert actor.shutdown_event.is_set()

    def test_is_shutdown(self):
        """Test is_shutdown method."""
        actor = RolloutQueueActor()

        assert actor.is_shutdown() is False

        actor.shutdown()

        assert actor.is_shutdown() is True

    def test_init_done(self):
        """Test init_done method."""
        actor = RolloutQueueActor()

        result = actor.init_done()

        assert result is None

class TestGetRolloutQueueActor:
    """Test cases for get_rollout_queue_actor function."""

    @patch('agentic_rl.controllers.rollout_controller.rollout_queue.ray.get_actor')
    def test_get_rollout_queue_actor(self, mock_get_actor):
        """Test get_rollout_queue_actor function."""
        mock_actor = MagicMock()
        mock_get_actor.return_value = mock_actor

        result = get_rollout_queue_actor()

        mock_get_actor.assert_called_once_with("rollout_queue", namespace="controller_raygroup")
        assert result == mock_actor


@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after each test."""
    import sys
    mocked_modules = []
    modules_to_check = ['ray']
    for mod in modules_to_check:
        if mod not in sys.modules:
            mocked_modules.append(mod)
    yield
    for mod in mocked_modules:
        if mod in sys.modules:
            del sys.modules[mod]
