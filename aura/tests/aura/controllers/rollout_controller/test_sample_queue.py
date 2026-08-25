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
import sys
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Module-level mocks (BEFORE importing the code under test)
#
# sample_queue.py imports:
#   - asyncio
#   - ray
#   - aura.base.log.loggers.Loggers (which imports torch, torch.distributed)
# ---------------------------------------------------------------------------
mock_ray = MagicMock()

mock_ray.remote = MagicMock(side_effect=lambda *args, **kwargs: (lambda cls: cls))
mock_ray.get_actor = MagicMock()

mock_loggers_module = MagicMock()
mock_loggers_module.Loggers.return_value.get_logger.return_value = MagicMock()

with patch.dict(sys.modules, {
    'ray': mock_ray,
    'aura.base.log.loggers': mock_loggers_module,
}):
    from aura.controllers.rollout_controller.sample_queue import (
        SAMPLE_QUEUE_ACTOR_NAME,
        SAMPLE_QUEUE_NAMESPACE,
        SampleQueue,
        create_sample_queue,
        get_sample_queue,
    )


class TestSampleQueue(unittest.IsolatedAsyncioTestCase):
    """Test cases for the SampleQueue actor class."""

    async def asyncSetUp(self):
        self.queue = SampleQueue(max_queue_size=2)

    async def test_init(self):
        self.assertEqual(self.queue.max_queue_size, 2)
        self.assertTrue(self.queue.running)
        self.assertEqual(self.queue.total_produced, 0)
        self.assertEqual(self.queue.total_consumed, 0)
        self.assertEqual(self.queue.dropped_samples, 0)
        self.assertEqual(self.queue.queue.maxsize, 2)
        self.assertEqual(self.queue.queue.qsize(), 0)

    async def test_put_sample_basic(self):
        result = await self.queue.put_sample({"data": 1})
        self.assertTrue(result)
        self.assertEqual(self.queue.total_produced, 1)
        self.assertEqual(await self.queue.get_queue_size(), 1)

    async def test_put_and_get_fifo(self):
        s1 = {"id": 1}
        s2 = {"id": 2}
        await self.queue.put_sample(s1)
        await self.queue.put_sample(s2)

        self.assertEqual(await self.queue.get_sample(), s1)
        self.assertEqual(await self.queue.get_sample(), s2)
        self.assertEqual(self.queue.total_consumed, 2)
        self.assertEqual(await self.queue.get_queue_size(), 0)

    async def test_put_sample_full_drops_oldest(self):
        s1 = {"weight_version": 1}
        s2 = {"weight_version": 2}
        s3 = {"weight_version": 3}
        await self.queue.put_sample(s1)
        await self.queue.put_sample(s2)

        # queue is full (maxsize=2); putting s3 drops the oldest sample (s1).
        await self.queue.put_sample(s3)

        self.assertEqual(self.queue.dropped_samples, 1)
        self.assertEqual(await self.queue.get_sample(), s2)
        self.assertEqual(await self.queue.get_sample(), s3)

    async def test_put_sample_after_shutdown(self):
        self.queue.running = False
        result = await self.queue.put_sample({"data": 1})
        self.assertFalse(result)
        self.assertEqual(self.queue.total_produced, 0)

    async def test_get_sample_shutdown_empty_returns_none(self):
        self.queue.running = False
        result = await self.queue.get_sample()
        self.assertIsNone(result)

    async def test_get_sample_waits_for_sample(self):
        async def put_later():
            await asyncio.sleep(0.02)
            await self.queue.put_sample({"id": 99})

        task = asyncio.create_task(put_later())
        result = await self.queue.get_sample()
        await task

        self.assertEqual(result, {"id": 99})
        self.assertEqual(self.queue.total_consumed, 1)

    async def test_get_statistics(self):
        await self.queue.put_sample({"data": 1})

        stats = await self.queue.get_statistics()

        self.assertEqual(stats["queue_size"], 1)
        self.assertEqual(stats["total_produced"], 1)
        self.assertEqual(stats["total_consumed"], 0)
        self.assertEqual(stats["dropped_samples"], 0)
        self.assertEqual(stats["max_queue_size"], 2)
        self.assertTrue(stats["running"])

    async def test_shutdown(self):
        await self.queue.shutdown()
        self.assertFalse(self.queue.running)


class TestSampleQueueFactory(unittest.TestCase):
    """Test cases for create_sample_queue / get_sample_queue."""

    def setUp(self):
        mock_ray.get_actor.reset_mock()

    def test_create_sample_queue(self):
        mock_utils = MagicMock()
        mock_actor = MagicMock()
        mock_utils.create_actor = MagicMock(return_value=mock_actor)

        with patch.dict(sys.modules, {'aura.controllers.utils.utils': mock_utils}):
            result = create_sample_queue(max_queue_size=128)

        mock_utils.create_actor.assert_called_once_with(
            name=SAMPLE_QUEUE_ACTOR_NAME,
            cls=SampleQueue,
            namespace=SAMPLE_QUEUE_NAMESPACE,
            options={"num_cpus": 1, "max_concurrency": 20},
            actor_kwargs={"max_queue_size": 128},
        )
        self.assertEqual(result, mock_actor)

    def test_create_sample_queue_default_max_queue_size(self):
        mock_utils = MagicMock()
        mock_actor = MagicMock()
        mock_utils.create_actor = MagicMock(return_value=mock_actor)

        with patch.dict(sys.modules, {'aura.controllers.utils.utils': mock_utils}):
            create_sample_queue()

        mock_utils.create_actor.assert_called_once_with(
            name=SAMPLE_QUEUE_ACTOR_NAME,
            cls=SampleQueue,
            namespace=SAMPLE_QUEUE_NAMESPACE,
            options={"num_cpus": 1, "max_concurrency": 20},
            actor_kwargs={"max_queue_size": 256},
        )

    def test_get_sample_queue(self):
        mock_actor = MagicMock()
        mock_ray.get_actor.return_value = mock_actor

        result = get_sample_queue()

        mock_ray.get_actor.assert_called_once_with(
            SAMPLE_QUEUE_ACTOR_NAME, namespace=SAMPLE_QUEUE_NAMESPACE
        )
        self.assertEqual(result, mock_actor)


if __name__ == '__main__':
    unittest.main()
