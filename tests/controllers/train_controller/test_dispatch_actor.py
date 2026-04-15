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
import unittest
from unittest.mock import MagicMock, patch, AsyncMock

# ---------------------------------------------------------------------------
# Module-level mocks -- created BEFORE the code-under-test is imported
# ---------------------------------------------------------------------------
mock_ray = MagicMock()
mock_ray.remote = MagicMock(side_effect=lambda cls: cls)

mock_torch = MagicMock()
mock_aiohttp = MagicMock()

mock_loggers_module = MagicMock()
mock_loggers_module.Loggers.return_value.get_logger.return_value = MagicMock()

mock_async_http_module = MagicMock()
mock_async_send_batch = AsyncMock(return_value={"status": 200})
mock_async_http_module.async_send_batch = mock_async_send_batch

mock_http_status_module = MagicMock()
mock_http_status_module.HTTP_OK_200 = 200

mock_sync_http_module = MagicMock()
mock_sync_send = MagicMock(return_value={"status": "ok"})
mock_sync_http_module.sync_send = mock_sync_send

mock_utils_module = MagicMock()
mock_post_with_url = MagicMock(return_value={"status": "ok"})
mock_utils_module.DEFAULT_URL_METHOD = "http"
mock_utils_module.post_with_url = mock_post_with_url
mock_utils_module.DEFAULT_RETRY_COUNT = 3
mock_utils_module.DEFAULT_BACKOFF_FACTOR = 30.0
mock_utils_module.MIN_BACKOFF_FACTOR = 5.0

with patch.dict('sys.modules', {
    'ray': mock_ray,
    'torch': mock_torch,
    'torch.distributed': mock_torch.distributed,
    'aiohttp': mock_aiohttp,
    'agentic_rl.base.log.loggers': mock_loggers_module,
    'agentic_rl.controllers.utils.async_http': mock_async_http_module,
    'agentic_rl.controllers.utils.http_status': mock_http_status_module,
    'agentic_rl.controllers.utils.sync_http': mock_sync_http_module,
    'agentic_rl.controllers.utils.utils': mock_utils_module,
}):
    from agentic_rl.controllers.train_controller.dispatch_actor import DispatchActor


class TestDispatchActor(unittest.TestCase):
    """Tests for DispatchActor covering init, sync helpers, and async workflows."""

    def setUp(self):
        self.mock_initialize = MagicMock(return_value=(iter([{"prompt": "a"}]), None, None))
        self.actor = DispatchActor(
            n_samples_per_prompt=2,
            validate_num_samples=4,
            global_batch_size=8,
            train_iters=10,
            data_loader=MagicMock(),
            initialize_train_dataloader=self.mock_initialize,
            consumed_train_samples=0,
            data_optimized=False,
            rollout_server_addr="127.0.0.1:4001",
        )

    def tearDown(self):
        mock_ray.reset_mock()
        mock_torch.reset_mock()
        mock_aiohttp.reset_mock()
        mock_async_send_batch.reset_mock()
        mock_sync_send.reset_mock()
        mock_post_with_url.reset_mock()
        mock_loggers_module.reset_mock()

    # ---- __init__ -----------------------------------------------------------

    def test_init_calls_initialize_train_dataloader(self):
        self.mock_initialize.assert_called_once()

    def test_init_sets_global_batch_size(self):
        self.assertEqual(self.actor.global_batch_size, 8)

    def test_init_sets_total_prompts_for_training(self):
        self.assertEqual(self.actor.total_prompts_for_training, 80)

    def test_init_total_prompts_for_rollout_starts_at_zero(self):
        self.assertEqual(self.actor.total_prompts_for_rollout_unit, 0)

    def test_init_rollout_unit_not_ready(self):
        self.assertFalse(self.actor.rollout_unit_ready)

    def test_init_session_is_none(self):
        self.assertIsNone(self.actor.session)

    def test_init_failed_groups_empty(self):
        self.assertEqual(len(self.actor.failed_groups), 0)

    def test_init_current_training_iter_default(self):
        self.assertEqual(self.actor.current_training_iter, 1)

    # ---- _get_from_data_iters -----------------------------------------------

    def test_get_from_data_iters_returns_next(self):
        self.actor.train_data_iters = iter([{"x": 1}, {"x": 2}])
        result = self.actor._get_from_data_iters()
        self.assertEqual(result, {"x": 1})

    def test_get_from_data_iters_returns_none_on_stop(self):
        self.actor.train_data_iters = iter([])
        result = self.actor._get_from_data_iters()
        self.assertIsNone(result)

    # ---- _get_from_failed_groups --------------------------------------------

    def test_get_from_failed_groups_pops(self):
        self.actor.failed_groups.append({"retry": True})
        result = self.actor._get_from_failed_groups()
        self.assertEqual(result, {"retry": True})
        self.assertEqual(len(self.actor.failed_groups), 0)

    # ---- _get_batch_groups_remote -------------------------------------------

    def test_get_batch_groups_prefers_failed_groups(self):
        self.actor.failed_groups.append({"failed": 1})
        self.actor.train_data_iters = iter([{"data": 2}])
        groups = self.actor._get_batch_groups_remote(2)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0], {"failed": 1})
        self.assertEqual(groups[1], {"data": 2})

    def test_get_batch_groups_returns_empty_when_no_data(self):
        self.actor.train_data_iters = iter([])
        groups = self.actor._get_batch_groups_remote(3)
        self.assertEqual(len(groups), 0)

    def test_get_batch_groups_stops_at_none(self):
        self.actor.train_data_iters = iter([{"a": 1}])
        groups = self.actor._get_batch_groups_remote(5)
        self.assertEqual(len(groups), 1)

    # ---- _stat_succeed ------------------------------------------------------

    def test_stat_succeed_counts_200(self):
        groups = [{"g1": 1}, {"g2": 2}]
        results = [{"status": 200}, {"status": 200}]
        count = self.actor._stat_succeed(groups, results)
        self.assertEqual(count, 2)
        self.assertEqual(len(self.actor.failed_groups), 0)

    def test_stat_succeed_requeues_failures(self):
        groups = [{"g1": 1}, {"g2": 2}]
        results = [{"status": 200}, {"status": 500}]
        count = self.actor._stat_succeed(groups, results)
        self.assertEqual(count, 1)
        self.assertEqual(len(self.actor.failed_groups), 1)
        self.assertEqual(self.actor.failed_groups[0], {"g2": 2})

    def test_stat_succeed_handles_none_result(self):
        groups = [{"g1": 1}]
        results = [None]
        count = self.actor._stat_succeed(groups, results)
        self.assertEqual(count, 0)
        self.assertEqual(len(self.actor.failed_groups), 1)

    # ---- check_stop_batch ---------------------------------------------------

    def test_check_stop_batch_false_when_below(self):
        self.actor.total_prompts_for_rollout_unit = 0
        self.assertFalse(self.actor.check_stop_batch())

    def test_check_stop_batch_true_when_at_limit(self):
        self.actor.total_prompts_for_rollout_unit = 80
        self.assertTrue(self.actor.check_stop_batch())

    def test_check_stop_batch_true_when_above_limit(self):
        self.actor.total_prompts_for_rollout_unit = 100
        self.assertTrue(self.actor.check_stop_batch())

    # ---- set/check_rollout_unit_ready ---------------------------------------

    def test_set_rollout_unit_ready(self):
        self.actor.set_rollout_unit_ready()
        self.assertTrue(self.actor.rollout_unit_ready)

    def test_check_rollout_unit_ready_default_false(self):
        self.assertFalse(self.actor.check_rollout_unit_ready())

    def test_check_rollout_unit_ready_after_set(self):
        self.actor.set_rollout_unit_ready()
        self.assertTrue(self.actor.check_rollout_unit_ready())

    # ---- set_current_training_iter ------------------------------------------

    def test_set_current_training_iter(self):
        self.actor.set_current_training_iter(5)
        self.assertEqual(self.actor.current_training_iter, 5)

    # ---- init_done ----------------------------------------------------------

    def test_init_done_runs_without_error(self):
        self.actor.init_done()

    # ---- async: unlock_rollout_unit -----------------------------------------

    async def test_unlock_rollout_unit(self):
        mock_post_with_url.return_value = {"status": "ok"}
        result = await self.actor.unlock_rollout_unit()
        self.assertIsNotNone(result)

    # ---- async: lock_rollout_unit -------------------------------------------

    async def test_lock_rollout_unit(self):
        mock_post_with_url.return_value = {"status": "ok"}
        result = await self.actor.lock_rollout_unit()
        self.assertIsNotNone(result)

    # ---- async: shutdown_rollout_unit ---------------------------------------

    async def test_shutdown_rollout_unit(self):
        mock_post_with_url.return_value = {"status": "ok"}
        result = await self.actor.shutdown_rollout_unit()
        self.assertIsNotNone(result)

    # ---- async: shutdown ----------------------------------------------------

    async def test_shutdown_calls_shutdown_rollout_unit(self):
        mock_post_with_url.return_value = {"status": "ok"}
        await self.actor.shutdown()

    # ---- async: _create_session ---------------------------------------------

    async def test_create_session_when_none(self):
        self.actor.session = None
        await self.actor._create_session()
        self.assertIsNotNone(self.actor.session)

    async def test_create_session_when_closed(self):
        mock_session = MagicMock()
        mock_session.closed = True
        self.actor.session = mock_session
        await self.actor._create_session()
        # Should create a new session because old one was closed
        self.assertIsNotNone(self.actor.session)

    async def test_create_session_reuses_open_session(self):
        mock_session = MagicMock()
        mock_session.closed = False
        self.actor.session = mock_session
        await self.actor._create_session()
        # Should keep the same session
        self.assertIs(self.actor.session, mock_session)

    # ---- async: _do_batch_groups_send_remote --------------------------------

    async def test_do_batch_groups_send_remote(self):
        self.actor.session = MagicMock()
        mock_async_send_batch.return_value = {"status": 200}
        groups = [{"g1": 1}, {"g2": 2}]
        results = await self.actor._do_batch_groups_send_remote(groups)
        self.assertEqual(len(results), 2)

    # ---- async: send_batch_groups -------------------------------------------

    async def test_send_batch_groups_no_data(self):
        self.actor.train_data_iters = iter([])
        result = await self.actor.send_batch_groups(3)
        self.assertEqual(result, 0)

    async def test_send_batch_groups_with_data(self):
        self.actor.train_data_iters = iter([{"a": 1}])
        mock_async_send_batch.return_value = {"status": 200}
        result = await self.actor.send_batch_groups(1)
        self.assertEqual(result, 1)

    # ---- async: notify_weights_update ---------------------------------------

    async def test_notify_weights_update(self):
        mock_sync_send.return_value = {"status": "ok"}
        await self.actor.notify_weights_update("/weights/iter_0000001")


# ---------------------------------------------------------------------------
# Wrap all async test methods so they can run under unittest
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


for name in dir(TestDispatchActor):
    if name.startswith('test_') and asyncio.iscoroutinefunction(getattr(TestDispatchActor, name)):
        setattr(TestDispatchActor, name, async_test(getattr(TestDispatchActor, name)))


if __name__ == '__main__':
    unittest.main()
