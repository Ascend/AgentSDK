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
from unittest.mock import MagicMock, AsyncMock, patch

# Create mock objects before importing code under test
mock_torch = MagicMock()
mock_aiohttp = MagicMock()
mock_ray = MagicMock()
mock_requests = MagicMock()

mock_loggers_module = MagicMock()
mock_logger = MagicMock()
mock_loggers_module.Loggers.return_value.get_logger.return_value = mock_logger

# Mock for the http_status module
mock_http_status = MagicMock()
mock_http_status.HTTP_OK_200 = 200

# Mock for the utils module constants
mock_utils_module = MagicMock()
mock_utils_module.DEFAULT_RETRY_COUNT = 3
mock_utils_module.MIN_BACKOFF_FACTOR = 5.0
mock_utils_module.MIN_RETRY_COUNT = 1
mock_utils_module.DEFAULT_BACKOFF_FACTOR = 30.0

with patch.dict('sys.modules', {
    'torch': mock_torch,
    'aiohttp': mock_aiohttp,
    'ray': mock_ray,
    'requests': mock_requests,
    'agentic_rl.base.log.loggers': mock_loggers_module,
    'agentic_rl.controllers.utils.http_status': mock_http_status,
    'agentic_rl.controllers.utils.utils': mock_utils_module,
}):
    from agentic_rl.controllers.utils.async_http import _dumps, client_post, async_send_batch
    import agentic_rl.controllers.utils.async_http as _async_http_mod


class TestDumps(unittest.TestCase):

    def setUp(self):
        self.batch_dict = {"key1": "value1", "key2": "value2"}

    def tearDown(self):
        mock_torch.reset_mock()
        mock_aiohttp.reset_mock()
        mock_ray.reset_mock()
        mock_requests.reset_mock()
        mock_loggers_module.reset_mock()
        mock_logger.reset_mock()

    def test_dumps_calls_torch_save(self):
        """_dumps calls torch.save with the correct arguments."""
        with patch.object(_async_http_mod, 'torch') as patched_torch:
            _dumps(self.batch_dict)

            patched_torch.save.assert_called_once()
            call_args = patched_torch.save.call_args
            self.assertEqual(call_args[0][0], self.batch_dict)
            self.assertEqual(call_args[1]['pickle_protocol'], 5)
            self.assertEqual(call_args[1]['_use_new_zipfile_serialization'], False)

    def test_dumps_returns_bytes(self):
        """_dumps returns bytes from the BytesIO buffer."""
        with patch.object(_async_http_mod, 'torch') as patched_torch:
            result = _dumps(self.batch_dict)

            # The result comes from buf.getvalue(), which is a bytes object
            # Since torch.save is mocked, buf.getvalue() returns b''
            self.assertIsInstance(result, bytes)

    def test_dumps_with_empty_dict(self):
        """_dumps handles an empty dictionary."""
        with patch.object(_async_http_mod, 'torch') as patched_torch:
            result = _dumps({})

            patched_torch.save.assert_called_once()
            self.assertIsInstance(result, bytes)


class TestClientPost(unittest.TestCase):

    def setUp(self):
        self.test_url = "http://127.0.0.1:8000/api/batch"
        self.test_body = b"fake_binary_data"

    def tearDown(self):
        mock_torch.reset_mock()
        mock_aiohttp.reset_mock()
        mock_ray.reset_mock()
        mock_requests.reset_mock()
        mock_loggers_module.reset_mock()
        mock_logger.reset_mock()

    async def test_client_post_success(self):
        """client_post returns status dict on successful POST."""
        mock_response = MagicMock()
        mock_response.status = 200

        mock_session = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.post.return_value = mock_ctx

        result = await client_post(mock_session, self.test_url, self.test_body)

        self.assertEqual(result, {"status": 200})
        mock_session.post.assert_called_once()

    async def test_client_post_with_custom_timeout(self):
        """client_post passes the timeout parameter to session.post."""
        mock_response = MagicMock()
        mock_response.status = 200

        mock_session = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.post.return_value = mock_ctx

        result = await client_post(mock_session, self.test_url, self.test_body, timeout_s=60)

        self.assertEqual(result, {"status": 200})
        call_kwargs = mock_session.post.call_args[1]
        self.assertEqual(call_kwargs['timeout'], 60)

    async def test_client_post_exception(self):
        """client_post returns status 0 and error on exception."""
        mock_session = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(side_effect=Exception("Network error"))
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.post.return_value = mock_ctx

        result = await client_post(mock_session, self.test_url, self.test_body)

        self.assertEqual(result["status"], 0)
        self.assertIn("Network error", result["error"])


class TestAsyncSendBatch(unittest.TestCase):

    def setUp(self):
        self.test_batch = {"tensor_data": [1, 2, 3]}
        self.test_url = "http://127.0.0.1:8000/api/train"

    def tearDown(self):
        mock_torch.reset_mock()
        mock_aiohttp.reset_mock()
        mock_ray.reset_mock()
        mock_requests.reset_mock()
        mock_loggers_module.reset_mock()
        mock_logger.reset_mock()

    async def test_async_send_batch_success_first_attempt(self):
        """async_send_batch returns result on first successful attempt."""
        mock_session = MagicMock()
        mock_session.close = AsyncMock()

        with patch.object(_async_http_mod, 'client_post',
                          new_callable=AsyncMock) as mock_client_post, \
             patch.object(_async_http_mod, '_executor') as mock_exec:
            mock_client_post.return_value = {"status": 200}

            loop = asyncio.get_running_loop()
            future = loop.create_future()
            future.set_result(b"serialized_bytes")
            with patch.object(loop, 'run_in_executor', return_value=future):
                result = await async_send_batch(
                    self.test_batch, self.test_url, retry=3, backoff=0.1, session=mock_session
                )

        self.assertEqual(result, {"status": 200})
        mock_session.close.assert_not_called()

    async def test_async_send_batch_creates_own_session(self):
        """async_send_batch creates and closes its own session when none provided."""
        mock_session_instance = MagicMock()
        mock_session_instance.close = AsyncMock()

        with patch.object(_async_http_mod, 'client_post',
                          new_callable=AsyncMock) as mock_client_post, \
             patch.object(_async_http_mod, '_executor') as mock_exec, \
             patch.object(_async_http_mod, 'aiohttp') as patched_aiohttp:
            mock_client_post.return_value = {"status": 200}
            patched_aiohttp.ClientSession.return_value = mock_session_instance

            loop = asyncio.get_running_loop()
            future = loop.create_future()
            future.set_result(b"serialized_bytes")
            with patch.object(loop, 'run_in_executor', return_value=future):
                result = await async_send_batch(
                    self.test_batch, self.test_url, retry=1, backoff=0.1
                )

        self.assertEqual(result, {"status": 200})
        patched_aiohttp.ClientSession.assert_called_once()
        mock_session_instance.close.assert_awaited_once()

    async def test_async_send_batch_retry_then_success(self):
        """async_send_batch retries on non-200 status and succeeds eventually."""
        mock_session = MagicMock()
        mock_session.close = AsyncMock()

        with patch.object(_async_http_mod, 'client_post',
                          new_callable=AsyncMock) as mock_client_post, \
             patch.object(_async_http_mod, '_executor') as mock_exec, \
             patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            mock_client_post.side_effect = [
                {"status": 500},
                {"status": 200},
            ]

            loop = asyncio.get_running_loop()
            future = loop.create_future()
            future.set_result(b"serialized_bytes")
            with patch.object(loop, 'run_in_executor', return_value=future):
                result = await async_send_batch(
                    self.test_batch, self.test_url, retry=3, backoff=0.1, session=mock_session
                )

        self.assertEqual(result, {"status": 200})
        self.assertEqual(mock_client_post.call_count, 2)
        mock_sleep.assert_awaited_once_with(0.1)

    async def test_async_send_batch_all_retries_fail(self):
        """async_send_batch returns last failure result when all retries exhausted."""
        mock_session = MagicMock()
        mock_session.close = AsyncMock()

        with patch.object(_async_http_mod, 'client_post',
                          new_callable=AsyncMock) as mock_client_post, \
             patch.object(_async_http_mod, '_executor') as mock_exec, \
             patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            mock_client_post.return_value = {"status": 500}

            loop = asyncio.get_running_loop()
            future = loop.create_future()
            future.set_result(b"serialized_bytes")
            with patch.object(loop, 'run_in_executor', return_value=future):
                result = await async_send_batch(
                    self.test_batch, self.test_url, retry=3, backoff=0.1, session=mock_session
                )

        self.assertEqual(result, {"status": 500})
        self.assertEqual(mock_client_post.call_count, 3)
        # Sleep called twice: after attempt 1 and attempt 2, not after attempt 3
        self.assertEqual(mock_sleep.call_count, 2)

    async def test_async_send_batch_exception_returns_none(self):
        """async_send_batch returns None when an unexpected exception occurs."""
        mock_session = MagicMock()
        mock_session.close = AsyncMock()

        with patch.object(_async_http_mod, 'client_post',
                          new_callable=AsyncMock) as mock_client_post, \
             patch.object(_async_http_mod, '_executor') as mock_exec:
            # Make run_in_executor raise an exception
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            future.set_exception(RuntimeError("Serialization failed"))
            with patch.object(loop, 'run_in_executor', return_value=future):
                result = await async_send_batch(
                    self.test_batch, self.test_url, retry=1, backoff=0.1, session=mock_session
                )

        self.assertIsNone(result)


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


for name in dir(TestClientPost):
    if name.startswith('test_') and asyncio.iscoroutinefunction(getattr(TestClientPost, name)):
        setattr(TestClientPost, name, async_test(getattr(TestClientPost, name)))

for name in dir(TestAsyncSendBatch):
    if name.startswith('test_') and asyncio.iscoroutinefunction(getattr(TestAsyncSendBatch, name)):
        setattr(TestAsyncSendBatch, name, async_test(getattr(TestAsyncSendBatch, name)))


if __name__ == '__main__':
    unittest.main()
