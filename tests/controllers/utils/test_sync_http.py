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
import unittest
from unittest.mock import MagicMock, patch

# Create mock objects before importing code under test
mock_torch = MagicMock()
mock_ray = MagicMock()
mock_requests = MagicMock()

mock_loggers_module = MagicMock()
mock_logger = MagicMock()
mock_loggers_module.Loggers.return_value.get_logger.return_value = mock_logger

# Mock for the utils module constants
mock_utils_module = MagicMock()
mock_utils_module.MIN_RETRY_COUNT = 1
mock_utils_module.DEFAULT_BACKOFF_FACTOR = 30.0
mock_utils_module.DEFAULT_RETRY_COUNT = 3
mock_utils_module.MIN_BACKOFF_FACTOR = 5.0

with patch.dict('sys.modules', {
    'torch': mock_torch,
    'ray': mock_ray,
    'requests': mock_requests,
    'agentic_rl.base.log.loggers': mock_loggers_module,
    'agentic_rl.controllers.utils.utils': mock_utils_module,
}):
    from agentic_rl.controllers.utils.sync_http import sync_send
    import agentic_rl.controllers.utils.sync_http as _sync_http_mod


class TestSyncSend(unittest.TestCase):

    def setUp(self):
        self.test_address = "http://127.0.0.1:8000"
        self.test_url = "http://127.0.0.1:8000/api/send"

    def test_success_first_attempt(self):
        """sync_send returns response JSON on first successful attempt."""
        with patch.object(_sync_http_mod, 'requests') as patched_requests:
            mock_response = MagicMock()
            mock_response.json.return_value = {"result": "ok"}
            patched_requests.post.return_value = mock_response

            result = sync_send(self.test_address, self.test_url)

            self.assertEqual(result, {"result": "ok"})
            patched_requests.post.assert_called_once_with(
                self.test_url,
                data=self.test_address,
                headers={"Content-Type": "text/plain"},
            )
            mock_response.raise_for_status.assert_called_once()
            mock_response.json.assert_called_once()

    @patch('time.sleep')
    def test_retry_then_success(self, mock_sleep):
        """sync_send retries on failure and returns JSON on eventual success."""
        with patch.object(_sync_http_mod, 'requests') as patched_requests:
            mock_fail_response = MagicMock()
            mock_fail_response.raise_for_status.side_effect = Exception("Connection error")

            mock_success_response = MagicMock()
            mock_success_response.json.return_value = {"result": "recovered"}

            patched_requests.post.side_effect = [mock_fail_response, mock_success_response]

            result = sync_send(self.test_address, self.test_url, retry=3, backoff=1.0)

            self.assertEqual(result, {"result": "recovered"})
            self.assertEqual(patched_requests.post.call_count, 2)
            mock_sleep.assert_called_once_with(1.0)

    @patch('time.sleep')
    def test_all_retries_fail_returns_none(self, mock_sleep):
        """sync_send returns None when all retry attempts are exhausted."""
        with patch.object(_sync_http_mod, 'requests') as patched_requests:
            mock_fail_response = MagicMock()
            mock_fail_response.raise_for_status.side_effect = Exception("Server down")

            patched_requests.post.return_value = mock_fail_response

            result = sync_send(self.test_address, self.test_url, retry=3, backoff=0.5)

            self.assertIsNone(result)
            self.assertEqual(patched_requests.post.call_count, 3)
            # Sleep is called between retries, not after the last attempt
            self.assertEqual(mock_sleep.call_count, 2)

    def test_raise_for_status_called(self):
        """sync_send calls raise_for_status on the response."""
        with patch.object(_sync_http_mod, 'requests') as patched_requests:
            mock_response = MagicMock()
            mock_response.json.return_value = {"status": "healthy"}
            patched_requests.post.return_value = mock_response

            sync_send(self.test_address, self.test_url)

            mock_response.raise_for_status.assert_called_once()

    def test_default_params(self):
        """sync_send uses MIN_RETRY_COUNT and DEFAULT_BACKOFF_FACTOR as defaults."""
        with patch.object(_sync_http_mod, 'requests') as patched_requests:
            mock_response = MagicMock()
            mock_response.json.return_value = {"ok": True}
            patched_requests.post.return_value = mock_response

            # With default retry=1 (MIN_RETRY_COUNT), only one attempt is made
            result = sync_send(self.test_address, self.test_url)

            self.assertEqual(result, {"ok": True})
            self.assertEqual(patched_requests.post.call_count, 1)

    @patch('time.sleep')
    def test_custom_retry_and_backoff(self, mock_sleep):
        """sync_send respects custom retry count and backoff factor."""
        with patch.object(_sync_http_mod, 'requests') as patched_requests:
            mock_fail_response = MagicMock()
            mock_fail_response.raise_for_status.side_effect = Exception("timeout")

            mock_success_response = MagicMock()
            mock_success_response.json.return_value = {"data": 42}

            patched_requests.post.side_effect = [
                mock_fail_response,
                mock_fail_response,
                mock_fail_response,
                mock_success_response,
            ]

            result = sync_send(self.test_address, self.test_url, retry=5, backoff=2.5)

            self.assertEqual(result, {"data": 42})
            self.assertEqual(patched_requests.post.call_count, 4)
            # Sleep called 3 times (after attempts 1, 2, 3) with backoff=2.5
            self.assertEqual(mock_sleep.call_count, 3)
            for call_args in mock_sleep.call_args_list:
                self.assertEqual(call_args[0][0], 2.5)

    @patch('time.sleep')
    def test_sleeps_between_retries(self, mock_sleep):
        """sync_send sleeps with the specified backoff between retry attempts."""
        with patch.object(_sync_http_mod, 'requests') as patched_requests:
            mock_fail_response = MagicMock()
            mock_fail_response.raise_for_status.side_effect = Exception("fail")

            patched_requests.post.return_value = mock_fail_response

            sync_send(self.test_address, self.test_url, retry=4, backoff=10.0)

            # 4 attempts means 3 sleeps between them
            self.assertEqual(mock_sleep.call_count, 3)
            for call_args in mock_sleep.call_args_list:
                self.assertEqual(call_args[0][0], 10.0)


if __name__ == '__main__':
    unittest.main()
