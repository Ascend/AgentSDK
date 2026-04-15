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
from unittest.mock import MagicMock, patch, call

# Create mock objects before importing code under test
mock_torch = MagicMock()
mock_ray = MagicMock()
mock_requests = MagicMock()

mock_loggers_module = MagicMock()
mock_logger = MagicMock()
mock_loggers_module.Loggers.return_value.get_logger.return_value = mock_logger

with patch.dict('sys.modules', {
    'torch': mock_torch,
    'ray': mock_ray,
    'requests': mock_requests,
    'agentic_rl.base.log.loggers': mock_loggers_module,
}):
    from agentic_rl.controllers.utils.utils import (
        post_with_url,
        tensor_item,
        create_actor,
        collator,
        MIN_RETRY_COUNT,
        DEFAULT_RETRY_COUNT,
        DEFAULT_BACKOFF_FACTOR,
        MIN_BACKOFF_FACTOR,
        MIN_SLEEP_TIME,
        DEFAULT_SLEEP_TIME,
        DEFAULT_TIMEOUT,
        READ_TIMEOUT,
        MAX_TIMEOUT,
        HEALTH_CHECK_TIMEOUT,
        DEFAULT_URL_METHOD,
        DEFAULT_REPLICAS,
        MAX_ONGOING_REQUESTS,
        DEFAULT_CPUS,
        MAX_CPUS,
        MAX_CONCURRENCY,
    )
    import agentic_rl.controllers.utils.utils as _utils_mod


class TestConstants(unittest.TestCase):

    def test_retry_constants(self):
        """Verify retry-related constants have expected values."""
        self.assertEqual(MIN_RETRY_COUNT, 1)
        self.assertEqual(DEFAULT_RETRY_COUNT, 3)
        self.assertEqual(DEFAULT_BACKOFF_FACTOR, 30.0)
        self.assertEqual(MIN_BACKOFF_FACTOR, 5.0)

    def test_timing_constants(self):
        """Verify timing-related constants have expected values."""
        self.assertEqual(MIN_SLEEP_TIME, 0.1)
        self.assertEqual(DEFAULT_SLEEP_TIME, 2)
        self.assertEqual(DEFAULT_TIMEOUT, 30)
        self.assertEqual(READ_TIMEOUT, 600)
        self.assertEqual(MAX_TIMEOUT, 1800)
        self.assertEqual(HEALTH_CHECK_TIMEOUT, 300)

    def test_resource_constants(self):
        """Verify resource-related constants have expected values."""
        self.assertEqual(DEFAULT_URL_METHOD, "http")
        self.assertEqual(DEFAULT_REPLICAS, 1)
        self.assertEqual(MAX_ONGOING_REQUESTS, 64)
        self.assertEqual(DEFAULT_CPUS, 1)
        self.assertEqual(MAX_CPUS, 4)
        self.assertEqual(MAX_CONCURRENCY, 128)


class TestPostWithUrl(unittest.TestCase):

    def setUp(self):
        self.test_url = "http://127.0.0.1:8000/api/train"

    def test_post_with_url_success(self):
        """post_with_url returns response JSON on first successful attempt."""
        with patch.object(_utils_mod, 'requests') as patched_requests:
            mock_response = MagicMock()
            mock_response.json.return_value = {"result": "ok"}
            patched_requests.post.return_value = mock_response

            result = post_with_url(self.test_url)

            self.assertEqual(result, {"result": "ok"})
            patched_requests.post.assert_called_once_with(self.test_url)
            mock_response.raise_for_status.assert_called_once()
            mock_response.json.assert_called_once()

    @patch('time.sleep')
    def test_post_with_url_retry_then_success(self, mock_sleep):
        """post_with_url retries on failure and returns JSON on eventual success."""
        with patch.object(_utils_mod, 'requests') as patched_requests:
            mock_fail_response = MagicMock()
            mock_fail_response.raise_for_status.side_effect = Exception("Connection refused")

            mock_success_response = MagicMock()
            mock_success_response.json.return_value = {"status": "recovered"}

            patched_requests.post.side_effect = [mock_fail_response, mock_success_response]

            result = post_with_url(self.test_url, retry=3, backoff=1.0)

            self.assertEqual(result, {"status": "recovered"})
            self.assertEqual(patched_requests.post.call_count, 2)
            mock_sleep.assert_called_once_with(1.0)

    @patch('time.sleep')
    def test_post_with_url_all_retries_fail_raises(self, mock_sleep):
        """post_with_url raises the last exception when all retries are exhausted."""
        with patch.object(_utils_mod, 'requests') as patched_requests:
            mock_fail_response = MagicMock()
            mock_fail_response.raise_for_status.side_effect = Exception("Server down")

            patched_requests.post.return_value = mock_fail_response

            with self.assertRaises(Exception) as ctx:
                post_with_url(self.test_url, retry=3, backoff=0.5)

            self.assertIn("Server down", str(ctx.exception))
            self.assertEqual(patched_requests.post.call_count, 3)
            # Sleep called between retries (attempts 1->2, 2->3), not after final
            self.assertEqual(mock_sleep.call_count, 2)

    def test_post_with_url_default_params(self):
        """post_with_url uses MIN_RETRY_COUNT as default retry (single attempt)."""
        with patch.object(_utils_mod, 'requests') as patched_requests:
            mock_fail_response = MagicMock()
            mock_fail_response.raise_for_status.side_effect = Exception("Fail")

            patched_requests.post.return_value = mock_fail_response

            # Default retry=1 means single attempt, raises on failure
            with self.assertRaises(Exception):
                post_with_url(self.test_url)

            self.assertEqual(patched_requests.post.call_count, 1)


class TestTensorItem(unittest.TestCase):

    def test_tensor_item_with_tensor(self):
        """tensor_item calls .item() when input is a tensor."""
        with patch.object(_utils_mod, 'torch') as patched_torch:
            mock_tensor = MagicMock()
            mock_tensor.item.return_value = 42
            patched_torch.is_tensor.return_value = True

            result = tensor_item(mock_tensor)

            self.assertEqual(result, 42)
            patched_torch.is_tensor.assert_called_once_with(mock_tensor)
            mock_tensor.item.assert_called_once()

    def test_tensor_item_with_non_tensor(self):
        """tensor_item returns int(x) when input is not a tensor."""
        with patch.object(_utils_mod, 'torch') as patched_torch:
            patched_torch.is_tensor.return_value = False

            result = tensor_item(5.7)

            self.assertEqual(result, 5)
            self.assertIsInstance(result, int)
            patched_torch.is_tensor.assert_called_once_with(5.7)

    def test_tensor_item_with_integer(self):
        """tensor_item returns the same integer when input is already int."""
        with patch.object(_utils_mod, 'torch') as patched_torch:
            patched_torch.is_tensor.return_value = False

            result = tensor_item(10)

            self.assertEqual(result, 10)
            self.assertIsInstance(result, int)


class TestCreateActor(unittest.TestCase):

    def setUp(self):
        self.mock_cls = MagicMock()
        self.mock_actor_handle = MagicMock()
        self.mock_cls.options.return_value.remote.return_value = self.mock_actor_handle

    def test_create_actor_existing_actor_killed_and_recreated(self):
        """create_actor kills an existing actor and creates a new one."""
        with patch.object(_utils_mod, 'ray') as patched_ray:
            existing_actor = MagicMock()
            patched_ray.get_actor.return_value = existing_actor

            result = create_actor(name="test_actor", cls=self.mock_cls, namespace="ns")

            patched_ray.get_actor.assert_called_once_with("test_actor", namespace="ns")
            patched_ray.kill.assert_called_once_with(existing_actor)
            self.mock_cls.options.assert_called_once_with(
                name="test_actor", namespace="ns", lifetime="detached"
            )
            self.mock_cls.options.return_value.remote.assert_called_once_with()
            self.assertEqual(result, self.mock_actor_handle)

    def test_create_actor_no_existing_actor(self):
        """create_actor creates a new actor when no existing actor found."""
        with patch.object(_utils_mod, 'ray') as patched_ray:
            patched_ray.get_actor.side_effect = ValueError("Actor not found")

            result = create_actor(name="new_actor", cls=self.mock_cls)

            patched_ray.get_actor.assert_called_once_with("new_actor", namespace=None)
            patched_ray.kill.assert_not_called()
            self.mock_cls.options.assert_called_once_with(
                name="new_actor", namespace=None, lifetime="detached"
            )
            self.mock_cls.options.return_value.remote.assert_called_once_with()
            self.assertEqual(result, self.mock_actor_handle)

    def test_create_actor_with_options_and_kwargs(self):
        """create_actor passes custom options and actor_kwargs through."""
        with patch.object(_utils_mod, 'ray') as patched_ray:
            patched_ray.get_actor.side_effect = ValueError("Actor not found")

            result = create_actor(
                name="custom_actor",
                cls=self.mock_cls,
                namespace="custom_ns",
                lifetime="non-detached",
                options={"num_cpus": 2},
                actor_args=(1, 2),
                actor_kwargs={"key": "value"},
            )

            self.mock_cls.options.assert_called_once_with(
                name="custom_actor",
                namespace="custom_ns",
                lifetime="non-detached",
                num_cpus=2,
            )
            self.mock_cls.options.return_value.remote.assert_called_once_with(1, 2, key="value")
            self.assertEqual(result, self.mock_actor_handle)

    def test_create_actor_default_options_none(self):
        """create_actor handles None options and actor_kwargs by defaulting to empty."""
        with patch.object(_utils_mod, 'ray') as patched_ray:
            patched_ray.get_actor.side_effect = ValueError("Not found")

            result = create_actor(
                name="default_actor",
                cls=self.mock_cls,
                options=None,
                actor_kwargs=None,
            )

            self.mock_cls.options.assert_called_once_with(
                name="default_actor", namespace=None, lifetime="detached"
            )
            self.mock_cls.options.return_value.remote.assert_called_once_with()
            self.assertEqual(result, self.mock_actor_handle)


class TestCollator(unittest.TestCase):

    def test_collator_basic(self):
        """collator extracts prompts and responses from features."""
        with patch.object(_utils_mod, 'torch') as patched_torch:
            features = [
                {"input_ids": [1, 2, 3], "response_ids": [4, 5, 6]},
                {"input_ids": [7, 8, 9], "response_ids": [10, 11, 12]},
            ]

            patched_torch.tensor.side_effect = lambda x: x

            result = collator(features)

            self.assertIn("prompts", result)
            self.assertIn("responses", result)
            self.assertEqual(len(result["prompts"]), 2)
            self.assertEqual(len(result["responses"]), 2)
            self.assertEqual(result["prompts"][0], [1, 2, 3])
            self.assertEqual(result["prompts"][1], [7, 8, 9])
            self.assertEqual(result["responses"][0], [4, 5, 6])
            self.assertEqual(result["responses"][1], [10, 11, 12])

    def test_collator_no_additional_keys(self):
        """collator with no additional keys returns only prompts and responses."""
        with patch.object(_utils_mod, 'torch') as patched_torch:
            features = [
                {"input_ids": [1], "response_ids": [2]},
            ]

            patched_torch.tensor.side_effect = lambda x: x

            result = collator(features, dataset_additional_keys=None)

            self.assertEqual(set(result.keys()), {"prompts", "responses"})
            self.assertEqual(len(result["prompts"]), 1)
            self.assertEqual(len(result["responses"]), 1)

    def test_collator_with_additional_keys(self):
        """collator includes additional keys when specified."""
        with patch.object(_utils_mod, 'torch') as patched_torch:
            features = [
                {"input_ids": [1], "response_ids": [2], "reward": [0.5], "mask": [1, 1]},
                {"input_ids": [3], "response_ids": [4], "reward": [0.8], "mask": [1, 0]},
            ]

            patched_torch.tensor.side_effect = lambda x: x

            result = collator(features, dataset_additional_keys=["reward", "mask"])

            self.assertIn("prompts", result)
            self.assertIn("responses", result)
            self.assertIn("reward", result)
            self.assertIn("mask", result)
            self.assertEqual(len(result["reward"]), 2)
            self.assertEqual(len(result["mask"]), 2)
            self.assertEqual(result["reward"][0], [0.5])
            self.assertEqual(result["reward"][1], [0.8])
            self.assertEqual(result["mask"][0], [1, 1])
            self.assertEqual(result["mask"][1], [1, 0])

    def test_collator_calls_torch_tensor(self):
        """collator wraps each value with torch.tensor."""
        with patch.object(_utils_mod, 'torch') as patched_torch:
            features = [
                {"input_ids": [10], "response_ids": [20]},
            ]

            patched_torch.tensor.side_effect = lambda x: x

            collator(features)

            # torch.tensor called once for input_ids and once for response_ids
            self.assertEqual(patched_torch.tensor.call_count, 2)
            patched_torch.tensor.assert_any_call([10])
            patched_torch.tensor.assert_any_call([20])

    def test_collator_empty_features(self):
        """collator returns empty lists for prompts and responses with no features."""
        with patch.object(_utils_mod, 'torch') as patched_torch:
            patched_torch.tensor.side_effect = lambda x: x

            result = collator([])

            self.assertEqual(result["prompts"], [])
            self.assertEqual(result["responses"], [])


if __name__ == '__main__':
    unittest.main()
