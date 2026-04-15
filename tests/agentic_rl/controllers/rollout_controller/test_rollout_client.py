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

import sys
import importlib.util
from unittest.mock import MagicMock
import pytest
import io
from unittest.mock import patch, call
import requests

# Mock openai before importing the module under test
mock_openai = MagicMock()
mock_openai.__spec__ = importlib.util.spec_from_loader('openai', loader=None)
sys.modules['openai'] = mock_openai

# Mock vertexai and google.cloud before importing the module under test
sys.modules['vertexai'] = MagicMock()
sys.modules['google'] = MagicMock()
sys.modules['google.cloud'] = MagicMock()
sys.modules['google.cloud.aiplatform_v1beta1'] = MagicMock()
sys.modules['google.cloud.aiplatform_v1beta1.types'] = MagicMock()
sys.modules['google.cloud.aiplatform_v1beta1.types.content'] = MagicMock()

# Mock sentence_transformers before importing the module under test
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['vertexai.generative_models'] = MagicMock()

# Mock ray before importing the module under test
mock_ray = MagicMock()
mock_ray.remote = lambda func_or_class: func_or_class
mock_ray.get = MagicMock(return_value=None)
mock_ray.get_actor = MagicMock()
mock_ray.kill = MagicMock()
mock_ray.is_initialized = MagicMock(return_value=True)
mock_ray.available_resources = MagicMock(return_value={"CPU": 8})

sys.modules['ray'] = mock_ray

from agentic_rl.controllers.rollout_controller.rollout_client import (
    send_ready_to_train_remote,
    send_outputs_to_train_server_remote,
    RolloutClient,
)
from agentic_rl.controllers.utils.http_status import HTTP_OK_200
from agentic_rl.controllers.utils.utils import DEFAULT_SLEEP_TIME, READ_TIMEOUT, DEFAULT_URL_METHOD


class TestSendReadyToTrainRemote:
    """Test cases for send_ready_to_train_remote function."""

    @patch('agentic_rl.controllers.rollout_controller.rollout_client.time.sleep')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.requests.post')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.get_rollout_queue_actor')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.ray.get')
    def test_send_ready_success_first_attempt(self, mock_ray_get, mock_get_actor, mock_post, mock_sleep):
        """Test successful send on first attempt."""
        mock_actor = MagicMock()
        mock_get_actor.return_value = mock_actor
        mock_ray_get.return_value = False

        mock_response = MagicMock()
        mock_response.status_code = HTTP_OK_200
        mock_post.return_value = mock_response

        send_ready_to_train_remote("localhost:8080")

        mock_post.assert_called_once_with(f"{DEFAULT_URL_METHOD}://localhost:8080/train/is_ready")
        mock_sleep.assert_not_called()

    @patch('agentic_rl.controllers.rollout_controller.rollout_client.time.sleep')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.requests.post')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.get_rollout_queue_actor')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.ray.get')
    def test_send_ready_retry_on_failure(self, mock_ray_get, mock_get_actor, mock_post, mock_sleep):
        """Test retry on non-200 status code."""
        mock_actor = MagicMock()
        mock_get_actor.return_value = mock_actor
        mock_ray_get.return_value = False

        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 500
        mock_response_success = MagicMock()
        mock_response_success.status_code = HTTP_OK_200
        mock_post.side_effect = [mock_response_fail, mock_response_success]

        send_ready_to_train_remote("localhost:8080")

        assert mock_post.call_count == 2
        mock_sleep.assert_called_with(DEFAULT_SLEEP_TIME)

    @patch('agentic_rl.controllers.rollout_controller.rollout_client.time.sleep')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.requests.post')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.get_rollout_queue_actor')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.ray.get')
    def test_send_ready_retry_on_exception(self, mock_ray_get, mock_get_actor, mock_post, mock_sleep):
        """Test retry on request exception."""
        mock_actor = MagicMock()
        mock_get_actor.return_value = mock_actor
        mock_ray_get.return_value = False

        mock_response = MagicMock()
        mock_response.status_code = HTTP_OK_200
        mock_post.side_effect = [requests.RequestException("Connection error"), mock_response]

        send_ready_to_train_remote("localhost:8080")

        assert mock_post.call_count == 2
        mock_sleep.assert_called_with(DEFAULT_SLEEP_TIME)

    @patch('agentic_rl.controllers.rollout_controller.rollout_client.time.sleep')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.requests.post')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.get_rollout_queue_actor')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.ray.get')
    def test_send_ready_stops_on_shutdown(self, mock_ray_get, mock_get_actor, mock_post, mock_sleep):
        """Test that loop stops when shutdown is signaled."""
        mock_actor = MagicMock()
        mock_get_actor.return_value = mock_actor
        mock_ray_get.side_effect = [False, True]

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        send_ready_to_train_remote("localhost:8080")

        assert mock_post.call_count == 1
        assert mock_ray_get.call_count == 2


class TestSendOutputsToTrainServerRemote:
    """Test cases for send_outputs_to_train_server_remote function."""

    @patch('agentic_rl.controllers.rollout_controller.rollout_client.time.sleep')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.torch.save')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.requests.post')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.get_rollout_queue_actor')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.ray.get')
    def test_send_outputs_success(self, mock_ray_get, mock_get_actor, mock_post, mock_torch_save, mock_sleep):
        """Test successful send outputs."""
        mock_actor = MagicMock()
        mock_get_actor.return_value = mock_actor
        mock_ray_get.return_value = False

        mock_response = MagicMock()
        mock_response.status_code = HTTP_OK_200
        mock_response.json.return_value = {"status": "ok"}
        mock_post.return_value = mock_response

        outputs = {"tensor": "data"}
        metric = {"accuracy": 0.95}

        result = send_outputs_to_train_server_remote("localhost:8080", outputs, metric)

        assert result == {"status": "ok"}
        mock_post.assert_called_once()
        mock_sleep.assert_not_called()

    @patch('agentic_rl.controllers.rollout_controller.rollout_client.time.sleep')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.torch.save')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.requests.post')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.get_rollout_queue_actor')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.ray.get')
    def test_send_outputs_retry_on_non_200(self, mock_ray_get, mock_get_actor, mock_post, mock_torch_save, mock_sleep):
        """Test retry on non-200 status code."""
        mock_actor = MagicMock()
        mock_get_actor.return_value = mock_actor
        mock_ray_get.return_value = False

        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 500
        mock_response_success = MagicMock()
        mock_response_success.status_code = HTTP_OK_200
        mock_response_success.json.return_value = {"status": "ok"}
        mock_post.side_effect = [mock_response_fail, mock_response_success]

        outputs = {"tensor": "data"}
        metric = {"accuracy": 0.95}

        result = send_outputs_to_train_server_remote("localhost:8080", outputs, metric)

        assert result == {"status": "ok"}
        assert mock_post.call_count == 2
        mock_sleep.assert_called_with(DEFAULT_SLEEP_TIME)

    @patch('agentic_rl.controllers.rollout_controller.rollout_client.time.sleep')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.torch.save')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.requests.post')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.get_rollout_queue_actor')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.ray.get')
    def test_send_outputs_retry_on_request_exception(self, mock_ray_get, mock_get_actor, mock_post, mock_torch_save, mock_sleep):
        """Test retry on RequestException."""
        mock_actor = MagicMock()
        mock_get_actor.return_value = mock_actor
        mock_ray_get.return_value = False

        mock_response = MagicMock()
        mock_response.status_code = HTTP_OK_200
        mock_response.json.return_value = {"status": "ok"}
        mock_post.side_effect = [requests.RequestException("Connection error"), mock_response]

        outputs = {"tensor": "data"}
        metric = {"accuracy": 0.95}

        result = send_outputs_to_train_server_remote("localhost:8080", outputs, metric)

        assert result == {"status": "ok"}
        assert mock_post.call_count == 2

    @patch('agentic_rl.controllers.rollout_controller.rollout_client.time.sleep')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.torch.save')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.requests.post')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.get_rollout_queue_actor')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.ray.get')
    def test_send_outputs_retry_on_generic_exception(self, mock_ray_get, mock_get_actor, mock_post, mock_torch_save, mock_sleep):
        """Test retry on generic exception."""
        mock_actor = MagicMock()
        mock_get_actor.return_value = mock_actor
        mock_ray_get.return_value = False

        mock_response = MagicMock()
        mock_response.status_code = HTTP_OK_200
        mock_response.json.return_value = {"status": "ok"}
        mock_post.side_effect = [Exception("Generic error"), mock_response]

        outputs = {"tensor": "data"}
        metric = {"accuracy": 0.95}

        result = send_outputs_to_train_server_remote("localhost:8080", outputs, metric)

        assert result == {"status": "ok"}
        assert mock_post.call_count == 2

    @patch('agentic_rl.controllers.rollout_controller.rollout_client.time.sleep')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.torch.save')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.requests.post')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.get_rollout_queue_actor')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.ray.get')
    def test_send_outputs_stops_on_shutdown(self, mock_ray_get, mock_get_actor, mock_post, mock_torch_save, mock_sleep):
        """Test that loop stops when shutdown is signaled."""
        mock_actor = MagicMock()
        mock_get_actor.return_value = mock_actor
        mock_ray_get.side_effect = [False, True, True]

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        outputs = {"tensor": "data"}
        metric = {"accuracy": 0.95}

        result = send_outputs_to_train_server_remote("localhost:8080", outputs, metric)

        assert result is None
        assert mock_post.call_count == 1

    @patch('agentic_rl.controllers.rollout_controller.rollout_client.time.sleep')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.torch.save')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.requests.post')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.get_rollout_queue_actor')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.ray.get')
    def test_send_outputs_with_custom_backoff(self, mock_ray_get, mock_get_actor, mock_post, mock_torch_save, mock_sleep):
        """Test send outputs with custom backoff time."""
        mock_actor = MagicMock()
        mock_get_actor.return_value = mock_actor
        mock_ray_get.return_value = False

        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 500
        mock_response_success = MagicMock()
        mock_response_success.status_code = HTTP_OK_200
        mock_response_success.json.return_value = {"status": "ok"}
        mock_post.side_effect = [mock_response_fail, mock_response_success]

        outputs = {"tensor": "data"}
        metric = {"accuracy": 0.95}
        custom_backoff = 5

        result = send_outputs_to_train_server_remote("localhost:8080", outputs, metric, backoff=custom_backoff)

        assert result == {"status": "ok"}
        mock_sleep.assert_called_with(custom_backoff)


class TestRolloutClient:
    """Test cases for RolloutClient class."""

    @patch('agentic_rl.controllers.rollout_controller.rollout_client.ControllerConfig')
    def test_rollout_client_init(self, mock_config_class):
        """Test RolloutClient initialization."""
        mock_config = MagicMock()
        mock_config.train_server_addr = "localhost:8080"
        mock_config_class.return_value = mock_config

        client = RolloutClient()

        assert client.train_server_addr == "localhost:8080"

    @patch('agentic_rl.controllers.rollout_controller.rollout_client.send_outputs_to_train_server_remote')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.ControllerConfig')
    def test_send_outputs_to_train_server(self, mock_config_class, mock_send_outputs):
        """Test send_outputs_to_train_server method."""
        mock_config = MagicMock()
        mock_config.train_server_addr = "localhost:8080"
        mock_config_class.return_value = mock_config

        client = RolloutClient()
        outputs = {"tensor": "data"}
        metric = {"accuracy": 0.95}

        client.send_outputs_to_train_server(outputs, metric)

        mock_send_outputs.assert_called_once_with("localhost:8080", outputs, metric)

    @patch('agentic_rl.controllers.rollout_controller.rollout_client.send_ready_to_train_remote')
    @patch('agentic_rl.controllers.rollout_controller.rollout_client.ControllerConfig')
    def test_send_ready_to_train(self, mock_config_class, mock_send_ready):
        """Test send_ready_to_train method."""
        mock_config = MagicMock()
        mock_config.train_server_addr = "localhost:8080"
        mock_config_class.return_value = mock_config

        client = RolloutClient()

        client.send_ready_to_train()

        mock_send_ready.assert_called_once_with("localhost:8080")

    @patch('agentic_rl.controllers.rollout_controller.rollout_client.ControllerConfig')
    def test_rollout_client_singleton(self, mock_config_class):
        """Test that RolloutClient is a singleton."""
        mock_config = MagicMock()
        mock_config.train_server_addr = "localhost:8080"
        mock_config_class.return_value = mock_config

        client1 = RolloutClient()
        client2 = RolloutClient()

        assert client1 is client2


@pytest.fixture(scope="module", autouse=True)
def cleanup_module():
    """Reset singleton and cleanup mock modules after all tests in this module."""
    yield
    if hasattr(RolloutClient, '__wrapped__'):
        RolloutClient.__wrapped__ = None
    modules_to_clean = ['openai', 'vertexai', 'google', 'google.cloud', 
                       'google.cloud.aiplatform_v1beta1', 'google.cloud.aiplatform_v1beta1.types',
                       'google.cloud.aiplatform_v1beta1.types.content', 'sentence_transformers',
                       'vertexai.generative_models', 'ray']
    for mod in modules_to_clean:
        if mod in sys.modules:
            del sys.modules[mod]
