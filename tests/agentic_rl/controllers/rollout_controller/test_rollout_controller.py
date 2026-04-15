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
import os
import pytest
import tempfile
from unittest.mock import patch, call, PropertyMock
import threading

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

# Mock FastAPI before importing the module under test
class HTTPException(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)

mock_fastapi = MagicMock()
mock_fastapi.HTTPException = HTTPException
mock_fastapi.APIRouter = MagicMock()
mock_fastapi.Request = MagicMock()
mock_fastapi.UploadFile = MagicMock()
mock_fastapi.File = MagicMock()

sys.modules['fastapi'] = mock_fastapi

# Mock ray before importing the module under test
mock_ray = MagicMock()
mock_ray.remote = lambda func_or_class: func_or_class
mock_ray.get = MagicMock(return_value=None)
mock_ray.get_actor = MagicMock()
mock_ray.kill = MagicMock()
mock_ray.is_initialized = MagicMock(return_value=True)
mock_ray.available_resources = MagicMock(return_value={"CPU": 8})

sys.modules['ray'] = mock_ray
sys.modules['ray.serve'] = MagicMock()
sys.modules['ray.util'] = MagicMock()
sys.modules['ray.util.placement_group'] = MagicMock()
sys.modules['ray.util.scheduling_strategies'] = MagicMock()
sys.modules['ray.exceptions'] = MagicMock()

# Mock uvicorn
sys.modules['uvicorn'] = MagicMock()

from agentic_rl.controllers.rollout_controller.rollout_controller import (
    clean_rollout_weights,
    RolloutController,
)
from agentic_rl.base.utils.globals import ROLLOUT_WEIGHTS_PREFIX


class TestCleanRolloutWeights:
    """Test cases for clean_rollout_weights function."""

    def test_clean_rollout_weights_removes_files(self):
        """Test that clean_rollout_weights removes files in the rollout directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rollout_dir = tmpdir + ROLLOUT_WEIGHTS_PREFIX
            os.makedirs(rollout_dir)

            test_file1 = os.path.join(rollout_dir, "weight1.pt")
            test_file2 = os.path.join(rollout_dir, "weight2.pt")
            with open(test_file1, 'w') as f:
                f.write("test1")
            with open(test_file2, 'w') as f:
                f.write("test2")

            assert os.path.exists(test_file1)
            assert os.path.exists(test_file2)

            clean_rollout_weights(tmpdir)

            assert not os.path.exists(test_file1)
            assert not os.path.exists(test_file2)

    def test_clean_rollout_weights_nonexistent_dir(self):
        """Test that clean_rollout_weights handles nonexistent directory."""
        clean_rollout_weights("/nonexistent/path")

    def test_clean_rollout_weights_empty_dir(self):
        """Test that clean_rollout_weights handles empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rollout_dir = tmpdir + ROLLOUT_WEIGHTS_PREFIX
            os.makedirs(rollout_dir)

            clean_rollout_weights(tmpdir)

            assert os.path.exists(rollout_dir)

    def test_clean_rollout_weights_nested_files(self):
        """Test that clean_rollout_weights removes nested files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rollout_dir = tmpdir + ROLLOUT_WEIGHTS_PREFIX
            nested_dir = os.path.join(rollout_dir, "nested")
            os.makedirs(nested_dir)

            test_file1 = os.path.join(rollout_dir, "weight1.pt")
            test_file2 = os.path.join(nested_dir, "weight2.pt")
            with open(test_file1, 'w') as f:
                f.write("test1")
            with open(test_file2, 'w') as f:
                f.write("test2")

            clean_rollout_weights(tmpdir)

            assert not os.path.exists(test_file1)
            assert not os.path.exists(test_file2)


class TestRolloutController:
    """Test cases for RolloutController class."""

    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.clean_rollout_weights')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.threading.Thread')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.FastAPI')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.RolloutServer')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.RolloutClient')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.create_actor')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.ray.get')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.ControllerConfig')
    def test_init(self, mock_config_class, mock_ray_get, mock_create_actor,
                  mock_rollout_client_class, mock_rollout_server_class,
                  mock_fastapi_class, mock_thread_class, mock_clean_weights):
        """Test RolloutController initialization."""
        mock_config = MagicMock()
        mock_config.rollout_server_addr = "localhost:8080"
        mock_config_class.return_value = mock_config

        mock_client = MagicMock()
        mock_rollout_client_class.return_value = mock_client

        mock_weight_manager = MagicMock()
        mock_queue_actor = MagicMock()
        mock_create_actor.side_effect = [mock_weight_manager, mock_queue_actor]

        mock_server = MagicMock()
        mock_rollout_server_class.return_value = mock_server

        mock_app = MagicMock()
        mock_fastapi_class.return_value = mock_app

        mock_thread = MagicMock()
        mock_thread_class.return_value = mock_thread

        controller = RolloutController(weight_save_dir="/tmp/weights")

        assert controller.rollout_server_addr == "localhost:8080"
        assert controller.rollout_client == mock_client
        assert controller.rollout_weight_manager == mock_weight_manager
        assert controller.rollout_queue_actor == mock_queue_actor
        assert controller.rollout_server == mock_server

    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.clean_rollout_weights')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.threading.Thread')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.FastAPI')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.RolloutServer')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.RolloutClient')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.create_actor')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.ray.get')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.ControllerConfig')
    def test_send_ready_to_train(self, mock_config_class, mock_ray_get, mock_create_actor,
                                 mock_rollout_client_class, mock_rollout_server_class,
                                 mock_fastapi_class, mock_thread_class, mock_clean_weights):
        """Test send_ready_to_train method."""
        mock_config = MagicMock()
        mock_config.rollout_server_addr = "localhost:8080"
        mock_config_class.return_value = mock_config

        mock_client = MagicMock()
        mock_rollout_client_class.return_value = mock_client

        mock_weight_manager = MagicMock()
        mock_queue_actor = MagicMock()
        mock_create_actor.side_effect = [mock_weight_manager, mock_queue_actor]

        mock_server = MagicMock()
        mock_rollout_server_class.return_value = mock_server

        mock_app = MagicMock()
        mock_fastapi_class.return_value = mock_app

        controller = RolloutController(weight_save_dir="/tmp/weights")
        controller.send_ready_to_train()

        mock_client.send_ready_to_train.assert_called_once()

    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.clean_rollout_weights')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.threading.Thread')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.FastAPI')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.RolloutServer')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.RolloutClient')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.create_actor')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.ray.get')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.ControllerConfig')
    def test_get_weight_manager(self, mock_config_class, mock_ray_get, mock_create_actor,
                                mock_rollout_client_class, mock_rollout_server_class,
                                mock_fastapi_class, mock_thread_class, mock_clean_weights):
        """Test get_weight_manager method."""
        mock_config = MagicMock()
        mock_config.rollout_server_addr = "localhost:8080"
        mock_config_class.return_value = mock_config

        mock_client = MagicMock()
        mock_rollout_client_class.return_value = mock_client

        mock_weight_manager = MagicMock()
        mock_queue_actor = MagicMock()
        mock_create_actor.side_effect = [mock_weight_manager, mock_queue_actor]

        mock_server = MagicMock()
        mock_rollout_server_class.return_value = mock_server

        mock_app = MagicMock()
        mock_fastapi_class.return_value = mock_app

        controller = RolloutController(weight_save_dir="/tmp/weights")
        result = controller.get_weight_manager()

        assert result == mock_weight_manager

    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.clean_rollout_weights')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.threading.Thread')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.FastAPI')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.RolloutServer')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.RolloutClient')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.create_actor')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.ray.get')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.ControllerConfig')
    def test_initialize_rollout_queue_actor(self, mock_config_class, mock_ray_get, mock_create_actor,
                                            mock_rollout_client_class, mock_rollout_server_class,
                                            mock_fastapi_class, mock_thread_class, mock_clean_weights):
        """Test initialize_rollout_queue_actor method."""
        mock_config = MagicMock()
        mock_config.rollout_server_addr = "localhost:8080"
        mock_config_class.return_value = mock_config

        mock_client = MagicMock()
        mock_rollout_client_class.return_value = mock_client

        mock_weight_manager = MagicMock()
        mock_queue_actor = MagicMock()
        mock_create_actor.side_effect = [mock_weight_manager, mock_queue_actor]

        mock_server = MagicMock()
        mock_rollout_server_class.return_value = mock_server

        mock_app = MagicMock()
        mock_fastapi_class.return_value = mock_app

        controller = RolloutController(weight_save_dir="/tmp/weights")

        assert controller.rollout_queue_actor == mock_queue_actor
        mock_ray_get.assert_called()

    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.clean_rollout_weights')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.threading.Thread')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.FastAPI')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.RolloutServer')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.RolloutClient')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.create_actor')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.ray.get')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.ControllerConfig')
    def test_initialize_rollout_server(self, mock_config_class, mock_ray_get, mock_create_actor,
                                       mock_rollout_client_class, mock_rollout_server_class,
                                       mock_fastapi_class, mock_thread_class, mock_clean_weights):
        """Test initialize_rollout_server method."""
        mock_config = MagicMock()
        mock_config.rollout_server_addr = "localhost:8080"
        mock_config_class.return_value = mock_config

        mock_client = MagicMock()
        mock_rollout_client_class.return_value = mock_client

        mock_weight_manager = MagicMock()
        mock_queue_actor = MagicMock()
        mock_create_actor.side_effect = [mock_weight_manager, mock_queue_actor]

        mock_server = MagicMock()
        mock_router = MagicMock()
        mock_server.router = mock_router
        mock_rollout_server_class.return_value = mock_server

        mock_app = MagicMock()
        mock_fastapi_class.return_value = mock_app

        mock_thread = MagicMock()
        mock_thread_class.return_value = mock_thread

        controller = RolloutController(weight_save_dir="/tmp/weights")

        assert controller.rollout_server == mock_server
        mock_app.include_router.assert_called_once_with(mock_router)
        mock_thread_class.assert_called_once()
        mock_thread.start.assert_called_once()

    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.clean_rollout_weights')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.threading.Thread')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.FastAPI')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.RolloutServer')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.RolloutClient')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.create_actor')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.ray.get')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.ControllerConfig')
    def test_initialize_rollout_weight_manager(self, mock_config_class, mock_ray_get, mock_create_actor,
                                               mock_rollout_client_class, mock_rollout_server_class,
                                               mock_fastapi_class, mock_thread_class, mock_clean_weights):
        """Test initialize_rollout_weight_manager method."""
        mock_config = MagicMock()
        mock_config.rollout_server_addr = "localhost:8080"
        mock_config_class.return_value = mock_config

        mock_client = MagicMock()
        mock_rollout_client_class.return_value = mock_client

        mock_weight_manager = MagicMock()
        mock_queue_actor = MagicMock()
        mock_create_actor.side_effect = [mock_weight_manager, mock_queue_actor]

        mock_server = MagicMock()
        mock_rollout_server_class.return_value = mock_server

        mock_app = MagicMock()
        mock_fastapi_class.return_value = mock_app

        controller = RolloutController(weight_save_dir="/tmp/weights")

        assert controller.rollout_weight_manager == mock_weight_manager
        mock_clean_weights.assert_called_once_with("/tmp/weights")

    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.clean_rollout_weights')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.threading.Thread')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.FastAPI')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.RolloutServer')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.RolloutClient')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.create_actor')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.ray.get')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.ControllerConfig')
    def test_running(self, mock_config_class, mock_ray_get, mock_create_actor,
                     mock_rollout_client_class, mock_rollout_server_class,
                     mock_fastapi_class, mock_thread_class, mock_clean_weights):
        """Test running method."""
        mock_config = MagicMock()
        mock_config.rollout_server_addr = "localhost:8080"
        mock_config_class.return_value = mock_config

        mock_client = MagicMock()
        mock_rollout_client_class.return_value = mock_client

        mock_weight_manager = MagicMock()
        mock_queue_actor = MagicMock()
        mock_create_actor.side_effect = [mock_weight_manager, mock_queue_actor]

        mock_server = MagicMock()
        mock_server.running.return_value = True
        mock_rollout_server_class.return_value = mock_server

        mock_app = MagicMock()
        mock_fastapi_class.return_value = mock_app

        controller = RolloutController(weight_save_dir="/tmp/weights")
        result = controller.running()

        assert result is True
        mock_server.running.assert_called_once()

    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.time.sleep')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.clean_rollout_weights')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.threading.Thread')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.FastAPI')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.RolloutServer')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.RolloutClient')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.create_actor')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.ray.get')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.ray.kill')
    @patch('agentic_rl.controllers.rollout_controller.rollout_controller.ControllerConfig')
    def test_finish_rollout(self, mock_config_class, mock_ray_kill, mock_ray_get,
                            mock_create_actor, mock_rollout_client_class,
                            mock_rollout_server_class, mock_fastapi_class,
                            mock_thread_class, mock_clean_weights, mock_sleep):
        """Test finish_rollout method."""
        mock_config = MagicMock()
        mock_config.rollout_server_addr = "localhost:8080"
        mock_config_class.return_value = mock_config

        mock_client = MagicMock()
        mock_rollout_client_class.return_value = mock_client

        mock_weight_manager = MagicMock()
        mock_queue_actor = MagicMock()
        mock_create_actor.side_effect = [mock_weight_manager, mock_queue_actor]

        mock_server = MagicMock()
        type(mock_server).is_shutdown = PropertyMock(side_effect=[False, True])
        mock_rollout_server_class.return_value = mock_server

        mock_app = MagicMock()
        mock_fastapi_class.return_value = mock_app

        controller = RolloutController(weight_save_dir="/tmp/weights")
        controller.finish_rollout()

        mock_sleep.assert_called_once_with(3)
        assert mock_ray_kill.call_count == 2
        mock_ray_kill.assert_any_call(mock_queue_actor)
        mock_ray_kill.assert_any_call(mock_weight_manager)


@pytest.fixture(scope="module", autouse=True)
def cleanup_module():
    """Cleanup mock modules after all tests in this module."""
    yield
    modules_to_clean = ['openai', 'vertexai', 'google', 'google.cloud', 
                       'google.cloud.aiplatform_v1beta1', 'google.cloud.aiplatform_v1beta1.types',
                       'google.cloud.aiplatform_v1beta1.types.content', 'sentence_transformers',
                       'vertexai.generative_models', 'fastapi', 'ray', 'ray.serve', 'ray.util',
                       'ray.util.placement_group', 'ray.util.scheduling_strategies', 
                       'ray.exceptions', 'uvicorn']
    for mod in modules_to_clean:
        if mod in sys.modules:
            del sys.modules[mod]
