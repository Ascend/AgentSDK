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

import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import io

# Mock FastAPI before importing the module under test
import sys

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

from agentic_rl.controllers.rollout_controller.rollout_server import RolloutServer
from agentic_rl.controllers.utils.http_status import HTTP_ERROR_400


# Create a module-level event loop for all tests
_module_loop = asyncio.new_event_loop()


def run_async(coro):
    return _module_loop.run_until_complete(coro)


class TestRolloutServer:
    """Test cases for RolloutServer class."""

    def test_init(self):
        """Test RolloutServer initialization."""
        mock_queue = MagicMock()
        mock_weight_manager = MagicMock()

        server = RolloutServer(mock_queue, mock_weight_manager)

        assert server.running is False
        assert server.rollout_queue == mock_queue
        assert server.rollout_weight_manager == mock_weight_manager
        assert server.is_shutdown is False
        assert server.router is not None

    def test_unlock(self):
        """Test unlock method."""
        mock_queue = MagicMock()
        mock_queue.set_running = AsyncMock()
        mock_weight_manager = MagicMock()

        server = RolloutServer(mock_queue, mock_weight_manager)
        result = run_async(server.unlock())

        assert server.running is True
        mock_queue.set_running.remote.assert_called_once()
        assert result == {"Status": "Running"}

    def test_lock(self):
        """Test lock method."""
        mock_queue = MagicMock()
        mock_queue.set_block = AsyncMock()
        mock_weight_manager = MagicMock()

        server = RolloutServer(mock_queue, mock_weight_manager)
        server.running = True

        result = run_async(server.lock())

        assert server.running is False
        mock_queue.set_block.remote.assert_called_once()
        assert result == {"Status": "Blocked"}

    def test_receive_batch_success(self):
        """Test receive_batch method with successful batch loading."""
        mock_queue = MagicMock()
        mock_queue.add_queue = AsyncMock()
        mock_weight_manager = MagicMock()

        mock_file = MagicMock()
        mock_file.read = AsyncMock(return_value=b"test_data")

        server = RolloutServer(mock_queue, mock_weight_manager)

        with patch('agentic_rl.controllers.rollout_controller.rollout_server.torch.load') as mock_load:
            mock_load.return_value = {"batch": "data"}

            result = run_async(server.receive_batch(mock_file))

            mock_file.read.assert_called_once()
            mock_load.assert_called_once()
            mock_queue.add_queue.remote.assert_called_once()
            assert result == {"Status": "ok"}

    def test_receive_batch_invalid_file(self):
        """Test receive_batch method with invalid file."""
        from agentic_rl.controllers.rollout_controller.rollout_server import HTTPException
        
        mock_queue = MagicMock()
        mock_queue.add_queue = AsyncMock()
        mock_weight_manager = MagicMock()

        mock_file = MagicMock()
        mock_file.read = AsyncMock(return_value=b"invalid_data")

        server = RolloutServer(mock_queue, mock_weight_manager)

        with patch('agentic_rl.controllers.rollout_controller.rollout_server.torch.load') as mock_load:
            mock_load.side_effect = RuntimeError("Invalid file format")

            with pytest.raises(HTTPException) as exc_info:
                _should_raise_exception = run_async(server.receive_batch(mock_file))
                del _should_raise_exception

            assert exc_info.value.status_code == HTTP_ERROR_400
            assert "Invalid torch file" in exc_info.value.detail

    def test_handle_weights_update(self):
        """Test handle_weights_update method."""
        mock_queue = MagicMock()
        mock_weight_manager = MagicMock()
        mock_weight_manager.sync_weights_update = MagicMock()

        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=b"/path/to/weights")

        server = RolloutServer(mock_queue, mock_weight_manager)

        result = run_async(server.handle_weights_update(mock_request))

        mock_request.body.assert_called_once()
        mock_weight_manager.sync_weights_update.remote.assert_called_once_with("/path/to/weights")
        assert result == {"received": "/path/to/weights"}

    def test_shutdown(self):
        """Test shutdown method."""
        mock_queue = MagicMock()
        mock_queue.shutdown = AsyncMock()
        mock_weight_manager = MagicMock()

        server = RolloutServer(mock_queue, mock_weight_manager)

        result = run_async(server.shutdown())

        assert server.is_shutdown is True
        mock_queue.shutdown.remote.assert_called_once()
        assert result == {"Status": "ok"}

    def test_router_routes(self):
        """Test that router has all expected routes."""
        mock_queue = MagicMock()
        mock_weight_manager = MagicMock()

        server = RolloutServer(mock_queue, mock_weight_manager)

        assert server.router is not None
        assert hasattr(server.router, 'post')


@pytest.fixture(autouse=True)
def ensure_mock():
    """Ensure fastapi mock exists before each test."""
    if 'fastapi' not in sys.modules:
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
    yield


@pytest.fixture(scope="module", autouse=True)
def cleanup_module():
    """Cleanup mock modules after all tests in this module."""
    yield
    import sys
    modules_to_clean = ['fastapi']
    for mod in modules_to_clean:
        if mod in sys.modules:
            del sys.modules[mod]
