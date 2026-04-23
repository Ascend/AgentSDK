#!/usr/bin/env python3
# coding=utf-8
# -------------------------------------------------------------------------# This file is part of the AgentSDK project.
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
import pytest
import unittest.mock as mock

from agentic_rl.base.utils.http_server import start_server


class TestHttpServer:
    @mock.patch("agentic_rl.base.utils.http_server.uvicorn.run")
    @mock.patch("agentic_rl.base.utils.http_server.logger")
    def test_start_server_default_params(self, mock_logger, mock_uvicorn_run):
        """Test start_server with default parameters."""
        # Create mock server application
        mock_app = mock.Mock()
        
        # Call function
        server_name = "TestServer"
        start_server(server_name, mock_app)
        
        # Verify log recording
        mock_logger.info.assert_called_once_with(f"Start {server_name}: 0.0.0.0:8000")
        
        # Verify uvicorn.run call
        mock_uvicorn_run.assert_called_once_with(
            mock_app,
            host='0.0.0.0',
            port=8000,
            log_level="info",
            reload=False
        )

    @mock.patch("agentic_rl.base.utils.http_server.uvicorn.run")
    @mock.patch("agentic_rl.base.utils.http_server.logger")
    def test_start_server_custom_params(self, mock_logger, mock_uvicorn_run):
        """Test start_server with custom parameters."""
        # Create mock server application
        mock_app = mock.Mock()
        
        # Call function with custom parameters
        server_name = "CustomServer"
        server_host = "127.0.0.1"
        server_port = 9000
        start_server(server_name, mock_app, server_host=server_host, server_port=server_port)
        
        # Verify log recording
        mock_logger.info.assert_called_once_with(f"Start {server_name}: {server_host}:{server_port}")
        
        # Verify uvicorn.run call
        mock_uvicorn_run.assert_called_once_with(
            mock_app,
            host=server_host,
            port=server_port,
            log_level="info",
            reload=False
        )

    @mock.patch("agentic_rl.base.utils.http_server.uvicorn.run")
    @mock.patch("agentic_rl.base.utils.http_server.logger")
    def test_start_server_port_as_string(self, mock_logger, mock_uvicorn_run):
        """Test start_server with port as string."""
        # Create mock server application
        mock_app = mock.Mock()
        
        # Call function with port as string
        server_name = "TestServer"
        server_port = "8080"
        start_server(server_name, mock_app, server_port=server_port)
        
        # Verify log recording
        mock_logger.info.assert_called_once_with(f"Start {server_name}: 0.0.0.0:{server_port}")
        
        # Verify uvicorn.run call, port should be converted to integer
        mock_uvicorn_run.assert_called_once_with(
            mock_app,
            host='0.0.0.0',
            port=8080,
            log_level="info",
            reload=False
        )

    @mock.patch("agentic_rl.base.utils.http_server.uvicorn.run")
    @mock.patch("agentic_rl.base.utils.http_server.logger")
    def test_start_server_exception_handling(self, mock_logger, mock_uvicorn_run):
        """Test start_server exception handling."""
        # Create mock server application
        mock_app = mock.Mock()
        
        # Mock uvicorn.run to raise exception
        mock_uvicorn_run.side_effect = Exception("Server startup failed")
        
        # Call function should propagate exception
        server_name = "TestServer"
        with pytest.raises(Exception, match="Server startup failed"):
            start_server(server_name, mock_app)
        
        # Verify log recording still occurs
        mock_logger.info.assert_called_once_with(f"Start {server_name}: 0.0.0.0:8000")