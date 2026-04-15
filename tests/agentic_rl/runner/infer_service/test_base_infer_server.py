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
import asyncio
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True, scope="function")
def mock_dependencies():
    """Mock all external dependencies for base_infer_server tests."""
    with patch("agentic_rl.runner.infer_service.base_infer_server.logger") as mock_logger:
        yield {
            "logger": mock_logger,
        }

class TestBaseInferServer:
    def setup_method(self):
        from agentic_rl.runner.infer_service.base_infer_server import BaseInferServer
        self.server = BaseInferServer()

    @pytest.mark.asyncio
    async def test_completions_no_exception(self):
        await self.server.completions({})

    @pytest.mark.asyncio
    async def test_chat_completions_no_exception(self):
        await self.server.chat_completions({})

    @pytest.mark.asyncio
    async def test_stream_chat_completions_no_exception(self):
        await self.server.stream_chat_completions({})

    @pytest.mark.asyncio
    async def test_launch_server(self, mock_dependencies):
        await self.server.launch_server()
        mock_dependencies["logger"].info.assert_called_once_with("BaseInferServer launch")

    @pytest.mark.asyncio
    async def test_wake_up(self, mock_dependencies):
        await self.server.wake_up()
        mock_dependencies["logger"].info.assert_called_once_with("BaseInferServer wake_up")

    @pytest.mark.asyncio
    async def test_sleep(self, mock_dependencies):
        await self.server.sleep()
        mock_dependencies["logger"].info.assert_called_once_with("BaseInferServer sleep")

    @pytest.mark.asyncio
    async def test_collective_rpc(self, mock_dependencies):
        await self.server.collective_rpc("test_method")
        mock_dependencies["logger"].info.assert_called_once_with("BaseInferServer collective_rpc")

    @pytest.mark.asyncio
    async def test_collective_rpc_with_args(self, mock_dependencies):
        await self.server.collective_rpc("test_method", timeout=10.0, args=(1, 2, 3), kwargs={"key": "value"})
        mock_dependencies["logger"].info.assert_called_once_with("BaseInferServer collective_rpc")

    @pytest.mark.asyncio
    async def test_collective_rpc_with_callable(self, mock_dependencies):
        async def dummy():
            pass

        await self.server.collective_rpc(dummy)
        mock_dependencies["logger"].info.assert_called_once_with("BaseInferServer collective_rpc")

    @pytest.mark.asyncio
    async def test_collective_rpc_kwargs_none(self, mock_dependencies):
        await self.server.collective_rpc("test_method", kwargs=None)
        mock_dependencies["logger"].info.assert_called_once_with("BaseInferServer collective_rpc")

