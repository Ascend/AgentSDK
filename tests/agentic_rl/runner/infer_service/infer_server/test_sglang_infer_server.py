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

from agentic_rl.runner.infer_service.infer_server.sglang_infer_server import SGLangInferServer


class TestSGLangInferServer:
    def setup_method(self):
        self.server = SGLangInferServer()
    
    @pytest.mark.asyncio
    async def test_completions(self):
        result = await self.server.completions({})
        assert result is None
    
    @pytest.mark.asyncio
    async def test_chat_completions(self):
        result = await self.server.chat_completions({})
        assert result is None
    
    @pytest.mark.asyncio
    async def test_stream_chat_completions(self):
        result = await self.server.stream_chat_completions({})
        assert result is None
    
    @pytest.mark.asyncio
    @patch('agentic_rl.runner.infer_service.base_infer_server.logger')
    async def test_launch_server(self, mock_logger):
        await self.server.launch_server()
        mock_logger.info.assert_called_once_with("BaseInferServer launch")
    
    @pytest.mark.asyncio
    @patch('agentic_rl.runner.infer_service.base_infer_server.logger')
    async def test_wake_up(self, mock_logger):
        await self.server.wake_up()
        mock_logger.info.assert_called_once_with("BaseInferServer wake_up")
    
    @pytest.mark.asyncio
    @patch('agentic_rl.runner.infer_service.base_infer_server.logger')
    async def test_sleep(self, mock_logger):
        await self.server.sleep()
        mock_logger.info.assert_called_once_with("BaseInferServer sleep")
