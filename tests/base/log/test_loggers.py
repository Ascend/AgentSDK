#!/usr/bin/env python3
# coding=utf-8
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

AgentSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""
import os
from unittest import mock

import pytest

from agentic_rl.base.log.loggers import Loggers


class TestLoggersInfo:

    @pytest.fixture
    def logger(self):
        logger = Loggers("agentic_rl")
        for h in logger.logger.handlers:
            logger.logger.removeHandler(h)
        logger.logger.propagate = True
        yield logger

    def test_info_log_success(
            self, logger, caplog):
        with caplog.at_level('INFO', logger="agentic_rl"):
            logger.info('test message', 1, 10)

        assert "INFO     agentic_rl:test_loggers.py" in caplog.text
        assert "iteration: 1 / 10 |  test message" in caplog.text

    def test_debug_log_success(self, logger, caplog):
        with caplog.at_level('DEBUG', logger="agentic_rl"):
            logger.debug("test message", 1, 10)

        assert "DEBUG    agentic_rl:test_loggers.py" in caplog.text
        assert "iteration: 1 / 10 |  test message" in caplog.text

    def test_warning_log_success(self, logger, caplog):
        with caplog.at_level('WARNING', logger="agentic_rl"):
            logger.warning("test message", 1, 10)

        assert "WARNING  agentic_rl:test_loggers.py" in caplog.text
        assert "iteration: 1 / 10 |  test message" in caplog.text

    def test_error_log_success(self, logger, caplog):
        with caplog.at_level('ERROR', logger="agentic_rl"):
            logger.error("test message", 1, 10)

        assert "ERROR    agentic_rl:test_loggers.py" in caplog.text
        assert "iteration: 1 / 10 |  test message" in caplog.text

    @mock.patch('torch.distributed.is_initialized', return_value=True)
    @mock.patch('torch.distributed.get_rank', return_value=0)
    @mock.patch('torch.distributed.get_world_size', return_value=2)
    def test_info_distributed_non_last_rank(self, mock_world_size, mock_rank, mock_is_initialized, logger, caplog):
        with caplog.at_level('INFO', logger="agentic_rl"):
            logger.info("test message", 1, 10)

        assert caplog.text == ""

    @mock.patch('torch.distributed.is_initialized', return_value=True)
    @mock.patch('torch.distributed.get_rank', return_value=1)
    @mock.patch('torch.distributed.get_world_size', return_value=2)
    def test_info_distributed_last_rank(self, mock_world_size, mock_rank, mock_is_initialized, logger, caplog):
        with caplog.at_level('INFO', logger="agentic_rl"):
            logger.info("test message", 1, 10)

        assert "INFO     agentic_rl:test_loggers.py" in caplog.text
        assert "iteration: 1 / 10 |  test message" in caplog.text

    @mock.patch('torch.distributed.is_initialized', return_value=True)
    @mock.patch.dict(os.environ, {'RANK': 'not_an_integer', 'WORLD_SIZE': '2'})
    def test_info_invalid_rank_env(self, mock_is_initialized, logger, caplog):
        with pytest.raises(ValueError) as excinfo:
            logger.info("test message", 1, 10)

        assert "RANK environment variable must be an integer" in str(excinfo.value)

    def test_filter_invalid_chars(self):
        test_cases = [
            # Test case 1: String with no invalid characters
            {
                'input': 'Hello, World!',
                'expected': 'Hello, World!'
            },
            # Test case 2: String with one invalid character
            {
                'input': 'Hello,\nWorld!',
                'expected': 'Hello, World!'
            },
            # Test case 3: String with multiple invalid characters
            {
                'input': 'Hello,\nWorld!\r',
                'expected': 'Hello, World! '
            },
            # Test case 4: String with all invalid characters
            {
                'input': '\n\f\r\b\t\v\u000D\u000A\u000C\u000B\u0009\u0008\u0007',
                'expected': ' '
            },
            # Test case 5: Empty string
            {
                'input': '',
                'expected': ''
            },
        ]

        for test_case in test_cases:
            assert Loggers._filter_invalid_chars(test_case['input']) == test_case['expected']
