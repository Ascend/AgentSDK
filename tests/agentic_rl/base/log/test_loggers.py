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
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import logging
import os
import unittest
from unittest import mock

from agentic_rl.base.log.loggers import generate_iteration_msg, handle_msg, Loggers


class TestLoggers(unittest.TestCase):
    def setUp(self):
        # Save original environment variables for restoration in tearDown
        self.original_rank = os.environ.get('RANK')
        self.original_world_size = os.environ.get('WORLD_SIZE')
        # Clear any existing environment variables
        if 'RANK' in os.environ:
            del os.environ['RANK']
        if 'WORLD_SIZE' in os.environ:
            del os.environ['WORLD_SIZE']

    def tearDown(self):
        # Restore original environment variables
        if self.original_rank is not None:
            os.environ['RANK'] = self.original_rank
        elif 'RANK' in os.environ:
            del os.environ['RANK']

        if self.original_world_size is not None:
            os.environ['WORLD_SIZE'] = self.original_world_size
        elif 'WORLD_SIZE' in os.environ:
            del os.environ['WORLD_SIZE']

    def test_generate_iteration_msg_string(self):
        """Test generate_iteration_msg function with string type msg"""
        msg = "test message"
        iteration = 1
        steps = 10
        expected = "iteration: 1 / 10 |  test message"
        result = generate_iteration_msg(msg, iteration, steps)
        self.assertEqual(result, expected)

    def test_generate_iteration_msg_dict(self):
        """Test generate_iteration_msg function with dict type msg"""
        msg = {"loss": 0.123456, "acc": 0.987654}
        iteration = 2
        steps = 20
        expected = "iteration: 2 / 20 | loss : 0.1235 | acc : 0.9877 "
        result = generate_iteration_msg(msg, iteration, steps)
        self.assertEqual(result, expected)

    def test_generate_iteration_msg_dict_with_lr(self):
        """Test generate_iteration_msg function with dict type msg containing 'param/lr' key"""
        msg = {"loss": 0.123456, "param/lr": 0.000123456}
        iteration = 3
        steps = 30
        expected = "iteration: 3 / 30 | loss : 0.1235 | param/lr : 1.234560e-04 "
        result = generate_iteration_msg(msg, iteration, steps)
        self.assertEqual(result, expected)

    def test_handle_msg_with_iteration(self):
        """Test handle_msg function when both iteration and steps are not None"""
        msg = "test message"
        iteration = 1
        steps = 10
        expected = "iteration: 1 / 10 |  test message"
        with mock.patch('agentic_rl.base.log.loggers.generate_iteration_msg', return_value=expected) as mock_generate:
            result = handle_msg(msg, iteration, steps)
            mock_generate.assert_called_once_with(msg, iteration, steps)
            self.assertEqual(result, expected)

    def test_handle_msg_without_iteration(self):
        """Test handle_msg function when either iteration or steps is None"""
        msg = "test message"
        iteration = None
        steps = 10
        expected = "test message"
        result = handle_msg(msg, iteration, steps)
        self.assertEqual(result, expected)

        msg = "test message"
        iteration = 1
        steps = None
        expected = "test message"
        result = handle_msg(msg, iteration, steps)
        self.assertEqual(result, expected)

        msg = "test message"
        iteration = None
        steps = None
        expected = "test message"
        result = handle_msg(msg, iteration, steps)
        self.assertEqual(result, expected)

    def test_loggers_init(self):
        """Test Loggers class initialization"""
        logger = Loggers(name="test_logger", logger_level=logging.DEBUG)
        self.assertEqual(logger.logger.name, "test_logger")
        self.assertEqual(logger.logger.level, logging.DEBUG)
        self.assertEqual(len(logger.logger.handlers), 1)
        self.assertIsInstance(logger.logger.handlers[0], logging.StreamHandler)
        self.assertEqual(logger.logger.propagate, False)

    def test_loggers_get_logger(self):
        """Test Loggers class get_logger method"""
        logger = Loggers(name="test_logger")
        self.assertEqual(logger.get_logger(), logger.logger)

    def test_format_info_not_distributed(self):
        """Test Loggers class format_info method when distributed is not initialized"""
        logger = Loggers(name="test_logger")
        msg = "test message"
        iteration = 1
        steps = 10
        expected = "iteration: 1 / 10 |  test message"

        with mock.patch('torch.distributed.is_initialized', return_value=False):
            with mock.patch.object(logger.logger, 'info') as mock_info:
                logger.format_info(msg, iteration, steps)
                mock_info.assert_called_once_with(expected)

    def test_format_info_distributed_not_last_rank(self):
        """Test Loggers class format_info method when distributed is initialized but not the last rank"""
        logger = Loggers(name="test_logger")
        msg = "test message"
        iteration = 1
        steps = 10

        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '2'

        with mock.patch('torch.distributed.is_initialized', return_value=True):
            with mock.patch('torch.distributed.get_rank', return_value=0):
                with mock.patch('torch.distributed.get_world_size', return_value=2):
                    with mock.patch.object(logger.logger, 'info') as mock_info:
                        logger.format_info(msg, iteration, steps)
                        mock_info.assert_not_called()

    def test_format_info_distributed_last_rank(self):
        """Test Loggers class format_info method when distributed is initialized and it's the last rank"""
        logger = Loggers(name="test_logger")
        msg = "test message"
        iteration = 1
        steps = 10
        expected = "iteration: 1 / 10 |  test message"

        os.environ['RANK'] = '1'
        os.environ['WORLD_SIZE'] = '2'

        with mock.patch('torch.distributed.is_initialized', return_value=True):
            with mock.patch('torch.distributed.get_rank', return_value=1):
                with mock.patch('torch.distributed.get_world_size', return_value=2):
                    with mock.patch.object(logger.logger, 'info') as mock_info:
                        logger.format_info(msg, iteration, steps)
                        mock_info.assert_called_once_with(expected)

    def test_format_warning(self):
        """Test Loggers class format_warning method"""
        logger = Loggers(name="test_logger")
        msg = "test warning"
        iteration = 1
        steps = 10
        expected = "iteration: 1 / 10 |  test warning"

        with mock.patch.object(logger.logger, 'warning') as mock_warning:
            logger.format_warning(msg, iteration, steps)
            mock_warning.assert_called_once_with(expected)

    def test_format_debug(self):
        """Test Loggers class format_debug method"""
        logger = Loggers(name="test_logger", logger_level=logging.DEBUG)
        msg = "test debug"
        iteration = 1
        steps = 10
        expected = "iteration: 1 / 10 |  test debug"

        with mock.patch.object(logger.logger, 'debug') as mock_debug:
            logger.format_debug(msg, iteration, steps)
            mock_debug.assert_called_once_with(expected)

    def test_format_error(self):
        """Test Loggers class format_error method"""
        logger = Loggers(name="test_logger")
        msg = "test error"
        iteration = 1
        steps = 10
        expected = "iteration: 1 / 10 |  test error"

        with mock.patch.object(logger.logger, 'error') as mock_error:
            logger.format_error(msg, iteration, steps)
            mock_error.assert_called_once_with(expected)


if __name__ == '__main__':
    unittest.main()
