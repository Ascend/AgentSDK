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

import os
import sys
import unittest
from unittest import mock


# Add the project root to sys.path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from agentic_rl.base.misc.misc import (
    colorful_print,
    colorful_warning,
    get_image,
    pad_from_left,
    merge_dicts,
    RequestIdStat,
    AppIdStat,
    ScheduleStat,
    AppStats,
    app_stats
)


class TestMiscFunctions(unittest.TestCase):
    """Test cases for miscellaneous functions."""

    def test_colorful_print(self):
        """Test colorful_print function."""
        with mock.patch('builtins.print') as mock_print:
            colorful_print("test", fg="red")
            mock_print.assert_called_once()
            args, kwargs = mock_print.call_args
            self.assertEqual(len(args), 1)
            self.assertIsInstance(args[0], str)
            self.assertEqual(kwargs.get('end'), '\n')
            self.assertEqual(kwargs.get('flush'), True)
            mock_print.reset_mock()
            
            # Test with custom end parameter
            colorful_print("test", fg="blue", end="!")
            mock_print.assert_called_once()
            _, kwargs = mock_print.call_args
            self.assertEqual(kwargs.get('end'), '!')
            
            # Test with bold style
            mock_print.reset_mock()
            colorful_print("test", bold=True)
            mock_print.assert_called_once()

    def test_colorful_warning(self):
        """Test colorful_warning function."""
        with mock.patch('warnings.warn') as mock_warn:
            colorful_warning("test warning", fg="yellow")
            mock_warn.assert_called_once()
            args, kwargs = mock_warn.call_args
            self.assertEqual(len(args), 1)
            self.assertIsInstance(args[0], str)
            self.assertEqual(kwargs.get('stacklevel'), 2)
            
            # Test with multiple styles
            mock_warn.reset_mock()
            colorful_warning("error warning", fg="red", bold=True)
            mock_warn.assert_called_once()

    @mock.patch('PIL.Image.open')
    def test_get_image(self, mock_open):
        """Test get_image function."""
        # Setup mock
        mock_img = mock.Mock()
        mock_img.convert.return_value = "converted_image"
        mock_open.return_value.__enter__.return_value = mock_img
        
        # Test normal case
        result = get_image("test_image.jpg")
        
        # Verify
        mock_open.assert_called_once_with("test_image.jpg")
        mock_img.convert.assert_called_once_with("RGB")
        self.assertEqual(result, "converted_image")
        
        # Test with different image formats
        mock_open.reset_mock()
        mock_img.reset_mock()
        get_image("test_image.png")
        mock_open.assert_called_once_with("test_image.png")
        
        # Test exception case
        mock_open.reset_mock()
        mock_open.side_effect = FileNotFoundError("File not found")
        with self.assertRaises(FileNotFoundError):
            get_image("non_existent.jpg")

    def test_pad_from_left(self):
        """Test pad_from_left function."""
        # Test with batch size > 1
        input_id_list = [[1, 2, 3], [4, 5], [6]]
        pad_token_id = 0
        result = pad_from_left(input_id_list, pad_token_id)
        # The first list is already the maximum length, so it shouldn't be padded
        expected = [[1, 2, 3], [0, 4, 5], [0, 0, 6]]
        self.assertEqual(result, expected)

        # Test with batch size = 1 (should add random padding)
        input_id_list = [[1, 2, 3]]
        pad_token_id = 0
        with mock.patch('random.randint', return_value=5):
            result = pad_from_left(input_id_list, pad_token_id)
            expected = [[0, 0, 0, 0, 0, 1, 2, 3]]
            self.assertEqual(result, expected)
        
        # Test with empty list
        input_id_list = []
        pad_token_id = 0
        # The function should raise ValueError when input list is empty
        with self.assertRaises(ValueError):
            pad_from_left(input_id_list, pad_token_id)
        
        # Test with all lists same length
        input_id_list = [[1, 2], [3, 4], [5, 6]]
        pad_token_id = 0
        result = pad_from_left(input_id_list, pad_token_id)
        expected = [[1, 2], [3, 4], [5, 6]]
        self.assertEqual(result, expected)
        
        # Test with negative pad_token_id
        input_id_list = [[1, 2, 3], [4, 5]]
        pad_token_id = -1
        result = pad_from_left(input_id_list, pad_token_id)
        expected = [[1, 2, 3], [-1, 4, 5]]
        self.assertEqual(result, expected)

    def test_merge_dicts(self):
        """Test merge_dicts function."""
        dict_list = [
            {"a": 1, "b": 2},
            {"a": 3, "c": 4},
            {"b": 5, "d": 6}
        ]
        result = merge_dicts(dict_list)
        expected = {
            "a": [1, 3],
            "b": [2, 5],
            "c": [4],
            "d": [6]
        }
        self.assertEqual(result, expected)

        # Test with empty list
        result = merge_dicts([])
        self.assertEqual(result, {})

        # Test with single dict
        result = merge_dicts([{"a": 1, "b": 2}])
        expected = {"a": [1], "b": [2]}
        self.assertEqual(result, expected)
        
        # Test with nested dicts
        dict_list = [
            {"nested": {"a": 1}},
            {"nested": {"b": 2}}
        ]
        result = merge_dicts(dict_list)
        expected = {"nested": [{"a": 1}, {"b": 2}]}
        self.assertEqual(result, expected)
        
        # Test with different value types
        dict_list = [
            {"num": 1, "str": "a"},
            {"num": 2, "list": [1, 2]}
        ]
        result = merge_dicts(dict_list)
        expected = {
            "num": [1, 2],
            "str": ["a"],
            "list": [[1, 2]]
        }
        self.assertEqual(result, expected)


class TestRequestIdStat(unittest.TestCase):
    """Test cases for RequestIdStat class."""

    def test_init(self):
        """Test RequestIdStat initialization."""
        # Test with req_id
        req_stat = RequestIdStat("test_req_id")
        self.assertEqual(req_stat.req_id, "test_req_id")
        self.assertEqual(req_stat.address, "")
        self.assertEqual(req_stat.step_idx, 0)
        self.assertEqual(req_stat.prompt_len, 0)
        self.assertEqual(req_stat.terminal_reason, "")
        self.assertIsInstance(req_stat.route_tick, float)
        self.assertEqual(req_stat.vllm_start_tick, 0)
        self.assertEqual(req_stat.vllm_end_tick, 0)
        self.assertEqual(req_stat.env_start_tick, 0)
        self.assertEqual(req_stat.env_end_tick, 0)
        
        # Test without req_id (default)
        req_stat2 = RequestIdStat()
        self.assertEqual(req_stat2.req_id, "")

    def test_to_dict(self):
        """Test RequestIdStat.to_dict method."""
        # Test with all fields set
        req_stat = RequestIdStat("test_req_id")
        req_stat.address = "127.0.0.1:8000"
        req_stat.step_idx = 5
        req_stat.prompt_len = 100
        req_stat.terminal_reason = "max_steps"
        req_stat.vllm_start_tick = 1.0
        req_stat.vllm_end_tick = 2.0
        req_stat.env_start_tick = 3.0
        req_stat.env_end_tick = 4.0
        
        result = req_stat.to_dict()
        
        expected = {
            "request_id": "test_req_id",
            "address": "127.0.0.1:8000",
            "step_idx": 5,
            "prompt_len": 100,
            "terminal_reason": "max_steps",
            "route_tick": req_stat.route_tick,
            "vllm_start_tick": 1.0,
            "vllm_end_tick": 2.0,
            "env_start_tick": 3.0,
            "env_end_tick": 4.0,
            "vllm_delay": 1.0,
            "env_delay": 1.0
        }
        
        self.assertEqual(result, expected)
        
        # Test with minimum fields set (only req_id)
        req_stat2 = RequestIdStat("test_req_id_2")
        result2 = req_stat2.to_dict()
        self.assertEqual(result2["request_id"], "test_req_id_2")
        self.assertEqual(result2["vllm_delay"], 0.0)  # start and end are 0, so delay is 0
        self.assertEqual(result2["env_delay"], 0.0)  # start and end are 0, so delay is 0


class TestAppIdStat(unittest.TestCase):
    """Test cases for AppIdStat class."""

    def test_init(self):
        """Test AppIdStat initialization."""
        # Test with app_id
        app_stat = AppIdStat("test_app_id")
        self.assertEqual(app_stat.app_id, "test_app_id")
        self.assertEqual(app_stat.req_stats, {})
        self.assertIsNone(app_stat.trajectory_id)
        self.assertEqual(app_stat.total_vllm_delay, 0)
        self.assertEqual(app_stat.total_env_delay, 0)
        self.assertEqual(app_stat.total_delay, 0)
        
        # Test without app_id (default)
        app_stat2 = AppIdStat()
        self.assertEqual(app_stat2.app_id, "")

    def test_stat_route(self):
        """Test AppIdStat.stat_route method."""
        app_stat = AppIdStat("test_app_id")
        app_stat.stat_route("test_req_id", "127.0.0.1:8000", 100)
        
        self.assertIn("test_req_id", app_stat.req_stats)
        self.assertEqual(app_stat.req_stats["test_req_id"].address, "127.0.0.1:8000")
        self.assertEqual(app_stat.req_stats["test_req_id"].prompt_len, 100)
        
        # Test updating existing request
        app_stat.stat_route("test_req_id", "127.0.0.1:8001", 200)
        self.assertEqual(app_stat.req_stats["test_req_id"].address, "127.0.0.1:8001")
        self.assertEqual(app_stat.req_stats["test_req_id"].prompt_len, 200)

    def test_stat_vllm_step(self):
        """Test AppIdStat.stat_vllm_step method."""
        app_stat = AppIdStat("test_app_id")
        app_stat.stat_vllm_step("test_req_id", 5, 1.0, 2.0)
        
        self.assertIn("test_req_id", app_stat.req_stats)
        self.assertEqual(app_stat.req_stats["test_req_id"].step_idx, 5)
        self.assertEqual(app_stat.req_stats["test_req_id"].vllm_start_tick, 1.0)
        self.assertEqual(app_stat.req_stats["test_req_id"].vllm_end_tick, 2.0)
        self.assertEqual(app_stat.total_vllm_delay, 1.0)
        
        # Test with multiple vllm steps
        app_stat.stat_vllm_step("test_req_id", 6, 3.0, 5.0)
        self.assertEqual(app_stat.req_stats["test_req_id"].step_idx, 6)  # Should be updated
        self.assertEqual(app_stat.total_vllm_delay, 3.0)  # 1.0 + 2.0
        
        # Test with new request
        app_stat.stat_vllm_step("test_req_id_2", 1, 0.0, 1.0)
        self.assertEqual(app_stat.total_vllm_delay, 4.0)  # 3.0 + 1.0

    def test_stat_env_step(self):
        """Test AppIdStat.stat_env_step method."""
        app_stat = AppIdStat("test_app_id")
        app_stat.stat_env_step("test_req_id", 5, 3.0, 4.0, "max_steps")
        
        self.assertIn("test_req_id", app_stat.req_stats)
        self.assertEqual(app_stat.req_stats["test_req_id"].step_idx, 5)
        self.assertEqual(app_stat.req_stats["test_req_id"].env_start_tick, 3.0)
        self.assertEqual(app_stat.req_stats["test_req_id"].env_end_tick, 4.0)
        self.assertEqual(app_stat.req_stats["test_req_id"].terminal_reason, "max_steps")
        self.assertEqual(app_stat.total_env_delay, 1.0)
        
        # Test with multiple env steps
        app_stat.stat_env_step("test_req_id", 6, 5.0, 7.0, "success")
        self.assertEqual(app_stat.req_stats["test_req_id"].step_idx, 6)  # Should be updated
        self.assertEqual(app_stat.req_stats["test_req_id"].terminal_reason, "success")  # Should be updated
        self.assertEqual(app_stat.total_env_delay, 3.0)  # 1.0 + 2.0

    def test_stat_env_state(self):
        """Test AppIdStat.stat_env_state method."""
        app_stat = AppIdStat("test_app_id")
        app_stat.stat_env_state("test_req_id", "success")
        
        self.assertIn("test_req_id", app_stat.req_stats)
        self.assertEqual(app_stat.req_stats["test_req_id"].terminal_reason, "success")
        
        # Test updating existing request
        app_stat.stat_env_state("test_req_id", "timeout")
        self.assertEqual(app_stat.req_stats["test_req_id"].terminal_reason, "timeout")

    def test_stat_trajectory(self):
        """Test AppIdStat.stat_trajectory method."""
        app_stat = AppIdStat("test_app_id")
        app_stat.stat_trajectory("test_trajectory_id")
        
        self.assertEqual(app_stat.trajectory_id, "test_trajectory_id")
        
        # Test updating trajectory_id
        app_stat.stat_trajectory("test_trajectory_id_2")
        self.assertEqual(app_stat.trajectory_id, "test_trajectory_id_2")

    def test_to_dict(self):
        """Test AppIdStat.to_dict method."""
        # Test with multiple requests
        app_stat = AppIdStat("test_app_id")
        app_stat.stat_trajectory("test_trajectory_id")
        
        # First request
        app_stat.stat_route("test_req_id_1", "127.0.0.1:8000", 100)
        app_stat.stat_vllm_step("test_req_id_1", 5, 1.0, 2.0)
        app_stat.stat_env_step("test_req_id_1", 5, 3.0, 4.0, "max_steps")
        
        # Second request
        app_stat.stat_route("test_req_id_2", "127.0.0.1:8001", 200)
        app_stat.stat_vllm_step("test_req_id_2", 1, 0.0, 1.0)
        app_stat.stat_env_step("test_req_id_2", 1, 2.0, 3.0, "success")
        
        result = app_stat.to_dict()
        
        self.assertEqual(result["application_id"], "test_app_id")
        self.assertEqual(result["total_vllm_delay"], 2.0)  # 1.0 + 1.0
        self.assertEqual(result["total_env_delay"], 2.0)  # 1.0 + 1.0
        self.assertEqual(result["total_delay"], 4.0)  # 2.0 + 2.0
        self.assertEqual(result["request_count"], 2)
        self.assertEqual(result["trajectory_id"], "test_trajectory_id")
        self.assertEqual(len(result["requests"]), 2)
        
        # Test with minimum fields
        app_stat2 = AppIdStat("test_app_id_2")
        result2 = app_stat2.to_dict()
        self.assertEqual(result2["request_count"], 0)
        self.assertEqual(result2["total_vllm_delay"], 0)
        self.assertEqual(result2["total_env_delay"], 0)
        self.assertEqual(result2["total_delay"], 0)


class TestScheduleStat(unittest.TestCase):
    """Test cases for ScheduleStat class."""

    def test_init(self):
        """Test ScheduleStat initialization."""
        # Test with address
        schedule_stat = ScheduleStat("127.0.0.1:8000")
        self.assertEqual(schedule_stat.address, "127.0.0.1:8000")
        self.assertEqual(schedule_stat.reqs, [])
        self.assertEqual(schedule_stat.total_prompt_len, 0)
        
        # Test without address (default)
        schedule_stat2 = ScheduleStat()
        self.assertEqual(schedule_stat2.address, "")

    def test_stat_add(self):
        """Test ScheduleStat.stat_add method."""
        schedule_stat = ScheduleStat("127.0.0.1:8000")
        schedule_stat.stat_add("test_req_id_1", 100)
        schedule_stat.stat_add("test_req_id_2", 200)
        
        self.assertEqual(schedule_stat.reqs, ["test_req_id_1", "test_req_id_2"])
        self.assertEqual(schedule_stat.total_prompt_len, 300)
        
        # Test with multiple calls for same request (should add multiple times)
        schedule_stat.stat_add("test_req_id_1", 50)
        self.assertEqual(schedule_stat.reqs, ["test_req_id_1", "test_req_id_2", "test_req_id_1"])
        self.assertEqual(schedule_stat.total_prompt_len, 350)
        
        # Test with zero prompt length
        schedule_stat.stat_add("test_req_id_3", 0)
        self.assertEqual(schedule_stat.reqs, ["test_req_id_1", "test_req_id_2", "test_req_id_1", "test_req_id_3"])
        self.assertEqual(schedule_stat.total_prompt_len, 350)

    def test_to_dict(self):
        """Test ScheduleStat.to_dict method."""
        # Test with multiple requests
        schedule_stat = ScheduleStat("127.0.0.1:8000")
        schedule_stat.stat_add("test_req_id_1", 100)
        schedule_stat.stat_add("test_req_id_2", 200)
        
        result = schedule_stat.to_dict()
        expected = {
            "address": "127.0.0.1:8000",
            "processed_tokens": 300,
            "request_count": 2,
            "requests": ["test_req_id_1", "test_req_id_2"]
        }
        
        self.assertEqual(result, expected)
        
        # Test with empty stats
        schedule_stat2 = ScheduleStat("127.0.0.1:8001")
        result2 = schedule_stat2.to_dict()
        expected2 = {
            "address": "127.0.0.1:8001",
            "processed_tokens": 0,
            "request_count": 0,
            "requests": []
        }
        self.assertEqual(result2, expected2)


class TestAppStats(unittest.TestCase):
    """Test cases for AppStats class."""

    def setUp(self):
        """Set up test fixtures."""
        # Ensure we start with a clean instance for each test
        AppStats._instance = None
        # Save original environment variable
        self.original_env = os.environ.get('GTS_STATS_ENABLE')

    def tearDown(self):
        """Clean up test fixtures."""
        # Restore original environment variable
        if self.original_env is not None:
            os.environ['GTS_STATS_ENABLE'] = self.original_env
        else:
            os.environ.pop('GTS_STATS_ENABLE', None)
        # Reset singleton instance
        AppStats._instance = None

    def test_singleton(self):
        """Test AppStats is a singleton."""
        instance1 = AppStats()
        instance2 = AppStats()
        self.assertIs(instance1, instance2)

    def test_get_request_id(self):
        """Test AppStats.get_request_id method."""
        req_id = AppStats.get_request_id("test_app_id", 5)
        self.assertEqual(req_id, "cmpl-test_app_id--5-0")

    @mock.patch('agentic_rl.base.misc.misc.AppStats.enabled', True)
    def test_stat_route(self):
        """Test AppStats.stat_route method."""
        app_stats = AppStats()
        app_stats.stat_route("test_app_id", "test_req_id", "127.0.0.1:8000", 100)
        
        self.assertIn("test_app_id", app_stats.appid_stats)
        self.assertIn("cmpl-test_req_id-0", app_stats.appid_stats["test_app_id"].req_stats)
        self.assertIn("127.0.0.1:8000", app_stats.dp_stats)
        self.assertIn("cmpl-test_req_id-0", app_stats.dp_stats["127.0.0.1:8000"].reqs)
        
        # Test with another address
        app_stats.stat_route("test_app_id", "test_req_id_2", "127.0.0.1:8001", 200)
        self.assertIn("127.0.0.1:8001", app_stats.dp_stats)
        self.assertIn("cmpl-test_req_id_2-0", app_stats.dp_stats["127.0.0.1:8001"].reqs)

    @mock.patch('agentic_rl.base.misc.misc.AppStats.enabled', True)
    def test_stat_vllm_step(self):
        """Test AppStats.stat_vllm_step method."""
        app_stats = AppStats()
        app_stats.stat_vllm_step("test_app_id", 5, 1.0, 2.0)
        
        self.assertIn("test_app_id", app_stats.appid_stats)
        self.assertIn("cmpl-test_app_id--5-0", app_stats.appid_stats["test_app_id"].req_stats)
        
        # Test with multiple steps
        app_stats.stat_vllm_step("test_app_id", 6, 3.0, 5.0)
        self.assertIn("cmpl-test_app_id--6-0", app_stats.appid_stats["test_app_id"].req_stats)

    @mock.patch('agentic_rl.base.misc.misc.AppStats.enabled', True)
    def test_stat_env_step(self):
        """Test AppStats.stat_env_step method."""
        app_stats = AppStats()
        app_stats.stat_env_step("test_app_id", 5, 3.0, 4.0, "max_steps")
        
        self.assertIn("test_app_id", app_stats.appid_stats)
        self.assertIn("cmpl-test_app_id--5-0", app_stats.appid_stats["test_app_id"].req_stats)
        
        # Test with another request
        app_stats.stat_env_step("test_app_id_2", 1, 0.0, 1.0, "success")
        self.assertIn("test_app_id_2", app_stats.appid_stats)

    @mock.patch('agentic_rl.base.misc.misc.AppStats.enabled', True)
    def test_stat_env_state(self):
        """Test AppStats.stat_env_state method."""
        app_stats = AppStats()
        app_stats.stat_env_state("test_app_id", 5, "success")
        
        self.assertIn("test_app_id", app_stats.appid_stats)
        self.assertIn("cmpl-test_app_id--5-0", app_stats.appid_stats["test_app_id"].req_stats)
        
        # Test updating existing state
        app_stats.stat_env_state("test_app_id", 5, "timeout")
        req_stat = app_stats.appid_stats["test_app_id"].req_stats["cmpl-test_app_id--5-0"]
        self.assertEqual(req_stat.terminal_reason, "timeout")

    @mock.patch('agentic_rl.base.misc.misc.AppStats.enabled', True)
    def test_stat_trajectory(self):
        """Test AppStats.stat_trajectory method."""
        app_stats = AppStats()
        app_stats.appid_stats["test_app_id"] = AppIdStat("test_app_id")
        app_stats.stat_trajectory("test_app_id", "test_trajectory_id")
        
        self.assertEqual(app_stats.appid_stats["test_app_id"].trajectory_id, "test_trajectory_id")
        
        # Test updating trajectory_id
        app_stats.stat_trajectory("test_app_id", "updated_trajectory_id")
        self.assertEqual(app_stats.appid_stats["test_app_id"].trajectory_id, "updated_trajectory_id")
    
    @mock.patch('agentic_rl.base.misc.misc.AppStats.enabled', False)
    def test_disabled_stats(self):
        """Test that no stats are collected when AppStats.enabled is False."""
        app_stats = AppStats()
        app_stats.stat_route("test_app_id", "test_req_id", "127.0.0.1:8000", 100)
        app_stats.stat_vllm_step("test_app_id", 5, 1.0, 2.0)
        app_stats.stat_env_step("test_app_id", 5, 3.0, 4.0, "max_steps")
        
        # No stats should be collected
        self.assertEqual(app_stats.appid_stats, {})
        self.assertEqual(app_stats.dp_stats, {})

    def test_clear(self):
        """Test AppStats.clear method."""
        app_stats = AppStats()
        app_stats.appid_stats["test_app_id"] = AppIdStat("test_app_id")
        app_stats.dp_stats["127.0.0.1:8000"] = ScheduleStat("127.0.0.1:8000")
        
        app_stats.clear()
        
        self.assertEqual(app_stats.appid_stats, {})
        self.assertEqual(app_stats.dp_stats, {})

    @mock.patch('agentic_rl.base.misc.misc.AppStats.enabled', True)
    @mock.patch('os.makedirs')
    @mock.patch('builtins.open', new_callable=mock.mock_open)
    @mock.patch('json.dumps')
    @mock.patch('time.time', return_value=1234567890.0)
    def test_print(self, mock_time, mock_json_dumps, mock_open, mock_makedirs):
        """Test AppStats.print method."""
        # Setup mock datetime using a different approach
        mock_date = mock.Mock()
        mock_date.strftime.return_value = "20260409"
        
        # Mock the entire datetime.datetime class
        with mock.patch('datetime.datetime', autospec=True) as mock_datetime_class:
            mock_datetime_class.now.return_value = mock_date
            
            # Setup test data
            app_stats = AppStats()
            app_stats.appid_stats["test_app_id"] = AppIdStat("test_app_id")
            app_stats.dp_stats["127.0.0.1:8000"] = ScheduleStat("127.0.0.1:8000")
            
            # Test
            app_stats.print(inner_iter=10)
            
            # Verify
            mock_makedirs.assert_called_once()
            mock_open.assert_called_once()
            mock_json_dumps.assert_called_once()
            # Check that clear was called
            self.assertEqual(app_stats.appid_stats, {})
            self.assertEqual(app_stats.dp_stats, {})
    
    @mock.patch('agentic_rl.base.misc.misc.AppStats.enabled', True)
    @mock.patch('os.makedirs')
    @mock.patch('builtins.open', new_callable=mock.mock_open)
    @mock.patch('json.dumps')
    @mock.patch('time.time', return_value=1234567890.0)
    def test_print_with_exception(self, mock_time, mock_json_dumps, mock_open, mock_makedirs):
        """Test AppStats.print method with exception."""
        # Setup mock datetime
        mock_date = mock.Mock()
        mock_date.strftime.return_value = "20260409"
        
        with mock.patch('datetime.datetime', autospec=True) as mock_datetime_class:
            mock_datetime_class.now.return_value = mock_date
            
            # Setup test data
            app_stats = AppStats()
            app_stats.appid_stats["test_app_id"] = AppIdStat("test_app_id")
            
            # Mock json.dumps to raise exception
            mock_json_dumps.side_effect = Exception("JSON serialization error")
            
            # Test should not raise exception
            app_stats.print(inner_iter=10)
            
            # Verify clear was still called
            self.assertEqual(app_stats.appid_stats, {})
            self.assertEqual(app_stats.dp_stats, {})
    
    @mock.patch('agentic_rl.base.misc.misc.AppStats.enabled', False)
    @mock.patch('os.makedirs')
    @mock.patch('builtins.open', new_callable=mock.mock_open)
    def test_print_disabled(self, mock_open, mock_makedirs):
        """Test AppStats.print method when disabled."""
        app_stats = AppStats()
        app_stats.appid_stats["test_app_id"] = AppIdStat("test_app_id")
        
        # Test
        app_stats.print(inner_iter=10)
        
        # Verify no file operations were called
        mock_makedirs.assert_not_called()
        mock_open.assert_not_called()
        # Verify stats were not cleared
        self.assertIn("test_app_id", app_stats.appid_stats)

    def test_enabled_env_variable(self):
        """Test AppStats.enabled property with environment variable."""
        import importlib
        from agentic_rl.base.misc import misc
        
        # Test with GTS_STATS_ENABLE=0
        os.environ['GTS_STATS_ENABLE'] = "0"
        importlib.reload(misc)
        AppStats = misc.AppStats
        AppStats._instance = None  # Reset singleton
        app_stats = AppStats()
        self.assertFalse(AppStats.enabled)
        
        # Test with GTS_STATS_ENABLE=1
        os.environ['GTS_STATS_ENABLE'] = "1"
        importlib.reload(misc)
        AppStats = misc.AppStats
        AppStats._instance = None  # Reset singleton
        app_stats = AppStats()
        self.assertTrue(AppStats.enabled)
        
        # Test with GTS_STATS_ENABLE not set
        os.environ.pop('GTS_STATS_ENABLE')
        importlib.reload(misc)
        AppStats = misc.AppStats
        AppStats._instance = None  # Reset singleton
        app_stats = AppStats()
        self.assertTrue(AppStats.enabled)


class TestAppStatsGlobalInstance(unittest.TestCase):
    """Test cases for the global app_stats instance."""

    def setUp(self):
        """Set up test fixtures."""
        # Import the module once and keep a reference
        import agentic_rl.base.misc.misc
        import importlib
        self.misc_module = agentic_rl.base.misc.misc
        # Reload to ensure clean state
        importlib.reload(self.misc_module)
        # Save original instance
        self.original_instance = self.misc_module.AppStats._instance

    def tearDown(self):
        """Clean up test fixtures."""
        # Restore original instance
        self.misc_module.AppStats._instance = self.original_instance

    def test_global_instance(self):
        """Test that the global app_stats is an instance of AppStats."""
        # Use the same module reference throughout the test
        app_stats = self.misc_module.app_stats
        AppStats = self.misc_module.AppStats
        
        self.assertIsInstance(app_stats, AppStats)
        
        # Test singleton behavior by checking that the _instance attribute is set
        self.assertIsNotNone(AppStats._instance)
        self.assertIs(app_stats, AppStats._instance)


if __name__ == '__main__':
    unittest.main()