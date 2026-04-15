#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import importlib
import importlib.util
import os
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open


class TestRequestStat(unittest.TestCase):
    """Test the RequestStat class"""

    @classmethod
    def setUpClass(cls):
        cls._import_module_under_test()

    @classmethod
    def _import_module_under_test(cls):
        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(test_file_dir, '..', '..', '..', '..', '..', '..', '..'))
        module_path = os.path.join(project_root, 'agentic_rl', 'runner', 'infer_adapter', 'vllm', 'patch', 'comm', 'scheduler_stat.py')
        spec = importlib.util.spec_from_file_location('scheduler_stat', module_path)
        cls.scheduler_stat = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.scheduler_stat)
        cls.RequestStat = cls.scheduler_stat.RequestStat

    def test_init_default(self):
        """Test default initialization"""
        stat = self.RequestStat()
        self.assertEqual(stat.request_id, "")
        self.assertEqual(stat.add_tick, 0)
        self.assertEqual(stat.schedule_tick, 0)
        self.assertEqual(stat.prefill_done_tick, 0)
        self.assertEqual(stat.finish_tick, 0)
        self.assertEqual(stat.prompt_len, 0)
        self.assertEqual(stat.output_len, 0)

    def test_init_with_parameters(self):
        """Test initialization with parameters"""
        request_id = "test_request_123"
        add_tick = 123456789.0
        schedule_tick = 123456790.0
        prefill_done_tick = 123456791.0
        finish_tick = 123456792.0

        stat = self.RequestStat(
            request_id=request_id,
            add_tick=add_tick,
            schedule_tick=schedule_tick,
            prefill_done_tick=prefill_done_tick,
            finish_tick=finish_tick
        )

        self.assertEqual(stat.request_id, request_id)
        self.assertEqual(stat.add_tick, add_tick)
        self.assertEqual(stat.schedule_tick, schedule_tick)
        self.assertEqual(stat.prefill_done_tick, prefill_done_tick)
        self.assertEqual(stat.finish_tick, finish_tick)
        self.assertEqual(stat.prompt_len, 0)
        self.assertEqual(stat.output_len, 0)

    def test_to_dict(self):
        """Test conversion to dictionary"""
        request_id = "test_request_123"
        add_tick = 123456789.0
        schedule_tick = 123456790.0
        prefill_done_tick = 123456791.0
        finish_tick = 123456792.0
        prompt_len = 100
        output_len = 200

        stat = self.RequestStat(
            request_id=request_id,
            add_tick=add_tick,
            schedule_tick=schedule_tick,
            prefill_done_tick=prefill_done_tick,
            finish_tick=finish_tick
        )
        stat.prompt_len = prompt_len
        stat.output_len = output_len

        result = stat.to_dict()
        expected = {
            "request_id": request_id,
            "add_tick": add_tick,
            "schedule_tick": schedule_tick,
            "prefill_done_tick": prefill_done_tick,
            "finish_tick": finish_tick,
            "prompt_len": prompt_len,
            "output_len": output_len
        }

        self.assertEqual(result, expected)


class TestRequestStats(unittest.TestCase):
    """Test the RequestStats class"""

    @classmethod
    def setUpClass(cls):
        cls._import_module_under_test()

    @classmethod
    def _import_module_under_test(cls):
        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(test_file_dir, '..', '..', '..', '..', '..', '..', '..'))
        module_path = os.path.join(project_root, 'agentic_rl', 'runner', 'infer_adapter', 'vllm', 'patch', 'comm', 'scheduler_stat.py')
        spec = importlib.util.spec_from_file_location('scheduler_stat', module_path)
        cls.scheduler_stat = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.scheduler_stat)
        cls.RequestStats = cls.scheduler_stat.RequestStats

    def _reload_module_with_env(self):
        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(test_file_dir, '..', '..', '..', '..', '..', '..', '..'))
        module_path = os.path.join(project_root, 'agentic_rl', 'runner', 'infer_adapter', 'vllm', 'patch', 'comm', 'scheduler_stat.py')
        spec = importlib.util.spec_from_file_location('scheduler_stat', module_path)
        scheduler_stat = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scheduler_stat)
        return scheduler_stat

    def setUp(self):
        self.original_gts_stats_enable = os.environ.get("GTS_STATS_ENABLE")
        self.original_index = self.RequestStats.index_
        self.RequestStats.index_ = 0

    def tearDown(self):
        if self.original_gts_stats_enable is None:
            if "GTS_STATS_ENABLE" in os.environ:
                del os.environ["GTS_STATS_ENABLE"]
        else:
            os.environ["GTS_STATS_ENABLE"] = self.original_gts_stats_enable
        self.RequestStats.index_ = self.original_index

    def test_init(self):
        """Test initialization"""
        stats = self.RequestStats()
        self.assertEqual(len(stats.req_stats), 0)

    @patch.dict(os.environ, {"GTS_STATS_ENABLE": "1"})
    def test_statistics_enabled(self):
        """Test that statistics are enabled"""
        scheduler_stat = self._reload_module_with_env()
        self.assertTrue(scheduler_stat.RequestStats.enabled)

    @patch.dict(os.environ, {"GTS_STATS_ENABLE": "0"})
    def test_statistics_disabled(self):
        """Test that statistics are disabled"""
        scheduler_stat = self._reload_module_with_env()
        self.assertFalse(scheduler_stat.RequestStats.enabled)

    @patch.dict(os.environ, {"GTS_STATS_ENABLE": "1"})
    @patch('time.time', return_value=123456789.0)
    def test_stat_add(self, mock_time):
        """Test adding request statistics"""
        scheduler_stat = self._reload_module_with_env()
        stats = scheduler_stat.RequestStats()
        request_id = "test_request_123"

        stats.stat_add(request_id)

        self.assertIn(request_id, stats.req_stats)
        self.assertEqual(stats.req_stats[request_id].add_tick, 123456789.0)
        mock_time.assert_called_once()

    @patch.dict(os.environ, {"GTS_STATS_ENABLE": "1"})
    @patch('time.time', return_value=123456789.0)
    def test_stat_schedule(self, mock_time):
        """Test recording request schedule time"""
        scheduler_stat = self._reload_module_with_env()
        stats = scheduler_stat.RequestStats()
        request_id = "test_request_123"

        stats.stat_add(request_id)
        mock_time.reset_mock()
        mock_time.return_value = 123456790.0

        stats.stat_schedule(request_id)

        self.assertEqual(stats.req_stats[request_id].schedule_tick, 123456790.0)
        mock_time.assert_called_once()

    @patch.dict(os.environ, {"GTS_STATS_ENABLE": "1"})
    @patch('time.time', return_value=123456789.0)
    def test_stat_prefill_done(self, mock_time):
        """Test recording prefill completion time"""
        scheduler_stat = self._reload_module_with_env()
        stats = scheduler_stat.RequestStats()
        request_id = "test_request_123"

        stats.stat_add(request_id)
        mock_time.reset_mock()
        mock_time.return_value = 123456791.0

        stats.stat_prefill_done(request_id)

        self.assertEqual(stats.req_stats[request_id].prefill_done_tick, 123456791.0)
        mock_time.assert_called_once()

    @patch.dict(os.environ, {"GTS_STATS_ENABLE": "1"})
    @patch('time.time', return_value=123456789.0)
    def test_stat_finish(self, mock_time):
        """Test recording request completion time and lengths"""
        scheduler_stat = self._reload_module_with_env()
        stats = scheduler_stat.RequestStats()
        request_id = "test_request_123"
        prompt_len = 100
        output_len = 200

        stats.stat_add(request_id)
        mock_time.reset_mock()
        mock_time.return_value = 123456792.0

        stats.stat_finish(request_id, prompt_len=prompt_len, output_len=output_len)

        self.assertEqual(stats.req_stats[request_id].finish_tick, 123456792.0)
        self.assertEqual(stats.req_stats[request_id].prompt_len, prompt_len)
        self.assertEqual(stats.req_stats[request_id].output_len, output_len)
        mock_time.assert_called_once()

    @patch.dict(os.environ, {"GTS_STATS_ENABLE": "1"})
    def test_reset(self):
        """Test reset functionality"""
        scheduler_stat = self._reload_module_with_env()
        stats = scheduler_stat.RequestStats()
        request_id = "test_request_123"

        stats.stat_add(request_id)
        self.assertEqual(len(stats.req_stats), 1)

        stats.reset()
        self.assertEqual(len(stats.req_stats), 0)

    @patch.dict(os.environ, {"GTS_STATS_ENABLE": "1"})
    @patch('os.makedirs')
    @patch('os.path.exists', return_value=False)
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('os.getpid', return_value=12345)
    @patch('time.time', return_value=123456789.0)
    def test_print(self, mock_time, mock_getpid, mock_json_dump, mock_open_file, mock_exists, mock_makedirs):
        """Test printing and saving statistics"""
        mock_date = MagicMock()
        mock_date.strftime.return_value = "20230402"

        with patch('datetime.datetime') as mock_datetime_class:
            mock_datetime_class.now.return_value = mock_date

            scheduler_stat = self._reload_module_with_env()

            stats = scheduler_stat.RequestStats()
            request_id = "test_request_123"
            stats.stat_add(request_id)
            stats.stat_schedule(request_id)
            stats.stat_prefill_done(request_id)
            stats.stat_finish(request_id, prompt_len=100, output_len=200)

            stats.print()

            mock_exists.assert_called_once()
            exists_call_args = mock_exists.call_args[0][0]
            self.assertIn('20230402', exists_call_args)
            self.assertIn('logs', exists_call_args)
            self.assertIn('vllm_schedule', exists_call_args)

            mock_makedirs.assert_called_once()
            makedirs_call_args = mock_makedirs.call_args[0][0]
            self.assertIn('20230402', makedirs_call_args)
            self.assertIn('logs', makedirs_call_args)
            self.assertIn('vllm_schedule', makedirs_call_args)
            self.assertEqual(mock_makedirs.call_args[1]['exist_ok'], True)

            mock_open_file.assert_called_once()
            open_call_args = mock_open_file.call_args[0][0]
            self.assertTrue(open_call_args.endswith('vllm_schedule_0_12345_123456789.json'))
            self.assertEqual(mock_open_file.call_args[0][1], 'w')

            mock_json_dump.assert_called_once()
            args, kwargs = mock_json_dump.call_args
            self.assertEqual(kwargs['indent'], 2)
            self.assertEqual(kwargs['ensure_ascii'], False)
            self.assertIn('timestamp', args[0])
            self.assertIn('request', args[0])
            self.assertIn(request_id, args[0]['request'])

            self.assertEqual(len(stats.req_stats), 0)
            self.assertEqual(scheduler_stat.RequestStats.index_, 1)

    @patch.dict(os.environ, {"GTS_STATS_ENABLE": "0"})
    def test_disabled_operations(self):
        """Test operations when statistics are disabled"""
        scheduler_stat = self._reload_module_with_env()
        stats = scheduler_stat.RequestStats()
        request_id = "test_request_123"

        stats.stat_add(request_id)
        stats.stat_schedule(request_id)
        stats.stat_prefill_done(request_id)
        stats.stat_finish(request_id, prompt_len=100, output_len=200)

        self.assertEqual(len(stats.req_stats), 0)


if __name__ == '__main__':
    unittest.main()
